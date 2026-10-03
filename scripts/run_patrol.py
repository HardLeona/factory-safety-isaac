"""순찰: 작업자가 정해진 경로를 걷고, 안전 에이전트가 바디캠 YOLO 판정, CCTV 거리 경고, CCTV 확대 재확인을 거쳐
조치 지시서를 만든다.

실행 (Isaac Sim 파이썬):
    <isaac>/python.sh scripts/run_patrol.py                        # 바디캠 화면, 한 바퀴
    <isaac>/python.sh scripts/run_patrol.py --view top             # 위에서 보는 관제 화면
    <isaac>/python.sh scripts/run_patrol.py --view ptz             # CCTV 확대(PTZ) 화면
    <isaac>/python.sh scripts/run_patrol.py --headless --sim-dt 0.0333 --seed 3    # 평가용 (같은 시드면 같은 결과)
    <isaac>/python.sh scripts/run_patrol.py --headless --sim-dt 0.0333 --seed 1 --yolo-every 3 --record outputs/record/seed1

끝나면 조치 지시서(outputs/agent/dashboard_seed<시드>.html)를 만들고,
정답표(outputs/eval/answer_key_seed<시드>.json)와 맞춘 채점 결과를 출력하고 저장한다.
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

parser = argparse.ArgumentParser(description="창고 안전 순찰 (바디캠 + CCTV + 에이전트 재확인)")
parser.add_argument("--weights", default=os.path.join(ROOT, "outputs", "yolo", "warehouse_v3", "weights", "best.pt"))
parser.add_argument("--seed", type=int, default=None, help="위험 요소 배치 시드 (비우면 무작위)")
parser.add_argument("--laps", type=float, default=1.0, help="몇 바퀴 돌지")
parser.add_argument("--view", default="bodycam", help="bodycam | top | ptz | cctv_west | cctv_east | cctv_south")
parser.add_argument("--speed", type=float, default=1.0, help="배속 (실시간 모드)")
parser.add_argument("--sim-dt", type=float, default=0.0, help="0보다 크면 프레임마다 이 시간만큼 진행 (평가 재현용)")
parser.add_argument("--yolo-every", type=int, default=6, help="몇 프레임마다 YOLO")
parser.add_argument("--no-cctv", action="store_true", help="CCTV 거리 측정과 확대 재확인 끄기")
parser.add_argument("--no-recheck", action="store_true", help="에이전트 재확인(CCTV 확대) 끄기 (비교용)")
parser.add_argument("--record", default=None, help="영상용: 바디캠, CCTV, 확대 화면과 에이전트 상태를 저장할 폴더")
parser.add_argument("--result", default=None, help="채점 결과 JSON 경로 (기본 outputs/eval/inspection_seed<시드>.json)")
parser.add_argument("--headless", action="store_true")
parser.add_argument("--sound", default="auto", help="음성 경고 소리: auto (창이 있을 때만), on, off")
parser.add_argument("--gestures", default="off", help="작업자 손동작 명령: off | demo (시연 순서대로 손가락 1~5 를 보임) | watch (손 인식만)")
parser.add_argument("--lang", default=None, help="작업자 언어 ko/en/zh/ja (쉼표로 여러 개면 명령마다 돌아가며). 기본 zh,en,ja, --story 면 en")
parser.add_argument("--story", action="store_true", help="시연 이야기: TBM → 카트 설명 → 상자 싣고 끌기 → 공장 스캔 → 한 바퀴 뒤 관리자 호출 (factory_safety/story.py)")
args, _ = parser.parse_known_args()
if args.story and args.gestures == "off":
    args.gestures = "watch"
if args.lang is None:
    args.lang = "en" if args.story else "zh,en,ja"

from factory_safety.isaac_utils import make_app  # noqa: E402

app = make_app(headless=args.headless)

import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402

from factory_safety import warehouse as W  # noqa: E402
from factory_safety.agent import SafetyAgent, action_summary, evaluation_summary  # noqa: E402
from factory_safety.config import CLASSES, IMG_H, IMG_W  # noqa: E402
from factory_safety.dashboard import write_dashboard  # noqa: E402
from factory_safety import overlay  # noqa: E402
from factory_safety.voice import ensure_voice, play_async, with_alarm  # noqa: E402
from factory_safety.detector import YoloDetector  # noqa: E402
from factory_safety.inspection import bodycam_summary, cctv_summary, object_id_from_path  # noqa: E402
from factory_safety.isaac_utils import (DebugBoxes, attach, get_annotator, isaac_labeler, new_stage,  # noqa: E402
                                        parse_bboxes, rgb_array, set_viewport_camera, set_viewport_top_view,
                                        timeline_setter)
from factory_safety.scenario import sample_scenario  # noqa: E402
from factory_safety.scene import WarehouseScene  # noqa: E402
from factory_safety.walker import PathWalker  # noqa: E402

if not os.path.exists(args.weights):
    sys.exit(f"[오류] YOLO 가중치가 없어요: {args.weights}\n       python scripts/train_yolo.py 로 먼저 학습하세요.")

# ---------------------------------------------------------------- 장면 (배치는 첫 렌더 전에)
stage = new_stage()
scene = WarehouseScene(stage, labeler=isaac_labeler, set_time=timeline_setter()).build()
PTZ_PATH = scene.add_camera("/World/CCTV/ptz")
seed = args.seed if args.seed is not None else int(np.random.default_rng().integers(1 << 30))
scenario = sample_scenario(seed)
scene.set_scenario(scenario)
eval_dir = os.path.join(ROOT, "outputs", "eval")
agent_dir = os.path.join(ROOT, "outputs", "agent")
for d in (eval_dir, agent_dir, args.record):
    if d:
        os.makedirs(d, exist_ok=True)
key_path = os.path.join(eval_dir, f"answer_key_seed{seed}.json")
scenario.save_answer_key(key_path)
n_h = len(scenario.hazards)
print(f"[시나리오] 시드 {seed}: 물체 {len(scenario.objects)}개 (위험 {n_h}, 안전 {len(scenario.objects) - n_h}), 정답표 {key_path}")

walker = PathWalker(seed=seed)
story = None
if args.story:
    from factory_safety.story import Story
    story = Story(walker)
    story.build(scene)          # 카트, 팔레트, 상자 (첫 렌더 전에)
cctv_names = [] if args.no_cctv else [c[0] for c in W.CCTVS]
agent = SafetyAgent(IMG_W, IMG_H, cctv_names)
if args.no_recheck:
    agent.JOB_DELAY_S = 1e9
for n, pose in agent.cctvs.items():
    scene.set_camera(pose, path=scene.cctv_paths[n])
if cctv_names:
    scene.set_camera(agent.aim(cctv_names[0], np.array([0.0, 0.0, 0.0]), 8.0), path=PTZ_PATH)

# 카메라 출력: 바디캠 + CCTV + 확대(PTZ), 각각 화면과 정답 박스
outs = {}
cams = [("bodycam", scene.cam_path)] + [(n, scene.cctv_paths[n]) for n in cctv_names] + ([("ptz", PTZ_PATH)] if cctv_names else [])
for name, path in cams:
    rp = rep.create.render_product(path, (IMG_W, IMG_H))
    a_rgb, a_box = get_annotator("rgb"), get_annotator("bounding_box_2d_tight")
    attach(a_rgb, rp)
    attach(a_box, rp)
    outs[name] = (a_rgb, a_box)
if args.view == "top":
    set_viewport_top_view(stage)
elif args.view == "ptz":
    set_viewport_camera(PTZ_PATH)
elif args.view in scene.cctv_paths:
    set_viewport_camera(scene.cctv_paths[args.view])
else:
    set_viewport_camera(scene.cam_path)
boxes3d = DebugBoxes()
yolo = YoloDetector(args.weights)
RT_SUBFRAMES = 2   # Isaac Sim 6.0: 1 이면 어노테이터에 한 단계 전 화면이 들어 있음
RT_SUBFRAMES_PTZ = 6   # 확대 카메라를 크게 돌린 직후에는 이전 화면 잔상이 남아서 더 여러 번 렌더
agent.plan(walker.path.length)
VOICE = ensure_voice()
SOUND = args.sound == "on" or (args.sound == "auto" and not args.headless)
if SOUND:
    agent.on_voice = lambda t: play_async(VOICE)

# 손동작 명령 (손 인식·다국어 음성 도우미는 따로 띄운 .venv-assistant 프로세스)
assistant = client = demo = None
if args.gestures != "off":
    from factory_safety.assistant import DemoScript, SiteAssistant, gesture_eval
    from factory_safety.assistant_client import AssistantClient
    client = AssistantClient()
    if client.ok:
        assistant = SiteAssistant(agent, langs=[v.strip() for v in args.lang.split(",") if v.strip()])
        demo = DemoScript() if args.gestures == "demo" else None
        print(f"[손동작] 손 인식 + 다국어 안내 켬 (작업자 언어 {args.lang})")
    else:
        print(f"[손동작] 도우미를 못 띄워서 끔: {client.error}")
gest = {"pending": None, "n_ev": 0, "pts": None, "count": 0}


def speak(t, ev):
    """명령 안내를 작업자 언어 음성으로 (SOS 는 경보음 먼저)."""
    wav, dur = client.tts(ev["text"], ev["lang"])
    if wav and ev["count"] == 5:
        wav, dur = with_alarm(wav), dur + 1.35
    ev["wav"], ev["dur"] = wav, round(dur, 2)
    if wav and SOUND:
        play_async(wav)
    if demo:
        demo.said(t, dur, ev["count"])
    if story:
        story.said(t, dur, ev["count"])
    print(f"[손동작] {ev['count']} {ev['cmd']} ({ev['lang_name']}, {dur:.1f}초): {ev['text']}")


def draw_cctv(img, dets, meas, title, cam):
    from PIL import ImageDraw
    im = overlay.draw(img, dets, lambda b: agent._tool_state(cam, b), size=14, width=2)
    d = ImageDraw.Draw(im)
    d.text((8, 8), title, fill=(255, 255, 255))
    for k, (wp, n, hp, dist) in enumerate(meas["pairs"] if meas else []):
        d.text((8, 24 + 14 * k), f"worker - {n}: {dist:.1f} m" + ("  !!" if dist < agent.prox.warn else ""),
               fill=(255, 80, 80) if dist < agent.prox.warn else (200, 255, 200))
    return im


def record_state(k, t, x, y, yaw, cam, ptz, sosv=None):
    if not args.record:
        return
    st = {"k": k, "t": round(t, 3), "worker": [x, y, yaw], "bodycam": [float(v) for v in cam.pos[:2]] + [cam.yaw, cam.vfov],
          "ptz": ({"cam": ptz[0], "target": [float(v) for v in agent.job["target"]], "vfov": ptz[1].vfov,
                   "finding": agent.job["finding"].fid if agent.job["finding"] else None,
                   "checkpoint": agent.job["checkpoint"].cid if agent.job["checkpoint"] else None} if ptz and agent.job else
                  {"cam": sosv[0], "vfov": sosv[1].vfov, "sos": sosv[2] == "SOS", "label": sosv[2],
                   "finding": None, "checkpoint": None} if sosv else None),
          "story": story.label if story else None,
          "n_timeline": len(agent.timeline),
          "hand": {"alpha": round(walker.gesture_alpha, 2), "shown": walker.gesture["count"] if walker.gesture else 0,
                   "count": gest["count"], "pts": gest["pts"]} if assistant else None,
          "n_assist": len(assistant.events) if assistant else 0,
          "zones": [{"id": z.zid, "source": z.source, "poly": [[round(float(a), 2), round(float(b), 2)] for a, b in z.poly]}
                    for z in agent.zones.values()],
          "voice": [v["t"] for v in agent.voice_events if v["t"] > t - 0.05],
          "findings": [{"id": f.fid, "x": float(f.xy[0]), "y": float(f.xy[1]), "cls": f.cls, "status": f.status,
                        "near_miss": f.near_miss} for f in agent.findings]}
    with open(os.path.join(args.record, "state.jsonl"), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(st, ensure_ascii=False) + "\n")


def run_ptz(t, k, ptz):
    """확대 촬영 한 장을 읽어 에이전트에 넘긴다 (렌더는 이미 끝난 상태)."""
    p_rgb, _ = outs["ptz"]
    pimg = rgb_array(p_rgb.get_data())
    if pimg is None:
        return
    pdets, pres = yolo(pimg)
    pose = agent.job["pose"] if agent.job else None
    agent.on_ptz(t, pdets)
    if args.record:
        im = overlay.draw(pimg, pdets, (lambda b: agent._tool_state(pose, b)) if pose is not None else None, size=20)
        im.save(os.path.join(args.record, f"ptz_{k:05d}.jpg"), quality=88)


if args.record:
    open(os.path.join(args.record, "state.jsonl"), "w").close()

# ---------------------------------------------------------------- 순찰 루프
print(f"[안내] 순찰 시작: 경로 {walker.path.length:.0f} m, {args.laps:g}바퀴. 창을 닫거나 Ctrl+C로 끝낼 수 있어요.")
t_sim, frame, step_k, last = 0.0, 0, 0, time.time()
x = y = yaw = 0.0
try:
    lap_flag = False
    while app.is_running() and (not story.done if story else walker.laps < args.laps):
        now = time.time()
        dt = args.sim_dt if args.sim_dt > 0 else min(0.05, now - last) * args.speed
        last = now
        if dt <= 0:
            app.update()
            continue
        t_sim += dt
        cam = walker.step(dt)
        if story:
            story.update(t_sim)
            story.apply()
            cam = walker.camera()
            if story.lap_done and not lap_flag and cctv_names and not args.no_recheck:
                agent.finish_patrol(t_sim)      # 한 바퀴 끝: 남은 점검 지점을 CCTV 확대로 (작업자가 관리자를 부르는 동안)
                lap_flag = True
        x, y, yaw, at = walker.base_pose
        scene.set_worker(x, y, yaw, at)
        scene.set_camera(cam)
        if frame % args.yolo_every != 0 or frame < 10:
            app.update()
            frame += 1
            continue
        sosv = assistant.ptz_view(t_sim) if assistant else None
        ptz = None if sosv else (agent.next_ptz(t_sim) if cctv_names else None)
        if ptz or sosv:
            scene.set_camera((ptz or sosv)[1], path=PTZ_PATH)
        # Isaac Sim 6.0: app.update() 만으로는 어노테이터가 비어서 나와서 Replicator 로 렌더
        rep.orchestrator.step(delta_time=0.0, rt_subframes=RT_SUBFRAMES_PTZ if (ptz or sosv) else RT_SUBFRAMES)
        a_rgb, a_box = outs["bodycam"]
        img = rgb_array(a_rgb.get_data())
        agent.log_worker(t_sim, (x, y))
        if img is not None:
            dets, res = yolo.track(img, "bodycam")
            agent.on_bodycam(t_sim, cam, dets, worker_xy=(x, y), gt=parse_bboxes(a_box.get_data(), CLASSES, with_paths=True))
            if assistant:
                if walker.gesture is None:
                    assistant.observe(t_sim, cam, dets)       # 손으로 가린 화면은 장비·위험 안내에 안 씀
                gest["count"], gest["pts"] = client.hand(img)
                ev = assistant.on_hand(t_sim, gest["count"], cam, (x, y), yaw, gest["pts"])
                if ev:
                    speak(t_sim, ev)
                if demo and walker.gesture is None and gest["pending"] is None:
                    c = demo.next(t_sim, walker.laps / args.laps, agent, cam, dets, (x, y), True)
                    if c:
                        walker.start_gesture(c, hold=demo.HOLD)
                        gest["pending"], gest["n_ev"] = t_sim, len(assistant.events)
                        print(f"[손동작] (작업자) 바디캠 앞에 손가락 {c}개")
            if args.record:
                im = overlay.draw(img, dets, lambda b: agent._tool_state(cam, b), size=19)
                if assistant and gest["pts"]:
                    last = assistant.events[-1] if assistant.events and t_sim - assistant.events[-1]["t"] < 2.5 else None
                    overlay.draw_hand(im, gest["pts"], gest["count"], last["cmd"] if last and last["count"] == gest["count"] else None)
                im.save(os.path.join(args.record, f"body_{step_k:05d}.jpg"), quality=88)
        if demo and gest["pending"] is not None and walker.gesture is None:
            if len(assistant.events) == gest["n_ev"]:
                demo.missed(t_sim)
                print("[손동작] 인식 못 함")
            gest["pending"] = None
        lines = []
        for n in cctv_names:
            c_rgb, c_box = outs[n]
            cimg = rgb_array(c_rgb.get_data())
            if cimg is None:
                continue
            cdets, _ = yolo(cimg)
            events, meas = agent.on_cctv(t_sim, n, agent.cctvs[n], cdets)
            vis = {object_id_from_path(b[6]) for b in parse_bboxes(c_box.get_data(), CLASSES, with_paths=True)}
            agent.prox.score_frame(t_sim, n, meas, (x, y), scenario.hazards, vis)
            for wp, nm, hp, dist in meas["pairs"]:
                if dist < agent.prox.warn:
                    lines.append((np.array([wp[0], wp[1], 0.05]), np.array([hp[0], hp[1], 0.05])))
            if args.record:
                draw_cctv(cimg, cdets, meas, n, agent.cctvs[n]).save(os.path.join(args.record, f"{n}_{step_k:05d}.jpg"), quality=85)
        if ptz:
            run_ptz(t_sim, step_k, ptz)
        elif sosv and args.record:
            # SOS·공장 스캔 확대 화면 (스캔은 YOLO 박스도 그려서 무엇을 비추는지 보이게)
            simg = rgb_array(outs["ptz"][0].get_data())
            if simg is not None:
                sdets = yolo(simg)[0] if sosv[2] != "SOS" else []
                overlay.draw(simg, sdets, lambda b: agent._tool_state(sosv[1], b), size=20).save(
                    os.path.join(args.record, f"ptz_{step_k:05d}.jpg"), quality=88)
        record_state(step_k, t_sim, x, y, yaw, cam, ptz, sosv)
        for z in agent.zones.values():
            pts = [np.array([a, b, 0.03]) for a, b in z.poly]
            lines += [(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]
        if boxes3d.draw is not None:
            boxes3d.draw.clear_lines()
            if lines:
                boxes3d.draw.draw_lines([tuple(a) for a, b in lines], [tuple(b) for a, b in lines],
                                        [(1.0, 0.2, 0.2, 1.0)] * len(lines), [6.0] * len(lines))
        frame += 1
        step_k += 1

    # ---------------------------------------------------------------- 순찰 끝: 남은 재확인과 점검표
    if cctv_names and not args.no_recheck and app.is_running():
        if not story:
            agent.finish_patrol(t_sim)
        guard = 0
        while agent.busy() and guard < 200 and app.is_running():
            ptz = agent.next_ptz(t_sim)
            if ptz is None:
                break
            scene.set_camera(ptz[1], path=PTZ_PATH)
            rep.orchestrator.step(delta_time=0.0, rt_subframes=RT_SUBFRAMES_PTZ)
            run_ptz(t_sim, step_k, ptz)
            record_state(step_k, t_sim, x, y, yaw, walker.camera(), ptz)
            t_sim += 0.4
            step_k += 1
            guard += 1
except KeyboardInterrupt:
    pass

# ---------------------------------------------------------------- 조치 지시서와 채점
key = scenario.answer_key()
report = agent.report()
evaluation = agent.evaluate(key)
if assistant:
    report["assistant"] = assistant.report()
    if demo or story:
        evaluation["gestures"] = gesture_eval((demo or story).shown, assistant.events)
        g = evaluation["gestures"]
        print(f"[손동작] 보인 손동작 {g['shown']}번 중 맞게 인식 {g['recognized']}, 틀림 {g['wrong']}, 못 함 {g['missed']}, 잘못 실행 {g['extra']}")
    client.close()
print("\n" + action_summary(report))
rows, summ = agent.scorer.report(key)
print("\n" + bodycam_summary(rows, summ))
cs = agent.prox.report() if cctv_names else None
if cs:
    print(cctv_summary(cs))
print(evaluation_summary(evaluation))
dash = write_dashboard(os.path.join(agent_dir, f"dashboard_seed{seed}.html"), report, evaluation, seed=seed)
out = args.result or os.path.join(eval_dir, f"inspection_seed{seed}.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump({"seed": seed, "sim_time": t_sim, "bodycam": summ, "objects": rows, "cctv": cs,
               "agent": {"evaluation": evaluation, "report": {k: v for k, v in report.items() if k != "timeline"}}},
              f, ensure_ascii=False, indent=1)
with open(os.path.join(agent_dir, f"report_seed{seed}.json"), "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=1)
if args.record:
    with open(os.path.join(args.record, "final.json"), "w", encoding="utf-8") as f:
        json.dump({"seed": seed, "report": report, "evaluation": evaluation, "bodycam": summ, "cctv": cs and
                   {k: v for k, v in cs.items() if k != "log"}}, f, ensure_ascii=False, indent=1)
print(f"[저장] 조치 지시서 {dash}\n[저장] 채점 결과 {out}")
app.close()
