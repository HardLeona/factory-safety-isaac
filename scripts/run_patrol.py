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
parser.add_argument("--weights", default=os.path.join(ROOT, "outputs", "yolo", "warehouse", "weights", "best.pt"))
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
args, _ = parser.parse_known_args()

from factory_safety.isaac_utils import make_app  # noqa: E402

app = make_app(headless=args.headless)

import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402

from factory_safety import warehouse as W  # noqa: E402
from factory_safety.agent import SafetyAgent, action_summary, evaluation_summary  # noqa: E402
from factory_safety.config import CLASSES, IMG_H, IMG_W  # noqa: E402
from factory_safety.dashboard import write_dashboard  # noqa: E402
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


def draw_cctv(img, dets, meas, title):
    from PIL import Image, ImageDraw
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    for n, c, b, *_ in dets:
        d.rectangle(b, outline=(255, 255, 255) if n == "worker" else (255, 60, 60), width=2)
        d.text((b[0] + 2, b[1] - 11), f"{n} {c:.2f}", fill=(255, 255, 0))
    d.text((8, 8), title, fill=(255, 255, 255))
    for k, (wp, n, hp, dist) in enumerate(meas["pairs"] if meas else []):
        d.text((8, 24 + 14 * k), f"worker - {n}: {dist:.1f} m" + ("  !!" if dist < agent.prox.warn else ""),
               fill=(255, 80, 80) if dist < agent.prox.warn else (200, 255, 200))
    return im


def record_state(k, t, x, y, yaw, cam, ptz):
    if not args.record:
        return
    st = {"k": k, "t": round(t, 3), "worker": [x, y, yaw], "bodycam": [float(v) for v in cam.pos[:2]] + [cam.yaw, cam.vfov],
          "ptz": ({"cam": ptz[0], "target": [float(v) for v in agent.job["target"]], "vfov": ptz[1].vfov,
                   "finding": agent.job["finding"].fid if agent.job["finding"] else None,
                   "checkpoint": agent.job["checkpoint"].cid if agent.job["checkpoint"] else None} if ptz and agent.job else None),
          "n_timeline": len(agent.timeline),
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
    agent.on_ptz(t, pdets)
    if args.record:
        from PIL import Image
        Image.fromarray(pres.plot()[..., ::-1]).save(os.path.join(args.record, f"ptz_{k:05d}.jpg"), quality=88)


if args.record:
    open(os.path.join(args.record, "state.jsonl"), "w").close()

# ---------------------------------------------------------------- 순찰 루프
print(f"[안내] 순찰 시작: 경로 {walker.path.length:.0f} m, {args.laps:g}바퀴. 창을 닫거나 Ctrl+C로 끝낼 수 있어요.")
t_sim, frame, step_k, last = 0.0, 0, 0, time.time()
x = y = yaw = 0.0
try:
    while app.is_running() and walker.laps < args.laps:
        now = time.time()
        dt = args.sim_dt if args.sim_dt > 0 else min(0.05, now - last) * args.speed
        last = now
        if dt <= 0:
            app.update()
            continue
        t_sim += dt
        cam = walker.step(dt)
        x, y, yaw, at = walker.base_pose
        scene.set_worker(x, y, yaw, at)
        scene.set_camera(cam)
        if frame % args.yolo_every != 0 or frame < 10:
            app.update()
            frame += 1
            continue
        ptz = agent.next_ptz(t_sim) if cctv_names else None
        if ptz:
            scene.set_camera(ptz[1], path=PTZ_PATH)
        # Isaac Sim 6.0: app.update() 만으로는 어노테이터가 비어서 나와서 Replicator 로 렌더
        rep.orchestrator.step(delta_time=0.0, rt_subframes=RT_SUBFRAMES_PTZ if ptz else RT_SUBFRAMES)
        a_rgb, a_box = outs["bodycam"]
        img = rgb_array(a_rgb.get_data())
        if img is not None:
            dets, res = yolo.track(img, "bodycam")
            agent.on_bodycam(t_sim, cam, dets)
            agent.scorer.score_frame(dets, parse_bboxes(a_box.get_data(), CLASSES, with_paths=True))
            if args.record:
                from PIL import Image
                Image.fromarray(res.plot()[..., ::-1]).save(os.path.join(args.record, f"body_{step_k:05d}.jpg"), quality=88)
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
                draw_cctv(cimg, cdets, meas, n).save(os.path.join(args.record, f"{n}_{step_k:05d}.jpg"), quality=85)
        if ptz:
            run_ptz(t_sim, step_k, ptz)
        record_state(step_k, t_sim, x, y, yaw, cam, ptz)
        if boxes3d.draw is not None:
            boxes3d.draw.clear_lines()
            if lines:
                boxes3d.draw.draw_lines([tuple(a) for a, b in lines], [tuple(b) for a, b in lines],
                                        [(1.0, 0.2, 0.2, 1.0)] * len(lines), [6.0] * len(lines))
        frame += 1
        step_k += 1

    # ---------------------------------------------------------------- 순찰 끝: 남은 재확인과 점검표
    if cctv_names and not args.no_recheck and app.is_running():
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
