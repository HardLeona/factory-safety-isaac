"""순찰: 작업자가 정해진 경로를 걷고, 바디캠 하나로만 안전 에이전트가 판정·거리 경고·재관측 재판단을 거쳐
조치 지시서를 만든다. CCTV/PTZ 는 쓰지 않는다 (착용자 시점 바디캠 단독).

실행 (Isaac Sim 파이썬):
    <isaac>/python.sh scripts/run_patrol.py                        # 바디캠 화면, 한 바퀴
    <isaac>/python.sh scripts/run_patrol.py --view top             # 위에서 보는 관제 화면
    <isaac>/python.sh scripts/run_patrol.py --headless --sim-dt 0.0333 --seed 3    # 평가용 (같은 시드면 같은 결과)
    <isaac>/python.sh scripts/run_patrol.py --headless --sim-dt 0.0333 --seed 1 --yolo-every 3 --record outputs/record/seed1

끝나면 조치 지시서(outputs/agent/dashboard_seed<시드>.html, --story 는 dashboard_story_seed<시드>.html)를 만들고,
정답표(outputs/eval/answer_key_seed<시드>.json)와 맞춘 채점 결과를 출력하고 저장한다.
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

parser = argparse.ArgumentParser(description="창고 안전 순찰 (바디캠 단독 + 에이전트 재관측 재판단)")
parser.add_argument("--weights", default=os.path.join(ROOT, "outputs", "yolo", "warehouse_v3", "weights", "best.pt"))
parser.add_argument("--seed", type=int, default=None, help="위험 요소 배치 시드 (비우면 무작위)")
parser.add_argument("--laps", type=float, default=1.0, help="몇 바퀴 돌지")
parser.add_argument("--view", default="bodycam", help="bodycam | top")
parser.add_argument("--speed", type=float, default=1.0, help="배속 (실시간 모드)")
parser.add_argument("--sim-dt", type=float, default=0.0, help="0보다 크면 프레임마다 이 시간만큼 진행 (평가 재현용)")
parser.add_argument("--yolo-every", type=int, default=6, help="몇 프레임마다 YOLO")
parser.add_argument("--record", default=None, help="영상용: 바디캠 화면과 에이전트 상태를 저장할 폴더")
parser.add_argument("--result", default=None, help="채점 결과 JSON 경로 (기본 outputs/eval/inspection_seed<시드>.json)")
parser.add_argument("--headless", action="store_true")
parser.add_argument("--sound", default="auto", help="음성 경고 소리: auto (창이 있을 때만), on, off")
parser.add_argument("--gestures", default="off", help="작업자 손동작 명령: off | demo (시연 순서대로 손가락 1~5 를 보임) | watch (손 인식만)")
parser.add_argument("--lang", default=None, help="작업자 언어 ko/en/zh/ja (쉼표로 여러 개면 명령마다 돌아가며). 기본 zh,en,ja, --story 면 en")
parser.add_argument("--llm", default="qwen2.5:7b", help="손동작 명령을 판단할 로컬 LLM (Ollama 모델 이름), off 면 규칙만")
parser.add_argument("--recheck-llm", default="qwen2.5:7b", help="'주의' 물체 재관측 재판단을 맡길 로컬 LLM, off 면 규칙(고위험 후보 재적용)만")
parser.add_argument("--agent", choices=["rule", "langgraph"], default="langgraph",
                    help="재관측 재판단 방식: langgraph (기본, --recheck-llm 사용) | rule (규칙만, --recheck-llm 무시)")
parser.add_argument("--report-llm", default="qwen2.5:7b", help="조치 지시서 문구를 매뉴얼로 보강할 로컬 LLM, off 면 고정 문구만")
parser.add_argument("--voice-max", type=int, default=None, help="음성 경고 최대 횟수 (--story 는 1)")
parser.add_argument("--story", action="store_true", help="시연 이야기: TBM → 카트 설명 → 상자 싣고 끌기 → 공장 스캔 → 한 바퀴 뒤 관리자 호출 (factory_safety/story.py)")
parser.add_argument("--pinch-weights", default=os.path.join(ROOT, "outputs", "yolo", "pinch_v1", "weights", "best.pt"),
                    help="끼임점(pinch_point) 1클래스 경량 모델 가중치. 없으면 등록 위치 폴백만 사용")
parser.add_argument("--pinch-demo", action="store_true",
                    help="시연: 컨베이어(M1)를 켜고 순찰 중 가장 가까이 지날 때 끼임 경보를 보인 뒤 끈다")
args, _ = parser.parse_known_args()
if args.agent == "rule":
    args.recheck_llm = "off"            # 비교 평가용: 재관측 재판단도 규칙만 (LLM 안 씀)
if args.story:
    if args.voice_max is None:
        args.voice_max = 1              # 시연: 음성 경고는 한 번만
    if args.gestures == "off":
        args.gestures = "watch"
if args.lang is None:
    args.lang = "en" if args.story else "zh,en,ja"

from factory_safety.isaac_utils import make_app  # noqa: E402

app = make_app(headless=args.headless)

import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402

from factory_safety.agent import SafetyAgent, action_summary, evaluation_summary  # noqa: E402
from factory_safety.assistant import load_tbm  # noqa: E402
from factory_safety.config import CLASSES, IMG_H, IMG_W, VOICE_TEXT_CAUTION, VOICE_TEXT_HAZARD  # noqa: E402
from factory_safety.dashboard import write_dashboard  # noqa: E402
from factory_safety import i18n  # noqa: E402
from factory_safety import overlay  # noqa: E402
from factory_safety.voice import VOICE_CAUTION_WAV, VOICE_WAV, ensure_voice, play_async, with_alarm  # noqa: E402
from factory_safety.detector import YoloDetector  # noqa: E402
from factory_safety.geometry import Projector  # noqa: E402
from factory_safety.inspection import bodycam_summary  # noqa: E402
from factory_safety.isaac_utils import (DebugBoxes, attach, depth_array, get_annotator, isaac_labeler, new_stage,  # noqa: E402
                                        parse_bboxes, rgb_array, set_viewport_camera, set_viewport_top_view,
                                        timeline_setter)
from factory_safety import pinch_detect as PD  # noqa: E402
from factory_safety.scenario import sample_scenario  # noqa: E402
from factory_safety.scene import WarehouseScene  # noqa: E402
from factory_safety import warehouse as W  # noqa: E402
from factory_safety.walker import PathWalker  # noqa: E402

if not os.path.exists(args.weights):
    sys.exit(f"[오류] YOLO 가중치가 없어요: {args.weights}\n       python scripts/train_yolo.py 로 먼저 학습하세요.")

# ---------------------------------------------------------------- 장면 (배치는 첫 렌더 전에)
stage = new_stage()
scene = WarehouseScene(stage, labeler=isaac_labeler, set_time=timeline_setter()).build()
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
agent = SafetyAgent(IMG_W, IMG_H)
agent.voice_max = args.voice_max
agent.set_task_context(load_tbm())      # 오늘 TBM 작업 대상은 거리 경고에서 제외 (제스처 기능 켜짐 여부와 무관하게 항상 적용)

# 카메라 출력: 바디캠 하나, 화면과 정답 박스. 깊이(distance_to_image_plane)도 받아서 손가락 끝·끼임점의
# 3D 위치를 구한다 (geometry.Projector.unproject, pinch_detect 참고)
rp = rep.create.render_product(scene.cam_path, (IMG_W, IMG_H))
a_rgb, a_box = get_annotator("rgb"), get_annotator("bounding_box_2d_tight")
attach(a_rgb, rp)
attach(a_box, rp)
a_depth = get_annotator("distance_to_image_plane")
attach(a_depth, rp)

pinch_yolo = None
if os.path.exists(args.pinch_weights):
    pinch_yolo = YoloDetector(args.pinch_weights)
    print(f"[끼임점] 경량 탐지 모델 사용: {args.pinch_weights}")
else:
    print(f"[끼임점] 탐지 모델 없음 ({args.pinch_weights}) → 등록 위치 폴백만 사용")
proj_cam = Projector(IMG_W, IMG_H)
MACHINE_ID = "M1"
agent.on_machine_visual = lambda mid, on: scene.set_machine_state(scene.machine_paths[mid], on)
if args.pinch_demo:
    agent.machines.set(MACHINE_ID, True)
s_machine = walker.path.nearest_s(W.MACHINE_CONVEYOR["x"], W.MACHINE_CONVEYOR["y"])
pinch_demo_state = {"fired": False, "off_at": None}
if args.view == "top":
    set_viewport_top_view(stage)
else:
    set_viewport_camera(scene.cam_path)
boxes3d = DebugBoxes()
yolo = YoloDetector(args.weights)
RT_SUBFRAMES = 2   # Isaac Sim 6.0: 1 이면 어노테이터에 한 단계 전 화면이 들어 있음
agent.plan(walker.path.length)
SOUND = args.sound == "on" or (args.sound == "auto" and not args.headless)
WORKER_LANGS = [v.strip() for v in args.lang.split(",") if v.strip()] or ["ko"]
WORKER_LANG = WORKER_LANGS[0]   # 근접 위험·주의 경보는 작업자 본인 언어 하나로 (여러 언어 순환은 손동작 응답 데모용)

# 손동작 명령 + 재관측 재판단 + 조치문구 보강 + 근접 경보 다국어 음성
# (손 인식·다국어 음성(edge-tts)·LangGraph LLM 도우미는 따로 띄운 .venv-assistant 프로세스)
assistant = client = demo = None
need_client = SOUND or args.gestures != "off" or args.recheck_llm != "off" or args.report_llm != "off"
if need_client:
    from factory_safety.assistant_client import AssistantClient
    client = AssistantClient()
    if not client.ok:
        print(f"[도우미] 못 띄워서 LangGraph·다국어 경보 기능은 꺼짐 (손동작·재판단·조치문구 규칙/고정문구, 근접 경보는 한국어로 동작): {client.error}")
    else:
        if args.recheck_llm != "off":
            agent.recheck_judge = lambda snap: client.recheck(snap, args.recheck_llm)   # LangGraph + 로컬 LLM 이 재관측 재판단
            print(f"[재관측] '주의' 물체를 바디캠이 다시 지나치면 로컬 LLM 으로 재판단 ({args.recheck_llm})")
        if args.report_llm != "off":
            agent.report_judge = lambda snap: client.report_action(snap, args.report_llm)  # LangGraph + 로컬 LLM 이 조치문구 보강
            print(f"[조치 지시서] 조치 문구를 매뉴얼 근거로 보강 ({args.report_llm})")
if args.gestures != "off" and client and client.ok:
    from factory_safety.assistant import DemoScript, SiteAssistant, gesture_eval
    assistant = SiteAssistant(agent, langs=WORKER_LANGS)
    if args.llm != "off":
        assistant.planner = lambda snap: client.agent(snap, args.llm)       # LangGraph + 로컬 LLM 이 판단
    demo = DemoScript() if args.gestures == "demo" else None
    print(f"[손동작] 손 인식 + 다국어 안내 켬 (작업자 언어 {args.lang}, 판단 {args.llm if args.llm != 'off' else '규칙'})")


def _proximity_voice(lang, i18n_text, fallback_text, fallback_path):
    """근접 위험/주의 경보 음성 (경보음 + 작업자 언어). 다국어 도우미(edge-tts)가 있으면 그 언어로, 없으면 한국어 Windows 음성으로."""
    if client and client.ok:
        wav, _ = client.tts(i18n_text.get(lang, i18n_text["ko"]), lang)
        if wav:
            return with_alarm(wav, 0.6)
        print(f"[음성] {lang} 음성 생성 실패, 한국어로 대신 경보합니다.")
    elif lang != "ko":
        print(f"[음성] {lang} 음성 도우미가 없어 한국어로 대신 경보합니다.")
    return ensure_voice(fallback_path, fallback_text)


VOICE_HAZARD = _proximity_voice(WORKER_LANG, i18n.VOICE, VOICE_TEXT_HAZARD, VOICE_WAV)
VOICE_CAUTION = _proximity_voice(WORKER_LANG, i18n.VOICE_CAUTION, VOICE_TEXT_CAUTION, VOICE_CAUTION_WAV)
if SOUND:
    agent.on_voice = lambda t, level: play_async(VOICE_HAZARD if level == "hazard" else VOICE_CAUTION)
gest = {"pending": None, "n_ev": 0, "pts": None, "count": 0, "region": None, "t_hand": -1e9}
HAND_HOLD_S = 1.0       # 손이 사라진 뒤에도 이 시간 동안은 손·팔 자리의 YOLO 박스를 판정에서 뺌 (손을 내리는 중)
badge = {"text": None, "until": -1e9, "n_seen": 0}
BADGE_HOLD_S = 4.0


def hand_region(pts):
    """손 관절로 손·팔이 가리는 화면 영역: 손 박스를 넓히고 손목 쪽 (팔이 들어오는 쪽) 은 화면 끝까지."""
    p = np.asarray(pts, float)
    x0, y0 = p.min(axis=0)
    x1, y1 = p.max(axis=0)
    m = 0.35 * max(x1 - x0, y1 - y0)
    x0, y0, x1, y1 = x0 - m, y0 - m, x1 + m, y1 + m
    wx = p[0, 0]
    if wx > (x0 + x1) / 2:
        x1 = IMG_W
    else:
        x0 = 0.0
    return (x0, y0, x1, y1)


def covered(b, r, frac=0.5):
    """박스 b 가 영역 r 에 frac 이상 들어가면 True."""
    ix = max(0.0, min(b[2], r[2]) - max(b[0], r[0]))
    iy = max(0.0, min(b[3], r[3]) - max(b[1], r[1]))
    area = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
    return ix * iy / area >= frac


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


def update_badge(t):
    """agent.timeline 에서 처리 경로(LangGraph/규칙)·상태 전환(주의→위험 등) 최신 내역을 뽑아 HUD 배지 문구로."""
    new = agent.timeline[badge["n_seen"]:]
    badge["n_seen"] = len(agent.timeline)
    for e in new:
        if e["kind"] in ("LLM 판단", "판정", "주의"):
            badge["text"], badge["until"] = f"[{e['kind']}] {e['text']}", t + BADGE_HOLD_S
    if t >= badge["until"]:
        return None
    return badge["text"], (new[-1]["kind"] if new else "판정")


def record_state(k, t, x, y, yaw, cam):
    if not args.record:
        return
    st = {"k": k, "t": round(t, 3), "worker": [x, y, yaw], "bodycam": [float(v) for v in cam.pos[:2]] + [cam.yaw, cam.vfov],
          "story": story.label if story else None,
          "n_timeline": len(agent.timeline),
          "hand": {"alpha": round(walker.gesture_alpha, 2), "shown": walker.gesture["count"] if walker.gesture else 0,
                   "count": gest["count"], "pts": gest["pts"]} if assistant else None,
          "n_assist": len(assistant.events) if assistant else 0,
          "zones": [{"id": z.zid, "source": z.source, "poly": [[round(float(a), 2), round(float(b), 2)] for a, b in z.poly]}
                    for z in agent.zones.values()],
          "voice": [{"t": v["t"], "level": v["level"]} for v in agent.voice_events if v["t"] > t - 0.05],
          "findings": [{"id": f.fid, "x": float(f.xy[0]), "y": float(f.xy[1]), "cls": f.cls, "status": f.status,
                        "near_miss": f.near_miss} for f in agent.findings]}
    with open(os.path.join(args.record, "state.jsonl"), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(st, ensure_ascii=False) + "\n")


if args.record:
    open(os.path.join(args.record, "state.jsonl"), "w").close()

# ---------------------------------------------------------------- 순찰 루프
print(f"[안내] 순찰 시작: 경로 {walker.path.length:.0f} m, {args.laps:g}바퀴. 창을 닫거나 Ctrl+C로 끝낼 수 있어요.")
t_sim, frame, step_k, last = 0.0, 0, 0, time.time()
x = y = yaw = 0.0
try:
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
        x, y, yaw, at = walker.base_pose
        scene.set_worker(x, y, yaw, at)
        scene.set_camera(cam)
        if args.pinch_demo and not pinch_demo_state["fired"] and abs(walker.s - s_machine) < 0.3:
            # 시연: 순찰 경로에서 기계 끼임점에 가장 가까운 지점을 지날 때, 손끝이 끼임점 바로 옆(3 cm)에
            # 있다고 보고 끼임 경보 규칙을 실제로 실행한다 (손을 뻗는 전신 애니메이션은 이번 범위 밖, README 참고)
            tip = agent.machines.pinch_xyz(MACHINE_ID) + np.array([0.0, 0.0, 0.03])
            ev = agent.check_pinch(t_sim, MACHINE_ID, tip)
            pinch_demo_state["fired"] = True
            pinch_demo_state["off_at"] = t_sim + 3.0
            if ev:
                print(f"[끼임 경보 시연] 손 빼세요! (거리 {ev['dist_cm']} cm)")
        if pinch_demo_state["off_at"] is not None and t_sim >= pinch_demo_state["off_at"]:
            agent.machines.set(MACHINE_ID, False)
            pinch_demo_state["off_at"] = None
        if frame % args.yolo_every != 0 or frame < 10:
            app.update()
            frame += 1
            continue
        # Isaac Sim 6.0: app.update() 만으로는 어노테이터가 비어서 나와서 Replicator 로 렌더
        rep.orchestrator.step(delta_time=0.0, rt_subframes=RT_SUBFRAMES)
        img = rgb_array(a_rgb.get_data())
        agent.log_worker(t_sim, (x, y))
        if img is not None:
            dets, res = yolo.track(img, "bodycam")
            if assistant:
                # 작업자 자기 손·팔은 YOLO 판정에서 뺌 (손을 물체로 잘못 볼 수 있어서)
                gest["count"], gest["pts"] = client.hand(img)
                if gest["pts"]:
                    gest["region"], gest["t_hand"] = hand_region(gest["pts"]), t_sim
                if gest["region"] is not None and t_sim - gest["t_hand"] <= HAND_HOLD_S:
                    dets = [d for d in dets if not covered(d[2], gest["region"])]
                if gest["pts"]:
                    # 손가락 끝(검지 끝) 3D 위치: 바디캠 깊이맵으로 역투영 (실제 탐지 경로, 시연 스크립트와 별개)
                    proj_cam.set_pose(cam)
                    depth_map = depth_array(a_depth.get_data())
                    fingertip = PD.fingertip_xyz(proj_cam, gest["pts"], depth_map) if depth_map is not None else None
                    if fingertip is not None:
                        pinch_xyxy = None
                        if pinch_yolo is not None:
                            pdets, _ = pinch_yolo(img)
                            muv, _ = proj_cam.project(agent.machines.pinch_xyz(MACHINE_ID)[None])
                            pinch_xyxy = PD.best_pinch_box(pdets, muv[0])
                        detected = PD.pinch_point_xyz(proj_cam, pinch_xyxy, depth_map) if pinch_xyxy is not None else None
                        agent.check_pinch(t_sim, MACHINE_ID, fingertip, detected)
            agent.on_bodycam(t_sim, cam, dets, worker_xy=(x, y), gt=parse_bboxes(a_box.get_data(), CLASSES, with_paths=True))
            if assistant:
                if walker.gesture is None:
                    assistant.observe(t_sim, cam, dets)       # 손으로 가린 화면은 장비·위험 안내에 안 씀
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
                    last_ev = assistant.events[-1] if assistant.events and t_sim - assistant.events[-1]["t"] < 2.5 else None
                    overlay.draw_hand(im, gest["pts"], gest["count"], last_ev["cmd"] if last_ev and last_ev["count"] == gest["count"] else None)
                b = update_badge(t_sim)
                if b:
                    overlay.draw_status_badge(im, b[0], b[1])
                im.save(os.path.join(args.record, f"body_{step_k:05d}.jpg"), quality=88)
            else:
                update_badge(t_sim)
        if demo and gest["pending"] is not None and walker.gesture is None:
            if len(assistant.events) == gest["n_ev"]:
                demo.missed(t_sim)
                print("[손동작] 인식 못 함")
            gest["pending"] = None
        record_state(step_k, t_sim, x, y, yaw, cam)
        lines = []
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

    # ---------------------------------------------------------------- 순찰 끝: 남은 점검 지점 정리
    if not story and app.is_running():
        agent.finish_patrol(t_sim)
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
if client:
    client.close()
print("\n" + action_summary(report))
for m in report["machines"]:
    print(f"[끼임 위험 기계] {m['id']} {m['name']}: {'작동' if m['on'] else '정지'} | "
          f"접근 경보 {m['zone_alerts']}회, 끼임 경보 {m['pinch_alerts']}회 (폴백 {m['pinch_fallback']}회)")
rows, summ = agent.scorer.report(key)
print("\n" + bodycam_summary(rows, summ))
print(evaluation_summary(evaluation))
tag = f"story_seed{seed}" if args.story else f"seed{seed}"      # 시연 이야기는 평가 결과를 덮지 않게 따로
dash = write_dashboard(os.path.join(agent_dir, f"dashboard_{tag}.html"), report, evaluation, seed=seed)
out = args.result or os.path.join(eval_dir, f"inspection_seed{seed}.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump({"seed": seed, "sim_time": t_sim, "bodycam": summ, "objects": rows,
               "agent": {"evaluation": evaluation, "report": {k: v for k, v in report.items() if k != "timeline"}}},
              f, ensure_ascii=False, indent=1)
with open(os.path.join(agent_dir, f"report_{tag}.json"), "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=1)
if args.record:
    with open(os.path.join(args.record, "final.json"), "w", encoding="utf-8") as f:
        json.dump({"seed": seed, "story": bool(args.story), "report": report, "evaluation": evaluation, "bodycam": summ},
                   f, ensure_ascii=False, indent=1)
print(f"[저장] 조치 지시서 {dash}\n[저장] 채점 결과 {out}")
app.close()
