"""BodyGuard v2 시연 영상용: 스토리보드 4장에 맞춘 짧은 시나리오를 각각 독립적으로 녹화한다 (Isaac Sim 파이썬).

    <isaac>/python.sh scripts/record_scenario.py --which pinch     --record outputs/record/v2_pinch
    <isaac>/python.sh scripts/record_scenario.py --which forklift --record outputs/record/v2_forklift
    <isaac>/python.sh scripts/record_scenario.py --which spill    --record outputs/record/v2_spill
    <isaac>/python.sh scripts/record_scenario.py --which sign     --record outputs/record/v2_sign

각 시나리오는 통짜 순찰이 아니라 그 장면 하나만 짧게 보여주는 독립 클립이다 (make_video2.py 가 스토리보드 이미지
+ TTS 내레이션 뒤에 이 클립들을 이어 붙인다). 프레임은 outputs/record/<이름>/body_XXXXX.jpg, 에이전트 상태는
state.jsonl, 끝나면 final.json 에 report()/evaluation 을 저장한다 (run_patrol.py 와 같은 형식).
"""
import argparse
import json
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

parser = argparse.ArgumentParser(description="BodyGuard v2 시나리오 클립 녹화")
parser.add_argument("--which", required=True, choices=["pinch", "forklift", "spill", "sign"])
parser.add_argument("--weights", default=os.path.join(ROOT, "outputs", "yolo", "warehouse_v3", "weights", "best.pt"))
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--record", required=True, help="프레임·상태를 저장할 폴더")
parser.add_argument("--headless", action="store_true")
parser.add_argument("--sim-dt", type=float, default=0.0333)
parser.add_argument("--yolo-every", type=int, default=3)
args, _ = parser.parse_known_args()

from factory_safety.isaac_utils import make_app  # noqa: E402

app = make_app(headless=args.headless)

import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402

from factory_safety import overlay  # noqa: E402
from factory_safety import warehouse as W  # noqa: E402
from factory_safety.agent import SafetyAgent, action_summary  # noqa: E402
from factory_safety.config import CLASSES, IMG_H, IMG_W  # noqa: E402
from factory_safety.dashboard import write_dashboard  # noqa: E402
from factory_safety.detector import YoloDetector  # noqa: E402
from factory_safety.geometry import CameraPose, Projector  # noqa: E402
from factory_safety.isaac_utils import (DebugBoxes, attach, depth_array, get_annotator, isaac_labeler, new_stage,  # noqa: E402
                                        parse_bboxes, rgb_array, set_viewport_camera, timeline_setter)
from factory_safety import pinch_detect as PD  # noqa: E402
from factory_safety.scenario import sample_scenario  # noqa: E402
from factory_safety.scene import WarehouseScene  # noqa: E402
from factory_safety.walker import PathWalker  # noqa: E402
from factory_safety.walk_anim import GESTURE_CAM_AHEAD, reach_info, reach_time  # noqa: E402

os.makedirs(args.record, exist_ok=True)
MACHINE_ID = "M1"
FORKLIFT_ID = "F1"

stage = new_stage()
scene = WarehouseScene(stage, labeler=isaac_labeler, set_time=timeline_setter()).build()
scenario = sample_scenario(args.seed)
if args.which != "sign":      # 표지판 시연은 경쟁할 다른 위험물 없이 표지만 뚜렷하게 보이게 (랜덤 시나리오를 안 씀)
    scene.set_scenario(scenario)
set_viewport_camera(scene.cam_path)

agent = SafetyAgent(IMG_W, IMG_H)
agent.on_machine_visual = lambda mid, on: scene.set_machine_state(scene.machine_paths[mid], on)
agent.plan(1.0)

rp = rep.create.render_product(scene.cam_path, (IMG_W, IMG_H))
a_rgb, a_box = get_annotator("rgb"), get_annotator("bounding_box_2d_tight")
attach(a_rgb, rp)
attach(a_box, rp)
a_depth = get_annotator("distance_to_image_plane")
attach(a_depth, rp)
boxes3d = DebugBoxes()
RT_SUBFRAMES = 2
yolo = YoloDetector(args.weights)
pinch_yolo_path = os.path.join(ROOT, "outputs", "yolo", "pinch_v1", "weights", "best.pt")
pinch_yolo = YoloDetector(pinch_yolo_path) if os.path.exists(pinch_yolo_path) else None
proj_cam = Projector(IMG_W, IMG_H)

state_f = open(os.path.join(args.record, "state.jsonl"), "w", encoding="utf-8")
step_k = [0]     # 저장한 프레임 번호 (record_state/save_frame 에서만 증가)
frame = [0]      # 매 틱 증가 (yolo_every 스킵 판정 전용, step_k 와 분리 — 한 번 스킵하면 영원히 스킵되는 걸 막음)


def record_state(t, x, y, yaw, cam, extra=None):
    st = {"k": step_k[0], "t": round(t, 3), "worker": [x, y, yaw],
          "bodycam": [float(v) for v in cam.pos[:2]] + [cam.yaw, cam.vfov],
          "zones": [{"id": z.zid, "source": z.source, "poly": [[round(float(a), 2), round(float(b), 2)] for a, b in z.poly]}
                    for z in agent.zones.values()],
          "voice": [{"t": v["t"], "level": v["level"]} for v in agent.voice_events if v["t"] > t - 0.05],
          "findings": [{"id": f.fid, "x": float(f.xy[0]), "y": float(f.xy[1]), "cls": f.cls, "status": f.status} for f in agent.findings],
          "n_timeline": len(agent.timeline)}
    if extra:
        st.update(extra)
    state_f.write(json.dumps(st, ensure_ascii=False) + "\n")
    step_k[0] += 1


def save_frame(img, cam, dets=None):
    im = overlay.draw(img, dets, lambda b: agent._tool_state(cam, b), size=19) if dets is not None else None
    if im is None:
        from PIL import Image
        im = Image.fromarray(img)
    im.save(os.path.join(args.record, f"body_{step_k[0]:05d}.jpg"), quality=88)


def render(cam, rt=RT_SUBFRAMES):
    scene.set_camera(cam)
    rep.orchestrator.step(delta_time=0.0, rt_subframes=rt)
    img = rgb_array(a_rgb.get_data())
    return img


# ---------------------------------------------------------------- ① 지게차 접근
def run_forklift():
    """순찰하며 걷는 중 뒤쪽 통로에서 지게차가 다가온다 (위치는 RTLS 태그 개념, 영상 인식 아님).
    위험 경보(음성 경고)가 실제로 뜨면, 그 직후 작업자가 뒤로 물러나며 지게차 쪽으로 돌아서서 보는
    회피 행동을 한다 (경보만 내고 가만히 서 있지 않음). 지게차가 지나가면 다시 정상 순찰로 돌아온다."""
    walker = PathWalker(seed=args.seed, speed=1.1)
    path = walker.path
    s_walker0 = path.length * 0.12
    walker.reset(s0=s_walker0)
    lag_m, fork_speed = 7.0, 2.3
    s_fork = s_walker0 - lag_m
    t = 0.0
    trigger_t = 2.0
    total_s = 18.0
    RETREAT_SPEED = 0.9     # 뒤로 물러나는 속도 (m/s)
    AVOID_HOLD_S = 2.0      # 경보 뒤 최소 이만큼은 물러나 지켜봄 (지게차가 금방 지나가도 회피 동작이 보이게)
    # 지게차와 작업자는 둘 다 같은 경로 중심선(path.point_at(s)) 위에 있어서, 앞뒤로만 비키면 스쳐
    # 지나가는 순간 같은 좌표에 겹칠 수 있다 (실측 최소 거리 7 cm 까지 겹친 적 있음). 옆으로도 비켜서게 함.
    LATERAL_OFFSET = 0.9    # 옆으로 비켜서는 거리 (m)
    LATERAL_RATE = 1.6      # 초당 비켜서는 비율 (약 0.6초에 다 비킴)
    warned, avoiding, avoid_t0, voice_n0, lateral_u = False, False, None, 0, 0.0
    while t < total_s and app.is_running():
        dt = args.sim_dt
        t += dt
        if t >= trigger_t:
            s_fork += fork_speed * dt
        fxy = path.point_at(s_fork)
        fyaw = path.heading_at(s_fork)
        scene.set_forklift(scene.forklift_path, fxy[0], fxy[1], fyaw)
        gap = s_fork - walker.s
        fk_active = t >= trigger_t and -lag_m - 1.0 < gap < 2.5
        agent.update_forklift(t, FORKLIFT_ID, fxy, fk_active)
        cam = walker.step(dt)
        if warned and (fk_active or (avoid_t0 is not None and t - avoid_t0 < AVOID_HOLD_S)):
            if avoid_t0 is None:
                avoid_t0 = t
            avoiding = True
            walker.s = max(0.0, walker.s - (RETREAT_SPEED + walker.speed) * dt)
            wp = path.point_at(walker.s)
            walker.face(math.atan2(fxy[1] - wp[1], fxy[0] - wp[0]))     # 돌아서서 다가오는 지게차를 봄
        elif avoiding:
            avoiding = False
            walker.face(None)
        lateral_u = max(0.0, min(1.0, lateral_u + LATERAL_RATE * dt * (1.0 if avoiding else -1.0)))
        x, y, yaw, at = walker.base_pose
        if lateral_u > 0:
            heading_here = path.heading_at(walker.s)
            x += -math.sin(heading_here) * LATERAL_OFFSET * lateral_u
            y += math.cos(heading_here) * LATERAL_OFFSET * lateral_u
        scene.set_worker(x, y, yaw, at)
        frame[0] += 1
        if frame[0] % args.yolo_every != 0:
            app.update()
            continue
        agent.on_bodycam(t, cam, [], worker_xy=(x, y))
        if not warned and any(v["level"] == "hazard" for v in agent.voice_events[voice_n0:]):
            warned = True       # 위험 경보가 실제로 뜬 다음부터만 회피 행동 시작
        voice_n0 = len(agent.voice_events)
        # 녹화 화면은 3인칭 추격 카메라로: 작업자 바디캠(1인칭)은 원래 뒤에서 오는 지게차가 안 보여야
        # 맞지만(그래서 에이전트가 위치 신호로 대신 경고하는 것), 시연 영상에서는 지게차가 다가오는
        # 모습 자체를 시청자에게 보여줘야 설득력이 있어 녹화용 카메라만 따로 둔다 (에이전트 판단은 그대로 cam/worker_xy 기준)
        # 지게차보다 3m, 작업자보다 1.5m 는 뒤에 (둘 중 더 뒤쪽) 카메라를 둬서, 작업자가 회피 행동으로
        # 뒤로 물러날 때도 카메라가 작업자 쪽으로 너무 가까워지며 각도가 무너지지 않게 함
        chase_s = min(s_fork - 3.0, walker.s - 1.5)
        chase_xy = path.point_at(chase_s)
        chase_pos = np.array([chase_xy[0], chase_xy[1], 2.3])
        d = np.array([x, y, 1.2]) - chase_pos
        chase_yaw = math.atan2(d[1], d[0])
        chase_pitch = math.atan2(d[2], math.hypot(d[0], d[1]))
        chase_cam = CameraPose(pos=chase_pos, yaw=chase_yaw, pitch=chase_pitch, vfov=65.0)
        img = render(chase_cam)
        if img is not None:
            save_frame(img, chase_cam)
        record_state(t, x, y, yaw, cam, {"forklift": [float(fxy[0]), float(fxy[1])], "forklift_active": fk_active})
    scene.set_forklift(scene.forklift_path, 0.0, -100.0, 0.0)
    agent.update_forklift(t, FORKLIFT_ID, path.point_at(s_fork), False)


# ---------------------------------------------------------------- ② 바닥 미끄러움·적재물 방치 (주의 → 관리자 알림)
def run_spill():
    """표지 없는 유출을 잠깐만 보고 지나쳐 '주의' 로 분류되고, 관리자 알림 피드에 자동으로 쌓인다."""
    walker = PathWalker(seed=args.seed, speed=1.1)
    # 정답표의 경로 옆 유출 자리(NEAR_PATH_SPILL_SLOTS)에 가장 가까운 경로 위 지점부터 시작
    sx, sy = W.NEAR_PATH_SPILL_SLOTS[0]
    s0 = max(0.0, walker.path.nearest_s(sx, sy) - 4.0)
    walker.reset(s0=s0)
    t, total_s = 0.0, 10.0
    while t < total_s and app.is_running():
        dt = args.sim_dt
        t += dt
        cam = walker.step(dt)
        x, y, yaw, at = walker.base_pose
        scene.set_worker(x, y, yaw, at)
        frame[0] += 1
        if frame[0] % args.yolo_every != 0:
            app.update()
            continue
        img = render(cam)
        if img is not None:
            dets, _ = yolo.track(img, "bodycam")
            agent.on_bodycam(t, cam, dets, worker_xy=(x, y))
            save_frame(img, cam, dets)
        record_state(t, x, y, yaw, cam)


# ---------------------------------------------------------------- ③ 표지판이 궁금할 때 (손동작 1 + 매뉴얼 RAG)
# 작업자 페르소나(베트남에서 온 민 씨)에 맞춰 베트남어로 답한다 (i18n.py 에 이 흐름에 필요한 범위만 검수해 넣음).
SIGN_DEMO_LANG = "vi"
assistant_events = []   # run_sign() 이 채우면 final.json 에 report["assistant"] 로 저장 (실제 음성 wav 포함)


def run_sign():
    """DANGER 표지 앞에서 실제로 손가락 1개를 들어 올리는 손동작을 보여주고, 매뉴얼 근거로 답한다 (LLM 필요).
    표지는 시나리오 무작위 배치에 기대지 않고 이 클립 전용으로 경로 옆에 직접 세운다 (첫 렌더 전).
    이 장면은 '궁금해서 묻는' 흐름이지 위험 장면이 아니라서, 표지판에 다가가도 평소의 자동 위험 경보(음성 경고)는
    울리지 않게 한다 (agent.on_bodycam 에는 빈 dets 만 넘기고, 표지판은 assistant.observe() 로만 보여줌)."""
    from factory_safety.assistant import SiteAssistant
    from factory_safety.assistant_client import AssistantClient
    walker = PathWalker(seed=args.seed, speed=1.1)
    s_sign = walker.path.length * 0.3
    p = walker.path.point_at(s_sign)
    heading = walker.path.heading_at(s_sign)
    nx, ny = -math.sin(heading), math.cos(heading)
    zx, zy = float(p[0] + nx * 1.3), float(p[1] + ny * 1.3)
    sign_yaw = heading - math.pi / 2     # 표지 앞면이 경로(작업자) 쪽을 보게
    sg = scene._danger_sign("/World/DemoSign", zx, zy, sign_yaw)
    scene.labeler(sg.GetPrim(), "danger_sign")

    client = AssistantClient()
    assistant = SiteAssistant(agent, langs=[SIGN_DEMO_LANG]) if client.ok else None
    if assistant:
        assistant.planner = lambda snap: client.agent(snap, "qwen2.5:7b")
    walker.reset(s0=max(0.0, s_sign - 3.0))
    t, total_s, asked = 0.0, 9.5, False
    while t < total_s and app.is_running():
        dt = args.sim_dt
        t += dt
        if walker.s >= s_sign - 0.8 and not asked and not walker.gesture:
            walker.start_gesture(1, hold=2.2)     # 멈춰 서서 손가락 1개를 들어 보임 (실제 제스처 애니메이션)
        cam = walker.step(dt)
        x, y, yaw, at = walker.base_pose
        scene.set_worker(x, y, yaw, at)
        frame[0] += 1
        if frame[0] % args.yolo_every != 0:
            app.update()
            continue
        img = render(cam)
        if img is not None:
            dets, _ = yolo.track(img, "bodycam")
            agent.on_bodycam(t, cam, [], worker_xy=(x, y))   # 빈 dets: 이 클립은 위험 경보 대상이 아님
            if assistant and not asked:
                # 이 데모 클립은 표지판+RAG 흐름만 보여주는 게 목적이라, 기본 창고 선반의 오탐(먼 "적재" 등)이
                # LLM 의 장비 선택(거리 제한 없음, assistant.py 기존 동작)을 가로채지 않게 표지판만 넘긴다
                assistant.observe(t, cam, [d for d in dets if d[0] == "danger_sign"])
            if assistant and not asked and walker.gesture and walker.gesture_alpha >= 0.999:
                ev = assistant.run(t, 1, cam, (x, y), yaw, lang=SIGN_DEMO_LANG)
                wav, dur = client.tts(ev["text"], SIGN_DEMO_LANG)
                ev["wav"], ev["dur"] = wav, round(dur, 2)
                assistant_events.append(ev)
                total_s = max(total_s, t + dur + 1.0)   # 음성 답변이 길면, 다 끝날 때까지 클립이 안 끊기게 늘림
                print(f"[손동작 1] {ev['text']} (wav={wav}, {dur:.1f}초)")
                asked = True
            save_frame(img, cam, dets)
        record_state(t, x, y, yaw, cam)
    if client:
        client.close()


# ---------------------------------------------------------------- ④ 손이 끼이기 직전 (끼임점에 손 뻗기)
def run_pinch():
    """컨베이어 진입 롤러 쪽으로 손을 뻗는 모습을 바디캠으로 보여주고, 끼임 경보가 실제로 뜬다."""
    agent.machines.set(MACHINE_ID, True)
    info = reach_info()
    sx, sy = info["stand_xy"]
    syaw = info["stand_yaw"]
    pinch_world = info["pinch_world"]
    raise_s = info["raise_s"]
    cam_h = 1.18   # 가슴 높이(1.38)보다 조금 아래서 찍어, 손-롤러 간격(위험 상황)이 화면에 더 크게 보이게

    def bodycam(alpha):
        # 끼임점을 계속 조준 (손이 거기로 다가가는 걸 보는 시점이라, 평소 걷기보다 훨씬 아래를 봐야 함:
        # 바디캠은 가슴 높이(cam_h) 인데 끼임점은 허리 높이(pinch_z) 라 단순 전방 시선으로는 프레임 밖으로 벗어남)
        ahead = GESTURE_CAM_AHEAD
        pos = np.array([sx + ahead * math.cos(syaw), sy + ahead * math.sin(syaw), cam_h])
        d = pinch_world - pos
        yaw = math.atan2(d[1], d[0])
        pitch = math.atan2(d[2], math.hypot(d[0], d[1])) - 0.08   # 끼임점보다 살짝 위를 봐서 다가오는 손도 같이 보이게
        return CameraPose(pos=pos, yaw=yaw, pitch=pitch, roll=0.0, vfov=68.0)

    def fingertip_at(alpha):
        """alpha(0 내림~1 다 뻗음) 에서 손 끝의 대략적인 세계 좌표 (끼임점 위쪽에서 다가오는 근사).
        '닿기 직전' 이 아니라 '가까워지는 중'에도 실제 거리로 경보가 뜨게 하려고, 손 끝 위치를 alpha 로 보간한다."""
        e = max(0.0, min(1.0, alpha)) ** 0.5
        far, near = 0.45, 0.05
        return pinch_world + np.array([0.0, 0.0, far * (1 - e) + near * e])

    # 타임라인: 1.0s 평소 서 있는 모습 -> 1.0s 손 뻗기 -> 1.6s 유지(끼임 경보 발생) -> 1.0s 손 빼기 -> 0.8s 정지 뒤 끝
    phases = [("idle", 1.0, lambda u: 0.0), ("raise", raise_s, lambda u: u),
              ("hold", 1.6, lambda u: 1.0), ("lower", raise_s, lambda u: 1.0 - u), ("end", 0.8, lambda u: 0.0)]
    t = 0.0
    fired = False
    for name, dur, fn in phases:
        t0 = t
        while t - t0 < dur and app.is_running():
            dt = args.sim_dt
            t += dt
            u = (t - t0) / dur
            alpha = fn(min(1.0, u))
            at = reach_time(alpha)
            scene.set_worker(sx, sy, syaw, at)
            cam = bodycam(alpha)
            frame[0] += 1
            if frame[0] % args.yolo_every != 0:
                app.update()
                continue
            img = render(cam)
            depth_map = depth_array(a_depth.get_data())
            if img is not None:
                if name in ("raise", "hold") and not fired:
                    # raise 단계부터 매 프레임 실제 거리로 검사 -> 손이 끼임점에 닿기 직전, 가까워지는 중에
                    # (hold 에 이르러서야가 아니라) 실제 임계 거리를 넘는 순간 바로 경보가 뜬다
                    pinch_xyxy = None
                    if pinch_yolo is not None:
                        pdets, _ = pinch_yolo(img)
                        proj_cam.set_pose(cam)
                        muv, _ = proj_cam.project(pinch_world[None])
                        pinch_xyxy = PD.best_pinch_box(pdets, muv[0])
                    detected = PD.pinch_point_xyz(proj_cam, pinch_xyxy, depth_map) if pinch_xyxy is not None else None
                    fingertip = fingertip_at(alpha)
                    ev = agent.check_pinch(t, MACHINE_ID, fingertip, detected)
                    if ev:
                        fired = True
                        print(f"[끼임 경보] alpha={alpha:.2f}, {ev['dist_cm']} cm, 폴백={ev['fallback']}")
                save_frame(img, cam)
            record_state(t, sx, sy, syaw, cam, {"reach_alpha": round(alpha, 2)})
    agent.machines.set(MACHINE_ID, False)
    # 꺼진 뒤 한 번 더: 위험구역 해제를 보여줌
    for _ in range(int(1.0 / args.sim_dt)):
        if not app.is_running():
            break
        t += args.sim_dt
        cam = bodycam(0.0)
        frame[0] += 1
        if frame[0] % args.yolo_every == 0:
            img = render(cam)
            if img is not None:
                save_frame(img, cam)
            record_state(t, sx, sy, syaw, cam, {"reach_alpha": 0.0})
        else:
            app.update()


RUN = {"pinch": run_pinch, "forklift": run_forklift, "spill": run_spill, "sign": run_sign}

t_start = time.time()
RUN[args.which]()
state_f.close()

report = agent.report()
if assistant_events:
    report["assistant"] = assistant_events
print("\n" + action_summary(report))
with open(os.path.join(args.record, "final.json"), "w", encoding="utf-8") as f:
    json.dump({"which": args.which, "seed": args.seed, "report": report}, f, ensure_ascii=False, indent=1)
dash = write_dashboard(os.path.join(args.record, "dashboard.html"), report, seed=args.seed,
                       title=f"BodyGuard 시나리오: {args.which}")
print(f"[완료] {args.which} ({time.time() - t_start:.0f}초) -> {args.record}")
app.close()
