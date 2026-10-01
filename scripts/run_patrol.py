"""순찰 시뮬레이션. 로봇(또는 작업자 바디캠)이 공장을 돌며 위험 요소를 찾는다.

실행 (Isaac Sim 파이썬):
    <isaac>/python.sh scripts/run_patrol.py                          # 로봇 시점, 가상 검출기
    <isaac>/python.sh scripts/run_patrol.py --camera bodycam         # 작업자 바디캠
    <isaac>/python.sh scripts/run_patrol.py --view top               # 위에서 관제 화면으로 보기
    <isaac>/python.sh scripts/run_patrol.py --detector yolo --weights outputs/yolo/run/weights/best.pt
    <isaac>/python.sh scripts/run_patrol.py --record outputs/frames      # 카메라 화면을 이미지로 저장 (영상 제작용)
"""
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

parser = argparse.ArgumentParser(description="공장 안전 순찰")
parser.add_argument("--camera", choices=["robot", "bodycam"], default="robot")
parser.add_argument("--detector", choices=["sim", "yolo"], default="sim")
parser.add_argument("--weights", default=None, help="YOLO 가중치 (.pt)")
parser.add_argument("--view", choices=["pov", "top"], default="pov", help="pov: 카메라 시점, top: 관제 화면")
parser.add_argument("--seed", type=int, default=None, help="위험 요소 배치 시드 (비우면 무작위)")
parser.add_argument("--layout-seed", type=int, default=0)
parser.add_argument("--speed", type=float, default=1.0, help="시뮬레이션 배속")
parser.add_argument("--duration", type=float, default=0.0, help="순찰 시간(초), 0이면 창을 닫을 때까지")
parser.add_argument("--yolo-every", type=int, default=6, help="YOLO를 몇 프레임마다 돌릴지")
parser.add_argument("--save-frames", default=None, help="YOLO 결과 이미지를 저장할 폴더")
parser.add_argument("--record", default=None, help="카메라 원본 화면을 저장할 폴더")
parser.add_argument("--record-every", type=int, default=2, help="몇 프레임마다 저장할지")
parser.add_argument("--no-zoom", action="store_true",
                    help="압력계 줌 판독 끄기 (끄면 정상/부족을 YOLO 클래스로만 판정)")
parser.add_argument("--save-zoom", default=None, help="압력계 줌 화면과 판독 결과를 저장할 폴더")
parser.add_argument("--dump-dets", default=None,
                    help="YOLO 검출, 카메라 자세, 줌 판독을 JSON 으로 저장 (scripts/replay_dets.py 로 Isaac 없이 다시 판단)")
parser.add_argument("--sim-dt", type=float, default=0.0,
                    help="0보다 크면 매 프레임 이 시간(초)만큼 진행 (실제 시간과 무관, 평가 재현용). 예: 0.0333")
parser.add_argument("--headless", action="store_true")
args, _ = parser.parse_known_args()

from factory_safety.isaac_utils import make_app  # noqa: E402

app = make_app(headless=args.headless)

import numpy as np  # noqa: E402

from factory_safety import layout as L  # noqa: E402
from factory_safety.config import IMG_H, IMG_W, RISK_RGBA  # noqa: E402
from factory_safety.detector import SimDetector  # noqa: E402
from factory_safety.isaac_utils import (DebugBoxes, attach, get_annotator, isaac_labeler, new_stage,  # noqa: E402
                                        rgb_array, set_viewport_camera, set_viewport_top_view)
from factory_safety.patrol import BodycamWalker, RobotPatrol  # noqa: E402
from factory_safety.report import box_color, event_line, summary  # noqa: E402
from factory_safety.scenario import sample_scenario  # noqa: E402
from factory_safety.usd_scene import FactoryStage  # noqa: E402
from factory_safety.yolo_judge import RISK, YoloJudge, judge_line, judge_summary  # noqa: E402

# ---------------------------------------------------------------- 장면
stage = new_stage()
scene = FactoryStage(stage, os.path.join(ROOT, "outputs", "textures"), static_seed=args.layout_seed,
                     labeler=isaac_labeler).build()
seed = args.seed if args.seed is not None else int(np.random.default_rng().integers(1 << 30))
scenario = sample_scenario(seed)
scene.set_scenario(scenario)
print(f"[시나리오] 시드 {seed}, 위험 요소 {len(scenario.targets)}개")

static = scene.static_models
s_min, s_max = L.occluder_arrays(static)
ex = scenario.occluder_boxes()
if ex:
    s_min = np.vstack([s_min, [b[0] for b in ex]])
    s_max = np.vstack([s_max, [b[1] for b in ex]])
det = SimDetector(scenario.hazards, s_min, s_max)
# YOLO 모드: 화면 박스 + 로봇 자기 위치 + 공장 지도만으로 판단 (정답 위치는 마지막 채점에만 씀)
# 로봇은 머리에 줌 카메라가 하나 더 있어서 압력계는 크게 찍어 바늘을 읽는다 (바디캠은 없음)
use_zoom = args.detector == "yolo" and args.camera == "robot" and not args.no_zoom
judge = YoloJudge(*L.map_boxes_3d(static), zoom=use_zoom) if args.detector == "yolo" else None

agent = RobotPatrol() if args.camera == "robot" else BodycamWalker(seed=seed)
scene.set_robot_visible(args.camera == "robot" and args.view == "top")
scene.set_worker_visible(args.camera == "bodycam" and args.view == "top")
if args.view == "pov":
    set_viewport_camera(scene.cam_path)
else:
    set_viewport_top_view(stage)
boxes = DebugBoxes()

yolo, rgb_annot = None, None
if args.detector == "yolo" and not args.weights:
    sys.exit("[오류] --detector yolo 는 --weights 가 필요해요.")
if args.detector == "yolo" or args.record:
    import omni.replicator.core as rep
    rp = rep.create.render_product(scene.cam_path, (IMG_W, IMG_H))
    rgb_annot = get_annotator("rgb")
    attach(rgb_annot, rp)
if args.detector == "yolo":
    from factory_safety.detector import YoloDetector
    yolo = YoloDetector(args.weights)
zoom_annot, ZOOM_PX = None, 320
# Isaac Sim 6.0 에서 rt_subframes=1 로 한 번 렌더하면 어노테이터에는 한 단계 전 화면이 들어 있다
# (카메라를 돌리고 바로 찍으면 돌리기 전 화면). 2 이상이면 지금 자세의 화면이 나온다.
RT_SUBFRAMES = 2
zoom_rp = None
ZOOM_RESET_AFTER = 12   # 줌 판독이 연속으로 이만큼 실패하면 줌 render product 를 새로 만든다


def make_zoom_render():
    global zoom_annot, zoom_rp
    if zoom_rp is not None:
        try:
            zoom_annot.detach()
            zoom_rp.destroy()
        except Exception:
            pass
    zoom_annot = get_annotator("rgb")
    zoom_rp = rep.create.render_product(zoom_path, (ZOOM_PX, ZOOM_PX))
    attach(zoom_annot, zoom_rp)


if use_zoom:
    from factory_safety.gauge_reader import read_gauge, zoom_pose
    zoom_path = scene.add_camera("/World/ZoomCam")
    make_zoom_render()
for d in (args.save_frames, args.record, args.save_zoom):
    if d:
        os.makedirs(d, exist_ok=True)
zoom_fail_streak = 0


def zoom_read(cam, dets):
    """줌 카메라로 압력계 하나를 읽어서 판단 모듈에 넣는다. 확정/재판정된 Track 목록 반환."""
    global zoom_fail_streak
    target = judge.zoom_target(cam, dets)
    if target is None:
        return []
    scene.set_camera(zoom_pose(cam, target, IMG_W, IMG_H), ZOOM_PX, ZOOM_PX, path=zoom_path)
    rep.orchestrator.step(delta_time=0.0, rt_subframes=RT_SUBFRAMES)
    zimg = rgb_array(zoom_annot.get_data())
    if zimg is None:
        return []
    reading, info = read_gauge(zimg)
    # 평가 중 한 번, 중간부터 줌 화면이 계속 판독 불가로 나온 적이 있다 (원인 미상, 재현 안 됨).
    # 연속으로 실패하면 그 화면을 남기고 줌 render product 를 새로 만들어 스스로 복구한다.
    zoom_fail_streak = 0 if reading else zoom_fail_streak + 1
    if zoom_fail_streak >= ZOOM_RESET_AFTER:
        from PIL import Image
        stuck = os.path.join(ROOT, "outputs", "logs", f"zoom_stuck_{frame:06d}.jpg")
        os.makedirs(os.path.dirname(stuck), exist_ok=True)
        Image.fromarray(zimg).save(stuck)
        print(f"[경고] 줌 판독이 {zoom_fail_streak}번 연속 실패해서 줌 카메라를 다시 만들어요 (화면: {stuck}, 정보: {info})")
        make_zoom_render()
        zoom_fail_streak = 0
    if args.save_zoom:
        from PIL import Image, ImageDraw
        # 파일 이름에 시뮬레이션 정답도 남긴다 (판독 정확도 채점용, 판단에는 안 씀)
        xy = judge.locate(cam, target, wall_mounted=True)
        exts = [h for h in scenario.hazards if h.type == "ext"]
        near = min(exts, key=lambda h: np.linalg.norm(h.center[:2] - xy)) if xy is not None else None
        gt = "gtnone" if near is None or np.linalg.norm(near.center[:2] - xy) > 2.0 else ("gtok" if near.ok else "gtlow")
        im = Image.fromarray(zimg)
        ImageDraw.Draw(im).text((6, 6), f"{reading or 'none'} {info.get('cos', '')}", fill=(255, 255, 0))
        im.save(os.path.join(args.save_zoom, f"zoom_{frame:06d}_{reading or 'none'}_{gt}.jpg"), quality=90)
    global last_zoom
    last_zoom = [target, reading, info, float(np.asarray(zimg).mean())]
    return judge.add_zoom_reading(t_sim, cam, target, reading) if reading else []

# ---------------------------------------------------------------- 순찰 루프
print("[안내] 순찰을 시작해요. 창을 닫거나 Ctrl+C로 끝낼 수 있어요.")
t_sim, frame, last = 0.0, 0, time.time()
dump = [] if args.dump_dets else None
last_zoom = None
try:
    while app.is_running():
        now = time.time()
        dt = args.sim_dt if args.sim_dt > 0 else min(0.05, now - last) * args.speed
        last = now
        if dt <= 0:
            app.update()
            continue
        t_sim += dt
        # 로봇의 능동 추적: sim 모드는 가상 검출기, yolo 모드는 화면 판단 결과를 보고 고개를 돌린다
        cam = agent.step(dt, judge.focus_view(t_sim) if judge else det)
        scene.set_camera(cam)
        if args.camera == "robot":
            x, y, yaw, head = agent.base_pose
            scene.set_robot(x, y, yaw, head, visible=args.view == "top")
        else:
            x, y, yaw, bob = agent.base_pose
            scene.set_worker(x, y, yaw, bob, visible=args.view == "top")
        scene.animate(t_sim)

        want_yolo = args.detector == "yolo" and frame % args.yolo_every == 0 and frame > 10
        want_rec = args.record and frame % args.record_every == 0 and frame > 10
        img = None
        if want_yolo or want_rec:
            # Isaac Sim 6.0 은 app.update() 만으로는 rgb 어노테이터가 비어서 나온다 (크기 0).
            # 화면이 필요한 프레임은 Replicator 로 렌더한다 (RT_SUBFRAMES 설명 참고).
            rep.orchestrator.step(delta_time=0.0, rt_subframes=RT_SUBFRAMES)
            img = rgb_array(rgb_annot.get_data())
        else:
            app.update()

        if args.detector == "sim":
            for i in det.update(dt, cam):
                print(event_line(det.hazards[i], t_sim, det.conf[i]))
            shown = []
            for i, h in enumerate(det.hazards):
                if det.detected[i] or (det.in_view[i] and det.conf[i] > det.SHOW_CONF):
                    shown.append((h.aabb_min, h.aabb_max, box_color(h, det.detected[i], det.conf[i])))
            boxes.show(shown)
        elif want_yolo and img is not None:
            dets, res = yolo(img)
            events = judge.update(t_sim, cam, dets)
            last_zoom = None
            if use_zoom:
                events += zoom_read(cam, dets)
            if dump is not None:
                dump.append({"t": t_sim, "cam": [*map(float, cam.pos), cam.yaw, cam.pitch, cam.roll, cam.vfov],
                             "dets": [[n, c, [float(v) for v in b]] for n, c, b in dets], "zoom": last_zoom})
            for tr in events:
                print(judge_line(tr, t_sim))
            boxes.show([(np.array([*tr.pos - 0.4, 0.0]), np.array([*tr.pos + 0.4, 0.8]),
                         RISK_RGBA[RISK[tr.verdict]]) for tr in judge.confirmed])
            if args.save_frames and dets:
                from PIL import Image
                Image.fromarray(res.plot()[..., ::-1]).save(os.path.join(args.save_frames, f"frame_{frame:06d}.jpg"))

        if want_rec and img is not None:
            from PIL import Image
            Image.fromarray(img).save(os.path.join(args.record, f"cam_{frame:06d}.jpg"), quality=90)

        frame += 1
        if args.duration and t_sim >= args.duration:
            break
except KeyboardInterrupt:
    pass

if args.detector == "sim":
    print(summary(det, t_sim))
else:
    print(judge_summary(judge, t_sim, scenario.hazards))
if dump is not None:
    import json
    os.makedirs(os.path.dirname(os.path.abspath(args.dump_dets)), exist_ok=True)
    with open(args.dump_dets, "w", encoding="utf-8") as f:
        json.dump({"seed": seed, "layout_seed": args.layout_seed, "zoom": use_zoom, "frames": dump}, f)
    print(f"[저장] 검출 기록 {len(dump)}프레임: {args.dump_dets}  (python scripts/replay_dets.py {args.dump_dets})")
app.close()
