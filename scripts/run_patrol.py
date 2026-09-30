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
parser.add_argument("--headless", action="store_true")
args, _ = parser.parse_known_args()

from factory_safety.isaac_utils import make_app  # noqa: E402

app = make_app(headless=args.headless)

import numpy as np  # noqa: E402

from factory_safety import layout as L  # noqa: E402
from factory_safety.config import IMG_H, IMG_W  # noqa: E402
from factory_safety.detector import SimDetector  # noqa: E402
from factory_safety.isaac_utils import (DebugBoxes, attach, get_annotator, isaac_labeler, new_stage,  # noqa: E402
                                        rgb_array, set_viewport_camera, set_viewport_top_view)
from factory_safety.patrol import BodycamWalker, RobotPatrol  # noqa: E402
from factory_safety.report import box_color, clock, event_line, summary  # noqa: E402
from factory_safety.scenario import sample_scenario  # noqa: E402
from factory_safety.usd_scene import FactoryStage  # noqa: E402

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

agent = RobotPatrol(track=args.detector == "sim") if args.camera == "robot" else BodycamWalker(seed=seed)
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
for d in (args.save_frames, args.record):
    if d:
        os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------- 순찰 루프
print("[안내] 순찰을 시작해요. 창을 닫거나 Ctrl+C로 끝낼 수 있어요.")
t_sim, frame, last = 0.0, 0, time.time()
last_seen = {}
try:
    while app.is_running():
        now = time.time()
        dt = min(0.05, now - last) * args.speed
        last = now
        if dt <= 0:
            app.update()
            continue
        t_sim += dt
        cam = agent.step(dt, det)
        scene.set_camera(cam)
        if args.camera == "robot":
            x, y, yaw, head = agent.base_pose
            scene.set_robot(x, y, yaw, head, visible=args.view == "top")
        else:
            x, y, yaw, bob = agent.base_pose
            scene.set_worker(x, y, yaw, bob, visible=args.view == "top")
        scene.animate(t_sim)

        if args.detector == "sim":
            for i in det.update(dt, cam):
                print(event_line(det.hazards[i], t_sim, det.conf[i]))
            shown = []
            for i, h in enumerate(det.hazards):
                if det.detected[i] or (det.in_view[i] and det.conf[i] > det.SHOW_CONF):
                    shown.append((h.aabb_min, h.aabb_max, box_color(h, det.detected[i], det.conf[i])))
            boxes.show(shown)
        elif frame % args.yolo_every == 0 and frame > 10:
            img = rgb_array(rgb_annot.get_data())
            if img is not None:
                dets, res = yolo(img)
                for name, conf, xyxy in dets:
                    if t_sim - last_seen.get(name, -99) > 3.0:
                        print(f"[{clock(t_sim)}] YOLO {name} {conf * 100:.0f}%  box {[round(v) for v in xyxy]}")
                    last_seen[name] = t_sim
                if args.save_frames and dets:
                    from PIL import Image
                    Image.fromarray(res.plot()[..., ::-1]).save(os.path.join(args.save_frames, f"frame_{frame:06d}.jpg"))

        if args.record and frame % args.record_every == 0 and frame > 10:
            img = rgb_array(rgb_annot.get_data())
            if img is not None:
                from PIL import Image
                Image.fromarray(img).save(os.path.join(args.record, f"cam_{frame:06d}.jpg"), quality=90)

        app.update()
        frame += 1
        if args.duration and t_sim >= args.duration:
            break
except KeyboardInterrupt:
    pass

if args.detector == "sim":
    print(summary(det, t_sim))
app.close()
