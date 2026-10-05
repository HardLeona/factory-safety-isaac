"""끼임점(pinch_point) 1클래스 경량 YOLO 학습용 합성 데이터 생성 (Isaac Sim Replicator).

기존 generate_dataset.py 와 같은 방식(바디캠 시점 중심, 조명·흔들림·센서 노이즈 후처리, Replicator
bounding_box_2d_tight 로 자동 라벨링)을 컨베이어 롤러 끼임점 1클래스에 맞게 다시 쓴다.
기존 19클래스 YOLO 모델은 재학습하지 않는다 (scene.py 의 진입 롤러에 붙은 pinch_point 라벨만 본다).

카메라는 끼임점 주변 0.5~3.2 m, 눈높이~허리높이에서 전 방위로 두고, 조명을 무작위화하고, 가끔
작은 상자로 끼임점 일부를 가려 가림 조건도 섞는다.

실행 (Isaac Sim 파이썬):
    <isaac>/python.sh scripts/generate_pinch_dataset.py --num 1500
    <isaac>/python.sh scripts/generate_pinch_dataset.py --num 100 --gui       # 찍는 장면을 보면서
"""
import argparse
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

parser = argparse.ArgumentParser(description="끼임점(pinch_point) 1클래스 합성 데이터 생성")
parser.add_argument("--num", type=int, default=1500, help="이미지 수")
parser.add_argument("--out", default=os.path.join(ROOT, "outputs", "pinch_dataset"))
parser.add_argument("--width", type=int, default=960)
parser.add_argument("--height", type=int, default=540)
parser.add_argument("--rt-subframes", type=int, default=8)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--val-every", type=int, default=7, help="몇 장 중 1장을 검증용으로")
parser.add_argument("--occlude-p", type=float, default=0.3, help="끼임점 일부를 작은 상자로 가릴 확률")
parser.add_argument("--max-occlusion", type=float, default=0.85)
parser.add_argument("--no-light-random", action="store_true")
parser.add_argument("--no-post", action="store_true", help="흔들림 블러, 노이즈 끄기")
parser.add_argument("--gui", action="store_true")
args, _ = parser.parse_known_args()

from factory_safety.isaac_utils import make_app  # noqa: E402

app = make_app(headless=not args.gui)

import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
from PIL import Image  # noqa: E402

from factory_safety import warehouse as W  # noqa: E402
from factory_safety.config import ASSETS  # noqa: E402
from factory_safety.dataset import filter_boxes, post_process, yolo_line  # noqa: E402
from factory_safety.geometry import CameraPose  # noqa: E402
from factory_safety.isaac_utils import (attach, disable_capture_on_play, get_annotator, isaac_labeler,  # noqa: E402
                                        new_stage, parse_bboxes, rgb_array, set_viewport_camera, timeline_setter)
from factory_safety.scene import WarehouseScene  # noqa: E402

CLASS_NAMES = ["pinch_point"]
M = W.MACHINE_CONVEYOR
MX, MY, MZ = M["x"], M["y"], M["pinch_z"]


def _look_at(pos, target):
    d = np.asarray(target, float) - np.asarray(pos, float)
    return math.atan2(d[1], d[0]), math.atan2(d[2], math.hypot(d[0], d[1]))


def sample_pose(rng):
    """끼임점 주변 바디캠 시점: 거리 0.5~3.2 m, 전 방위, 높이 0.5~1.9 m (쪼그려~서서 보는 눈높이)."""
    dist = rng.uniform(0.5, 3.2)
    az = rng.uniform(0, 2 * math.pi)
    z = rng.uniform(0.5, 1.9)
    x, y = MX + dist * math.cos(az), MY + dist * math.sin(az)
    yaw, pitch = _look_at((x, y, z), (MX, MY, MZ + rng.uniform(-0.08, 0.08)))
    return CameraPose(pos=np.array([x, y, z]), yaw=yaw + rng.uniform(-0.3, 0.3),
                      pitch=float(np.clip(pitch + rng.uniform(-0.15, 0.15), -1.1, 0.3)),
                      roll=rng.uniform(-0.05, 0.05), vfov=rng.uniform(55, 78))


def write_data_yaml(out_dir):
    with open(os.path.join(out_dir, "data.yaml"), "w", encoding="utf-8") as f:
        f.write("train: images/train\nval: images/val\n\nnames:\n  0: pinch_point\n")


def write_readme(out_dir, n, n_train, n_val, count):
    with open(os.path.join(out_dir, "README.txt"), "w", encoding="utf-8") as f:
        f.write(f"끼임점(pinch_point) 1클래스 합성 데이터셋 (YOLO 형식, Isaac Sim Replicator)\n\n"
                f"이미지: {n}장 (학습 {n_train}, 검증 {n_val})\n라벨 수: {count}\n\n"
                f"학습: python scripts/train_yolo.py --data <이 폴더>/data.yaml --model yolo26n.pt --name pinch_v1\n")


disable_capture_on_play()
stage = new_stage()
scene = WarehouseScene(stage, labeler=isaac_labeler, set_time=timeline_setter()).build()
if args.gui:
    set_viewport_camera(scene.cam_path)

occ_path = scene.add_box("/World/PinchOcc", "box_d")

rp = rep.create.render_product(scene.cam_path, (args.width, args.height))
rgb_annot = get_annotator("rgb")
bbox_annot = get_annotator("bounding_box_2d_tight")
attach(rgb_annot, rp)
attach(bbox_annot, rp)
for split in ("train", "val"):
    os.makedirs(os.path.join(args.out, "images", split), exist_ok=True)
    os.makedirs(os.path.join(args.out, "labels", split), exist_ok=True)

rng = np.random.default_rng(args.seed)
count = 0
n_train = n_val = empties = 0
t0 = time.time()
for _ in range(3):
    rep.orchestrator.step(delta_time=0.0, rt_subframes=16)

for i in range(args.num):
    if not args.no_light_random:
        scene.randomize_lighting(rng)
    pose = sample_pose(rng)
    scene.set_camera(pose, args.width, args.height)
    occlude = rng.random() < args.occlude_p
    if occlude:
        a = rng.uniform(0, 2 * math.pi)
        r = rng.uniform(0.08, 0.2)
        from pxr import Gf
        scene.set_matrix(occ_path, Gf.Matrix4d().SetTranslate(
            Gf.Vec3d(MX + r * math.cos(a), MY + r * math.sin(a), MZ + rng.uniform(-0.1, 0.1))))
    scene._set_visible(occ_path, occlude)
    rep.orchestrator.step(delta_time=0.0, rt_subframes=args.rt_subframes)

    img = rgb_array(rgb_annot.get_data())
    if img is None:
        print(f"[경고] {i}번 이미지를 읽지 못해서 건너뛰어요.")
        continue
    boxes = filter_boxes(parse_bboxes(bbox_annot.get_data(), CLASS_NAMES), args.width, args.height,
                         max_occlusion=args.max_occlusion)
    if not args.no_post:
        img = post_process(img, rng)
    split = "val" if i % args.val_every == args.val_every // 2 else "train"
    n_val += split == "val"
    n_train += split == "train"
    name = f"pinch_{i + 1:05d}"
    Image.fromarray(img).save(os.path.join(args.out, "images", split, name + ".jpg"), quality=90)
    with open(os.path.join(args.out, "labels", split, name + ".txt"), "w", encoding="utf-8") as f:
        for cls, x0, y0, x1, y1 in boxes:
            f.write(yolo_line(cls, x0, y0, x1, y1, args.width, args.height) + "\n")
            count += 1
    empties += not boxes
    if (i + 1) % 100 == 0 or i + 1 == args.num:
        el = time.time() - t0
        print(f"[{i + 1}/{args.num}] {el / (i + 1):.2f}초/장, 라벨 {count}개, 빈 이미지 {empties}", flush=True)

rep.orchestrator.wait_until_complete()
write_data_yaml(args.out)
write_readme(args.out, args.num, n_train, n_val, count)
print(f"[완료] {args.out}  (학습 {n_train}, 검증 {n_val}, 빈 이미지 {empties})")
print(f"       다음: python scripts/train_yolo.py --data {os.path.join(args.out, 'data.yaml')} --model yolo26n.pt --name pinch_v1")
app.close()
