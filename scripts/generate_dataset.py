"""YOLO 학습용 합성 데이터 생성 (Isaac Sim Replicator).

정답 박스는 Replicator의 bounding_box_2d_tight 어노테이터가 실제로 보이는 픽셀 기준으로 계산한다.
가려진 부분이 많은 물체(occlusion > 0.8)와 너무 작은 박스는 뺀다.

실행 (Isaac Sim 파이썬):
    <isaac>/python.sh scripts/generate_dataset.py --num 1000
    <isaac>/python.sh scripts/generate_dataset.py --num 300 --gui       # 찍는 장면을 보면서
"""
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

parser = argparse.ArgumentParser(description="YOLO 합성 데이터 생성")
parser.add_argument("--num", type=int, default=500, help="이미지 수")
parser.add_argument("--out", default=os.path.join(ROOT, "outputs", "dataset"))
parser.add_argument("--width", type=int, default=960)
parser.add_argument("--height", type=int, default=540)
parser.add_argument("--rt-subframes", type=int, default=8, help="프레임마다 렌더링 반복 수 (높을수록 깨끗, 느림)")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--layout-seed", type=int, default=0)
parser.add_argument("--scenario-every", type=int, default=25, help="몇 장마다 위험 요소 배치를 새로 섞을지")
parser.add_argument("--val-every", type=int, default=7, help="몇 장 중 1장을 검증용으로")
parser.add_argument("--no-light-random", action="store_true", help="조명 무작위화 끄기")
parser.add_argument("--no-post", action="store_true", help="흔들림 블러, 노이즈 끄기")
parser.add_argument("--max-occlusion", type=float, default=0.8)
parser.add_argument("--gui", action="store_true", help="창을 띄워서 보면서 생성")
args, _ = parser.parse_known_args()

from factory_safety.isaac_utils import make_app  # noqa: E402

app = make_app(headless=not args.gui)

import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
from PIL import Image  # noqa: E402

from factory_safety.config import CLASSES  # noqa: E402
from factory_safety.dataset import (filter_boxes, post_process, sample_capture_pose, write_data_yaml,  # noqa: E402
                                    write_readme, yolo_line)
from factory_safety.isaac_utils import (attach, disable_capture_on_play, get_annotator, isaac_labeler,  # noqa: E402
                                        new_stage, parse_bboxes, rgb_array, set_viewport_camera)
from factory_safety.patrol import ClosedPath  # noqa: E402
from factory_safety.scenario import sample_scenario  # noqa: E402
from factory_safety.usd_scene import FactoryStage  # noqa: E402

disable_capture_on_play()
stage = new_stage()
scene = FactoryStage(stage, os.path.join(ROOT, "outputs", "textures"), static_seed=args.layout_seed,
                     labeler=isaac_labeler).build()
scene.set_robot_visible(False)
if args.gui:
    set_viewport_camera(scene.cam_path)

rp = rep.create.render_product(scene.cam_path, (args.width, args.height))
rgb_annot = get_annotator("rgb")
bbox_annot = get_annotator("bounding_box_2d_tight")
attach(rgb_annot, rp)
attach(bbox_annot, rp)

for split in ("train", "val"):
    os.makedirs(os.path.join(args.out, "images", split), exist_ok=True)
    os.makedirs(os.path.join(args.out, "labels", split), exist_ok=True)

rng = np.random.default_rng(args.seed)
path = ClosedPath()
counts = [0] * len(CLASSES)
n_train = n_val = empties = 0
scenario = None
t0 = time.time()

# 첫 몇 프레임은 셰이더와 텍스처가 덜 올라와서 버린다
scene.set_scenario(sample_scenario(int(rng.integers(1 << 30))))
for _ in range(3):
    rep.orchestrator.step(delta_time=0.0, rt_subframes=16)

for i in range(args.num):
    if i % args.scenario_every == 0:
        scenario = sample_scenario(int(rng.integers(1 << 30)))
        scene.set_scenario(scenario)
    if not args.no_light_random:
        scene.randomize_lighting(rng)
    scene.animate(rng.uniform(0, 100))
    pose = sample_capture_pose(rng, scenario, path)
    scene.set_camera(pose, args.width, args.height)
    rep.orchestrator.step(delta_time=0.0, rt_subframes=args.rt_subframes)

    img = rgb_array(rgb_annot.get_data())
    if img is None:
        print(f"[경고] {i}번 이미지를 읽지 못해서 건너뛰어요.")
        continue
    boxes = filter_boxes(parse_bboxes(bbox_annot.get_data(), CLASSES), args.width, args.height,
                         max_occlusion=args.max_occlusion)
    if not args.no_post:
        img = post_process(img, rng)

    split = "val" if i % args.val_every == args.val_every // 2 else "train"
    n_val += split == "val"
    n_train += split == "train"
    name = f"factory_{i + 1:05d}"
    Image.fromarray(img).save(os.path.join(args.out, "images", split, name + ".jpg"), quality=90)
    with open(os.path.join(args.out, "labels", split, name + ".txt"), "w", encoding="utf-8") as f:
        for cls, x0, y0, x1, y1 in boxes:
            f.write(yolo_line(cls, x0, y0, x1, y1, args.width, args.height) + "\n")
            counts[cls] += 1
    empties += not boxes
    if (i + 1) % 25 == 0 or i + 1 == args.num:
        el = time.time() - t0
        print(f"[{i + 1}/{args.num}] {el / (i + 1):.2f}초/장, 라벨 " + ", ".join(f"{c} {n}" for c, n in zip(CLASSES, counts)))

rep.orchestrator.wait_until_complete()
write_data_yaml(args.out)
write_readme(args.out, args.num, n_train, n_val, counts, args.width, args.height)
print(f"[완료] {args.out}  (학습 {n_train}, 검증 {n_val}, 빈 이미지 {empties})")
print(f"       다음: python scripts/train_yolo.py --data {os.path.join(args.out, 'data.yaml')}")
app.close()
