"""YOLO 학습용 합성 데이터 생성 (Isaac Sim Replicator, NVIDIA 실사 창고).

정답 박스는 Replicator 의 bounding_box_2d_tight 가 실제로 보이는 픽셀 기준으로 계산한다.
물체 묶음(유출+표지판, 소화기+앞을 막은 상자 등)마다 의미 라벨이 붙어 있어서 묶음 전체를 감싼다.

실행 (Isaac Sim 파이썬):
    <isaac>/python.sh scripts/generate_dataset.py --num 4000
    <isaac>/python.sh scripts/generate_dataset.py --num 200 --gui       # 찍는 장면을 보면서
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
parser.add_argument("--scenario-every", type=int, default=40, help="몇 장마다 위험 요소 배치를 바꿀지")
parser.add_argument("--val-every", type=int, default=7, help="몇 장 중 1장을 검증용으로")
parser.add_argument("--no-light-random", action="store_true", help="조명 무작위화 끄기")
parser.add_argument("--no-post", action="store_true", help="흔들림 블러, 노이즈 끄기")
parser.add_argument("--max-occlusion", type=float, default=0.85)
parser.add_argument("--gui", action="store_true", help="창을 띄워서 보면서 생성")
args, _ = parser.parse_known_args()

from factory_safety.isaac_utils import make_app  # noqa: E402

app = make_app(headless=not args.gui)

import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
from PIL import Image  # noqa: E402

from factory_safety.config import CLASSES  # noqa: E402
from factory_safety.dataset import filter_boxes, post_process, sample_capture, write_data_yaml, write_readme, yolo_line  # noqa: E402
from factory_safety.isaac_utils import (attach, disable_capture_on_play, get_annotator, isaac_labeler,  # noqa: E402
                                        new_stage, parse_bboxes, rgb_array, set_viewport_camera, timeline_setter)
from factory_safety.scenario import sample_scenario  # noqa: E402
from factory_safety.scene import WarehouseScene  # noqa: E402
from factory_safety.warehouse import PatrolPath  # noqa: E402

disable_capture_on_play()
stage = new_stage()
scene = WarehouseScene(stage, labeler=isaac_labeler, set_time=timeline_setter()).build()
if args.gui:
    set_viewport_camera(scene.cam_path)

rng = np.random.default_rng(args.seed)
# 위험 요소 배치는 첫 렌더 전에 전부 만들어 두고 보이기/숨기기로만 바꾼다
# (Isaac Sim 6.0 Replicator 는 첫 렌더 뒤에 새로 만든 라벨 물체를 제대로 못 읽음)
n_scen = (args.num + args.scenario_every - 1) // args.scenario_every
print(f"[준비] 작업자 걷기 애니메이션 {'있음' if scene.has_walk else '없음'}, 창고 소화기 자리 {len(scene.ext_mats)}개", flush=True)
print(f"[준비] 배치 {n_scen}개를 미리 만들어요...", flush=True)
for _ in range(n_scen):
    scene.add_scenario(sample_scenario(int(rng.integers(1 << 30))))
scene.show_scenario(0)

rp = rep.create.render_product(scene.cam_path, (args.width, args.height))
rgb_annot = get_annotator("rgb")
bbox_annot = get_annotator("bounding_box_2d_tight")
attach(rgb_annot, rp)
attach(bbox_annot, rp)
for split in ("train", "val"):
    os.makedirs(os.path.join(args.out, "images", split), exist_ok=True)
    os.makedirs(os.path.join(args.out, "labels", split), exist_ok=True)

path = PatrolPath()
counts = [0] * len(CLASSES)
kinds = {"bodycam": 0, "cctv": 0, "free": 0}
n_train = n_val = empties = 0
t0 = time.time()
for _ in range(3):      # 셰이더와 텍스처가 다 올라오게
    rep.orchestrator.step(delta_time=0.0, rt_subframes=16)

for i in range(args.num):
    if i % args.scenario_every == 0:
        scene.show_scenario(i // args.scenario_every)
        for _ in range(2):
            rep.orchestrator.step(delta_time=0.0, rt_subframes=4)
    if not args.no_light_random:
        scene.randomize_lighting(rng)
    pose, (wx, wy, wyaw, wt, wvis), kind = sample_capture(rng, scene.scenario, path)
    kinds[kind] += 1
    scene.set_worker(wx, wy, wyaw, wt, visible=wvis)
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
    name = f"wh_{i + 1:05d}"
    Image.fromarray(img).save(os.path.join(args.out, "images", split, name + ".jpg"), quality=90)
    with open(os.path.join(args.out, "labels", split, name + ".txt"), "w", encoding="utf-8") as f:
        for cls, x0, y0, x1, y1 in boxes:
            f.write(yolo_line(cls, x0, y0, x1, y1, args.width, args.height) + "\n")
            counts[cls] += 1
    empties += not boxes
    if (i + 1) % 50 == 0 or i + 1 == args.num:
        el = time.time() - t0
        print(f"[{i + 1}/{args.num}] {el / (i + 1):.2f}초/장, 라벨 " + ", ".join(f"{c} {n}" for c, n in zip(CLASSES, counts)),
              flush=True)

rep.orchestrator.wait_until_complete()
write_data_yaml(args.out)
write_readme(args.out, args.num, n_train, n_val, counts, args.width, args.height)
print(f"[완료] {args.out}  (학습 {n_train}, 검증 {n_val}, 빈 이미지 {empties}, 시점 {kinds})")
print(f"       다음: python scripts/train_yolo.py --data {os.path.join(args.out, 'data.yaml')}")
app.close()
