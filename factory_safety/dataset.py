"""학습 데이터 생성 도우미 (Isaac 없이 테스트 가능).

촬영 시점은 실제로 쓰일 화면과 비슷하게 섞는다:
  바디캠 (55%): 순찰 경로 위 작업자 가슴 높이, 걷는 방향 + 둘러보기. 일부는 물체 쪽을 봄
  CCTV  (30%): CCTV 3대 근처에서 위치와 각도를 흔들어서. 작업자가 화면에 들어오게 둠
  자유  (15%): 물체 주변 1.0~2.2 m 높이에서 그 물체를 봄
"""
import math
import os

import numpy as np

from . import warehouse as W
from .config import CLASS_KO, CLASSES
from .geometry import CameraPose
from .walk_anim import PERIOD

# 촬영 대상 물체를 고르는 가중치 (드문 클래스를 더 자주)
CAPTURE_WEIGHTS = {"spill": 1.2, "spill_marked": 1.2, "tool_floor": 2.0, "tool_stored": 1.0, "stack_unstable": 1.0,
                   "stack_stable": 1.0, "ext_fallen": 1.6, "ext_blocked": 1.6, "ext_ok": 0.6}


def _look_at(pos, target):
    d = np.asarray(target, float) - np.asarray(pos, float)
    return math.atan2(d[1], d[0]), math.atan2(d[2], math.hypot(d[0], d[1]))


def _pick_object(rng, scenario):
    objs = scenario.objects
    w = np.array([CAPTURE_WEIGHTS[o.cls] for o in objs])
    return objs[int(rng.choice(len(objs), p=w / w.sum()))]


def sample_capture(rng, scenario, path):
    """학습 이미지 한 장의 (카메라 자세, 작업자 (x, y, yaw, 애니메이션 시각, 보임), 종류)."""
    r = rng.random()
    s_worker = rng.uniform(0, path.length)
    wp = path.point_at(s_worker)
    worker = (float(wp[0]), float(wp[1]), path.heading_at(s_worker) + (math.pi if rng.random() < 0.3 else 0.0),
              float(rng.uniform(0, PERIOD)), rng.random() < 0.85)
    if r < 0.55:
        s = rng.uniform(0, path.length)
        p, yaw = path.point_at(s), path.heading_at(s)
        lat = rng.uniform(-1.0, 1.0)
        x, y = p[0] - math.sin(yaw) * lat, p[1] + math.cos(yaw) * lat
        z = rng.uniform(1.2, 1.55)
        if rng.random() < 0.5:
            o = _pick_object(rng, scenario)
            if 1.2 < math.hypot(o.x - x, o.y - y) < 9:
                yaw, pitch = _look_at((x, y, z), (o.x, o.y, 0.4))
                yaw += rng.uniform(-0.35, 0.35)
                pitch = float(np.clip(pitch + rng.uniform(-0.12, 0.12), -0.7, 0.1))
            else:
                yaw += rng.uniform(-0.5, 0.5)
                pitch = rng.uniform(-0.35, -0.05)
        else:
            yaw += rng.uniform(-0.6, 0.6) + (math.pi if rng.random() < 0.15 else 0.0)
            pitch = rng.uniform(-0.35, -0.05)
        pose = CameraPose(pos=np.array([x, y, z]), yaw=yaw, pitch=pitch, roll=rng.uniform(-0.06, 0.06),
                          vfov=rng.uniform(60, 78))
        # 바디캠 화면이면 작업자는 다른 곳에 있어야 자기 몸이 화면을 막지 않는다
        if math.hypot(worker[0] - x, worker[1] - y) < 1.5:
            wp = path.point_at(s + path.length / 2)
            worker = (float(wp[0]), float(wp[1]), worker[2], worker[3], worker[4])
        return pose, worker, "bodycam"
    if r < 0.85:
        name, cx, cy, cz, yaw_d, pitch_d, vfov = W.CCTVS[int(rng.integers(len(W.CCTVS)))]
        pos = np.array([cx + rng.uniform(-1.0, 1.0), cy + rng.uniform(-0.3, 0.3), cz + rng.uniform(-0.8, 0.6)])
        pose = CameraPose(pos=pos, yaw=math.radians(yaw_d + rng.uniform(-14, 14)),
                          pitch=math.radians(pitch_d + rng.uniform(-8, 8)), roll=0.0, vfov=vfov + rng.uniform(-8, 8))
        # 작업자를 그 CCTV 가 보는 쪽 경로 위에 둔다
        for _ in range(20):
            s = rng.uniform(0, path.length)
            q = path.point_at(s)
            d = q - pos[:2]
            ang = (math.atan2(d[1], d[0]) - pose.yaw + math.pi) % (2 * math.pi) - math.pi
            if abs(ang) < math.radians(vfov * 0.7) and 3 < np.linalg.norm(d) < 20:
                worker = (float(q[0]), float(q[1]), path.heading_at(s), float(rng.uniform(0, PERIOD)), True)
                break
        return pose, worker, "cctv"
    o = _pick_object(rng, scenario)
    a = rng.uniform(0, 2 * math.pi)
    dist = rng.uniform(1.5, 5.0)
    x, y, z = o.x + dist * math.cos(a), o.y + dist * math.sin(a), rng.uniform(1.0, 2.2)
    x, y = float(np.clip(x, -9.5, 9.5)), float(np.clip(y, -11.5, 17.5))
    yaw, pitch = _look_at((x, y, z), (o.x, o.y, 0.4))
    pose = CameraPose(pos=np.array([x, y, z]), yaw=yaw + rng.uniform(-0.25, 0.25),
                      pitch=float(np.clip(pitch + rng.uniform(-0.1, 0.1), -0.9, 0.1)), roll=rng.uniform(-0.05, 0.05),
                      vfov=rng.uniform(55, 75))
    return pose, worker, "free"


def post_process(img, rng, blur_prob=0.35):
    """흔들림 블러, 밝기/대비, 센서 노이즈 (바디캠, CCTV 느낌). img: (H, W, 3) uint8."""
    out = img.astype(np.float32)
    if rng.random() < blur_prob:
        ang, length, k = rng.uniform(0, 2 * math.pi), rng.uniform(2, 8), 5
        acc = np.zeros_like(out)
        for i in range(k):
            f = (i / (k - 1) - 0.5) * length
            dx, dy = int(round(math.cos(ang) * f)), int(round(math.sin(ang) * f))
            acc += np.roll(np.roll(out, dy, axis=0), dx, axis=1)
        out = acc / k
    bright, contrast, noise = rng.uniform(-28, 18), rng.uniform(0.8, 1.18), rng.uniform(0, 7)
    out = (out - 128) * contrast + 128 + bright
    if noise > 0:
        out += rng.normal(0, noise, out.shape[:2])[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def yolo_line(cls, x0, y0, x1, y1, w, h):
    cx, cy = (x0 + x1) / 2 / w, (y0 + y1) / 2 / h
    bw, bh = (x1 - x0) / w, (y1 - y0) / h
    return f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def filter_boxes(boxes, w, h, min_side=6, max_occlusion=0.85):
    keep = []
    for cls, x0, y0, x1, y1, occ, *rest in boxes:
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
        if x1 - x0 < min_side or y1 - y0 < min_side or occ > max_occlusion:
            continue
        keep.append((cls, x0, y0, x1, y1, *rest))
    return keep


def write_data_yaml(out_dir):
    p = os.path.join(out_dir, "data.yaml")
    with open(p, "w", encoding="utf-8") as f:
        f.write("train: images/train\nval: images/val\n\nnames:\n")
        for i, c in enumerate(CLASSES):
            f.write(f"  {i}: {c}\n")
    return p


def write_readme(out_dir, n, n_train, n_val, counts, width, height):
    lines = [
        "창고 위험 요소 합성 데이터셋 (YOLO 형식, Isaac Sim Replicator, NVIDIA 실사 창고)", "",
        f"이미지: {n}장 (학습 {n_train}, 검증 {n_val}), {width}x{height} JPEG",
        '라벨: labels/ 아래 같은 이름의 .txt, 한 줄에 "클래스 cx cy w h" (0~1 정규화)', "",
        "클래스 (라벨 수)",
    ]
    for i, c in enumerate(CLASSES):
        lines.append(f"  {i} {c:<15} {CLASS_KO[c]} ({counts[i]})")
    lines += ["", "학습: python scripts/train_yolo.py --data <이 폴더>/data.yaml", ""]
    with open(os.path.join(out_dir, "README.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
