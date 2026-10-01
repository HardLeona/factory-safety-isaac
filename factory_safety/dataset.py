"""학습 데이터 생성에 쓰는 순수 파이썬 도우미 (Isaac 없이 테스트 가능)."""
import math
import os

import numpy as np

from .config import CLASS_KO, CLASSES
from .geometry import CameraPose


# 드문 클래스가 더 자주 찍히도록 가중치 (generate_dataset.py --tool-weight 로 공구만 바꿀 수 있음)
CAPTURE_WEIGHTS = {"ext_ok": 0.35, "ext_low": 1.8, "tool": 1.7, "puddle": 1.2}


def capture_weight(h):
    """드문 클래스가 더 자주 찍히도록 가중치."""
    if h.type == "ext":
        return CAPTURE_WEIGHTS["ext_ok" if h.ok else "ext_low"]
    return CAPTURE_WEIGHTS.get(h.type, 1.0)


def sample_capture_pose(rng, scenario, path):
    """통로 위 임의 지점에서 임의 높이 (로봇 1.55 / 가슴 1.2~1.5 / 헬멧 1.5~1.8)로 찍는 시점.
    75%는 위험 요소 하나를 골라 그 근처에서 그쪽을 바라본다."""
    subject = None
    hz = scenario.hazards
    if hz and rng.random() < 0.75:
        w = np.array([capture_weight(h) for h in hz])
        subject = hz[int(rng.choice(len(hz), p=w / w.sum()))]
    if subject is not None:
        # 작은 물체는 가까이서 찍는다 (공구는 0.4 m 라 7 m 밖에서는 박스가 6픽셀 안 돼서 라벨에서 빠짐)
        span = {"ext": 3.5, "tool": 4.0}.get(subject.type, 7.0) / path.length
        u = (path.nearest_u(subject.center[0], subject.center[1]) + rng.uniform(-1, 1) * span) % 1.0
    else:
        u = rng.random()
    p, t = path.point_at(u), path.tangent_at(u)
    lat = rng.uniform(-1.2, 1.2)
    x, y = p[0] - t[1] * lat, p[1] + t[0] * lat
    cam_z = rng.uniform(1.2, 1.5) if rng.random() < 0.5 else rng.uniform(1.5, 1.8)
    if subject is not None:
        d = math.hypot(subject.center[0] - x, subject.center[1] - y)
        if d < 1.2 or d > 14:
            subject = None
    if subject is not None:
        dx, dy = subject.center[0] - x, subject.center[1] - y
        yaw = math.atan2(dy, dx) + rng.uniform(-0.45, 0.45)
        pitch = float(np.clip(math.atan2(subject.center[2] - cam_z, math.hypot(dx, dy)) + rng.uniform(-0.18, 0.12), -0.75, 0.2))
    else:
        yaw = math.atan2(t[1], t[0]) + (math.pi if rng.random() < 0.3 else 0.0) + rng.uniform(-1.3, 1.3)
        pitch = rng.uniform(-0.45, 0.05)
    return CameraPose(pos=np.array([x, y, cam_z]), yaw=yaw, pitch=pitch, roll=rng.uniform(-0.07, 0.07),
                      vfov=rng.uniform(55, 78))


def post_process(img, rng, blur_prob=0.35):
    """흔들림 블러, 밝기/대비, 센서 노이즈 (바디캠 느낌). img: (H, W, 3) uint8."""
    out = img.astype(np.float32)
    if rng.random() < blur_prob:
        ang, length, k = rng.uniform(0, 2 * math.pi), rng.uniform(2, 8), 5
        acc = np.zeros_like(out)
        for i in range(k):
            f = (i / (k - 1) - 0.5) * length
            dx, dy = int(round(math.cos(ang) * f)), int(round(math.sin(ang) * f))
            acc += np.roll(np.roll(out, dy, axis=0), dx, axis=1)
        out = acc / k
    bright, contrast, noise = rng.uniform(-24, 18), rng.uniform(0.8, 1.18), rng.uniform(0, 6)
    out = (out - 128) * contrast + 128 + bright
    if noise > 0:
        out += rng.normal(0, noise, out.shape[:2])[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def yolo_line(cls, x0, y0, x1, y1, w, h):
    cx, cy = (x0 + x1) / 2 / w, (y0 + y1) / 2 / h
    bw, bh = (x1 - x0) / w, (y1 - y0) / h
    return f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def filter_boxes(boxes, w, h, min_side=6, max_occlusion=0.8):
    keep = []
    for cls, x0, y0, x1, y1, occ in boxes:
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
        min_s = 4 if CLASSES[cls].startswith("gauge") else min_side
        if x1 - x0 < min_s or y1 - y0 < min_s or occ > max_occlusion:
            continue
        keep.append((cls, x0, y0, x1, y1))
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
        "공장 위험 요소 합성 데이터셋 (YOLO 형식, Isaac Sim Replicator)", "",
        f"이미지: {n}장 (학습 {n_train}, 검증 {n_val}), {width}x{height} JPEG",
        '라벨: labels/ 아래 같은 이름의 .txt, 한 줄에 "클래스 cx cy w h" (0~1 정규화)', "",
        "클래스 (라벨 수)",
    ]
    for i, c in enumerate(CLASSES):
        lines.append(f"  {i} {c:<15} {CLASS_KO[c]} ({counts[i]})")
    lines += ["", "학습: python scripts/train_yolo.py --data <이 폴더>/data.yaml", ""]
    with open(os.path.join(out_dir, "README.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
