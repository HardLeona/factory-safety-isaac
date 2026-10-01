"""줌 카메라 화면으로 소화기 압력계를 읽는다 (영상 처리 규칙, 학습 없음).

광각 카메라(960x540)에서 압력계는 3 m 거리에서도 13픽셀 정도라 바늘(0.4 cm)이 1픽셀도 안 된다.
YOLO 는 압력계 위치는 잘 찾지만 정상/부족 구분은 거의 못 해서 (부족을 정상으로 58% 오판),
실제 점검 로봇처럼 줌 카메라로 압력계를 크게 찍어서 바늘 방향을 읽는다.

판독 방법 (회전, 좌우 반전, 비스듬히 본 경우에도 성립):
  압력계 판 = 흰 바탕 + 녹색 구간(위쪽, 정상) + 적색 구간(양옆 아래, 이상) + 검은 바늘과 가운데 축.
  1) 흰색+녹색 덩어리 중 녹색이 가장 많은 것을 판으로 고른다 (판 밖의 흰 벽은 빠짐).
  2) 판 테두리에 타원을 맞춰 판 중심(= 바늘 축)을 구한다. 압력계 아래쪽은 소화기 몸통에 가려 잘려
     보이는 일이 많아서, 빨간 몸통과 맞닿은 경계는 빼고 진짜 테두리에만 맞춘다.
     비스듬히 보면 판이 타원이 되니 타원 좌표를 원으로 펴서 계산한다.
  3) 중심에서 본 바늘(판 안 어두운 픽셀, 축 둘레는 뺌)의 방향과 녹색 픽셀들의 방향이
     가까우면 정상 (바늘이 녹색 구간), 멀면 부족 (바늘이 적색 구간).
"""
import math

import cv2
import numpy as np
from scipy import ndimage

OK_COS = 0.5      # 바늘과 녹색 방향 사이 각이 60도 안이면 정상
LOW_COS = -0.05   # 약 93도 넘게 벌어지면 부족 (부족 바늘은 95~128도), 그 사이는 판독 보류
ROI_FRAC = 0.42   # 화면 가운데 이 비율(짧은 변 기준 반지름) 안만 압력계로 본다
MIN_RIM_PTS = 12  # 타원을 맞출 테두리 점이 이보다 적으면 보류


def _unit_mean(pts):
    """점들(기준점에서 본 상대 좌표)의 평균 방향 단위벡터. 사방으로 흩어져 있으면 None."""
    n = np.linalg.norm(pts, axis=1)
    keep = n > 1e-6
    if not keep.any():
        return None
    v = (pts[keep] / n[keep, None]).mean(axis=0)
    norm = np.linalg.norm(v)
    return v / norm if norm > 0.3 else None


def read_gauge(img):
    """img: (H, W, 3) uint8 RGB. 반환: ("gauge_normal" | "gauge_low" | None, info dict)."""
    a = np.asarray(img)[..., :3].astype(np.int16)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx, mn = a.max(axis=2), a.min(axis=2)
    hgt, wid = a.shape[:2]
    yy, xx = np.mgrid[0:hgt, 0:wid]
    # 줌 카메라가 압력계를 화면 가운데로 겨누니 가운데 원 안만 본다 (밖에는 흰 벽, 빨간 소화기 몸통이 있음)
    roi = np.hypot(xx - wid / 2.0, yy - hgt / 2.0) < ROI_FRAC * min(hgt, wid)
    # 흰 바탕 기준은 조명에 맞춘다: 가운데 영역의 가장 밝은 무채색(99%)의 80% (어두운 곳에선 판이 160 정도)
    gray = (mx - mn < 45) & roi
    bright = float(np.percentile(mn[gray], 99)) if gray.any() else 200.0
    white = gray & (mn > max(80.0, 0.8 * bright))
    green = (g > r + 35) & (g > b + 20) & (g > 80) & roi
    red = (r > g + 60) & (r > b + 50)
    info = {"green_px": int(green.sum())}
    if info["green_px"] < 15:
        return None, info
    # 1) 판: 흰색+녹색이 이어진 덩어리 중 녹색이 가장 많은 것. 구멍(적색 구간, 바늘, 축)은 메운다
    lab, n = ndimage.label(white | green)
    if n == 0:
        return None, info
    k = int(np.argmax(ndimage.sum(green, lab, index=np.arange(1, n + 1)))) + 1
    face = ndimage.binary_fill_holes(lab == k)
    info["face_px"] = int(face.sum())
    if info["face_px"] < 120:
        return None, info
    green &= face
    # 2) 테두리 타원. 빨간 것(몸통, 또는 몸통과 이어져 메워지지 않은 적색 구간) 옆 경계는 가려진 경계라 뺀다
    contours, _ = cv2.findContours(face.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None, info
    rim = max(contours, key=len)[:, 0, :]
    red_near = ndimage.binary_dilation(red & ~face, iterations=3)
    rim = rim[~red_near[rim[:, 1], rim[:, 0]]]
    info["rim_pts"] = int(len(rim))
    if len(rim) < MIN_RIM_PTS:
        return None, info
    (cx, cy), (ax1, ax2), ang = cv2.fitEllipse(rim.astype(np.float32))
    if min(ax1, ax2) < 6 or max(ax1, ax2) > 4 * min(hgt, wid):
        return None, info
    # 타원 좌표를 원으로 편다 (판 가장자리가 거리 1)
    t = math.radians(ang)
    ca, sa = math.cos(t), math.sin(t)
    dx, dy = xx - cx, yy - cy
    u = (dx * ca + dy * sa) / (ax1 / 2.0)
    v = (-dx * sa + dy * ca) / (ax2 / 2.0)
    dist = np.hypot(u, v)
    rel = np.stack([u, v], axis=-1)
    # 3) 바늘: 판 안 어두운 픽셀, 가운데 축(반지름 0.2 안)과 테두리 근처는 뺀다. 어두움 기준은 판 밝기에 비례
    face_v = float(np.median(mx[white & face])) if (white & face).any() else 200.0
    needle = (mx < min(100.0, 0.45 * face_v)) & face & (dist > 0.2) & (dist < 0.85)
    info["needle_px"] = int(needle.sum())
    if info["needle_px"] < 6:
        return None, info
    ndir = _unit_mean(rel[needle])
    gdir = _unit_mean(rel[green])
    if ndir is None or gdir is None:
        return None, info
    cos = float(ndir @ gdir)
    info["cos"] = round(cos, 3)
    if cos > OK_COS:
        return "gauge_normal", info
    if cos < LOW_COS:
        return "gauge_low", info
    return None, info


def zoom_pose(cam, xyxy, img_w, img_h, fill=0.45, min_fov=2.0, max_fov=20.0):
    """광각 카메라의 압력계 박스 쪽을 겨누는 줌 카메라 자세 (같은 위치, 박스가 화면의 fill 만큼 차게)."""
    from .geometry import CameraPose, Projector
    p = Projector(img_w, img_h)
    p.set_pose(cam)
    u, v = (xyxy[0] + xyxy[2]) / 2.0, (xyxy[1] + xyxy[3]) / 2.0
    d = p.f + p.r * (u - p.w / 2.0) / p.fx + p.u * (p.h / 2.0 - v) / p.fy
    d = d / np.linalg.norm(d)
    size = max(xyxy[2] - xyxy[0], xyxy[3] - xyxy[1])
    ang = 2.0 * math.degrees(math.atan((size / 2.0) / p.fy))
    vfov = float(np.clip(ang / fill, min_fov, max_fov))
    return CameraPose(pos=np.asarray(cam.pos, dtype=float), yaw=math.atan2(d[1], d[0]),
                      pitch=math.asin(float(np.clip(d[2], -1, 1))), roll=0.0, vfov=vfov)
