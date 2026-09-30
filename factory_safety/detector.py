"""검출기.

SimDetector : 정답 위치를 알고 거리, 시야각, 가림 여부로 탐지를 흉내내는 가상 검출기.
              순찰 로직 개발과 강화학습 보상 계산에 쓴다.
YoloDetector: 학습한 YOLO 모델로 실제 카메라 영상에서 탐지 (ultralytics 필요).
"""
import math

import numpy as np

from .config import IMG_H, IMG_W
from .geometry import Projector, aabb_corners, segments_blocked


class SimDetector:
    CONFIRM_CONF = 0.70
    SHOW_CONF = 0.42

    def __init__(self, hazards, occ_mins, occ_maxs, img_w=IMG_W, img_h=IMG_H):
        self.hazards = list(hazards)
        self.static_mins, self.static_maxs = np.asarray(occ_mins), np.asarray(occ_maxs)
        self.proj = Projector(img_w, img_h)
        n = len(self.hazards)
        # 다른 위험 요소도 서로를 가릴 수 있으니 가림 목록에 포함 (자기 자신은 제외)
        self.h_mins = np.array([h.aabb_min for h in self.hazards]) if n else np.zeros((0, 3))
        self.h_maxs = np.array([h.aabb_max for h in self.hazards]) if n else np.zeros((0, 3))
        self.centers = np.array([h.center for h in self.hazards]) if n else np.zeros((0, 3))
        self.corners = [aabb_corners(h.aabb_min, h.aabb_max) for h in self.hazards]
        self.time = 0.0
        self.reset()

    def reset(self):
        n = len(self.hazards)
        self.conf = np.zeros(n)
        self.seen = np.zeros(n)
        self.detected = np.zeros(n, dtype=bool)
        self.detected_at = np.full(n, np.nan)
        self.in_view = np.zeros(n, dtype=bool)
        self.los = np.ones(n, dtype=bool)
        self.last_ray = np.full(n, -1.0)
        self.rects = [None] * n
        self.time = 0.0

    def _occluders_except(self, i):
        keep = np.ones(len(self.hazards), dtype=bool)
        keep[i] = False
        return (np.vstack([self.static_mins, self.h_mins[keep]]),
                np.vstack([self.static_maxs, self.h_maxs[keep]]))

    def line_of_sight(self, origin, target, i):
        mins, maxs = self._occluders_except(i)
        return not bool(segments_blocked(origin, target[None], mins, maxs, end_margin=0.1)[0])

    def update(self, dt, cam):
        """cam: CameraPose. 이번 스텝에 새로 확정된 위험 요소 인덱스 목록을 반환."""
        self.time += dt
        self.proj.set_pose(cam)
        cp = np.asarray(cam.pos, dtype=float)
        k = 1 - math.exp(-dt * 5)
        new = []
        for i, h in enumerate(self.hazards):
            d = float(np.linalg.norm(self.centers[i] - cp))
            target = 0.0
            rect = self.proj.rect_of_points(self.corners[i], min_front=0.01) if d < h.range else None
            self.rects[i] = rect
            view = rect is not None
            if view and h.normal is not None:
                v = (cp - self.centers[i]) / max(d, 1e-9)
                if float(v @ h.normal) < 0.12:
                    view = False
            if view:
                if self.last_ray[i] < 0 or self.time - self.last_ray[i] > 0.15:
                    self.last_ray[i] = self.time
                    self.los[i] = self.line_of_sight(cp, self.centers[i], i)
                if self.los[i]:
                    uv, _ = self.proj.project(self.centers[i][None])
                    nx = (uv[0, 0] - self.proj.w / 2) / (self.proj.w / 2)
                    ny = (uv[0, 1] - self.proj.h / 2) / (self.proj.h / 2)
                    off = min(1.2, math.hypot(nx, ny * 0.8)) if np.all(np.isfinite(uv)) else 1.2
                    dist_f = max(0.0, 1 - d / h.range)
                    target = 0.3 + 0.69 * dist_f ** 0.55 * (1 - 0.32 * off)
                else:
                    view = False
            self.in_view[i] = view
            self.conf[i] += (target - self.conf[i]) * k
            ready = self.conf[i] > self.CONFIRM_CONF and (h.read_range is None or d < h.read_range)
            self.seen[i] = self.seen[i] + dt if ready else max(0.0, self.seen[i] - dt * 0.6)
            need = 0.8 if h.type == "ext" else 0.5
            if not self.detected[i] and self.seen[i] > need:
                self.detected[i] = True
                self.detected_at[i] = self.time
                new.append(i)
        return new

    @property
    def found_targets(self):
        return int(sum(1 for i, h in enumerate(self.hazards) if h.is_hazard and self.detected[i]))

    @property
    def total_targets(self):
        return int(sum(1 for h in self.hazards if h.is_hazard))


class YoloDetector:
    """ultralytics YOLO 래퍼. rgb: (H, W, 3|4) uint8."""

    def __init__(self, weights, conf=0.35, imgsz=640, device=None):
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError("ultralytics가 필요해요. Isaac Sim 파이썬에 설치: python.sh -m pip install ultralytics") from e
        self.model = YOLO(weights)
        self.conf, self.imgsz, self.device = conf, imgsz, device
        self.names = self.model.names

    def __call__(self, rgb):
        img = np.ascontiguousarray(np.asarray(rgb)[..., :3][..., ::-1])   # RGB -> BGR (ultralytics numpy 입력 규약)
        res = self.model.predict(img, conf=self.conf, imgsz=self.imgsz, device=self.device, verbose=False)[0]
        out = []
        for b in res.boxes:
            c = int(b.cls[0])
            out.append((self.names[c], float(b.conf[0]), [float(v) for v in b.xyxy[0].tolist()]))
        return out, res
