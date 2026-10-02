"""작업자가 미리 정한 경로를 걷고 가슴 바디캠으로 찍는다."""
import math

import numpy as np

from . import warehouse as W
from .geometry import CameraPose
from .walk_anim import PERIOD


class PathWalker:
    """순찰 경로를 일정 속도로 걷는 작업자. 바디캠은 가슴 앞 (몸통에 가리지 않게 30 cm 앞).

    사람처럼 좌우를 천천히 둘러보고 (몸 기준 ±20도 안팎), 걸음마다 바디캠이 조금 흔들린다."""

    def __init__(self, path=None, speed=1.25, cam_h=1.38, cam_ahead=0.30, vfov=70.0, seed=0):
        self.path = path or W.PatrolPath()
        self.speed, self.cam_h, self.cam_ahead, self.vfov = speed, cam_h, cam_ahead, vfov
        rng = np.random.default_rng(seed)
        self.ph = rng.uniform(0, 2 * math.pi, 3)
        self.reset()

    def reset(self, s0=0.0):
        self.s, self.t = s0, 0.0

    @property
    def laps(self):
        return self.s / self.path.length

    def step(self, dt):
        self.t += dt
        self.s += self.speed * dt
        return self.camera()

    @property
    def base_pose(self):
        """(x, y, 몸 방향 yaw, 걷기 애니메이션 시각)."""
        p = self.path.point_at(self.s)
        return float(p[0]), float(p[1]), self.path.heading_at(self.s), self.t % PERIOD

    def camera(self):
        x, y, yaw, _ = self.base_pose
        t = self.t
        look = 0.30 * math.sin(2 * math.pi * t / 6.5 + self.ph[0]) + 0.10 * math.sin(2 * math.pi * t / 2.3 + self.ph[1])
        step = 2 * math.pi * t * 2 / PERIOD          # 한 주기에 두 걸음
        bob = 0.018 * math.cos(step)
        cy = yaw + look
        pos = np.array([x + self.cam_ahead * math.cos(yaw), y + self.cam_ahead * math.sin(yaw), self.cam_h + bob])
        return CameraPose(pos=pos, yaw=cy, pitch=-0.20 + 0.03 * math.sin(step * 0.5 + self.ph[2]),
                          roll=0.025 * math.sin(step * 0.5), vfov=self.vfov)


def cctv_pose(name):
    for n, x, y, z, yaw, pitch, vfov in W.CCTVS:
        if n == name:
            return CameraPose(pos=np.array([x, y, z]), yaw=math.radians(yaw), pitch=math.radians(pitch), vfov=vfov)
    raise KeyError(name)
