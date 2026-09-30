"""순찰 경로와 카메라 움직임.

RobotPatrol  : 순찰 로봇. 좌우로 훑다가 의심 물체가 보이면 그쪽으로 카메라를 돌림
BodycamWalker: 작업자 가슴 바디캠. 걸음에 따른 흔들림, 멈춰서 두리번거리기
"""
import math

import numpy as np

from .config import BODY_H, CAM_H_ROBOT
from .geometry import CameraPose, wrap_angle
from .layout import PATROL_PATH


def _cr_centripetal(p0, p1, p2, p3, n, alpha=0.5):
    def tj(ti, a, b):
        return ti + max(np.linalg.norm(b - a), 1e-6) ** alpha

    t0 = 0.0
    t1 = tj(t0, p0, p1)
    t2 = tj(t1, p1, p2)
    t3 = tj(t2, p2, p3)
    t = np.linspace(t1, t2, n, endpoint=False)[:, None]
    a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
    a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
    a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
    b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
    b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
    return (t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2


class ClosedPath:
    """제어점을 지나는 닫힌 곡선 (centripetal Catmull-Rom). 길이 기준 매개변수 u ∈ [0, 1)."""

    def __init__(self, points=PATROL_PATH, samples_per_seg=80):
        pts = np.array(points, dtype=float)
        n = len(pts)
        segs = [_cr_centripetal(pts[(i - 1) % n], pts[i], pts[(i + 1) % n], pts[(i + 2) % n], samples_per_seg)
                for i in range(n)]
        self.pts = np.vstack(segs)
        loop = np.vstack([self.pts, self.pts[:1]])
        seg_len = np.linalg.norm(np.diff(loop, axis=0), axis=1)
        self.cum = np.concatenate([[0.0], np.cumsum(seg_len)])
        self.length = float(self.cum[-1])
        self._loop = loop

    def point_at(self, u):
        s = (u % 1.0) * self.length
        i = int(np.clip(np.searchsorted(self.cum, s, side="right") - 1, 0, len(self.cum) - 2))
        k = (s - self.cum[i]) / max(self.cum[i + 1] - self.cum[i], 1e-9)
        return self._loop[i] * (1 - k) + self._loop[i + 1] * k

    def tangent_at(self, u, eps=1e-3):
        d = self.point_at(u + eps) - self.point_at(u - eps)
        return d / max(np.linalg.norm(d), 1e-9)

    def nearest_u(self, x, y):
        d = np.sum((self.pts - [x, y]) ** 2, axis=1)
        i = int(np.argmin(d))
        return self.cum[i] / self.length


class RobotPatrol:
    """고정 경로를 도는 순찰 로봇 + 능동 시선 추적."""

    def __init__(self, path=None, speed=1.5, pitch=-0.13, vfov=66.0, track=True):
        self.path = path or ClosedPath()
        self.speed, self.pitch, self.vfov, self.track = speed, pitch, vfov, track
        self.reset()

    def reset(self, u=0.0):
        self.u, self.laps, self.t = u, 0, 0.0
        self.look_off, self.sweep_phase = 0.0, 0.0
        self.tracking = None
        self.heading = 0.0
        self.pos2 = self.path.point_at(u)

    def step(self, dt, detector=None):
        self.t += dt
        self.u += dt * self.speed / self.path.length
        if self.u >= 1.0:
            self.u -= 1.0
            self.laps += 1
        p = self.path.point_at(self.u)
        t = self.path.tangent_at(self.u)
        self.pos2 = p
        self.heading = math.atan2(t[1], t[0])

        want = None
        self.tracking = None
        if self.track and detector is not None:
            best = 0.0
            for i, h in enumerate(detector.hazards):
                if detector.detected[i] or not detector.in_view[i] or detector.conf[i] < 0.38:
                    continue
                if h.read_range and math.hypot(h.center[0] - p[0], h.center[1] - p[1]) > h.read_range + 1.5:
                    continue
                if detector.conf[i] > best:
                    best, self.tracking = detector.conf[i], i
            if self.tracking is not None:
                c = detector.hazards[self.tracking].center
                off = wrap_angle(math.atan2(c[1] - p[1], c[0] - p[0]) - self.heading)
                want = float(np.clip(off, -1.9, 1.9))
        if want is None:
            self.sweep_phase += dt * 0.6
            want = 1.0 * math.sin(self.sweep_phase)
        rate = 5.0 if self.tracking is not None else 6.0
        self.look_off += (want - self.look_off) * (1 - math.exp(-dt * rate))
        return self.camera()

    def camera(self):
        yaw = self.heading + self.look_off
        pos = np.array([self.pos2[0], self.pos2[1], CAM_H_ROBOT])
        return CameraPose(pos=pos, yaw=yaw, pitch=self.pitch, roll=0.0, vfov=self.vfov)

    @property
    def base_pose(self):
        """로봇 몸체 (x, y, yaw) 와 머리 회전각."""
        return float(self.pos2[0]), float(self.pos2[1]), self.heading, self.look_off


class BodycamWalker:
    """작업자 가슴 바디캠."""

    def __init__(self, path=None, speed=1.25, vfov=78.0, seed=None):
        self.path = path or ClosedPath()
        self.speed, self.vfov = speed, vfov
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self, u=0.0):
        r = self.rng
        self.u, self.laps, self.t = u, 0, 0.0
        self.walk, self.phase, self.pause = 0.0, 0.0, 0.0
        self.next_pause = r.uniform(10, 18)
        self.yaw, self.yaw_goal, self.yaw_t = 0.0, 0.0, 0.0
        self.pitch, self.pitch_goal, self.pitch_t = -0.18, -0.18, 0.0
        self.lat, self.lat_goal, self.lat_t = 0.0, 0.0, 0.0
        self.heading, self.pos2, self.cur_speed = 0.0, self.path.point_at(u), 0.0
        self._pose = None
        self.step(0.0)

    def step(self, dt, detector=None):
        r = self.rng
        self.t += dt
        v = self.speed
        self.next_pause -= dt
        if self.pause > 0:
            self.pause -= dt
            v = 0.0
        elif self.next_pause <= 0:
            self.pause, self.next_pause, self.yaw_t = r.uniform(1.5, 3.5), r.uniform(12, 24), 0.0
        self.cur_speed = v

        def e(rate):
            return 1 - math.exp(-dt * rate)

        self.walk += ((1.0 if v > 0 else 0.0) - self.walk) * e(5)
        self.u += dt * v / self.path.length
        if self.u >= 1.0:
            self.u -= 1.0
            self.laps += 1
        self.phase += dt * 2 * math.pi * 1.9 * self.walk
        self.yaw_t -= dt
        if self.yaw_t <= 0:
            self.yaw_t = r.uniform(1.2, 3.5)
            self.yaw_goal = r.uniform(-1.2, 1.2) if self.pause > 0 else r.uniform(-0.35, 0.35)
        self.pitch_t -= dt
        if self.pitch_t <= 0:
            self.pitch_t, self.pitch_goal = r.uniform(1.5, 4), r.uniform(-0.36, -0.05)
        self.lat_t -= dt
        if self.lat_t <= 0:
            self.lat_t, self.lat_goal = r.uniform(3, 7), r.uniform(-0.7, 0.7)
        self.yaw += (self.yaw_goal - self.yaw) * e(2.2)
        self.pitch += (self.pitch_goal - self.pitch) * e(1.8)
        self.lat += (self.lat_goal - self.lat) * e(0.8)

        p, t = self.path.point_at(self.u), self.path.tangent_at(self.u)
        self.heading = math.atan2(t[1], t[0])
        ph, a = self.phase, self.walk
        bob = 0.028 * a * math.cos(ph)
        off = self.lat + 0.03 * a * math.sin(ph / 2)
        n = np.array([-t[1], t[0]])                 # 진행 방향 왼쪽
        xy = p + n * off
        self.pos2 = xy
        tt = self.t
        shake_p = 0.012 * a * math.sin(ph + 0.6) + 0.003 * math.sin(tt * 11.3) + 0.002 * math.sin(tt * 17.9)
        shake_y = 0.01 * a * math.sin(ph / 2 + 0.3) + 0.003 * math.sin(tt * 9.1)
        roll = 0.035 * a * math.sin(ph / 2) + 0.004 * math.sin(tt * 13.7)
        self.body_yaw = self.heading + self.yaw
        self.bob = bob
        self._pose = CameraPose(pos=np.array([xy[0], xy[1], BODY_H + bob]), yaw=self.body_yaw + shake_y,
                                pitch=self.pitch + shake_p, roll=roll, vfov=self.vfov)
        return self._pose

    def camera(self):
        return self._pose

    @property
    def base_pose(self):
        """작업자 몸 (x, y, yaw) 와 위아래 흔들림. 몸은 카메라보다 0.145 m 뒤."""
        x = self.pos2[0] - math.cos(self.body_yaw) * 0.145
        y = self.pos2[1] - math.sin(self.body_yaw) * 0.145
        return float(x), float(y), self.body_yaw, self.bob
