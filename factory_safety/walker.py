"""작업자가 미리 정한 경로를 걷고 가슴 바디캠으로 찍는다."""
import math

import numpy as np

from . import warehouse as W
from .geometry import CameraPose
from .walk_anim import GESTURE_CAM_AHEAD, PERIOD, RAISE_S, gesture_time, pull_walk_time


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class PathWalker:
    """순찰 경로를 일정 속도로 걷는 작업자. 바디캠은 가슴 앞 (몸통에 가리지 않게 30 cm 앞).

    사람처럼 좌우를 천천히 둘러보고 (몸 기준 ±20도 안팎), 걸음마다 바디캠이 조금 흔들린다.
    손동작 (start_gesture): 멈춰 서서 정면을 보고 오른손을 바디캠 앞에 올려 손가락 k 개를 보인 뒤 내리고 다시 걷는다.
    시연 (story.py): pause / resume 으로 멈추고, face 로 몸을 어떤 방향으로 돌리고, pulling 이면 왼손으로 카트를 끈다."""
    TURN_RATE = 2.5         # 몸을 돌리는 속도 (rad/s)
    CALM_RATE = 3.0         # 멈추면 둘러보기·흔들림이 잦아드는 속도

    def __init__(self, path=None, speed=1.25, cam_h=1.38, cam_ahead=0.30, vfov=70.0, seed=0):
        self.path = path or W.PatrolPath()
        self.speed, self.cam_h, self.cam_ahead, self.vfov = speed, cam_h, cam_ahead, vfov
        rng = np.random.default_rng(seed)
        self.ph = rng.uniform(0, 2 * math.pi, 3)
        self.reset()

    def reset(self, s0=0.0):
        self.s, self.t = s0, 0.0
        self.gesture = None
        self.paused = False
        self.pulling = False
        self.face_yaw = None        # 몸을 돌려 볼 방향 (None 이면 경로 방향)
        self.body_yaw = None        # 돌리는 중인 몸 방향 (face 를 쓴 뒤에만)
        self.look_pitch = 0.0       # 멈춰 볼 때 더 숙이는 각도
        self.calm = 0.0

    @property
    def laps(self):
        return self.s / self.path.length

    def start_gesture(self, count, hold=1.6):
        self.gesture = {"count": int(count), "t0": self.t, "hold": float(hold)}

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def face(self, yaw, pitch=0.0):
        """몸을 yaw 쪽으로 돌림 (멈춰 있을 때). None 이면 다시 경로 방향."""
        if self.body_yaw is None:
            self.body_yaw = self.path.heading_at(self.s)
        self.face_yaw = yaw
        self.look_pitch = pitch if yaw is not None else 0.0

    @property
    def gesture_alpha(self):
        """손을 올린 정도 (0 내림 ~ 1 다 올림). 손동작 중이 아니면 0."""
        g = self.gesture
        if not g:
            return 0.0
        e = self.t - g["t0"]
        if e < RAISE_S:
            return e / RAISE_S
        if e < RAISE_S + g["hold"]:
            return 1.0
        return max(0.0, (2 * RAISE_S + g["hold"] - e) / RAISE_S)

    @property
    def gesture_count(self):
        """손이 다 올라가 있으면 보이는 손가락 수, 아니면 0."""
        return self.gesture["count"] if self.gesture and self.gesture_alpha >= 0.999 else 0

    @property
    def standing(self):
        return self.paused or self.gesture is not None

    def step(self, dt):
        self.t += dt
        g = self.gesture
        if g and self.t - g["t0"] >= 2 * RAISE_S + g["hold"]:
            self.gesture = g = None
        if not g and not self.paused:
            self.s += self.speed * dt
        if self.paused:
            self.calm = min(1.0, self.calm + self.CALM_RATE * dt)
        elif self.calm > 0:
            self.calm = max(0.0, self.calm - self.CALM_RATE * dt)
        if self.body_yaw is not None:
            target = self.face_yaw if self.face_yaw is not None else self.path.heading_at(self.s)
            d = _wrap(target - self.body_yaw)
            self.body_yaw += max(-self.TURN_RATE * dt, min(self.TURN_RATE * dt, d))
            if self.face_yaw is None and abs(d) < 0.01:
                self.body_yaw = None            # 경로 방향으로 다 돌아옴
        return self.camera()

    @property
    def base_pose(self):
        """(x, y, 몸 방향 yaw, 애니메이션 시각). 손동작 중이면 손동작 클립, 멈춰 있으면 선 자세."""
        p = self.path.point_at(self.s)
        if self.gesture:
            at = gesture_time(self.gesture["count"], self._smooth(), self.pulling)
        elif self.paused:
            at = gesture_time(1, 0.0, self.pulling)
        elif self.pulling:
            at = pull_walk_time(self.t)
        else:
            at = self.t % PERIOD
        yaw = self.body_yaw if self.body_yaw is not None else self.path.heading_at(self.s)
        return float(p[0]), float(p[1]), yaw, at

    def _smooth(self):
        a = self.gesture_alpha
        return a * a * (3 - 2 * a)

    def camera(self):
        x, y, yaw, _ = self.base_pose
        t = self.t
        look = 0.30 * math.sin(2 * math.pi * t / 6.5 + self.ph[0]) + 0.10 * math.sin(2 * math.pi * t / 2.3 + self.ph[1])
        step = 2 * math.pi * t * 2 / PERIOD          # 한 주기에 두 걸음
        bob = 0.018 * math.cos(step)
        pitch = -0.20 + 0.03 * math.sin(step * 0.5 + self.ph[2])
        roll = 0.025 * math.sin(step * 0.5)
        ahead = self.cam_ahead
        g = self._smooth() if self.gesture else 0.0
        calm = max(g, self.calm)
        if calm > 0:
            # 멈추거나 손동작: 정면을 보고 흔들림 없이 (멈춰 볼 때는 조금 숙여서)
            k = 1.0 - calm
            look, bob, roll = look * k, bob * k, roll * k
            pitch = -0.20 + (pitch + 0.20) * k + self.look_pitch * self.calm * (1.0 - g)
        if g > 0:
            # 손동작: 바디캠은 가슴 가까이 (팔 길이 안에 손이 오게)
            ahead = GESTURE_CAM_AHEAD + (self.cam_ahead - GESTURE_CAM_AHEAD) * (1.0 - g)
        pos = np.array([x + ahead * math.cos(yaw), y + ahead * math.sin(yaw), self.cam_h + bob])
        return CameraPose(pos=pos, yaw=yaw + look, pitch=pitch, roll=roll, vfov=self.vfov)


def cctv_pose(name):
    for n, x, y, z, yaw, pitch, vfov in W.CCTVS:
        if n == name:
            return CameraPose(pos=np.array([x, y, z]), yaw=math.radians(yaw), pitch=math.radians(pitch), vfov=vfov)
    raise KeyError(name)
