"""시연 이야기 (run_patrol.py --story): 작업자가 하는 일. 에이전트는 이걸 모르고 바디캠 화면과 손동작으로만 안다.

    ① 시작하자마자 손가락 2 → 오늘의 TBM (상자 4개를 카트로 북쪽 보관 구역에서 남쪽 작업 구역으로)
    ② 순찰하며 걷기 (에이전트가 위험 요소 판정)
    ③ 북쪽 보관 구역에서 멈춰 운반 카트를 3초 보고 손가락 1 → 카트 쓰는 법과 주의점
    ④ 옆 팔레트의 상자 4개를 카트에 싣고, 카트를 뒤로 끌며 걷기
    ⑤ 한 바퀴를 다 돌면 손가락 3 → 관리자 호출, 끝 (남은 점검 지점은 "현장 확인 필요"로 표시)
손동작이 인식되지 않으면 한 번 더 보이고, 앞 안내가 끝나야 다음 손동작을 한다.
"""
import math

import numpy as np

from .config import ASSETS

STATION_XY = (-0.3, 15.6)               # 작업자가 멈추는 경로 위 자리 (북쪽 보관 구역)
CART_PARK = (-1.05, 16.75, math.pi)     # 카트를 세워 둔 자리 (손잡이가 남쪽, 작업자 쪽)
PALLET_XY = (0.6, 16.9)                 # 실을 상자가 놓인 팔레트
PALLET_TOP = 0.21
BOX = "box_c"                           # 카트에 싣는 상자 (팔레트 위층)
BASE_BOX = "box_b"                      # 팔레트 아래층에 남는 상자 (반듯한 2단 적재로 보이게)
N_BOXES = 4
BOX_GRID = [(-0.27, -0.25), (0.27, -0.25), (-0.27, 0.25), (0.27, 0.25)]
LOAD_Z = PALLET_TOP + 0.5               # 실을 상자는 아래층 (0.5 m) 위에
FOLLOW_M = 1.25                         # 카트 바퀴 축은 작업자 뒤 이만큼 (경로를 따라)
PULL_TILT = 40.0                        # 끌 때 카트 기울기 (도)
LOOK_S = 3.0                            # 카트를 보는 시간
TURN_S = 0.8
START_S = 1.5                           # 경로 출발점에서 이만큼 앞에서 시작 (바로 옆 바닥 공구에 닿지 않게)
LOAD_EACH_S = 1.0                       # 상자 하나 싣는 시간
ATTACH_S = 1.2                          # 카트를 끄는 자세로 잡는 시간
HOLD = 2.0                              # 손을 들고 있는 시간
GAP_S = 0.5
LABELS = {"start": "준비", "tbm": "TBM 듣기", "walk": "순찰", "look": "운반 카트 확인", "load": "상자 싣기", "attach": "카트 잡기",
          "pull": "카트 끌며 순찰", "end": "순찰 끝", "call": "관리자 호출", "done": "끝"}


def _ease(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


def _lerp_angle(a, b, u):
    d = (b - a + math.pi) % (2 * math.pi) - math.pi
    return a + d * u


class Story:
    def __init__(self, walker, prefix="/World/Story"):
        self.w = walker
        self.path = walker.path
        self.prefix = prefix
        self.s_station = self.path.nearest_s(*STATION_XY)
        self.state, self.t_state = "start", 0.0
        self.shown, self.heard = [], set()
        self.busy_until = 0.0
        self.pending = None             # 지금 보이는 손동작 (수, 시작 시각)
        self.cart = (CART_PARK[0], CART_PARK[1], CART_PARK[2], 0.0)
        self.cart_from = None
        self.box_pos = [np.array([PALLET_XY[0] + dx, PALLET_XY[1] + dy, LOAD_Z]) for dx, dy in BOX_GRID]
        self.loaded = 0
        self.lap_done = False           # run_patrol 이 보고 에이전트 finish_patrol 을 부름
        self.done = False
        walker.reset(s0=START_S)

    @property
    def label(self):
        return LABELS.get(self.state.split("_")[0], self.state)

    # ------------------------------------------------------------ 장면
    def build(self, scene):
        """카트, 팔레트, 상자 4개 (첫 렌더 전에)."""
        self.scene = scene
        scene.add_cart(self.prefix + "/Cart", *self.cart)
        scene._ref(self.prefix + "/Pallet", ASSETS["pallet"], (PALLET_XY[0], PALLET_XY[1], 0.0))
        for k, (dx, dy) in enumerate(BOX_GRID):
            scene._ref(f"{self.prefix}/Base{k}", ASSETS[BASE_BOX], (PALLET_XY[0] + dx, PALLET_XY[1] + dy, PALLET_TOP))
        self.boxes = [scene.add_box(f"{self.prefix}/Box{k}", BOX) for k in range(N_BOXES)]
        self.apply()

    def apply(self):
        """카트와 상자 위치를 장면에 반영 (매 프레임)."""
        from pxr import Gf
        sc = self.scene
        x, y, yaw, tilt = self.cart
        sc.set_cart(self.prefix + "/Cart", x, y, yaw, tilt)
        cm = sc.cart_matrix(x, y, yaw, tilt)
        for k, path in enumerate(self.boxes):
            if k < self.loaded:
                bx, by, bz = sc.cart_slot(k, BOX)
                m = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), 3.0 * (k % 2) - 1.5)) * \
                    Gf.Matrix4d().SetTranslate(Gf.Vec3d(bx, by, bz)) * cm
            else:
                p = self.box_pos[k]
                m = Gf.Matrix4d().SetTranslate(Gf.Vec3d(float(p[0]), float(p[1]), float(p[2])))
            sc.set_matrix(path, m)

    def _slot_world(self, k):
        x, y, yaw, tilt = self.cart
        bx, by, bz = self.scene.cart_slot(k, BOX)
        c, s = math.cos(yaw), math.sin(yaw)
        return np.array([x + c * bx - s * by, y + s * bx + c * by, bz])

    def _follow_pose(self):
        s = self.w.s
        ax = self.path.point_at(max(0.0, s - FOLLOW_M))
        wp = self.path.point_at(s)
        d = wp - ax
        yaw = math.atan2(d[1], d[0]) - math.pi / 2          # 손잡이 (+Y) 가 작업자 쪽
        return float(ax[0]), float(ax[1]), yaw, PULL_TILT

    # ------------------------------------------------------------ 진행
    def said(self, t, dur, count):
        self.busy_until = t + dur + GAP_S
        self.heard.add(count)

    def _show(self, t, count):
        self.w.start_gesture(count, hold=HOLD)
        self.shown.append((float(t), count))
        self.pending = (count, t)

    def _gesture_over(self, t):
        """보이던 손동작이 끝났으면 True (인식이 안 됐으면 한 번 더 보임)."""
        if self.w.gesture is not None or self.pending is None:
            return self.pending is None
        count, _ = self.pending
        if count not in self.heard and sum(1 for _, c in self.shown if c == count) < 2:
            self._show(t, count)
            return False
        self.pending = None
        return True

    def _go(self, state, t):
        self.state, self.t_state = state, t

    def update(self, t):
        w, st, e = self.w, self.state, t - self.t_state
        if st == "start":
            w.pause()
            if t >= 0.8:
                self._show(t, 2)
                self._go("tbm", t)
        elif st == "tbm":
            if self._gesture_over(t) and t >= self.busy_until:      # TBM 을 끝까지 듣고 출발
                w.resume()
                self._go("walk", t)
        elif st == "walk":
            if w.s >= self.s_station:
                w.pause()
                cx, cy, _ = CART_PARK
                p = self.path.point_at(w.s)
                w.face(math.atan2(cy - p[1], cx - p[0]), pitch=-0.15)
                self._go("look", t)
        elif st == "look":
            if e >= TURN_S + LOOK_S and t >= self.busy_until:
                self._show(t, 1)
                self._go("look_ask", t)
        elif st == "look_ask":
            if self._gesture_over(t):
                p = self.path.point_at(w.s)
                mx, my = (CART_PARK[0] + PALLET_XY[0]) / 2, (CART_PARK[1] + PALLET_XY[1]) / 2
                w.face(math.atan2(my - p[1], mx - p[0]), pitch=-0.15)
                self._go("load", t)
        elif st == "load":
            # 상자를 하나씩 팔레트에서 들어 카트 짐 받침에 쌓음
            k = int(e // LOAD_EACH_S)
            u = (e - k * LOAD_EACH_S) / (LOAD_EACH_S * 0.85)
            if k < N_BOXES:
                start = np.array([PALLET_XY[0] + BOX_GRID[k][0], PALLET_XY[1] + BOX_GRID[k][1], LOAD_Z])
                end = self._slot_world(k)
                q = _ease(u)
                self.box_pos[k] = start + (end - start) * q + np.array([0.0, 0.0, 0.55 * math.sin(math.pi * min(u, 1.0))])
                self.loaded = k + (1 if u >= 1.0 else 0)
            else:
                self.loaded = N_BOXES
                self.cart_from = self.cart
                w.pulling = True
                w.face(None)
                self._go("attach", t)
        elif st == "attach":
            u = _ease(e / ATTACH_S)
            x0, y0, yaw0, t0 = self.cart_from
            x1, y1, yaw1, t1 = self._follow_pose()
            self.cart = (x0 + (x1 - x0) * u, y0 + (y1 - y0) * u, _lerp_angle(yaw0, yaw1, u), t0 + (t1 - t0) * u)
            if e >= ATTACH_S:
                w.resume()
                self._go("pull", t)
        elif st == "pull":
            self.cart = self._follow_pose()
            if w.s >= self.path.length - 0.05:
                w.s = self.path.length - 0.05
                w.pause()
                self.lap_done = True
                self._go("end", t)
        elif st == "end":
            if e >= 0.6 and t >= self.busy_until:
                self._show(t, 3)
                self._go("call", t)
        elif st == "call":
            # 관리자 호출 안내가 끝나면 끝 (두 번 보여도 인식이 안 되면 그냥 끝)
            if self._gesture_over(t) and (3 not in self.heard or t >= self.busy_until):
                self._go("done", t)
                self.done = True
        elif st == "done":
            self.done = True
