"""강화학습 환경: 로봇이 스스로 이동하고 카메라를 돌려서 위험 요소를 빨리 찾도록 학습.

Isaac Sim 없이 numpy로만 돌아가서 일반 PC에서 병렬로 빠르게 학습할 수 있고,
학습한 정책은 scripts/play_policy.py 로 Isaac Sim 화면에서 그대로 재생한다.

행동 (연속 3개, 각각 -1~1)
  0: 전진 속도  (-1 = 정지, 1 = 1.5 m/s)
  1: 회전 속도  (±1.2 rad/s)
  2: 카메라 좌우 회전 속도 (±1.8 rad/s, 몸 기준 ±1.6 rad 까지)

관측 (122차원)
  자기 위치, 방향, 카메라 각도, 속도        6
  주변 거리 센서 16방향 (최대 10 m)          16
  공장을 4 m 칸으로 나눈 '이미 본 곳' 지도     88
  지금 보이는 미확정 후보 3개 (방향, 거리, 신뢰도) 9
  남은 위험 요소 비율, 남은 소화기 비율, 시간     3

보상
  위험 요소 발견 +1.0, 정상 소화기 점검 +0.2, 처음 보는 칸 +0.01,
  매 스텝 -0.002, 충돌 -0.1, 장애물에 1 m 안쪽으로 붙으면 최대 -0.01,
  전부 찾으면 남은 시간 비례 보너스
"""
import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from . import layout as L
from .config import CAM_H_ROBOT, D, W
from .detector import SimDetector
from .geometry import CameraPose, ray_distances_2d
from .patrol import ClosedPath
from .scenario import sample_scenario

VMAX, WMAX, PAN_RATE, PAN_MAX = 1.5, 1.2, 1.8, 1.6
ROBOT_R = 0.45
N_RAYS, RAY_MAX = 16, 10.0
GRID_NX, GRID_NY = 11, 8
SEE_RANGE = 9.0
COLLIDE_PEN = 0.1                  # 충돌 벌점 (0.02 일 때는 벽에 비비며 도는 정책이 나왔음)
NEAR_DIST, NEAR_PEN = 1.0, 0.01    # 로봇 중심에서 장애물까지 이보다 가까우면 거리 비례 벌점


def _segments_blocked_2d(origin, targets, mins, maxs):
    if len(mins) == 0 or len(targets) == 0:
        return np.zeros(len(targets), dtype=bool)
    dvec = targets - origin
    dist = np.linalg.norm(dvec, axis=1)
    dirs = dvec / np.maximum(dist, 1e-9)[:, None]
    dirs = np.where(np.abs(dirs) < 1e-12, 1e-12, dirs)
    inv = 1.0 / dirs
    t1 = (mins[None] - origin) * inv[:, None, :]
    t2 = (maxs[None] - origin) * inv[:, None, :]
    tn = np.max(np.minimum(t1, t2), axis=2)
    tf = np.min(np.maximum(t1, t2), axis=2)
    hit = (tf >= np.maximum(tn, 0.0)) & (tn < (dist - 0.3)[:, None])
    return hit.any(axis=1)


class FactoryPatrolEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, max_steps=1500, dt=0.1, static_seed=0, require_all_ext=True, scenario_seed=None):
        super().__init__()
        self.max_steps, self.dt, self.require_all_ext = max_steps, dt, require_all_ext
        self.fixed_scenario_seed = scenario_seed
        self.static = L.build_static(static_seed)
        self.s_mins, self.s_maxs = L.occluder_arrays(self.static)
        self.f_mins, self.f_maxs = L.floor_obstacles_2d(self.static)
        tall = (self.s_maxs[:, 2] > 1.3) & (self.s_mins[:, 2] < 1.6)
        self.tall_mins, self.tall_maxs = self.s_mins[tall][:, :2], self.s_maxs[tall][:, :2]
        self.path = ClosedPath()
        gx = (np.arange(GRID_NX) + 0.5) * (W / GRID_NX) - W / 2
        gy = (np.arange(GRID_NY) + 0.5) * (D / GRID_NY) - D / 2
        self.cells = np.array([(x, y) for y in gy for x in gx])
        self.ray_angles = np.linspace(0, 2 * math.pi, N_RAYS, endpoint=False)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(-1.5, 1.5, shape=(6 + N_RAYS + GRID_NX * GRID_NY + 9 + 3,), dtype=np.float32)
        self.scenario = None

    # ------------------------------------------------------------ 상태 초기화
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        rng = self.np_random
        sc_seed = options.get("scenario_seed", self.fixed_scenario_seed)
        if sc_seed is None:
            sc_seed = int(rng.integers(1 << 30))
        self.scenario = sample_scenario(sc_seed)
        ex = self.scenario.occluder_boxes()
        mins = np.vstack([self.s_mins] + ([np.array([b[0] for b in ex])] if ex else []))
        maxs = np.vstack([self.s_maxs] + ([np.array([b[1] for b in ex])] if ex else []))
        self.detector = SimDetector(self.scenario.hazards, mins, maxs)
        # 로봇이 부딪히는 것: 고정 장애물 + 팔레트/적재물
        obs_m, obs_M = [self.f_mins], [self.f_maxs]
        for m in self.scenario.extras + [h.model for h in self.scenario.hazards if h.type == "stack"]:
            fm, fM = L.floor_obstacles_2d([m])
            if len(fm):
                obs_m.append(fm)
                obs_M.append(fM)
        self.o_mins, self.o_maxs = np.vstack(obs_m), np.vstack(obs_M)
        self.o_mins_r, self.o_maxs_r = self.o_mins - ROBOT_R, self.o_maxs + ROBOT_R

        u = float(rng.random())
        p, t = self.path.point_at(u), self.path.tangent_at(u)
        self.pos = np.array(p, dtype=float)
        self.heading = math.atan2(t[1], t[0]) + (math.pi if rng.random() < 0.5 else 0.0)
        self.pan, self.v = 0.0, 0.0
        self.steps, self.collisions = 0, 0
        self.seen_cells = np.zeros(len(self.cells), dtype=bool)
        self.n_targets = sum(1 for h in self.scenario.hazards if h.is_hazard)
        self.n_ext = sum(1 for h in self.scenario.hazards if h.type == "ext")
        self.detector.update(1e-4, self.camera())
        self.rays = self._ray_scan()
        return self._obs(), self._info([])

    def _ray_scan(self):
        return ray_distances_2d(self.pos, self.ray_angles + self.heading, self.o_mins, self.o_maxs, RAY_MAX)

    def camera(self):
        return CameraPose(pos=np.array([self.pos[0], self.pos[1], CAM_H_ROBOT]), yaw=self.heading + self.pan,
                          pitch=-0.13, roll=0.0, vfov=66.0)

    # ------------------------------------------------------------ 한 스텝
    def step(self, action):
        a = np.clip(np.asarray(action, dtype=float), -1, 1)
        dt = self.dt
        self.v = (a[0] + 1) / 2 * VMAX
        self.heading = (self.heading + a[1] * WMAX * dt + math.pi) % (2 * math.pi) - math.pi
        self.pan = float(np.clip(self.pan + a[2] * PAN_RATE * dt, -PAN_MAX, PAN_MAX))
        new_pos = self.pos + self.v * dt * np.array([math.cos(self.heading), math.sin(self.heading)])
        inside = np.all((new_pos >= self.o_mins_r) & (new_pos <= self.o_maxs_r), axis=1)
        collided = bool(inside.any())
        if collided:
            self.collisions += 1
        else:
            self.pos = new_pos

        cam = self.camera()
        new = self.detector.update(dt, cam)
        self.rays = self._ray_scan()
        near = max(0.0, (NEAR_DIST - float(self.rays.min())) / NEAR_DIST)
        reward = -0.002 - (COLLIDE_PEN if collided else 0.0) - NEAR_PEN * near
        for i in new:
            h = self.scenario.hazards[i]
            reward += 1.0 if h.is_hazard else 0.2
        reward += 0.01 * self._update_coverage(cam)

        self.steps += 1
        found = self.detector.found_targets
        ext_done = int(sum(1 for i, h in enumerate(self.scenario.hazards) if h.type == "ext" and self.detector.detected[i]))
        done = found == self.n_targets and (not self.require_all_ext or ext_done == self.n_ext)
        if done:
            reward += 2.0 * (1 - self.steps / self.max_steps)
        truncated = self.steps >= self.max_steps
        return self._obs(), float(reward), bool(done), bool(truncated), self._info(new, collided)

    def _update_coverage(self, cam):
        rel = self.cells - self.pos
        dist = np.linalg.norm(rel, axis=1)
        ang = np.arctan2(rel[:, 1], rel[:, 0]) - cam.yaw
        ang = np.abs((ang + math.pi) % (2 * math.pi) - math.pi)
        hf = math.atan(math.tan(math.radians(cam.vfov) / 2) * 16 / 9)
        cand = np.where((~self.seen_cells) & (dist < SEE_RANGE) & (ang < hf))[0]
        if len(cand) == 0:
            return 0
        blocked = _segments_blocked_2d(self.pos, self.cells[cand], self.tall_mins, self.tall_maxs)
        newly = cand[~blocked]
        self.seen_cells[newly] = True
        return len(newly)

    # ------------------------------------------------------------ 관측
    def _obs(self):
        det = self.detector
        o = [self.pos[0] / (W / 2), self.pos[1] / (D / 2), math.cos(self.heading), math.sin(self.heading),
             self.pan / PAN_MAX, self.v / VMAX]
        o += list(self.rays / RAY_MAX)
        o += list(self.seen_cells.astype(float))
        cam_yaw = self.heading + self.pan
        cands = []
        for i, h in enumerate(det.hazards):
            if det.detected[i] or not det.in_view[i]:
                continue
            rel = h.center[:2] - self.pos
            bearing = math.atan2(rel[1], rel[0]) - cam_yaw
            bearing = (bearing + math.pi) % (2 * math.pi) - math.pi
            cands.append((det.conf[i], bearing / math.pi, min(1.0, np.linalg.norm(rel) / 14.0)))
        cands.sort(reverse=True)
        for k in range(3):
            if k < len(cands):
                c, b, d = cands[k]
                o += [b, d, c]
            else:
                o += [0.0, 1.0, 0.0]
        ext_done = sum(1 for i, h in enumerate(det.hazards) if h.type == "ext" and det.detected[i])
        o += [1 - det.found_targets / max(1, self.n_targets), 1 - ext_done / max(1, self.n_ext), self.steps / self.max_steps]
        return np.clip(np.array(o, dtype=np.float32), -1.5, 1.5)

    def _info(self, new, collided=False):
        det = self.detector
        return {
            "new": list(new),
            "found": det.found_targets,
            "total": self.n_targets,
            "ext_checked": int(sum(1 for i, h in enumerate(det.hazards) if h.type == "ext" and det.detected[i])),
            "ext_total": self.n_ext,
            "collided": collided,
            "coverage": float(self.seen_cells.mean()),
        }


class PathFollower:
    """비교용 기준 정책: 고정 경로를 따라가며 카메라를 좌우로 훑고,
    track=True 면 의심 물체 쪽으로 카메라를 돌린다 (웹 버전 로봇과 같은 방식)."""

    def __init__(self, env, lookahead=2.0, track=True):
        self.env, self.look, self.track = env, lookahead, track
        self.phase = 0.0

    def reset(self):
        self.phase = 0.0

    def __call__(self, obs=None):
        e = self.env
        u = e.path.nearest_u(*e.pos)
        tgt = e.path.point_at(u + self.look / e.path.length)
        err = math.atan2(tgt[1] - e.pos[1], tgt[0] - e.pos[0]) - e.heading
        err = (err + math.pi) % (2 * math.pi) - math.pi
        w = float(np.clip(2.0 * err / WMAX, -1, 1))
        v = 1.0 if abs(err) < 0.6 else -0.4
        pan_goal = None
        if self.track:
            det, best = e.detector, 0.38
            for i, h in enumerate(det.hazards):
                if det.detected[i] or not det.in_view[i] or det.conf[i] < best:
                    continue
                if h.read_range and np.linalg.norm(h.center[:2] - e.pos) > h.read_range + 1.5:
                    continue
                best = det.conf[i]
                off = math.atan2(h.center[1] - e.pos[1], h.center[0] - e.pos[0]) - e.heading
                pan_goal = float(np.clip((off + math.pi) % (2 * math.pi) - math.pi, -PAN_MAX, PAN_MAX))
        if pan_goal is None:
            self.phase += e.dt * 0.6
            pan_goal = 1.0 * math.sin(self.phase)
        p = float(np.clip((pan_goal - e.pan) / (PAN_RATE * e.dt), -1, 1))
        return np.array([v, w, p], dtype=np.float32)


gym.register(id="FactoryPatrol-v0", entry_point="factory_safety.rl_env:FactoryPatrolEnv")
