"""위험 요소 무작위 배치 (웹 버전과 같은 규칙).

  웅덩이 6자리 중 2개, 공구 5자리 중 2개, 불안정 적재물 5자리 중 2개,
  소화기 8개 중 1~2개 압력 부족
"""
import math
from dataclasses import dataclass, field

import numpy as np

from . import layout as L
from .config import HAZARD_CLASS
from .models import GAUGE_X, GAUGE_Z, Model, box, extinguisher, pallet, power_tool, puddle, stable_stack, unstable_stack, C


@dataclass
class Hazard:
    id: int
    type: str            # puddle | tool | stack | ext
    model: Model
    label: str           # 로그에 쓰는 이름
    short: str
    detail: str
    risk: str            # high | mid | ok
    is_hazard: bool
    range: float         # 검출 가능한 최대 거리
    read_range: float = None   # 소화기: 압력계를 읽을 수 있는 거리
    ok: bool = None            # 소화기 정상 여부
    normal: np.ndarray = None  # 소화기 앞면 방향
    center: np.ndarray = None
    aabb_min: np.ndarray = None
    aabb_max: np.ndarray = None
    cls: int = 0
    gauge_cls: int = None

    @property
    def zone(self):
        return L.zone_name(self.center[0], self.center[1])


@dataclass
class Scenario:
    seed: int
    hazards: list = field(default_factory=list)
    extras: list = field(default_factory=list)   # 위험 요소가 아닌 슬롯 채움 (정상 적재물 등)

    @property
    def targets(self):
        return [h for h in self.hazards if h.is_hazard]

    def occluder_boxes(self):
        return [b for m in self.extras for b in m.occluder_boxes()]


def _finish(h):
    mn, mx = h.model.aabb(roles=("main",))
    h.aabb_min, h.aabb_max = mn - 0.03, mx + 0.03
    if h.center is None:
        h.center = (h.aabb_min + h.aabb_max) / 2.0
    return h


def sample_scenario(seed=None):
    rng = np.random.default_rng(seed)
    sc = Scenario(seed=seed if seed is not None else int(rng.integers(1 << 30)))
    hid = 0

    def pick(n_total, k):
        return list(rng.choice(n_total, size=k, replace=False))

    # 웅덩이
    for i in pick(len(L.PUDDLE_SLOTS), 2):
        x, y = L.PUDDLE_SLOTS[i]
        oil = rng.random() < 0.6
        m = Model(f"H{hid:02d}_puddle", puddle(rng, oil), pose=(x, y, 0.0, rng.uniform(0, math.pi)))
        sc.hazards.append(_finish(Hazard(
            hid, "puddle", m, "미끄러운 바닥 (기름 유출)" if oil else "미끄러운 바닥 (물기)", "미끄러운 바닥",
            "기름이 흘러 미끄럼 사고 위험" if oil else "바닥에 물기가 있어 미끄럼 사고 위험",
            "high", True, 13.0, cls=HAZARD_CLASS["puddle"])))
        hid += 1

    # 공구
    for i in pick(len(L.TOOL_SLOTS), 2):
        x, y, z, bench = L.TOOL_SLOTS[i]
        v = int(rng.random() < 0.5)
        yaw = (math.pi / 2 + rng.uniform(-0.4, 0.4)) if bench else rng.uniform(0, 2 * math.pi)
        m = Model(f"H{hid:02d}_tool", power_tool(v), pose=(x, y, z, yaw))
        name = "원형톱" if v else "그라인더"
        detail = ("톱날이" if v else "절단날이") + " 노출된 채 " + ("작업대 가장자리에 놓여 있음" if bench else "통로 바닥에 놓여 전선 걸림 위험")
        sc.hazards.append(_finish(Hazard(hid, "tool", m, f"방치된 {name}", f"{name} 방치", detail, "mid", True, 11.5,
                                         cls=HAZARD_CLASS["tool"])))
        hid += 1

    # 적재물
    chosen = set(pick(len(L.STACK_SLOTS), 2))
    for i, s in enumerate(L.STACK_SLOTS):
        yaw = s["yaw"] + (rng.uniform(-0.3, 0.3) if s["pallet"] else 0.0)
        base = 0.13 if s["pallet"] else 0.0
        parts = pallet() if s["pallet"] else []
        if i in chosen:
            st, tilt = unstable_stack(rng, s["max_h"], -0.25 if s["pallet"] else -0.22, base)
            m = Model(f"H{hid:02d}_stack", parts + st, pose=(s["x"], s["y"], s["z"], yaw))
            sc.hazards.append(_finish(Hazard(hid, "stack", m, f"불안정 적재물 ({s['where']})", "불안정 적재",
                                             f"약 {tilt}° 기울어져 낙하 위험", "high", True, 14.0,
                                             cls=HAZARD_CLASS["stack"])))
            hid += 1
        elif s["pallet"]:
            sc.extras.append(Model(f"Extra_pallet{i}", parts + stable_stack(rng, base), pose=(s["x"], s["y"], s["z"], yaw)))
        else:
            sc.extras.append(Model(f"Extra_box{i}", [box((0, 0, 0.275), (0.8, 0.7, 0.55), C["cardboard"][rng.integers(5)],
                                                         occ=True, role="extra")], pose=(s["x"], s["y"], s["z"], 0.0)))

    # 소화기
    n_low = 1 if rng.random() < 0.5 else 2
    lows = set(pick(len(L.EXT_MOUNTS), n_low))
    for i, (x, y, nx, ny) in enumerate(L.EXT_MOUNTS):
        ok = i not in lows
        yaw = math.atan2(ny, nx)
        m = Model(f"H{hid:02d}_ext", extinguisher(ok, int(rng.integers(3))), pose=(x, y, 0.95, yaw))
        gauge = m.transform([(GAUGE_X, 0.0, GAUGE_Z)])[0]
        h = Hazard(hid, "ext", m, "소화기 정상" if ok else "소화기 압력 부족", "소화기 정상" if ok else "소화기 압력 부족",
                   "압력계 바늘이 녹색 구간, 사용 가능" if ok else "압력계 바늘이 적색 구간, 충전이나 교체 필요",
                   "ok" if ok else "high", not ok, 14.0, read_range=6.5, ok=ok, normal=np.array([nx, ny, 0.0]),
                   center=gauge, cls=HAZARD_CLASS["ext"], gauge_cls=4 if ok else 5)
        sc.hazards.append(_finish(h))
        hid += 1
    return sc
