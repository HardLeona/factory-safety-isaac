"""위험 요소 배치와 정답표.

시나리오마다 위험한 상태와 안전한 상태의 물체를 섞어 놓고, 물체마다 정답(위험/안전)을 미리 기록한다.
위험 영역(라바콘으로 둘러친 곳, DANGER 표지를 세운 곳) 도 정답표에 넣는다.
YOLO 판정은 끝나고 이 정답표와 맞춰 채점한다 (판정에는 정답을 쓰지 않음).
"""
import json
import math
from dataclasses import asdict, dataclass, field

import numpy as np

from . import warehouse as W
from .config import CLASS_KO, HAZARD, KIND, POLYHAVEN_MODELS, STATES, TOOL_KO, TOOL_TYPES


@dataclass
class SceneObject:
    id: str                 # 예: "O03"
    cls: str                # 상태 (위험/안전까지 포함, config.STATES)
    x: float
    y: float
    yaw: float = 0.0
    params: dict = field(default_factory=dict)   # 모양 변형 (유출 종류와 크기, 공구 종류, 기울기 등)

    @property
    def kind(self):
        return KIND[self.cls]

    @property
    def hazard(self):
        return HAZARD[self.cls]

    @property
    def zone(self):
        return W.zone_name(self.x, self.y)

    @property
    def label(self):
        return CLASS_KO[self.cls]

    @property
    def state_id(self):
        return STATES.index(self.cls)

    @property
    def tools(self):
        """공구 묶음이면 공구 종류 목록."""
        return [it[0] for it in self.params.get("items", [])]


@dataclass
class ZoneSpec:
    """위험 영역: 라바콘으로 둘러친 곳 (cone) 또는 DANGER 표지만 세운 곳 (sign)."""
    id: str                 # 예: "Z1"
    type: str               # "cone" | "sign"
    x: float
    y: float
    radius: float
    params: dict = field(default_factory=dict)   # 라바콘 위치, 표지 여부, 안쪽 내용 (pit: 뚜껑 열린 바닥 구멍)

    @property
    def zone(self):
        return W.zone_name(self.x, self.y)


@dataclass
class EquipSpec:
    """장비 (위험/안전 판정 대상 아님, 정답표 objects 에 안 들어감): 운반 카트 (핸드트럭)."""
    id: str
    type: str               # "cart"
    x: float
    y: float
    yaw: float = 0.0
    params: dict = field(default_factory=dict)   # tilt (끌 때 뒤로 기운 각도, 도), boxes (실은 상자 수), box

    @property
    def cls(self):
        return self.type

    @property
    def zone(self):
        return W.zone_name(self.x, self.y)


@dataclass
class Scenario:
    seed: int
    objects: list
    zones: list = field(default_factory=list)
    equipment: list = field(default_factory=list)

    def answer_key(self):
        """정답표 (판정 전에 미리 만들어 두는 것)."""
        objs = []
        for o in self.objects:
            d = {"id": o.id, "class": o.cls, "kind": o.kind, "hazard": o.hazard, "label": o.label,
                 "x": round(o.x, 2), "y": round(o.y, 2), "zone": o.zone}
            if o.kind == "tool":
                d["tools"] = o.tools
                d["tools_ko"] = [TOOL_KO[t] for t in o.tools]
            if o.kind == "spill":
                d["radius"] = round(o.params["radius"], 2)
            objs.append(d)
        zones = [{"id": z.id, "type": z.type, "x": round(z.x, 2), "y": round(z.y, 2), "radius": z.radius,
                  "sign": bool(z.params.get("sign")), "n_cones": len(z.params.get("cones", [])), "inner": z.params.get("inner"),
                  "zone": z.zone} for z in self.zones]
        equip = [{"id": e.id, "type": e.type, "x": round(e.x, 2), "y": round(e.y, 2), "zone": e.zone} for e in self.equipment]
        return {"seed": self.seed, "objects": objs, "zones": zones, "equipment": equip}

    def save_answer_key(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.answer_key(), f, ensure_ascii=False, indent=1)

    @property
    def hazards(self):
        return [o for o in self.objects if o.hazard]


def _free(x, y, used, d=W.ZONE_CLEAR_M):
    return all(math.hypot(x - ux, y - uy) >= d for ux, uy in used)


def sample_scenario(seed, n_spill=3, n_tool_floor=(2, 3), n_stack=4, n_zone=2, p_marked=0.45, p_unstable=0.5,
                    p_ext=(0.6, 0.2, 0.2), p_cone_zone=0.6, n_cart=0):
    rng = np.random.default_rng(seed)
    objs, zones = [], []

    def add(cls, x, y, yaw=0.0, **params):
        objs.append(SceneObject(f"O{len(objs):02d}", cls, float(x), float(y), float(yaw), params))

    # 위험 영역 먼저 (주변 자리를 비워야 해서)
    used = []
    for i in rng.choice(len(W.ZONE_SLOTS), n_zone, replace=False):
        zx, zy, r = W.ZONE_SLOTS[i]
        ztype = "cone" if rng.random() < p_cone_zone else "sign"
        params = {"inner": "pit" if rng.random() < 0.5 else "none", "sign_yaw": float(rng.uniform(0, 2 * math.pi))}
        if ztype == "cone":
            n = int(rng.integers(4, 7))
            a0 = rng.uniform(0, 2 * math.pi)
            params["cones"] = [(float(zx + r * math.cos(a0 + 2 * math.pi * k / n + rng.uniform(-0.15, 0.15))),
                                float(zy + r * math.sin(a0 + 2 * math.pi * k / n + rng.uniform(-0.15, 0.15)))) for k in range(n)]
            params["sign"] = bool(rng.random() < 0.5)
        else:
            params["cones"] = []
            params["sign"] = True
        zones.append(ZoneSpec(f"Z{len(zones) + 1}", ztype, float(zx), float(zy), float(r), params))
        used.append((zx, zy))

    def pick(slots, k):
        free = [s for s in slots if _free(s[0], s[1], used)]
        idx = rng.choice(len(free), min(k, len(free)), replace=False) if free else []
        out = [free[i] for i in idx]
        used.extend(out)
        return out

    # 경로 바로 옆 위험 하나 (작업자가 닿기 직전까지 다가감)
    near_tool = rng.random() < 0.5
    near = pick(W.NEAR_PATH_TOOL_SLOTS if near_tool else W.NEAR_PATH_SPILL_SLOTS, 1)

    def add_spill(x, y, marked):
        add("spill_marked" if marked else "spill", x + rng.uniform(-0.2, 0.2), y + rng.uniform(-0.2, 0.2),
            rng.uniform(0, 2 * math.pi),
            fluid="oil" if rng.random() < 0.5 else "water", radius=float(rng.uniform(0.45, 0.8)),
            shape_seed=int(rng.integers(1 << 30)),
            source=str(rng.choice(["bucket", "bottle", "barrel", "none"], p=[0.4, 0.3, 0.15, 0.15])),
            n_cones=int(rng.integers(1, 3)))

    def add_tools(x, y):
        add("tool_floor", x + rng.uniform(-0.2, 0.2), y + rng.uniform(-0.2, 0.2), rng.uniform(0, 2 * math.pi),
            items=_tool_items(rng, spread=0.35))

    n_spill_left, k_tool = n_spill, int(rng.integers(n_tool_floor[0], n_tool_floor[1] + 1))
    for x, y in near:
        if near_tool:
            add_tools(x, y)
            k_tool -= 1
        else:
            add_spill(x, y, marked=False)      # 경로 옆 유출은 조치 안 된 것
            n_spill_left -= 1
    for x, y in pick(W.SPILL_SLOTS, n_spill_left):
        add_spill(x, y, rng.random() < p_marked)
    for x, y in pick(W.TOOL_FLOOR_SLOTS, k_tool):
        add_tools(x, y)
    for tx, ty, _ in W.TABLES:
        add("tool_stored", tx, ty, 0.0, items=_tool_items(rng, spread=0.0, on_table=True))
    for x, y in pick(W.STACK_SLOTS, n_stack):
        unstable = rng.random() < p_unstable
        add("stack_unstable" if unstable else "stack_stable", x, y, float(rng.choice([0, math.pi / 2])) + rng.uniform(-0.1, 0.1),
            layers=int(rng.integers(2, 4)), box=str(rng.choice(["box_b", "box_a"])),
            tilt=float(rng.uniform(9, 18)) if unstable else 0.0, shift=float(rng.uniform(0.15, 0.3)) if unstable else 0.0,
            fallen_box=bool(unstable and rng.random() < 0.5), tilt_seed=int(rng.integers(1 << 30)))
    for mi, (x, y, (nx, ny)) in enumerate(W.EXT_MOUNTS):
        st = str(rng.choice(["ext_ok", "ext_fallen", "ext_blocked"], p=list(p_ext)))
        # 정답표 위치는 실제로 놓인 곳: 쓰러진 소화기는 바닥, 가로막힌 소화기는 앞을 막은 상자 자리
        off = {"ext_ok": 0.0, "ext_fallen": W.EXT_FALL_OUT, "ext_blocked": W.EXT_BLOCK_OUT}[st]
        add(st, x + nx * off, y + ny * off, 0.0, mount=mi, out=(nx, ny), fall_yaw=float(rng.uniform(-1, 1)),
            block_box=str(rng.choice(["box_a", "box_b"])), block_layers=int(rng.integers(2, 4)))
    # 운반 카트 (학습 데이터용, 기본 0대라 평가 시나리오는 그대로): 경로 옆 빈 바닥에 세워 두거나 끄는 자세로 기울임
    equip = []
    path = W.PatrolPath() if n_cart else None
    for k in range(n_cart):
        for _ in range(40):
            s = rng.uniform(0, path.length)
            p, h = path.point_at(s), path.heading_at(s)
            lat = float(rng.choice([-1.0, 1.0]) * rng.uniform(0.7, 1.8))
            x, y = float(p[0] - math.sin(h) * lat), float(p[1] + math.cos(h) * lat)
            inside = -9.4 < x < 9.4 and -11.2 < y < 17.3
            in_rack = any(a[0] - 0.5 < x < b[0] + 0.5 and a[1] - 0.5 < y < b[1] + 0.5 for a, b in W.RACKS)
            near_table = any(abs(x - tx) < 2.0 and abs(y - ty) < 1.2 for tx, ty, _ in W.TABLES)
            if inside and not in_rack and not near_table and _free(x, y, used, d=1.6) and \
                    all(math.hypot(x - o.x, y - o.y) > 1.6 for o in objs):
                break
        used.append((x, y))
        equip.append(EquipSpec(f"E{k + 1}", "cart", x, y, float(rng.uniform(0, 2 * math.pi)),
                               {"tilt": 0.0 if rng.random() < 0.45 else float(rng.uniform(25, 45)),
                                "boxes": int(rng.integers(0, 5)), "box": str(rng.choice(["box_c", "box_d"]))}))
    return Scenario(seed, objs, zones, equip)


# 작업대 위에 올리는 공구 (긴 곡괭이, 삽은 바닥에만)
TABLE_TOOLS = ["hammer", "screwdriver", "saw", "power_saw", "wrench", "drill"]


def _tool_items(rng, spread, on_table=False):
    """공구 묶음: [(종류, 모델, dx, dy, yaw)]. 바닥은 흩어져 있고, 작업대 위는 나란히."""
    n = int(rng.integers(1, 4))
    items = []
    kinds = TABLE_TOOLS if on_table else TOOL_TYPES
    for j in range(n):
        kind = str(rng.choice(kinds))
        models = POLYHAVEN_MODELS.get(kind, []) + (["ycb_drill"] if kind == "drill" else []) + (["procedural"] if kind == "power_saw" else [])
        model = str(rng.choice(models))
        if on_table:
            dx, dy = W.TABLE_TOOL_DX[j % len(W.TABLE_TOOL_DX)] + rng.uniform(-0.1, 0.1), rng.uniform(-0.08, 0.08)
        else:
            dx, dy = rng.uniform(-spread, spread), rng.uniform(-spread, spread)
        items.append((kind, model, float(dx), float(dy), float(rng.uniform(0, 2 * math.pi))))
    return items


def to_dict(sc):
    return {"seed": sc.seed, "objects": [asdict(o) for o in sc.objects], "zones": [asdict(z) for z in sc.zones],
            "equipment": [asdict(e) for e in sc.equipment]}
