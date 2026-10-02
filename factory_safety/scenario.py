"""위험 요소 배치와 정답표.

시나리오마다 위험한 상태와 안전한 상태의 물체를 섞어 놓고, 물체마다 정답(위험/안전)을 미리 기록한다.
YOLO 판정은 끝나고 이 정답표와 맞춰 채점한다 (판정에는 정답을 쓰지 않음).
"""
import json
import math
from dataclasses import asdict, dataclass, field

import numpy as np

from . import warehouse as W
from .config import CLASS_KO, CLASSES, HAZARD, KIND


@dataclass
class SceneObject:
    id: str                 # 예: "O03"
    cls: str                # YOLO 클래스 이름 (위험/안전 상태까지 포함)
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
    def cls_id(self):
        return CLASSES.index(self.cls)


@dataclass
class Scenario:
    seed: int
    objects: list

    def answer_key(self):
        """정답표 (판정 전에 미리 만들어 두는 것)."""
        return {"seed": self.seed, "objects": [
            {"id": o.id, "class": o.cls, "kind": o.kind, "hazard": o.hazard, "label": o.label,
             "x": round(o.x, 2), "y": round(o.y, 2), "zone": o.zone} for o in self.objects]}

    def save_answer_key(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.answer_key(), f, ensure_ascii=False, indent=1)

    @property
    def hazards(self):
        return [o for o in self.objects if o.hazard]


def sample_scenario(seed, n_spill=3, n_tool_floor=(2, 3), n_stack=4, p_marked=0.45, p_unstable=0.5,
                    p_ext=(0.6, 0.2, 0.2)):
    rng = np.random.default_rng(seed)
    objs = []

    def add(cls, x, y, yaw=0.0, **params):
        objs.append(SceneObject(f"O{len(objs):02d}", cls, float(x), float(y), float(yaw), params))

    for i in rng.choice(len(W.SPILL_SLOTS), n_spill, replace=False):
        x, y = W.SPILL_SLOTS[i]
        marked = rng.random() < p_marked
        add("spill_marked" if marked else "spill", x + rng.uniform(-0.3, 0.3), y + rng.uniform(-0.3, 0.3),
            rng.uniform(0, 2 * math.pi),
            fluid="oil" if rng.random() < 0.5 else "water", radius=float(rng.uniform(0.45, 0.8)),
            shape_seed=int(rng.integers(1 << 30)),
            source=str(rng.choice(["bucket", "bottle", "barrel", "none"], p=[0.4, 0.3, 0.15, 0.15])),
            n_cones=int(rng.integers(1, 3)))
    k = int(rng.integers(n_tool_floor[0], n_tool_floor[1] + 1))
    for i in rng.choice(len(W.TOOL_FLOOR_SLOTS), k, replace=False):
        x, y = W.TOOL_FLOOR_SLOTS[i]
        add("tool_floor", x + rng.uniform(-0.3, 0.3), y + rng.uniform(-0.3, 0.3), rng.uniform(0, 2 * math.pi),
            items=_tool_items(rng, spread=0.35))
    for tx, ty, _ in W.TABLES:
        add("tool_stored", tx, ty, 0.0, items=_tool_items(rng, spread=0.0, on_table=True))
    for i in rng.choice(len(W.STACK_SLOTS), n_stack, replace=False):
        x, y = W.STACK_SLOTS[i]
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
    return Scenario(seed, objs)


def _tool_items(rng, spread, on_table=False):
    """공구 묶음: [(종류, dx, dy, yaw)]. 바닥은 흩어져 있고, 작업대 위는 나란히."""
    kinds = ["drill", "clamp", "scissors", "wood"]
    n = int(rng.integers(1, 4))
    items = []
    for j in range(n):
        kind = str(rng.choice(kinds if not on_table else ["drill", "clamp", "scissors"]))
        if on_table:
            dx, dy = W.TABLE_TOOL_DX[j % len(W.TABLE_TOOL_DX)] + rng.uniform(-0.1, 0.1), rng.uniform(-0.1, 0.1)
        else:
            dx, dy = rng.uniform(-spread, spread), rng.uniform(-spread, spread)
        items.append((kind, float(dx), float(dy), float(rng.uniform(0, 2 * math.pi))))
    return items


def to_dict(sc):
    return {"seed": sc.seed, "objects": [asdict(o) for o in sc.objects]}
