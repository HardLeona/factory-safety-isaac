"""공장 고정 배치 (벽, 기둥, 랙, 기계, 작업대) 와 위험 요소가 놓일 수 있는 자리.

배치를 바꾸고 싶으면 이 파일만 고치면 된다. USD 장면, 검출기, 강화학습 환경이
모두 여기서 만든 목록을 그대로 쓴다.
"""
import math

import numpy as np

from .config import W, D, H, srgb
from .models import C, Model, Part, box, cyl, pallet, stable_stack

# ---------------------------------------------------------------- 통로, 경로
# 통로 사각형 (x0, y0, x1, y1): 바닥 텍스처, 평면도, 강화학습 시작 위치에 사용
AISLES = {
    "A": (-20.5, 7.5, 20.5, 10.5),     # 북쪽 가로 통로 (랙 앞)
    "B": (-20.5, -8.5, 20.5, -5.5),    # 남쪽 가로 통로 (작업대 앞)
    "W": (-20.5, -5.5, -17.5, 7.5),    # 서쪽 세로 통로
    "E": (17.5, -5.5, 20.5, 7.5),      # 동쪽 세로 통로
}

# 고정 순찰 경로 (닫힌 곡선의 제어점)
PATROL_PATH = [(-19, 9), (-9.5, 9), (0, 9), (9.5, 9), (19, 9), (19, 1), (19, -7),
               (9.5, -7), (0, -7), (-9.5, -7), (-19, -7), (-19, 1)]

PILLARS = [(x, y) for x in (-11.0, 0.0, 11.0) for y in (6.0, -4.0)]
RACK_X = [-15.5, -5.0, 5.0, 15.5]
RACK_Y = 13.3
RACK_LEVELS = [0.12, 1.5, 2.85, 4.2]
LAMPS = [(x, y) for x in range(-18, 19, 6) for y in (-9, -3, 3, 9)]

# 소화기 설치 위치: (x, y, 바깥쪽 법선 nx, ny)
EXT_MOUNTS = []
for _x in (-11.0, 0.0, 11.0):
    EXT_MOUNTS.append((_x, 6.0 + 0.3 + 0.14, 0.0, 1.0))     # A 통로 쪽 기둥면
    EXT_MOUNTS.append((_x, -4.0 - 0.3 - 0.14, 0.0, -1.0))   # B 통로 쪽 기둥면
EXT_MOUNTS.append((-W / 2 + 0.14, 1.0, 1.0, 0.0))            # 서쪽 벽
EXT_MOUNTS.append((W / 2 - 0.14, -2.0, -1.0, 0.0))           # 동쪽 벽

# 위험 요소 후보 자리
PUDDLE_SLOTS = [(-10, 9.2), (7.5, 8.7), (19.1, -1.5), (4.5, -7.2), (-13.5, -6.8), (-19, 3.5)]
TOOL_SLOTS = [  # (x, y, z, 작업대 위인지)
    (-14.8, -10.92, 0.94, True), (-5.2, -10.92, 0.94, True),
    (-9.2, 6.9, 0.0, False), (5.5, -4.8, 0.0, False), (20.6, 4.0, 0.0, False),
]
STACK_SLOTS = [  # x, y, z, 기울어지는 방향 yaw, 팔레트 여부, 최대 높이, 위치 이름
    dict(x=-6.5, y=RACK_Y, z=1.6, yaw=-math.pi / 2, pallet=False, max_h=1.15, where="랙 선반"),
    dict(x=13.5, y=RACK_Y, z=1.6, yaw=-math.pi / 2, pallet=False, max_h=1.15, where="랙 선반"),
    dict(x=2.5, y=-11.8, z=0.0, yaw=math.pi / 2, pallet=True, max_h=1.9, where="팔레트"),
    dict(x=13.0, y=-11.8, z=0.0, yaw=math.pi / 2, pallet=True, max_h=1.9, where="팔레트"),
    dict(x=-2.6, y=6.7, z=0.0, yaw=math.pi / 2, pallet=True, max_h=1.9, where="팔레트"),
]
RACK_RESERVED = {-5.0: [-6.5], 15.5: [13.5]}   # 랙 선반 위 비워둔 자리 (적재물 슬롯)


def zone_name(x, y):
    if y > 10.5:
        return "북측 랙 구역"
    if y >= 7.5:
        return "A 통로"
    if y <= -8.5:
        return "남측 작업·출하 구역"
    if y <= -5.5:
        return "B 통로"
    if x <= -17.5:
        return "서측 통로"
    if x >= 17.5:
        return "동측 통로"
    return "가공 구역"


# ---------------------------------------------------------------- 정적 모델
def _shell():
    t = 0.4
    p = [
        box((0, D / 2 + t / 2, H / 2), (W + 2 * t, t, H), C["wall"], occ=True, name="WallN"),
        box((0, -D / 2 - t / 2, H / 2), (W + 2 * t, t, H), C["wall"], occ=True, name="WallS"),
        box((-W / 2 - t / 2, 0, H / 2), (t, D, H), C["wall"], occ=True, name="WallW"),
        box((W / 2 + t / 2, 0, H / 2), (t, D, H), C["wall"], occ=True, name="WallE"),
        box((0, D / 2 - 0.02, 0.55), (W, 0.04, 1.1), C["band"], shadow=False, name="BandN"),
        box((0, -D / 2 + 0.02, 0.55), (W, 0.04, 1.1), C["band"], shadow=False, name="BandS"),
        box((-W / 2 + 0.02, 0, 0.55), (0.04, D, 1.1), C["band"], shadow=False, name="BandW"),
        box((W / 2 - 0.02, 0, 0.55), (0.04, D, 1.1), C["band"], shadow=False, name="BandE"),
        box((W / 2 - 0.03, -12, 2.3), (0.06, 5.2, 4.6), C["door"], rough=0.7, metal=0.3, name="Door"),
        box((W / 2 - 0.05, -12, 4.7), (0.08, 5.6, 0.2), C["yellow"], name="DoorTop"),
        box((W / 2 - 0.05, -9.3, 2.4), (0.08, 0.2, 4.8), C["yellow"], name="DoorL"),
        box((W / 2 - 0.05, -14.7, 2.4), (0.08, 0.2, 4.8), C["yellow"], name="DoorR"),
        Part("quad", pos=(0, 0, 0), size=(W, D), facing="+Z", texture="floor", rough=0.92, role="floor", name="Floor"),
    ]
    return Model("Shell", p)


def _ceiling():
    p = [Part("quad", pos=(0, 0, H), size=(W + 1, D + 1), facing="-Z", color=C["ceiling"], rough=0.95,
              shadow=False, name="Ceiling")]
    for i, y in enumerate((-12, -6, 0, 6, 12)):
        p.append(box((0, y, H - 0.25), (W, 0.25, 0.45), C["beam"], rough=0.8, metal=0.3, shadow=False, name=f"Beam{i}"))
    for i, (x, y) in enumerate(LAMPS):
        p.append(box((x, y, H - 0.55), (2.6, 0.34, 0.08), srgb(0xFFF8E6), emissive=srgb(0xFFF1CC), shadow=False, name=f"Lamp{i}"))
    for i, x in enumerate(range(-18, 19, 6)):
        for s in (1, -1):
            p.append(box((x, s * (D / 2 - 0.03), 5.6), (4, 0.05, 0.9), srgb(0xCFE3F2), emissive=srgb(0x9EC3DE),
                         shadow=False, name=f"Window{i}{'N' if s > 0 else 'S'}"))
    return Model("Ceiling", p)


def _pillars():
    p = []
    for i, (x, y) in enumerate(PILLARS):
        p.append(box((x, y, H / 2), (0.6, 0.6, H), C["pillar"], occ=True, name=f"Pillar{i}"))
        p.append(box((x, y, 0.5), (0.63, 0.63, 1.0), C["yellow"], shadow=False, name=f"Stripe{i}"))
        p.append(box((x, y, 0.33), (0.64, 0.64, 0.12), C["black"], shadow=False, name=f"StripeB{i}"))
        p.append(box((x, y, 0.67), (0.64, 0.64, 0.12), C["black"], shadow=False, name=f"StripeC{i}"))
    return Model("Pillars", p)


def _rack(cx, rng):
    cy, length, dep = RACK_Y, 7.0, 1.1
    reserved = RACK_RESERVED.get(cx, [])
    p = []
    for i in range(4):
        x = cx - length / 2 + i * length / 3
        for s in (-1, 1):
            p.append(box((x, cy + s * dep / 2, 2.3), (0.09, 0.09, 4.6), C["rack_up"], rough=0.55, metal=0.35, name=f"Up{i}{s}"))
    for li, z in enumerate(RACK_LEVELS):
        for s in (-1, 1):
            p.append(box((cx, cy + s * dep / 2, z), (length, 0.06, 0.12), C["rack_beam"], rough=0.55, metal=0.25, name=f"Beam{li}{s}"))
        p.append(box((cx, cy, z + 0.08), (length, dep, 0.04), C["rack_deck"], rough=0.8, metal=0.3, occ=True, name=f"Deck{li}"))
        top, max_h = z + 0.1, (1.1 if li < 3 else 0.85)
        x, k = cx - length / 2 + 0.15, 0
        while x < cx + length / 2 - 0.55:
            w = rng.uniform(0.5, 0.95)
            xc = x + w / 2
            blocked = li == 1 and any(abs(rx - xc) < 0.75 + w / 2 for rx in reserved)
            if not blocked and rng.random() > 0.14:
                h, d = rng.uniform(0.35, max_h), rng.uniform(0.6, 0.95)
                col = C["bin"] if rng.random() < 0.15 else C["cardboard"][rng.integers(5)]
                p.append(box((xc, cy + rng.uniform(-0.06, 0.06), top + h / 2), (w, d, h), col, rough=0.9, occ=True,
                             rot=(0, 0, math.degrees(rng.uniform(-0.03, 0.03))), name=f"Cargo{li}_{k}"))
                k += 1
            x += w + rng.uniform(0.05, 0.15)
    return Model(f"Rack_{int(cx * 10)}".replace("-", "m"), p)


def _cnc(x, y, name):
    dk = C["dark"]
    p = [
        box((0, 0, 1.15), (3.4, 2.2, 2.3), C["machine"], rough=0.55, metal=0.2, occ=True, name="Body"),
        box((0, 0, 0.12), (3.44, 2.24, 0.25), dk, rough=0.6, metal=0.3, name="Base"),
        box((-0.3, -1.111, 1.4), (1.6, 0.02, 1.0), srgb(0x6F8FA6), rough=0.1, metal=0.5, opacity=0.65, shadow=False, name="Window"),
        box((1.2, -1.2, 1.5), (0.7, 0.25, 0.9), dk, rough=0.6, metal=0.3, name="Panel"),
        box((1.2, -1.33, 1.6), (0.5, 0.02, 0.35), srgb(0x3FA9F5), emissive=srgb(0x1C6FB0), shadow=False, name="Screen"),
        box((0, 0, 2.2), (3.44, 2.24, 0.12), C["orange"], name="Stripe"),
    ]
    for i, c in enumerate((0xE03131, 0xF2C200, 0x2F9E44)):
        p.append(cyl((-1.5, 0.9, 2.46 + i * 0.15), 0.08, 0.14, srgb(c), emissive=srgb(c) if i == 2 else None, name=f"Tower{i}"))
    return Model(name, p, pose=(x, y, 0.0, 0.0))


def _conveyor():
    x0, x1, y = 1.0, 15.0, 1.0
    length, cx = x1 - x0, (x0 + x1) / 2
    p = [
        box((cx, y, 0.65), (length, 0.9, 0.35), C["steel"], rough=0.6, metal=0.4, occ=True, name="Frame"),
        box((cx, y, 0.86), (length, 1.0, 0.06), C["belt"], rough=0.9, name="Belt"),
        box((cx, y + 0.56, 0.92), (length, 0.07, 0.16), C["yellow"], name="RailN"),
        box((cx, y - 0.56, 0.92), (length, 0.07, 0.16), C["yellow"], name="RailS"),
    ]
    x, k = x0 + 0.4, 0
    while x < x1:
        for s in (-1, 1):
            p.append(box((x, y + s * 0.38, 0.25), (0.1, 0.1, 0.5), C["steel"], name=f"Leg{k}"))
            k += 1
        x += 2.2
    return Model("Conveyor", p)


def _cabinet_and_posts():
    p = [
        box((-11, 1.5, 1.0), (1.2, 0.6, 2.0), srgb(0x8D969E), rough=0.6, metal=0.3, occ=True, name="Cabinet"),
        box((-11.3, 1.19, 1.6), (0.12, 0.02, 0.12), srgb(0x51CF66), emissive=srgb(0x2F9E44), shadow=False, name="CabinetLed"),
        cyl((8.5, 3.1, 0.25), 0.42, 0.5, C["dark"], occ=True, name="ArmBase"),
    ]
    for i, (x, y) in enumerate([(7.1, 4.5), (10.4, 4.5), (7.1, 1.9), (10.4, 1.9)]):
        p.append(cyl((x, y, 0.5), 0.05, 1.0, C["yellow"], name=f"Post{i}"))
    return Model("Cell", p)


def _bench(cx, cy, length, name):
    p = [box((cx, cy, 0.9), (length, 1.1, 0.08), C["bench"], rough=0.7, name="Top")]
    for i, dx in enumerate((-length / 2 + 0.1, length / 2 - 0.1)):
        for j, dy in enumerate((-0.45, 0.45)):
            p.append(box((cx + dx, cy + dy, 0.43), (0.08, 0.08, 0.86), C["leg"], rough=0.6, metal=0.4, name=f"Leg{i}{j}"))
    p += [
        box((cx, cy - 0.55, 1.62), (length, 0.05, 1.3), C["peg"], rough=0.9, occ=True, name="Pegboard"),
        box((cx + length / 2 - 0.6, cy - 0.2, 1.07), (0.5, 0.26, 0.25), srgb(0xC92A2A), rough=0.5, metal=0.3, name="Toolbox"),
        box((cx - length / 2 + 0.5, cy - 0.1, 1.03), (0.22, 0.3, 0.18), srgb(0x3A3F45), rough=0.5, metal=0.6, name="Vise"),
    ]
    return Model(name, p)


def _barrels():
    pts = [(18.2, -13.1), (18.9, -13.1), (18.55, -13.7), (19.25, -13.7), (17.85, -13.7)]
    return Model("Barrels", [cyl((x, y, 0.45), 0.3, 0.9, C["barrel"], rough=0.5, metal=0.3, occ=True, name=f"Drum{i}")
                             for i, (x, y) in enumerate(pts)])


def _static_pallets(rng):
    out = []
    for i, (x, y, yaw) in enumerate([(6.8, -11.8, math.pi / 2), (10.0, -12.9, 0.0)]):
        out.append(Model(f"Pallet{i}", pallet() + stable_stack(rng, 0.13), pose=(x, y, 0.0, yaw)))
    return out


def _signs():
    p = []
    for i, (x, y, nx, ny) in enumerate(EXT_MOUNTS):
        p.append(Part("quad", pos=(x - nx * 0.12, y - ny * 0.12, 1.92), size=(0.36, 0.18), facing="+X",
                      rot=(0, 0, math.degrees(math.atan2(ny, nx))), texture="sign", color=(1, 1, 1),
                      shadow=False, name=f"Sign{i}"))
    return Model("Signs", p)


def build_static(seed=0):
    """정적 모델 목록. 같은 seed면 랙 적재물 배치도 항상 같다."""
    rng = np.random.default_rng(seed)
    models = [_shell(), _ceiling(), _pillars()]
    models += [_rack(x, rng) for x in RACK_X]
    models += [_cnc(-15.5, 1.5, "CNC1"), _cnc(-6.5, 1.5, "CNC2"), _conveyor(), _cabinet_and_posts()]
    models += [_bench(-14, -11.4, 6, "Bench1"), _bench(-6, -11.4, 6, "Bench2"), _barrels()]
    models += _static_pallets(rng)
    models.append(_signs())
    return models


def occluder_arrays(models):
    """시야를 가리는 박스들을 (mins, maxs) 배열로."""
    boxes = [b for m in models for b in m.occluder_boxes()]
    if not boxes:
        return np.zeros((0, 3)), np.zeros((0, 3))
    return np.array([b[0] for b in boxes]), np.array([b[1] for b in boxes])


def floor_obstacles_2d(models, max_z=1.2, min_z=-0.1):
    """바닥에서 부딪히는 장애물 (로봇 충돌용 2D 박스)."""
    mins, maxs = [], []
    for m in models:
        for p in m.parts:
            if p.role in ("floor",) or p.kind == "quad":
                continue
            c = m.transform(p.local_corners())
            mn, mx = c.min(axis=0), c.max(axis=0)
            if mn[2] < max_z and mx[2] > min_z + 0.15 and (mx[0] - mn[0]) > 0.04 and (mx[1] - mn[1]) > 0.04:
                mins.append(mn[:2])
                maxs.append(mx[:2])
    return np.array(mins), np.array(maxs)
