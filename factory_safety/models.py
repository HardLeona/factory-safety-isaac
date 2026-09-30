"""모든 물체를 기본 도형(박스, 원기둥, 구, 메시, 사각판) 묶음으로 정의.

같은 정의를 USD 장면 생성(usd_scene.py)과 검출기/강화학습의 충돌, 가림 계산이
함께 쓰기 때문에 모양을 바꾸면 양쪽이 자동으로 맞춰진다.
"""
import math
from dataclasses import dataclass, field

import numpy as np

from .config import srgb
from .geometry import rot_xyz_deg, rot_z


@dataclass
class Part:
    kind: str                           # box | cyl | sphere | mesh | quad
    pos: tuple = (0.0, 0.0, 0.0)        # 모델 로컬 좌표
    size: tuple = (1.0, 1.0, 1.0)       # box: (x, y, z) / quad: (가로, 세로)
    radius: float = 0.0
    height: float = 0.0
    axis: str = "Z"                     # cyl 축
    rot: tuple = (0.0, 0.0, 0.0)        # 도 단위, XYZ 순서
    color: tuple = (0.5, 0.5, 0.5)      # 선형 RGB
    rough: float = 0.8
    metal: float = 0.0
    emissive: tuple = None
    opacity: float = 1.0
    texture: str = None                 # quad 에 붙일 텍스처 키
    facing: str = "+X"                  # quad 방향: +X | +Z | -Z
    points: list = None                 # mesh: 로컬 XY 다각형
    role: str = "main"                  # USD 하위 그룹 이름 (라벨 대상 구분용)
    occ: bool = False                   # 시야를 가리는 물체인지
    shadow: bool = True
    name: str = ""

    def local_corners(self):
        if self.kind == "box":
            hx, hy, hz = (s / 2.0 for s in self.size)
        elif self.kind == "cyl":
            r, hh = self.radius, self.height / 2.0
            hx, hy, hz = {"X": (hh, r, r), "Y": (r, hh, r), "Z": (r, r, hh)}[self.axis]
        elif self.kind == "sphere":
            hx = hy = hz = self.radius
        elif self.kind == "quad":
            a, b = self.size[0] / 2.0, self.size[1] / 2.0
            hx, hy, hz = (0.002, a, b) if self.facing == "+X" else (a, b, 0.002)
        elif self.kind == "mesh":
            pts = np.array(self.points, dtype=float)
            c = np.column_stack([pts, np.zeros(len(pts))])
            c = np.vstack([c, c + [0, 0, 0.02]])
            return c @ rot_xyz_deg(*self.rot).T + np.array(self.pos)
        else:
            raise ValueError(self.kind)
        corners = np.array([[x, y, z] for x in (-hx, hx) for y in (-hy, hy) for z in (-hz, hz)])
        return corners @ rot_xyz_deg(*self.rot).T + np.array(self.pos)


@dataclass
class Model:
    name: str
    parts: list = field(default_factory=list)
    pose: tuple = (0.0, 0.0, 0.0, 0.0)  # x, y, z, yaw(rad)

    def transform(self, pts):
        x, y, z, yaw = self.pose
        return np.atleast_2d(pts) @ rot_z(yaw).T + np.array([x, y, z])

    def world_corners(self, roles=None):
        cs = [p.local_corners() for p in self.parts if roles is None or p.role in roles]
        if not cs:
            return np.zeros((0, 3))
        return self.transform(np.vstack(cs))

    def aabb(self, roles=None):
        c = self.world_corners(roles)
        return c.min(axis=0), c.max(axis=0)

    def occluder_boxes(self):
        out = []
        for p in self.parts:
            if p.occ:
                c = self.transform(p.local_corners())
                out.append((c.min(axis=0), c.max(axis=0)))
        return out


# ---------------------------------------------------------------- 색상
C = {
    "cardboard": [srgb(c) for c in (0xB58A55, 0xC89F69, 0xA47A49, 0xD2AF7C, 0x8F6B41)],
    "bin": srgb(0x2F6DB5),
    "rack_up": srgb(0x1F5FA8), "rack_beam": srgb(0xE8590C), "rack_deck": srgb(0x59616A),
    "wall": srgb(0xC4C9CE), "band": srgb(0x34506A), "pillar": srgb(0xAAB2B9),
    "yellow": srgb(0xF2C200), "black": srgb(0x1D2126), "dark": srgb(0x2C3238),
    "machine": srgb(0xDFE3E6), "orange": srgb(0xE8590C), "steel": srgb(0x5B636B),
    "belt": srgb(0x23272B), "wood": srgb(0xA37A48), "bench": srgb(0x7A5A3A),
    "leg": srgb(0x4A5157), "peg": srgb(0x9AA4AD), "red": srgb(0xD42A2A),
    "white": srgb(0xF1F1EC), "barrel": srgb(0x2B5FA8), "ceiling": srgb(0x2A2E34),
    "beam": srgb(0x3A4047), "door": srgb(0x98A3AC),
}


def box(pos, size, color, **kw):
    return Part("box", pos=tuple(pos), size=tuple(size), color=color, **kw)


def cyl(pos, radius, height, color, axis="Z", **kw):
    return Part("cyl", pos=tuple(pos), radius=radius, height=height, axis=axis, color=color, **kw)


def sphere(pos, radius, color, **kw):
    return Part("sphere", pos=tuple(pos), radius=radius, color=color, **kw)


# ---------------------------------------------------------------- 위험 요소 모델
def puddle(rng, oil):
    """불규칙한 웅덩이 다각형 + (기름이면) 쓰러진 오일 캔."""
    n, base_r = 30, rng.uniform(0.6, 1.05)
    ph = rng.uniform(0, 6, 3)
    pts = []
    for i in range(n):
        a = i / n * 2 * math.pi
        r = base_r * (1 + 0.22 * math.sin(3 * a + ph[0]) + 0.12 * math.sin(5 * a + ph[1]) + 0.07 * math.sin(7 * a + ph[2]))
        pts.append((math.cos(a) * r * 1.35, math.sin(a) * r))
    color = srgb(0x151A1F) if oil else srgb(0x3A4650)
    parts = [Part("mesh", pos=(0, 0, 0.004), points=pts, color=color, rough=0.03 if oil else 0.02,
                  metal=0.0, opacity=1.0 if oil else 0.8, role="main", name="Puddle")]
    if oil:
        parts.append(cyl((1.25, rng.uniform(-0.3, 0.3), 0.09), 0.09, 0.26, srgb(0xB3261E), axis="X",
                         rough=0.35, metal=0.5, role="extra", name="OilCan"))
    return parts


def power_tool(variant):
    """variant 0: 그라인더, 1: 원형톱. 로컬 +X 쪽이 날."""
    s = 1.3
    p = []
    if variant == 0:
        p.append(cyl((0, 0, 0.055 * s), 0.05 * s, 0.32 * s, srgb(0x1F7A8C), axis="X", rough=0.45, name="Body"))
        p.append(box((0.2 * s, 0, 0.055 * s), (0.1 * s, 0.1 * s, 0.09 * s), srgb(0x33383E), rough=0.4, metal=0.6, name="Head"))
        p.append(cyl((0.22 * s, 0, 0.012 * s), 0.078 * s, 0.008 * s, srgb(0xC3C9CF), rough=0.25, metal=0.85, name="Disc"))
        p.append(cyl((0.18 * s, -0.1 * s, 0.06 * s), 0.018 * s, 0.14 * s, srgb(0x1B1E22), axis="Y", name="Handle"))
    else:
        p.append(box((0, 0, 0.01 * s), (0.34 * s, 0.22 * s, 0.02 * s), srgb(0x9AA1A8), rough=0.4, metal=0.7, name="Plate"))
        p.append(cyl((-0.02 * s, -0.06 * s, 0.1 * s), 0.06 * s, 0.16 * s, srgb(0xF2B705), axis="Y", rough=0.5, name="Motor"))
        p.append(box((0.02 * s, 0.02 * s, 0.16 * s), (0.22 * s, 0.06 * s, 0.07 * s), srgb(0xF2B705), rough=0.5, name="Housing"))
        p.append(cyl((0.02 * s, 0.07 * s, 0.09 * s), 0.105 * s, 0.004 * s, srgb(0xD0D5DA), axis="Y", rough=0.2, metal=0.9, name="Blade"))
    # 전선: 얇은 박스 여러 개로 지그재그
    prev = np.array([-0.18 * s, 0.0])
    for i in range(1, 5):
        cur = np.array([(-0.18 - i * 0.16) * s, math.sin(i * 1.4) * 0.12 * s])
        mid, d = (prev + cur) / 2, cur - prev
        p.append(box((mid[0], mid[1], 0.012 * s), (np.linalg.norm(d), 0.018, 0.018), srgb(0x111111),
                     rot=(0, 0, math.degrees(math.atan2(d[1], d[0]))), role="extra", name=f"Cable{i}"))
        prev = cur
    return p


def pallet(z0=0.0):
    wood = C["wood"]
    p = [box((0, dy, z0 + 0.05), (1.2, 0.14, 0.1), wood, occ=True, role="extra", name=f"Stringer{i}")
         for i, dy in enumerate((-0.42, 0.0, 0.42))]
    p += [box((dx, 0, z0 + 0.115), (0.13, 1.0, 0.03), wood, role="extra", name=f"Board{i}")
          for i, dx in enumerate((-0.52, -0.26, 0.0, 0.26, 0.52))]
    return p


def stable_stack(rng, z0):
    col = C["cardboard"][rng.integers(len(C["cardboard"]))]
    layers = 2 + int(rng.random() < 0.5)
    p = []
    for layer in range(layers):
        for dx in (-0.29, 0.29):
            for dy in (-0.24, 0.24):
                p.append(box((dx, dy, z0 + 0.21 + layer * 0.42), (0.56, 0.46, 0.42), col, rough=0.9,
                             occ=True, role="extra", name=f"Box{len(p)}"))
    return p


def unstable_stack(rng, max_h, x0, z0):
    """위로 갈수록 +X 쪽으로 밀리고 기울어진 박스 더미. (parts, 기울기 도)"""
    n = (2 + int(rng.random() < 0.5)) if max_h < 1.3 else (3 + int(rng.random() < 0.6))
    tilt = int(round(rng.uniform(6, 12)))
    p, z, off = [], z0, 0.0
    for i in range(n):
        w, d = rng.uniform(0.5, 0.7), rng.uniform(0.42, 0.6)
        h = min(rng.uniform(0.32, 0.5), (max_h - 0.05) / n)
        if i > 0:
            off += rng.uniform(0.1, 0.18)
        ang = (i / max(1, n - 1)) * tilt
        col = C["cardboard"][rng.integers(len(C["cardboard"]))]
        p.append(box((x0 + off, rng.uniform(-0.04, 0.04), z + h / 2), (w, d, h), col, rough=0.9,
                     rot=(0, ang, math.degrees(rng.uniform(-0.18, 0.18))), role="main", name=f"Box{i}"))
        z += h * 0.97
    return p, tilt


GAUGE_Z = 0.62
GAUGE_X = 0.064


def extinguisher(ok, variant):
    """로컬 +X 가 앞면. 몸통은 role=main, 압력계는 role=gauge."""
    red, md = C["red"], srgb(0x2B2F34)
    p = [
        cyl((0, 0, 0.25), 0.115, 0.5, red, rough=0.35, metal=0.15, name="Body"),
        sphere((0, 0, 0.5), 0.115, red, rough=0.35, metal=0.15, name="Dome"),
        cyl((0, 0, 0.3), 0.117, 0.14, C["white"], rough=0.6, name="Band"),
        cyl((0, 0, 0.64), 0.033, 0.09, md, rough=0.4, metal=0.7, name="Neck"),
        box((0, 0.03, 0.7), (0.04, 0.17, 0.025), md, rough=0.4, metal=0.7, name="Handle"),
        box((0, 0.02, 0.665), (0.035, 0.15, 0.02), md, rough=0.4, metal=0.7, rot=(14, 0, 0), name="Lever"),
        cyl((0.02, -0.125, 0.42), 0.014, 0.42, srgb(0x111214), rot=(7, 0, 0), name="Hose"),
        box((-0.13, 0, 0.35), (0.03, 0.12, 0.34), md, rough=0.4, metal=0.7, role="extra", name="Bracket"),
        cyl((0.05, 0, GAUGE_Z), 0.056, 0.025, md, axis="X", rough=0.4, metal=0.7, role="gauge", name="Bezel"),
        Part("quad", pos=(GAUGE_X, 0, GAUGE_Z), size=(0.096, 0.096), color=(1, 1, 1), rough=0.5,
             texture=f"gauge_{'ok' if ok else 'low'}_{variant}", facing="+X", role="gauge", name="Face"),
    ]
    return p


# ---------------------------------------------------------------- 로봇, 작업자
def patrol_robot():
    """로컬 +X 가 앞. role=head 는 카메라 머리 (좌우로 돌아감)."""
    yel, dk = srgb(0xF2B705), srgb(0x24282D)
    p = [
        box((0, 0, 0.26), (0.9, 0.7, 0.28), yel, rough=0.5, name="Body"),
        box((0, 0, 0.1), (0.94, 0.74, 0.1), dk, rough=0.6, name="Bumper"),
        cyl((0, 0, 0.95), 0.035, 1.15, dk, name="Mast"),
        cyl((-0.32, 0.25, 0.44), 0.05, 0.08, srgb(0xFF8A1F), emissive=srgb(0xFF7A00), name="Beacon"),
    ]
    for i, (x, y) in enumerate([(0.3, 0.37), (0.3, -0.37), (-0.3, 0.37), (-0.3, -0.37)]):
        p.append(cyl((x, y, 0.1), 0.1, 0.06, dk, axis="Y", name=f"Wheel{i}"))
    p.append(box((0, 0, 0), (0.2, 0.26, 0.15), dk, role="head", name="Head"))
    p.append(cyl((0.11, 0, 0), 0.05, 0.05, srgb(0x4DABF7), axis="X", emissive=srgb(0x1C6FB0), role="head", name="Lens"))
    return p


def worker():
    """안전조끼를 입은 작업자. 로컬 +X 가 앞, 가슴에 바디캠."""
    vest, pants, dark = srgb(0xD4EF38), srgb(0x2D3A4F), srgb(0x1D2126)
    p = []
    for s in (1, -1):
        p.append(box((0, 0.11 * s, 0.49), (0.17, 0.14, 0.86), pants, name=f"Leg{s}"))
        p.append(box((0.04, 0.11 * s, 0.05), (0.27, 0.15, 0.1), srgb(0x2A2420), name=f"Boot{s}"))
        p.append(box((0, 0.27 * s, 1.2), (0.12, 0.1, 0.6), dark, name=f"Arm{s}"))
    p += [
        box((0, 0, 1.22), (0.25, 0.42, 0.62), vest, rough=0.6, name="Vest"),
        box((0, 0, 1.08), (0.26, 0.43, 0.05), srgb(0xE8ECEF), rough=0.3, metal=0.4, name="Stripe1"),
        box((0, 0, 1.3), (0.26, 0.43, 0.05), srgb(0xE8ECEF), rough=0.3, metal=0.4, name="Stripe2"),
        sphere((0, 0, 1.66), 0.12, srgb(0xD9A47E), rough=0.7, name="Head"),
        sphere((0, 0, 1.73), 0.13, C["yellow"], rough=0.4, name="Helmet"),
        box((0.13, 0, 1.72), (0.12, 0.3, 0.02), C["yellow"], rough=0.4, name="Brim"),
        box((0.145, 0, 1.38), (0.04, 0.08, 0.1), dark, name="BodyCam"),
    ]
    return p
