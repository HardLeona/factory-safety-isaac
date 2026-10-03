"""위험 영역: 라바콘으로 둘러친 곳, DANGER 표지가 선 곳, 에이전트가 스스로 영역으로 판단한 곳.

  라바콘   : 바닥 위치가 3 m 안으로 이어진 라바콘 2개 이상을 한 영역으로 (볼록 다각형 + 0.3 m 여유)
  표지     : DANGER 표지 주변 1.2 m (라바콘 영역 2 m 안에 있으면 그 영역에 합침)
  에이전트 : 방치된 유출 (미끄럼 1.3 m), 무너질 듯한 적재 (붕괴 1.6 m) 는 물건이 아니라 주변이 위험하다고 보고 영역으로.
             위험물 여러 개가 3 m 안에 모여 있으면 한 영역으로 묶는다. 이미 라바콘·표지 영역 안이면 따로 안 만듦
영역은 다각형 (꼭짓점 목록, 반시계) 으로 다룬다.
"""
import math
from dataclasses import dataclass, field

import numpy as np

AGENT_RADIUS = {"spill": 1.3, "stack_unstable": 1.6}
AGENT_REASON = {"spill": "방치된 유출 주변 미끄럼 위험", "stack_unstable": "무너질 듯한 적재 주변 붕괴 위험"}
CONE_LINK_M = 3.0
SIGN_JOIN_M = 2.0
SIGN_RADIUS = 1.2
CONE_PAD = 0.3
CLUSTER_M = 3.0


def circle(c, r, n=20):
    a = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return np.stack([c[0] + r * np.cos(a), c[1] + r * np.sin(a)], axis=1)


def hull(pts):
    """볼록 껍질 (반시계). 점이 1~2개면 그대로."""
    p = sorted(set(map(tuple, np.round(np.asarray(pts, float), 4))))
    if len(p) <= 2:
        return np.array(p, float)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for q in p:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], q) <= 0:
            lower.pop()
        lower.append(q)
    for q in reversed(p):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], q) <= 0:
            upper.pop()
        upper.append(q)
    return np.array(lower[:-1] + upper[:-1], float)


def padded_hull(points, pad):
    """점들을 pad 만큼 부풀린 볼록 다각형."""
    pts = np.concatenate([circle(p, pad, 12) for p in np.atleast_2d(points)])
    return hull(pts)


def inside(p, poly):
    x, y = p
    n, ins = len(poly), False
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1:
            ins = not ins
    return ins


def distance(p, poly):
    """다각형까지 거리 (안이면 0)."""
    if len(poly) >= 3 and inside(p, poly):
        return 0.0
    p = np.asarray(p, float)
    best = 1e9
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        ab = b - a
        t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0, 1)
        best = min(best, float(np.linalg.norm(p - (a + t * ab))))
    return best


@dataclass
class Zone:
    zid: str
    source: str            # "cone" | "sign" | "agent"
    key: str
    poly: np.ndarray
    reason: str
    members: list = field(default_factory=list)
    t0: float = 0.0
    n_cones: int = 0
    has_sign: bool = False

    @property
    def center(self):
        return self.poly.mean(axis=0)

    @property
    def radius(self):
        return float(np.max(np.linalg.norm(self.poly - self.center, axis=1)))


def _clusters(points, link):
    """한 줄로 이어 묶기 (single linkage). 반환: 인덱스 묶음 목록."""
    n = len(points)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(points[i] - points[j]) < link:
                parent[find(i)] = find(j)
    out = {}
    for i in range(n):
        out.setdefault(find(i), []).append(i)
    return list(out.values())


def build(cones, signs, hazards):
    """cones, signs: [(id, xy)]. hazards: [(fid, 상태, xy)] (에이전트가 확정한 위험물).
    반환: [(key, source, poly, reason, members, n_cones, has_sign)] (key 로 같은 영역을 이어서 알아봄)."""
    out = []
    used_signs = set()
    cpts = np.array([xy for _, xy in cones], float).reshape(-1, 2)
    for idx in _clusters(cpts, CONE_LINK_M):
        if len(idx) < 2:
            continue
        pts = cpts[idx]
        members = sorted(cones[i][0] for i in idx)
        cen = pts.mean(axis=0)
        near_signs = [(sid, sxy) for sid, sxy in signs if np.linalg.norm(np.asarray(sxy) - cen) < SIGN_JOIN_M + 0.5 * np.ptp(pts, axis=0).max()]
        for sid, _ in near_signs:
            used_signs.add(sid)
        if len(idx) == 2:
            poly = circle(cen, np.linalg.norm(pts[0] - pts[1]) / 2 + 0.4)
        else:
            poly = padded_hull(pts, CONE_PAD)
        reason = f"라바콘 {len(idx)}개로 둘러친 영역" + (" + DANGER 표지" if near_signs else "")
        out.append((f"cone:{members[0]}", "cone", poly, reason, members + [s for s, _ in near_signs], len(idx), bool(near_signs)))
    for sid, sxy in signs:
        if sid in used_signs:
            continue
        out.append((f"sign:{sid}", "sign", circle(np.asarray(sxy, float), SIGN_RADIUS), "DANGER 표지가 세워진 영역", [sid], 0, True))
    covered = [z[2] for z in out]
    hz = [(fid, st, np.asarray(xy, float)) for fid, st, xy in hazards
          if st in AGENT_RADIUS and not any(len(p) >= 3 and inside(xy, p) for p in covered)]
    hpts = np.array([xy for _, _, xy in hz], float).reshape(-1, 2)
    for idx in _clusters(hpts, CLUSTER_M):
        items = [hz[i] for i in idx]
        if len(items) == 1:
            fid, st, xy = items[0]
            poly = circle(xy, AGENT_RADIUS[st])
            reason = AGENT_REASON[st]
        else:
            poly = hull(np.concatenate([circle(xy, AGENT_RADIUS[st], 12) for _, st, xy in items]))
            reason = f"위험물 {len(items)}개가 모인 영역"
        members = sorted(fid for fid, _, _ in items)
        out.append((f"agent:{members[0]}", "agent", poly, reason, members, 0, False))
    return out
