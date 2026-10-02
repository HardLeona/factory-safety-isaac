"""창고 배치: 작업자 순찰 경로, 위험 요소 자리, 작업대, CCTV 위치.

NVIDIA warehouse_multiple_shelves.usd 를 그대로 쓴다 (24 x 30 m 안쪽, 벽 y=-12, y=18).
  랙 3줄: x = -9.8~-8.2 (서), -0.8~0.8 (가운데), 8.2~9.8 (동), y = -3.8 ~ 12.4
  통로 2개: 서쪽 x -8.2~-0.8 (가운데 -4.5), 동쪽 x 0.8~8.2 (가운데 4.5)
  남쪽 빈 공간 y -12 ~ -3.8, 북쪽 빈 공간 y 12.4 ~ 18 (손수레, 부품 상자가 x=-6, 4.5 근처에 있음)
창고에 원래 있던 소화기 6개(랙 끝)는 끄고, 같은 자리에 라벨을 붙인 소화기를 다시 놓는다.
"""
import math

import numpy as np

# 창고 안에 원래 있던 소화기 (끄고, 그 위치와 방향을 소화기 자리로 쓴다)
ENV_EXTINGUISHERS = [
    "/Root/Shelf_0/SM_FireExtinguisher_18", "/Root/Shelf_0/SM_FireExtinguisher_24",
    "/Root/Shelf_1/SM_FireExtinguisher_18", "/Root/Shelf_1/SM_FireExtinguisher_24",
    "/Root/Shelf_2/SM_FireExtinguisher_18", "/Root/Shelf_2/SM_FireExtinguisher_24",
]
# 소화기 자리: (x, y) 바닥 위치와 랙 바깥쪽 방향 (북쪽 끝은 +Y, 남쪽 끝은 -Y). 높이와 회전은 원래 소화기에서 가져옴
EXT_MOUNTS = [
    (-8.86, 12.45, (0.0, 1.0)), (-8.91, -3.81, (0.0, -1.0)),
    (0.14, 12.45, (0.0, 1.0)), (0.09, -3.81, (0.0, -1.0)),
    (9.14, 12.45, (0.0, 1.0)), (9.09, -3.81, (0.0, -1.0)),
]

EXT_FALL_OUT = 0.45    # 쓰러진 소화기: 자리에서 바깥쪽으로 이만큼 떨어진 바닥
EXT_BLOCK_OUT = 0.60   # 가로막은 상자: 소화기 앞 이만큼

# 작업자 순찰 경로 (닫힌 경로, 시계 반대 방향 아님: 남쪽 가운데 -> 서쪽 통로 북상 -> 북쪽 -> 동쪽 통로 남하)
PATH_WAYPOINTS = [
    (0.0, -7.6), (-4.5, -6.6), (-4.5, 11.6), (-3.0, 15.6), (3.0, 15.6), (4.5, 11.6), (4.5, -6.6),
]

# 바닥 유출 자리 (통로 바닥, 경로에서 옆으로 1~2 m)
SPILL_SLOTS = [(-5.9, 0.5), (-3.1, 7.8), (3.1, 2.2), (5.9, 9.4), (-2.4, -9.3), (2.6, -4.9), (0.2, 16.8)]
# 통로 바닥 공구 자리
TOOL_FLOOR_SLOTS = [(-7.3, 4.6), (-1.8, 10.6), (7.2, 6.4), (1.9, -1.6), (-6.2, -9.6), (6.4, -9.0)]
# 작업대 (남쪽 벽 앞, packing table 2.47 x 0.78 m, 윗면 높이 TABLE_TOP)
TABLES = [(-6.0, -11.45, 0.0), (6.0, -11.45, 0.0)]
TABLE_TOP = 0.995
# 작업대 위 공구 자리 (작업대 중심 기준 x 오프셋)
TABLE_TOOL_DX = (-0.8, 0.0, 0.8)
# 팔레트 적재 자리 (바닥)
STACK_SLOTS = [(-7.3, -5.4), (7.2, -5.6), (-1.6, -10.7), (1.8, -10.9), (-7.6, 16.6), (7.4, 16.7)]

# CCTV: (이름, x, y, z, yaw_deg, pitch_deg, 세로 화각)
CCTVS = [
    ("cctv_west", -4.5, -11.6, 5.0, 90.0, -24.0, 52.0),     # 서쪽 통로를 남쪽 벽에서 내려다봄
    ("cctv_east", 4.5, 17.6, 5.0, -90.0, -24.0, 52.0),      # 동쪽 통로를 북쪽 벽에서
    ("cctv_south", 9.4, -11.7, 4.6, 150.0, -26.0, 58.0),    # 남쪽 빈 공간 (작업대, 적재 자리)
]


def zone_name(x, y):
    if y < -3.8:
        return "남쪽 작업 구역"
    if y > 12.4:
        return "북쪽 구역"
    if x < -0.8:
        return "서쪽 통로"
    if x > 0.8:
        return "동쪽 통로"
    return "가운데 랙"


class PatrolPath:
    """순찰 경로 (닫힌 꺾은선, 모서리는 반지름 r 로 둥글게)."""

    def __init__(self, waypoints=PATH_WAYPOINTS, corner=1.2, step=0.05):
        pts = np.array(waypoints, dtype=float)
        n = len(pts)
        dense = []
        for i in range(n):
            a, b, c = pts[i - 1], pts[i], pts[(i + 1) % n]
            d1, d2 = (b - a) / np.linalg.norm(b - a), (c - b) / np.linalg.norm(c - b)
            r = min(corner, 0.45 * np.linalg.norm(b - a), 0.45 * np.linalg.norm(c - b))
            p_in, p_out = b - d1 * r, b + d2 * r
            for t in np.linspace(0, 1, 12, endpoint=False):     # 2차 베지어로 모서리
                dense.append((1 - t) ** 2 * p_in + 2 * (1 - t) * t * b + t ** 2 * p_out)
            nxt = pts[(i + 1) % n]
            nxt_c = pts[(i + 2) % n]
            d3 = (nxt_c - nxt) / np.linalg.norm(nxt_c - nxt)
            r2 = min(corner, 0.45 * np.linalg.norm(nxt - b), 0.45 * np.linalg.norm(nxt_c - nxt))
            seg_end = nxt - d2 * r2
            L = np.linalg.norm(seg_end - p_out)
            for t in np.linspace(0, 1, max(2, int(L / step)), endpoint=False):
                dense.append(p_out + (seg_end - p_out) * t)
        self.pts = np.array(dense)
        seg = np.linalg.norm(np.diff(np.vstack([self.pts, self.pts[:1]]), axis=0), axis=1)
        self.cum = np.concatenate([[0.0], np.cumsum(seg)])
        self.length = float(self.cum[-1])

    def point_at(self, s):
        s = s % self.length
        i = int(np.searchsorted(self.cum, s, side="right") - 1)
        i = min(max(i, 0), len(self.pts) - 1)
        a, b = self.pts[i], self.pts[(i + 1) % len(self.pts)]
        k = (s - self.cum[i]) / max(self.cum[i + 1] - self.cum[i], 1e-9)
        return a + (b - a) * k

    def heading_at(self, s, ahead=0.6):
        a, b = self.point_at(s - ahead * 0.5), self.point_at(s + ahead * 0.5)
        return math.atan2(b[1] - a[1], b[0] - a[0])

    def nearest_s(self, x, y):
        d = np.hypot(self.pts[:, 0] - x, self.pts[:, 1] - y)
        return float(self.cum[int(np.argmin(d))])
