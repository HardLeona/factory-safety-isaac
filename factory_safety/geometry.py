"""회전, 카메라 투영, 레이캐스트 같은 기하 계산 (numpy만 사용)."""
import math
from dataclasses import dataclass, field

import numpy as np


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def rot_xyz_deg(rx, ry, rz):
    """USD xformOp:rotateXYZ 와 같은 순서 (X 먼저, 그다음 Y, 마지막 Z)."""
    return rot_z(math.radians(rz)) @ rot_y(math.radians(ry)) @ rot_x(math.radians(rx))


def wrap_angle(a):
    return math.atan2(math.sin(a), math.cos(a))


def camera_basis(yaw, pitch, roll=0.0):
    """카메라 앞(f), 오른쪽(r), 위(u) 단위 벡터."""
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    f = np.array([cp * cy, cp * sy, sp])
    r = np.array([sy, -cy, 0.0])
    u = np.cross(r, f)
    if roll:
        cr, sr = math.cos(roll), math.sin(roll)
        r, u = r * cr + u * sr, -r * sr + u * cr
    return f, r, u


@dataclass
class CameraPose:
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    vfov: float = 66.0  # 세로 화각 (도)

    def basis(self):
        return camera_basis(self.yaw, self.pitch, self.roll)

    def usd_matrix(self):
        """USD 카메라용 4x4 행렬 (행 벡터 규약). USD 카메라는 로컬 -Z를 봄."""
        f, r, u = self.basis()
        m = np.eye(4)
        m[0, :3] = r
        m[1, :3] = u
        m[2, :3] = -f
        m[3, :3] = self.pos
        return m


class Projector:
    """핀홀 카메라 투영. 검출기와 라벨 계산에 같이 씀."""

    def __init__(self, width, height, near=0.05):
        self.w, self.h, self.near = width, height, near
        self.set_pose(CameraPose())

    def set_pose(self, pose):
        self.pose = pose
        self.f, self.r, self.u = pose.basis()
        self.pos = np.asarray(pose.pos, dtype=float)
        self.fy = (self.h / 2.0) / math.tan(math.radians(pose.vfov) / 2.0)
        self.fx = self.fy

    @property
    def hfov(self):
        return 2.0 * math.atan((self.w / 2.0) / self.fx)

    def to_cam(self, pts):
        d = np.atleast_2d(pts) - self.pos
        return np.stack([d @ self.r, d @ self.u, d @ self.f], axis=1)

    def project(self, pts):
        """반환: (uv (N,2) 픽셀, depth (N,)). depth <= near 이면 카메라 뒤."""
        c = self.to_cam(pts)
        z = c[:, 2]
        safe = np.where(z > self.near, z, np.nan)
        u = self.w / 2.0 + self.fx * c[:, 0] / safe
        v = self.h / 2.0 - self.fy * c[:, 1] / safe
        return np.stack([u, v], axis=1), z

    def unproject(self, uv, depth):
        """project() 의 역: 픽셀 (u, v) 와 z-depth(카메라 정면 축 거리, distance_to_image_plane 어노테이터와 같은 정의)
        -> 월드 좌표 (N, 3). 바디캠 깊이맵에서 손가락 끝·끼임점의 3D 위치를 복원할 때 씀."""
        uv = np.atleast_2d(uv).astype(float)
        depth = np.atleast_1d(depth).astype(float)
        cx = (uv[:, 0] - self.w / 2.0) * depth / self.fx
        cy = (self.h / 2.0 - uv[:, 1]) * depth / self.fy
        return self.pos + cx[:, None] * self.r + cy[:, None] * self.u + depth[:, None] * self.f

    def rect_of_points(self, pts, min_front=0.6):
        """점들을 투영한 화면 사각형 (화면 밖은 잘라냄). 안 보이면 None."""
        uv, z = self.project(pts)
        front = z > self.near
        if front.sum() < max(1, min_front * len(z)):
            return None
        uv = uv[front]
        x0, y0 = uv.min(axis=0)
        x1, y1 = uv.max(axis=0)
        full = max(1e-9, (x1 - x0) * (y1 - y0))
        cx0, cy0 = max(0.0, x0), max(0.0, y0)
        cx1, cy1 = min(self.w, x1), min(self.h, y1)
        if cx1 <= cx0 or cy1 <= cy0:
            return None
        return (cx0, cy0, cx1, cy1, ((cx1 - cx0) * (cy1 - cy0)) / full)


def aabb_corners(mn, mx):
    xs = (mn[0], mx[0])
    ys = (mn[1], mx[1])
    zs = (mn[2], mx[2])
    return np.array([[x, y, z] for x in xs for y in ys for z in zs], dtype=float)


def ray_hits(origin, direction, tmax, mins, maxs):
    """한 개의 광선과 여러 AABB 교차 여부 (bool 배열). 0 < t < tmax 구간만."""
    if len(mins) == 0:
        return np.zeros(0, dtype=bool)
    d = np.where(np.abs(direction) < 1e-12, 1e-12, direction)
    inv = 1.0 / d
    t1 = (mins - origin) * inv
    t2 = (maxs - origin) * inv
    tnear = np.max(np.minimum(t1, t2), axis=1)
    tfar = np.min(np.maximum(t1, t2), axis=1)
    return (tfar >= np.maximum(tnear, 0.0)) & (tnear < tmax)


def segments_blocked(origin, targets, mins, maxs, end_margin=0.1):
    """origin에서 여러 목표점까지의 선분이 AABB에 막히는지 (R,) bool."""
    targets = np.atleast_2d(targets)
    if len(mins) == 0:
        return np.zeros(len(targets), dtype=bool)
    dvec = targets - origin
    dist = np.linalg.norm(dvec, axis=1)
    dirs = dvec / np.maximum(dist, 1e-9)[:, None]
    dirs = np.where(np.abs(dirs) < 1e-12, 1e-12, dirs)
    inv = 1.0 / dirs                                  # (R,3)
    t1 = (mins[None, :, :] - origin) * inv[:, None, :]  # (R,N,3)
    t2 = (maxs[None, :, :] - origin) * inv[:, None, :]
    tnear = np.max(np.minimum(t1, t2), axis=2)
    tfar = np.min(np.maximum(t1, t2), axis=2)
    tmax = np.maximum(dist - end_margin, 0.05)[:, None]
    hit = (tfar >= np.maximum(tnear, 0.0)) & (tnear < tmax)
    return hit.any(axis=1)


def ray_distances_2d(origin, angles, mins2, maxs2, max_range):
    """2D 거리 센서 (라이다 비슷한 것). 각 각도별 가장 가까운 장애물까지 거리."""
    dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    dirs = np.where(np.abs(dirs) < 1e-12, 1e-12, dirs)
    inv = 1.0 / dirs
    t1 = (mins2[None] - origin) * inv[:, None, :]
    t2 = (maxs2[None] - origin) * inv[:, None, :]
    tnear = np.max(np.minimum(t1, t2), axis=2)
    tfar = np.min(np.maximum(t1, t2), axis=2)
    hit = (tfar >= np.maximum(tnear, 0.0)) & (tnear < max_range)
    t = np.where(hit, np.maximum(tnear, 0.0), max_range)
    return t.min(axis=1)
