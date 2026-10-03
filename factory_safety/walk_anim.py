"""작업자 캐릭터 걷기 애니메이션과 손동작을 직접 만든다 (UsdSkel).

NVIDIA 사람 모델(Reallusion 뼈대 RL_BoneRoot/...)과 NVIDIA 걷기 애니메이션(Root/Pelvis/... 뼈대)은
관절 이름이 달라서 그대로 붙지 않는다 (공식 사람 시뮬레이션은 리타기팅 기능을 따로 씀).
그래서 매 프레임 허벅지, 정강이, 위팔, 아래팔이 향할 방향을 정하고, 부모 관절부터 차례로
그 방향을 보도록 회전시켜 SkelAnimation 을 만든다 (나머지 관절은 기본 자세 그대로).

손동작: 걸음을 멈추고 오른손을 바디캠 앞에 가로로 내밀어 (손바닥이 카메라 쪽, 손가락은 왼쪽) 손가락 1~5개를 편다.
아래팔과 손은 방향만이 아니라 엄지 쪽 방향까지 맞춰야 손바닥이 카메라를 보므로 두 벡터로 회전을 정한다.

뼈대 공간: 미터, Z 위, 캐릭터는 -Y 를 보고 섬, 왼쪽(L_)이 +X. 기본 자세는 팔을 수평으로 벌린 T 자세 (손바닥이 아래, 엄지가 앞).
"""
import math

import numpy as np

PERIOD = 1.1        # 한 걸음 주기 (초, 왼발-오른발 한 번씩)
FPS = 30
HIP_SWING = math.radians(22)
KNEE_BEND = math.radians(45)
ARM_SWING = math.radians(18)
ELBOW_BEND = math.radians(18)
# 뼈: (관절, 방향을 맞출 자식 관절)
BONES = {
    "L_Thigh": "L_Calf", "L_Calf": "L_Foot", "R_Thigh": "R_Calf", "R_Calf": "R_Foot",
    "L_Upperarm": "L_Forearm", "L_Forearm": "L_Hand", "R_Upperarm": "R_Forearm", "R_Forearm": "R_Hand",
    "R_Hand": "R_Mid1",
}

# 손동작 클립: k 번 (손가락 k 개) 은 타임코드 GESTURE_T0 + k * GESTURE_STRIDE 부터 RAISE_S 동안 손을 올리는 동작
GESTURE_T0 = 1000
GESTURE_STRIDE = 100
RAISE_S = 0.5
GESTURE_CAM_AHEAD = 0.16    # 손동작 중 바디캠 위치 (몸 중심에서 앞으로, 걸을 때는 0.30)
HAND_AHEAD = 0.20           # 바디캠에서 손바닥까지 앞쪽 거리
HAND_DROP = 0.035           # 바디캠보다 아래
WRIST_TO_CENTER = 0.085     # 손목에서 손 가운데까지 (손 전체가 화면 가운데 오게)
FINGERS = ("Index", "Mid", "Ring", "Pinky")
CURL = (math.radians(80), math.radians(95), math.radians(55))     # 접은 손가락 마디별 각도
THUMB_TUCK = (math.radians(60), math.radians(45), math.radians(40))  # 엄지 접기 (손바닥 가로질러, 둘째·셋째 마디)
# 카트 끌기: 왼손을 뒤로 뻗어 핸드트럭 손잡이를 잡음 (손잡이는 몸 뒤 PULL_HAND, 뼈대 공간)
PULL_T0 = 2000              # 끌며 걷기 클립 시작 타임코드
PULL_GESTURE_T0 = 3000      # 끌다가 멈춰 손동작 클립 (k 번은 PULL_GESTURE_T0 + k * GESTURE_STRIDE)
PULL_HAND = np.array([0.20, 0.37, 1.05])
REST_THUMB_SIDE = np.array([0.0, -1.0, 0.0])    # 기본 자세에서 오른손 엄지 쪽
REST_PALM = np.array([0.0, 0.0, -1.0])          # 기본 자세에서 손바닥이 보는 쪽


def _mat_col(gf_m):
    """Gf.Matrix4d (행 벡터 규약) -> numpy 4x4 (열 벡터 규약)."""
    return np.array(gf_m, dtype=float).T


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def _axis_angle(axis, ang):
    k = _unit(axis)
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(ang) * kx + (1 - math.cos(ang)) * kx @ kx


def _rot_between(a, b):
    """단위 벡터 a 를 b 로 돌리는 회전 행렬."""
    a, b = _unit(a), _unit(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-9:
        if c > 0:
            return np.eye(3)
        p = np.cross(a, [1.0, 0, 0] if abs(a[0]) < 0.9 else [0, 1.0, 0])     # 정반대: 수직 축으로 180도
        return _axis_angle(p, math.pi)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def _frame(d, s):
    """방향 d 와 옆 방향 s 로 만든 직교 좌표계 (열: d, s, d x s)."""
    d = _unit(d)
    s = np.asarray(s, float) - np.dot(s, d) * d
    s = _unit(s)
    return np.column_stack([d, s, np.cross(d, s)])


def _targets(t):
    """시각 t 에서 각 뼈가 향할 방향 (뼈대 공간)."""
    ph = 2 * math.pi * t / PERIOD
    out = {}
    for side, sgn, lat in (("L", 1.0, 1.0), ("R", -1.0, -1.0)):
        th = sgn * HIP_SWING * math.sin(ph)                        # 허벅지 앞(+)/뒤(-)
        kn = math.radians(5) + KNEE_BEND * max(0.0, sgn * math.cos(ph)) ** 1.5   # 다리를 앞으로 옮길 때 무릎 굽힘
        out[f"{side}_Thigh"] = np.array([0.02 * lat, -math.sin(th), -math.cos(th)])
        out[f"{side}_Calf"] = np.array([0.01 * lat, -math.sin(th - kn), -math.cos(th - kn)])
        al = -sgn * ARM_SWING * math.sin(ph)                      # 팔은 다리와 반대로
        out[f"{side}_Upperarm"] = np.array([0.16 * lat, -math.sin(al), -math.cos(al)])
        out[f"{side}_Forearm"] = np.array([0.06 * lat, -math.sin(al + ELBOW_BEND), -math.cos(al + ELBOW_BEND)])
    return out


def _stand():
    """멈춰 선 자세 (다리 곧게, 팔 내림)."""
    out = {}
    kn = math.radians(5)
    for side, lat in (("L", 1.0), ("R", -1.0)):
        out[f"{side}_Thigh"] = np.array([0.02 * lat, 0.0, -1.0])
        out[f"{side}_Calf"] = np.array([0.01 * lat, math.sin(kn), -math.cos(kn)])
        out[f"{side}_Upperarm"] = np.array([0.16 * lat, -0.03, -1.0])
        out[f"{side}_Forearm"] = np.array([0.06 * lat, -math.sin(ELBOW_BEND), -math.cos(ELBOW_BEND)])
    return out


def _two_bone(sh, target, l1, l2, pole):
    """어깨 sh 에서 target 까지 두 마디 (길이 l1, l2) 팔: (팔꿈치, 손목). 팔꿈치는 pole 쪽으로."""
    d = target - sh
    dn = _unit(d)
    dist = min(np.linalg.norm(d), (l1 + l2) * 0.999)
    a = (l1 ** 2 - l2 ** 2 + dist ** 2) / (2 * dist)
    h = math.sqrt(max(l1 ** 2 - a ** 2, 0.0))
    pole = _unit(pole - np.dot(pole, dn) * dn)
    return sh + a * dn + h * pole, sh + dist * dn


class Rig:
    """뼈대 기본 자세 (관절별 부모 기준 4x4, 열 벡터 규약) 로 자세를 푼다."""

    def __init__(self, joints, rest):
        self.joints = list(joints)
        self.rest = [np.array(m, float) for m in rest]
        self.names = [j.split("/")[-1] for j in self.joints]
        index = {j: i for i, j in enumerate(self.joints)}
        self.parent = [index.get(j.rsplit("/", 1)[0], -1) if "/" in j else -1 for j in self.joints]
        self.idx = {n: i for i, n in enumerate(self.names)}
        self.child_of = {}
        for i, n in enumerate(self.names):
            if n in BONES:
                k = next((k for k, j in enumerate(self.joints) if self.parent[k] == i and self.names[k] == BONES[n]), None)
                if k is not None:
                    self.child_of[i] = k
        self.world_rest = []
        for i in range(len(self.joints)):
            pw = self.world_rest[self.parent[i]] if self.parent[i] >= 0 else np.eye(4)
            self.world_rest.append(pw @ self.rest[i])

    def pos_rest(self, name):
        return self.world_rest[self.idx[name]][:3, 3].copy()

    def solve(self, dirs, frames=None, bends=None):
        """dirs {관절: 방향}, frames {관절: (방향, 기본 자세 옆 벡터, 목표 옆 벡터)}, bends {관절: (기본 자세 축, 각도)}.
        관절별 부모 기준 회전 (3x3) 목록과 뼈대 공간 4x4 목록을 반환."""
        frames, bends = frames or {}, bends or {}
        world, rots = [None] * len(self.joints), []
        for i, n in enumerate(self.names):
            pw = world[self.parent[i]] if self.parent[i] >= 0 else np.eye(4)
            loc = self.rest[i].copy()
            cur = pw[:3, :3] @ loc[:3, :3]
            s = None
            if n in frames and i in self.child_of:
                d, side_rest, side = frames[n]
                off = self.rest[self.child_of[i]][:3, 3]
                cur_side = cur @ (self.world_rest[i][:3, :3].T @ side_rest)
                s = _frame(d, side) @ _frame(cur @ off, cur_side).T
            elif n in dirs and i in self.child_of:
                s = _rot_between(cur @ self.rest[self.child_of[i]][:3, 3], dirs[n])
            elif n in bends:
                axis_rest, ang = bends[n]
                s = _axis_angle(cur @ (self.world_rest[i][:3, :3].T @ axis_rest), ang)
            if s is not None:
                loc[:3, :3] = pw[:3, :3].T @ s @ cur
            world[i] = pw @ loc
            rots.append(loc[:3, :3])
        return rots, world

    def gesture(self, count, cam_h=1.38):
        """손가락 count 개를 펴서 바디캠 앞에 내민 자세의 (dirs, frames, bends)."""
        cam = np.array([0.0, -GESTURE_CAM_AHEAD, cam_h])
        f = np.array([1.0, 0.0, 0.0])                      # 손가락: 작업자 왼쪽 (가로)
        center = cam + np.array([0.0, -HAND_AHEAD, -HAND_DROP])
        palm = _unit(cam - center)                         # 손바닥이 카메라를 봄
        palm = _unit(palm - np.dot(palm, f) * f)
        thumb = np.cross(f, palm)                          # 오른손: 엄지 = 손가락 x 손바닥 (위쪽)
        wrist = center - WRIST_TO_CENTER * f
        sh = self.pos_rest("R_Upperarm")
        l1 = np.linalg.norm(self.pos_rest("R_Forearm") - sh)
        l2 = np.linalg.norm(self.pos_rest("R_Hand") - self.pos_rest("R_Forearm"))
        elbow, wrist = _two_bone(sh, wrist, l1, l2, np.array([-0.7, 0.25, -1.0]))     # 팔꿈치는 아래 바깥쪽
        dirs = {k: v for k, v in _stand().items() if k not in ("R_Upperarm", "R_Forearm")}
        dirs["R_Upperarm"] = elbow - sh
        frames = {"R_Forearm": (wrist - elbow, REST_THUMB_SIDE, thumb),
                  "R_Hand": (f, REST_THUMB_SIDE, thumb)}
        bends = {}
        axis = np.cross(np.array([-1.0, 0, 0]), REST_PALM)     # 손가락을 손바닥 쪽으로 굽히는 축 (기본 자세)
        for k, name in enumerate(FINGERS):
            if count == 5 or k < count:
                continue
            for m, ang in enumerate(CURL):
                bends[f"R_{name}{m + 1}"] = (axis, ang)
        if count != 5:
            bends["R_Thumb1"] = (REST_PALM, THUMB_TUCK[0])
            bends["R_Thumb2"] = (axis, THUMB_TUCK[1])
            bends["R_Thumb3"] = (axis, THUMB_TUCK[2])
        return dirs, frames, bends


def pull_arm(rig, dirs):
    """dirs 의 왼팔을 뒤로 뻗어 카트 손잡이를 잡는 방향으로 바꾼다."""
    sh = rig.pos_rest("L_Upperarm")
    l1 = np.linalg.norm(rig.pos_rest("L_Forearm") - sh)
    l2 = np.linalg.norm(rig.pos_rest("L_Hand") - rig.pos_rest("L_Forearm"))
    elbow, wrist = _two_bone(sh, PULL_HAND, l1, l2, np.array([0.7, 0.3, -1.0]))
    out = dict(dirs)
    out["L_Upperarm"], out["L_Forearm"] = elbow - sh, wrist - elbow
    return out


def _quat(r):
    """3x3 회전 (열 벡터 규약) -> 쿼터니언 (w, x, y, z)."""
    m = r
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def _slerp(q0, q1, a):
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + a * (q1 - q0)
        return q / np.linalg.norm(q)
    th = math.acos(d)
    return (math.sin((1 - a) * th) * q0 + math.sin(a * th) * q1) / math.sin(th)


def gesture_time(count, alpha, pull=False):
    """손동작 count 를 alpha (0 내림 ~ 1 다 올림) 만큼 한 자세의 타임라인 시각 (초). alpha 0 은 선 자세.
    pull: 카트 손잡이를 잡은 채."""
    n = int(round(RAISE_S * FPS))
    return ((PULL_GESTURE_T0 if pull else GESTURE_T0) + count * GESTURE_STRIDE + max(0.0, min(1.0, alpha)) * n) / FPS


def pull_walk_time(t):
    """카트를 끌며 걷는 클립의 타임라인 시각 (초)."""
    return PULL_T0 / FPS + t % PERIOD


def gesture_quats(rig, cam_h=1.38, pull=False):
    """선 자세와 손동작 1~5 의 관절 쿼터니언 (pull: 왼손은 카트 손잡이)."""
    fix = (lambda d: pull_arm(rig, d)) if pull else (lambda d: d)
    stand = [_quat(r) for r in rig.solve(fix(_stand()))[0]]
    poses = {}
    for c in range(1, 6):
        dirs, frames, bends = rig.gesture(c, cam_h)
        poses[c] = [_quat(r) for r in rig.solve(fix(dirs), frames, bends)[0]]
    return stand, poses


def build_walk_animation(stage, skel_prim, anim_path, cam_h=1.38):
    """skel_prim 에 맞는 걷기 + 손동작 SkelAnimation 을 anim_path 에 만들고 연결한다. 주기(초)를 반환.
    재생: 걷기는 타임라인 시각을 (t % PERIOD), 손동작은 gesture_time(k, alpha) 로 맞추면 된다 (timeCodesPerSecond = FPS)."""
    from pxr import Gf, UsdSkel, Vt

    skel = UsdSkel.Skeleton(skel_prim)
    joints = list(skel.GetJointsAttr().Get())
    rest = [_mat_col(m) for m in skel.GetRestTransformsAttr().Get()]
    rig = Rig(joints, rest)

    anim = UsdSkel.Animation.Define(stage, anim_path)
    anim.CreateJointsAttr().Set(Vt.TokenArray(joints))
    anim.CreateTranslationsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*m[:3, 3]) for m in rest]))
    anim.CreateScalesAttr().Set(Vt.Vec3hArray([Gf.Vec3h(1, 1, 1)] * len(joints)))
    rot_attr = anim.CreateRotationsAttr()

    def put(quats, tc):
        rot_attr.Set(Vt.QuatfArray([Gf.Quatf(float(q[0]), float(q[1]), float(q[2]), float(q[3])) for q in quats]), tc)

    for k in range(int(round(PERIOD * FPS)) + 1):
        put([_quat(r) for r in rig.solve(_targets(k / FPS))[0]], k)
        put([_quat(r) for r in rig.solve(pull_arm(rig, _targets(k / FPS)))[0]], PULL_T0 + k)
    n = int(round(RAISE_S * FPS))
    for pull, t0 in ((False, GESTURE_T0), (True, PULL_GESTURE_T0)):
        stand, poses = gesture_quats(rig, cam_h, pull)
        for count, g in poses.items():
            for k in range(n + 1):
                a = k / n
                a = a * a * (3 - 2 * a)
                put([_slerp(q0, q1, a) for q0, q1 in zip(stand, g)], t0 + count * GESTURE_STRIDE + k)
    UsdSkel.BindingAPI.Apply(skel_prim).CreateAnimationSourceRel().SetTargets([anim.GetPath()])
    return PERIOD
