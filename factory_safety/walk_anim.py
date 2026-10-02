"""작업자 캐릭터 걷기 애니메이션을 직접 만든다 (UsdSkel).

NVIDIA 사람 모델(Reallusion 뼈대 RL_BoneRoot/...)과 NVIDIA 걷기 애니메이션(Root/Pelvis/... 뼈대)은
관절 이름이 달라서 그대로 붙지 않는다 (공식 사람 시뮬레이션은 리타기팅 기능을 따로 씀).
그래서 매 프레임 허벅지, 정강이, 위팔, 아래팔이 향할 방향을 정하고, 부모 관절부터 차례로
그 방향을 보도록 회전시켜 SkelAnimation 을 만든다 (나머지 관절은 기본 자세 그대로).

뼈대 공간: 미터, Z 위, 캐릭터는 -Y 를 보고 섬, 왼쪽(L_)이 +X. 기본 자세는 팔을 수평으로 벌린 T 자세.
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
}


def _mat_col(gf_m):
    """Gf.Matrix4d (행 벡터 규약) -> numpy 4x4 (열 벡터 규약)."""
    return np.array(gf_m, dtype=float).T


def _rot_between(a, b):
    """단위 벡터 a 를 b 로 돌리는 회전 행렬."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else -np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


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


def build_walk_animation(stage, skel_prim, anim_path):
    """skel_prim 에 맞는 걷기 SkelAnimation 을 anim_path 에 만들고 연결한다. 주기(초)를 반환.
    재생: 타임라인 시각을 (t % PERIOD) 로 맞추면 된다 (스테이지 timeCodesPerSecond = FPS)."""
    from pxr import Gf, UsdSkel, Vt

    skel = UsdSkel.Skeleton(skel_prim)
    joints = list(skel.GetJointsAttr().Get())
    rest = [_mat_col(m) for m in skel.GetRestTransformsAttr().Get()]
    names = [j.split("/")[-1] for j in joints]
    index = {j: i for i, j in enumerate(joints)}
    parent = [index.get(j.rsplit("/", 1)[0], -1) if "/" in j else -1 for j in joints]
    child_of = {}
    for i, n in enumerate(names):
        if n in BONES:
            want = BONES[n]
            child_of[i] = next(k for k, j in enumerate(joints) if parent[k] == i and names[k] == want)

    anim = UsdSkel.Animation.Define(stage, anim_path)
    anim.CreateJointsAttr().Set(Vt.TokenArray(joints))
    anim.CreateTranslationsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*m[:3, 3]) for m in rest]))
    anim.CreateScalesAttr().Set(Vt.Vec3hArray([Gf.Vec3h(1, 1, 1)] * len(joints)))
    rot_attr = anim.CreateRotationsAttr()
    n_frames = int(round(PERIOD * FPS)) + 1
    for k in range(n_frames):
        tgt = _targets(k / FPS)
        world = [None] * len(joints)
        quats = []
        for i in range(len(joints)):
            pw = world[parent[i]] if parent[i] >= 0 else np.eye(4)
            loc = rest[i].copy()
            if i in child_of and names[i] in tgt:
                off = rest[child_of[i]][:3, 3]
                cur = pw[:3, :3] @ loc[:3, :3] @ off
                s = _rot_between(cur, tgt[names[i]])
                loc[:3, :3] = pw[:3, :3].T @ s @ pw[:3, :3] @ loc[:3, :3]
            world[i] = pw @ loc
            r_row = loc[:3, :3].T            # USD 는 행 벡터 규약
            q = Gf.Matrix3d(*r_row.flatten().tolist()).ExtractRotation().GetQuat()
            quats.append(Gf.Quatf(q))
        rot_attr.Set(Vt.QuatfArray(quats), k)
    UsdSkel.BindingAPI.Apply(skel_prim).CreateAnimationSourceRel().SetTargets([anim.GetPath()])
    return PERIOD
