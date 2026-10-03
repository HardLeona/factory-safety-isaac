"""안전 순찰 에이전트: 점검 계획 -> 바디캠 판정 -> 위험물 대장 -> CCTV 확대 재확인 -> 조치 지시서.

판정에는 정답표를 쓰지 않는다. 에이전트가 아는 것은 도면(경로, 랙, 소화기 자리, 작업대, CCTV 위치)뿐이다.

  1. 계획      : 도면에서 꼭 확인할 지점(소화기 6곳, 작업대 2곳)으로 점검표를 만든다.
                 통로 바닥(유출, 공구, 적재)은 순찰 경로를 걷는 바디캠이 맡는다.
  2. 판정      : 바디캠 YOLO 추적으로 물체마다 판정을 모아 위험물 대장에 올린다 (위치, 상태, 확신도, 본 횟수).
  3. 재확인 계획: 잠깐 보고 지나친 물체, 판정이 엇갈린 물체, CCTV 접근 경고로 처음 알게 된 물체,
                 순찰이 끝나도 확인 못 한 점검 지점을 재확인 목록에 올린다.
  4. 도구 선택  : 그 자리를 볼 수 있는 CCTV 를 고르고 (거리, 랙에 가리는지), 그 CCTV 의 PTZ 를 돌려 확대해서 YOLO 로 다시 판정.
                 못 찾으면 다음 CCTV 로 다시 하고, 다 안 되면 사람에게 "현장 확인" 을 넘긴다.
  5. 결과 반영  : 재확인 판정을 대장에 합치고 판정이 바뀌면 기록한다. CCTV 접근 경고는 그 위험물의 우선순위를 올린다.
  6. 위험 영역  : 라바콘으로 둘러친 곳, DANGER 표지가 선 곳, 스스로 판단한 위험 주변(유출, 무너질 듯한 적재)을 영역으로 설정.
  7. 음성 경고  : 작업자가 위험물에 1 m 안으로 다가가거나 위험 영역 0.5 m 안에 들어서면 "경고 경고 위험 요소가 식별되었습니다".
  8. 보고      : 위험물마다 조치 방법, 우선순위, 위치(공구는 이름까지)를 담은 조치 지시서.

채점(evaluate)에서만 정답표를 쓴다: 바디캠만 썼을 때(before)와 에이전트 최종 대장(after)을 창고 전체 물체와 비교.
"""
import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from . import warehouse as W
from . import zones as Z
from .config import CLASS_KO, EQUIPMENT, HAZARD, KIND, TOOL_KO, TOOL_TYPES, TOUCH_WARN_M
from .geometry import CameraPose, Projector, segments_blocked
from .inspection import BodycamInspector, CCTVProximity, floor_point
from .report import clock
from .walker import cctv_pose

# 위험 상태별 조치와 기본 심각도 (3 높음 ~ 1 낮음)
ACTIONS = {
    "spill": ("유출물 제거. 치우기 전까지 미끄럼 주의 표지판과 라바콘 설치", 3),
    "stack_unstable": ("기울어진 상단 상자를 내려 다시 쌓고 떨어진 상자 회수. 그 전까지 주변 통행 제한", 3),
    "ext_blocked": ("소화기 앞 적재물 치우기 (소화기 앞 통로 확보)", 2),
    "ext_fallen": ("소화기를 제자리에 다시 걸고 압력계, 안전핀 점검", 2),
    "tool_floor": ("통로 바닥의 공구를 작업대로 회수", 2),
}
SAFE_NOTES = {
    "spill_marked": "표지 조치됨. 청소가 끝나면 표지 회수",
    "tool_stored": "작업대 정리 상태 양호",
    "stack_stable": "적재 상태 양호",
    "ext_ok": "소화기 정상 비치",
}
CHECK_ACTION = ("현장에서 직접 확인 (카메라로 확정 못 함)", 1)
# 재확인 때 확대 카메라가 볼 높이 (물체 종류별)
AIM_Z = {"spill": 0.0, "tool": 0.15, "stack": 0.6, "ext": 0.9}
# CCTV 시야를 가리는 고정 구조물: 랙(6 m) + 기둥(9 m)
RACK_MIN = np.array([r[0] for r in W.RACKS + W.PILLARS], float)
RACK_MAX = np.array([r[1] for r in W.RACKS + W.PILLARS], float)
ACTIVE = ("확정", "재확인 완료")
INACTIVE = ("기각", "병합")


def group_of(cls):
    """같은 물체로 묶는 단위. 공구는 바닥/작업대가 서로 다른 물체라 나누고, 나머지는 위험/안전 짝이 같은 물체."""
    return cls if KIND[cls] == "tool" else KIND[cls]


def kind_of(group):
    return KIND.get(group, group)


def verdict(cls):
    return f"{'위험' if HAZARD[cls] else '안전'} {CLASS_KO[cls]}"


@dataclass
class Checkpoint:
    cid: str
    group: str               # "ext" 또는 "tool_stored"
    name: str
    xy: np.ndarray
    target: np.ndarray       # 확대 카메라가 볼 점
    radius: float
    status: str = "미확인"   # 미확인 / 바디캠 확인 / CCTV 확대 확인 / 현장 확인 필요
    finding: str = None


def build_checkpoints():
    cps = []
    for i, (x, y, (nx, ny)) in enumerate(W.EXT_MOUNTS):
        side = "서쪽" if x < -4 else ("동쪽" if x > 4 else "가운데")
        end = "북쪽" if ny > 0 else "남쪽"
        cps.append(Checkpoint(f"C{i + 1}", "ext", f"소화기 {i + 1} ({side} 랙 {end} 끝)", np.array([x, y], float),
                              np.array([x + 0.4 * nx, y + 0.4 * ny, AIM_Z["ext"]]), 1.6))
    for j, (x, y, _) in enumerate(W.TABLES):
        cps.append(Checkpoint(f"C{len(W.EXT_MOUNTS) + j + 1}", "tool_stored", f"작업대 {j + 1} ({'서쪽' if x < 0 else '동쪽'})",
                              np.array([x, y], float), np.array([x, y, W.TABLE_TOP + 0.1]), 2.0))
    return cps


@dataclass
class Finding:
    """위험물 대장 한 줄. 판정은 출처별 YOLO 신뢰도 합으로 투표 (CCTV 확대 한 장은 바디캠 두 장 몫)."""
    fid: str
    group: str
    xy: np.ndarray
    t0: float
    source: str
    body: dict = field(default_factory=lambda: defaultdict(float))
    ptz: dict = field(default_factory=lambda: defaultdict(float))
    cctv: dict = field(default_factory=lambda: defaultdict(float))
    n_body: int = 0
    n_ptz: int = 0
    body_confirmed: bool = False
    status: str = "확정"         # 확정 / 재확인 대기 / 재확인 완료 / 현장 확인 필요 / 기각 / 병합
    merged_into: str = None
    near_miss: int = 0
    rechecked: bool = False
    changed_by_recheck: bool = False
    checkpoint: str = None
    n_xy: float = 1.0
    types: dict = field(default_factory=lambda: defaultdict(float))   # 공구 종류별 YOLO 신뢰도 합

    PTZ_W = 2.0
    CCTV_W = 0.5

    def votes(self):
        v = defaultdict(float)
        for src, w in ((self.body, 1.0), (self.cctv, self.CCTV_W), (self.ptz, self.PTZ_W)):
            for c, s in src.items():
                v[c] += w * s
        return v

    @property
    def cls(self):
        v = self.votes()
        return max(v, key=v.get) if v else None

    @property
    def body_cls(self):
        return max(self.body, key=self.body.get) if self.body else None

    @property
    def share(self):
        v = self.votes()
        tot = sum(v.values())
        return max(v.values()) / tot if tot > 0 else 0.0

    @property
    def tool_names(self):
        """많이 보인 공구 종류 (1등의 35% 이상)."""
        if not self.types:
            return []
        top = max(self.types.values())
        return [t for t, v in sorted(self.types.items(), key=lambda kv: -kv[1]) if v >= 0.35 * top]

    def move_to(self, xy, w=1.0):
        self.xy = (self.xy * self.n_xy + np.asarray(xy, float) * w) / (self.n_xy + w)
        self.n_xy += w


class SafetyAgent:
    CONFIRM = 3              # 같은 추적 번호가 이만큼 보이면 확정
    TRACK_LOST_S = 0.8       # 이만큼 안 보이면 추적이 끝난 것으로 봄
    LOW_SHARE = 0.7          # 판정 표에서 1등 비율이 이보다 낮으면 "엇갈림"
    SAME_PLACE_M = 2.0       # 같은 묶음이 이 거리 안이면 같은 물체
    SNAP_M = 3.0             # 작업대 위 공구는 도면의 작업대 자리에 맞춤
    JOB_DELAY_S = 2.0        # 재확인을 요청하고 이만큼 기다려도 미확인이면 실행 (그 사이 바디캠이 확정할 수 있음)
    PTZ_WINDOWS = (4.0, 2.6)  # 확대 촬영 화면 세로 폭 (m): 넓게 한 장, 좁게 한 장
    PTZ_MAX_RANGE = 32.0
    MATCH_FRAC = 0.42        # 확대 화면에서 목표점과 박스 중심 거리 허용 (화면 높이 비율)
    PTZ_SURE = 0.5           # 확대 판정 신뢰도가 이보다 낮으면 다른 CCTV 로 한 번 더 보고, 그래도 낮으면 현장 확인
    OTHER_SURE = 0.7         # 그 자리에 다른 물체가 이만큼 확실히 보이면 바디캠이 잘못 본 것으로 판단
    NEAR_ALERT_M = 2.5       # CCTV 경고 위치와 대장 물체를 같은 것으로 보는 거리
    FAR_K = 0.2              # 멀리서 본 것일수록 위치가 부정확해서, 같은 물체로 보는 거리를 거리의 20% 까지 넓힘
    BIG_EXTRA_M = 1.0        # 적재물(팔레트 1.2 m)은 크고 바닥 투영 오차도 커서 같은 물체로 보는 거리를 1 m 더 넓힘
    PTZ_POS_W = 2.0          # 확대 화면(높이 5 m)에서 구한 위치의 가중치 (바디캠 한 장은 최대 0.44)
    EXT_REGROUP_M = 6.0      # 다른 종류로 잡혔다가 소화기로 바뀐 항목을 가장 가까운 소화기 자리에 붙이는 거리
    STACK_H = 1.7            # 대장에 있는 적재물을 CCTV 시야 가림으로 볼 때 높이 (팔레트 + 상자 3단)
    STACK_CLEAR_M = 1.2      # 목표점에서 이보다 가까운 적재물은 가림으로 안 봄 (소화기를 막은 상자 자체일 수 있음)
    ZONE_WARN_M = 0.5        # 위험 영역 경계에서 이 거리 안이면 음성 경고
    VOICE_GAP_S = 5.5        # 음성 경고끼리 최소 간격 (음성 길이)
    WARN_REPEAT_S = 12.0     # 같은 위험물·영역은 이 간격으로만 다시 경고
    MARKER_MIN = 2           # 라바콘·표지는 이만큼 여러 번 보여야 영역 계산에 씀
    MARKER_RANGE = 15.0      # CCTV 에서 라바콘·표지 위치를 쓰는 최대 거리

    def __init__(self, img_w, img_h, cctv_names, log=print):
        self.w, self.h = img_w, img_h
        self.cctvs = {n: cctv_pose(n) for n in cctv_names}
        self.checkpoints = build_checkpoints()
        self.findings = []
        self.tracks = {}
        self.jobs = []
        self.job = None
        self.timeline = []
        self.log = log
        self.stats = defaultdict(int)
        self.scorer = BodycamInspector(img_w, img_h)   # 채점용 (판정에는 안 씀)
        self.prox = CCTVProximity(img_w, img_h)
        self.markers = []        # 라바콘, DANGER 표지: {"mid", "kind", "xy", "w", "n"}
        self.zones = {}          # key -> Zone
        self.voice_events = []   # 음성 경고 기록
        self.on_voice = None     # 음성 경고 때 부를 함수 (순찰 화면에서 소리 재생)
        self._last_voice = -1e9
        self._warned = {}
        self.worker_log = []     # 채점용 실제 작업자 위치 (판정에는 안 씀)
        self.equip_tids = set()  # 운반 카트처럼 장비로 본 추적 번호 (그 번호로는 위험/안전 판정을 안 함)
        self._t = 0.0

    # ------------------------------------------------------------ 기록
    def say(self, t, kind, text):
        self.timeline.append({"t": float(t), "kind": kind, "text": text})
        if self.log:
            self.log(f"[{clock(t)}] <{kind}> {text}")

    # ------------------------------------------------------------ 1. 계획
    def plan(self, path_len):
        self.say(0.0, "계획", f"목표: 순찰 한 바퀴({path_len:.0f} m) 동안 위험물을 찾아 위험/안전을 판정하고, "
                              f"작업자 접근을 경고하고, 조치 지시서를 만든다")
        self.say(0.0, "계획", f"통로 바닥(유출, 공구, 적재)은 바디캠, 점검표 {len(self.checkpoints)}곳은 바디캠으로 보되 "
                              f"못 보면 CCTV 확대로 확인")
        for cp in self.checkpoints:
            cams = self.camera_options(cp.target)
            self.say(0.0, "계획", f"점검표 {cp.cid} {cp.name}  (확대 가능 CCTV: {', '.join(n for n, _ in cams) or '없음'})")

    def camera_options(self, target, exclude=None, why=None):
        """target 을 볼 수 있는 CCTV (거리순). 너무 멀거나, 랙(높이 6 m)·기둥이나 대장에 있는 적재물(높이 약 1.7 m)에
        가리면 뺀다. why 에 dict 를 주면 뺀 이유를 적어 준다."""
        obst = [f for f in self.findings if f is not exclude and f.group == "stack" and f.status in ACTIVE
                and np.linalg.norm(f.xy - target[:2]) > self.STACK_CLEAR_M]
        smin = np.array([[f.xy[0] - 0.7, f.xy[1] - 0.7, 0.0] for f in obst]).reshape(-1, 3)
        smax = np.array([[f.xy[0] + 0.7, f.xy[1] + 0.7, self.STACK_H] for f in obst]).reshape(-1, 3)
        out = []
        for n, pose in self.cctvs.items():
            d = float(np.linalg.norm(target - pose.pos))
            if d > self.PTZ_MAX_RANGE:
                continue
            if segments_blocked(pose.pos, target[None], RACK_MIN, RACK_MAX)[0]:
                continue
            if len(obst):
                hit = [f for f, mn, mx in zip(obst, smin, smax) if segments_blocked(pose.pos, target[None], mn[None], mx[None])[0]]
                if hit:
                    if why is not None:
                        why[n] = hit[0].fid
                    continue
            out.append((n, d))
        return sorted(out, key=lambda c: c[1])

    def aim(self, name, target, window):
        """CCTV name 의 PTZ 를 target 쪽으로 돌리고, 화면 세로가 window m 가 되게 확대."""
        pos = self.cctvs[name].pos
        v = target - pos
        dist = float(np.linalg.norm(v))
        vfov = math.degrees(2 * math.atan(window / 2 / dist))
        return CameraPose(pos=pos.copy(), yaw=math.atan2(v[1], v[0]), pitch=math.atan2(v[2], math.hypot(v[0], v[1])),
                          vfov=float(np.clip(vfov, 3.0, 50.0)))

    # ------------------------------------------------------------ 2. 바디캠 판정
    @staticmethod
    def _sanitize(xy):
        """바닥 위치를 창고 안으로 넣고, 랙 안쪽에 떨어진 추정은 가장 가까운 랙 면으로 밀어낸다.
        벽 밖으로 1 m 넘게 나간 추정은 믿을 수 없어서 버린다 (None)."""
        if not (-11.3 < xy[0] < 11.3 and -13.0 < xy[1] < 19.0):
            return None
        x, y = float(np.clip(xy[0], -10.1, 10.1)), float(np.clip(xy[1], -11.9, 17.9))
        for (a, b) in W.RACKS:
            if a[0] < x < b[0] and a[1] < y < b[1]:
                d = [(x - a[0], (a[0] - 0.1, y)), (b[0] - x, (b[0] + 0.1, y)),
                     (y - a[1], (x, a[1] - 0.1)), (b[1] - y, (x, b[1] + 0.1))]
                x, y = min(d)[1]
        return np.array([x, y])

    def _locate(self, cam, xyxy, cls):
        """박스 -> (바닥 위치, 카메라에서 거리). 못 구하면 (None, 0)."""
        if KIND[cls] == "ext":
            xy = self._snap_ext(cam, xyxy)
        else:
            z = W.TABLE_TOP if cls == "tool_stored" else 0.0
            xy = floor_point(cam, xyxy, cls, self.w, self.h, z=z)
            if xy is not None and cls == "tool_stored":
                cp = min((c for c in self.checkpoints if c.group == "tool_stored"), key=lambda c: np.linalg.norm(c.xy - xy))
                xy = cp.xy.copy() if np.linalg.norm(cp.xy - xy) < self.SNAP_M else None
            elif xy is not None:
                xy = self._sanitize(xy)
        if xy is None:
            return None, 0.0
        return xy, float(np.linalg.norm(xy - np.asarray(cam.pos[:2], float)))

    def _snap_ext(self, cam, xyxy):
        """소화기는 거치대(벽걸이 1.4~1.6 m) 나 그 앞 바닥에 있으니, 화면 박스와 겹치는 도면의 소화기 자리를 찾는다."""
        P = Projector(self.w, self.h)
        P.set_pose(cam)
        x0, y0, x1, y1 = xyxy
        bw, bh = max(x1 - x0, 20.0), y1 - y0
        best = None
        for i, (mx, my, (nx, ny)) in enumerate(W.EXT_MOUNTS):
            pts = np.array([[mx, my, W.EXT_HEIGHTS[i] - 0.25], [mx + W.EXT_FALL_OUT * nx, my + W.EXT_FALL_OUT * ny, 0.1]])
            uv, z = P.project(pts)
            for (u, v), d in zip(uv, z):
                if not (0.3 < d < 25.0) or np.isnan(u):
                    continue
                du = abs(u - (x0 + x1) / 2) / bw
                if du < 1.0 and y0 - 0.5 * bh <= v <= y1 + 0.5 * bh and (best is None or du < best[0]):
                    best = (du, i)
        if best is None:
            return None
        mx, my, _ = W.EXT_MOUNTS[best[1]]
        return np.array([mx, my], float)

    @staticmethod
    def _weight(rng):
        return 1.0 / max(rng, 1.5) ** 2        # 가까이서 본 위치일수록 정확

    @staticmethod
    def _track_xy(tr):
        w = np.array([w for _, w in tr["xy"]])
        p = np.array([p for p, _ in tr["xy"]])
        return (p * w[:, None]).sum(axis=0) / w.sum(), float(w.sum())

    def _tool_state(self, cam, xyxy):
        """공구 박스 아래쪽을 작업대 윗면 높이로 투영해서 작업대 위면 정리된 공구, 아니면 통로 바닥에 방치된 공구."""
        p = floor_point(cam, xyxy, "tool", self.w, self.h, z=W.TABLE_TOP)
        if p is not None:
            for tx, ty, _ in W.TABLES:
                if abs(p[0] - tx) < W.TABLE_HALF[0] + 0.15 and abs(p[1] - ty) < W.TABLE_HALF[1] + 0.15:
                    return "tool_stored"
        return "tool_floor"

    def _normalize(self, cam, dets):
        """YOLO 결과 -> (상태 관찰 [(상태, 신뢰도, xyxy, 추적 번호, 공구 종류)], 라바콘·표지 [(이름, 신뢰도, xyxy)])."""
        obs, marks = [], []
        for name, conf, xyxy, tid, *_ in dets:
            if name in EQUIPMENT:                           # 운반 카트는 위험/안전 판정 대상이 아님
                if tid is not None:
                    self._equipment_track(tid)
                continue
            if name == "worker" or (tid is not None and tid in self.equip_tids):
                continue
            if name in ("cone", "danger_sign"):
                marks.append((name, conf, xyxy))
                continue
            tool = None
            if name in TOOL_TYPES:
                tool, name = name, self._tool_state(cam, xyxy)
            obs.append((name, conf, xyxy, tid, tool))
        return obs, marks

    def _equipment_track(self, tid):
        """이 추적 번호를 장비로 봄. 앞서 이 번호로만 (바디캠에서) 만든 판정이 있으면 장비를 잘못 본 것으로 뺌
        (카트를 돌아보는 중에 화면 가장자리에서 잠깐 적재로 잡히는 경우)."""
        if tid in self.equip_tids:
            return
        self.equip_tids.add(tid)
        tr = self.tracks.get(tid)
        f = tr["finding"] if tr else None
        if f is not None and f.status in ACTIVE and f.n_ptz == 0 and f.n_body <= tr["n"] + 1:
            f.status = "기각"
            self.stats["equip_rejected"] += 1
            self.say(self._t, "판정 수정", f"{f.fid} 같은 물체를 가까이서 보니 운반 카트 (장비) → {CLASS_KO[f.cls]} 판정을 대장에서 뺌")

    def _label(self, f):
        names = [TOOL_KO[t] for t in f.tool_names] if KIND.get(f.cls) == "tool" else []
        return verdict(f.cls) + (f" ({', '.join(names)})" if names else "")

    def on_bodycam(self, t, cam, dets, worker_xy=None, gt=None):
        """dets: [(클래스, 신뢰도, xyxy, 추적 번호)]. worker_xy: 작업자 위치 (바디캠 위치 추적, 없으면 카메라 위치).
        gt: 채점용 정답 박스 (판정에는 안 씀)."""
        self._t = t
        obs, marks = self._normalize(cam, dets)
        if gt is not None:
            self.scorer.score_frame([(n, c, b, tid) for n, c, b, tid, _ in obs], gt)
        for name, conf, xyxy in marks:
            self._on_marker(t, cam, name, xyxy, 25.0)
        seen = set()
        for name, conf, xyxy, tid, tool in obs:
            if tid is None:
                continue
            seen.add(tid)
            tr = self.tracks.get(tid)
            if tr is None:
                tr = self.tracks[tid] = {"votes": defaultdict(float), "types": defaultdict(float), "n": 0, "xy": [],
                                         "rng": 1e9, "finding": None, "last": t}
            tr["votes"][name] += conf
            if tool:
                tr["types"][tool] += conf
            tr["n"] += 1
            tr["last"] = t
            xy, rng = self._locate(cam, xyxy, name)
            if xy is not None:
                tr["xy"].append((xy, self._weight(rng)))
                tr["rng"] = min(tr["rng"], rng)
            f = tr["finding"]
            if f is not None:
                before = f.cls
                f.body[name] += conf
                f.n_body += 1
                if tool:
                    f.types[tool] += conf
                if xy is not None and f.group not in ("ext", "tool_stored") and group_of(name) == f.group:
                    f.move_to(xy, self._weight(rng))
                if f.cls != before:
                    self.say(t, "판정 수정", f"{f.fid} 가까이서 다시 보니 {CLASS_KO[before]} → {verdict(f.cls)}")
                    tr["finding"] = self._regroup(t, f)
            elif tr["n"] >= self.CONFIRM and tr["xy"]:
                self._confirm_track(t, tr)
        for tid in [k for k, tr in self.tracks.items() if k not in seen and t - tr["last"] >= self.TRACK_LOST_S]:
            self._end_track(t, self.tracks.pop(tid))
        self._tick_checkpoints(t)
        self._update_zones(t)
        wxy = np.asarray(worker_xy if worker_xy is not None else cam.pos[:2], float)
        self._check_warnings(t, wxy, cam, obs)

    def _add_track_votes(self, f, tr):
        for c, s in tr["votes"].items():
            f.body[c] += s
        for c, s in tr.get("types", {}).items():
            f.types[c] += s
        f.n_body += tr["n"]

    def _confirm_track(self, t, tr):
        best = max(tr["votes"], key=tr["votes"].get)
        xy, w = self._track_xy(tr)
        f, new = self._attach(t, group_of(best), xy, "바디캠", tr["rng"], w)
        before = f.cls
        self._add_track_votes(f, tr)
        tr["finding"] = f
        was = f.body_confirmed
        f.body_confirmed = True
        if f.status in ("재확인 대기", "현장 확인 필요"):
            if f.status == "현장 확인 필요":
                self.stats["recheck_escalated"] -= 1
                self.say(t, "판정 수정", f"{f.fid} 현장 확인을 요청했던 물체를 바디캠이 다시 보고 확정 → 현장 확인 취소")
            f.status = "확정"
            self._cancel_jobs(t, f)
        if new or not was:
            conf = tr["votes"][best] / tr["n"]
            self.say(t, "판정", f"바디캠 {self._label(f)}  |  {W.zone_name(*f.xy)}  |  YOLO {conf * 100:.0f}%  → 대장 {f.fid}")
        elif before and f.cls != before:
            self.say(t, "판정 수정", f"{f.fid} 다시 보니 {CLASS_KO[before]} → {verdict(f.cls)}")
        tr["finding"] = self._regroup(t, f)

    def _end_track(self, t, tr):
        f = tr["finding"]
        if f is None:
            if not tr["xy"]:
                return
            best = max(tr["votes"], key=tr["votes"].get)
            xy, w = self._track_xy(tr)
            f, new = self._attach(t, group_of(best), xy, "바디캠", tr["rng"], w)
            self._add_track_votes(f, tr)
            if new:
                f.status = "재확인 대기"
                self._request(t, f, None, f"바디캠에 {tr['n']}프레임만 보이고 지나감 ({CLASS_KO[best]}?)", "잠깐 보임")
            else:
                self._regroup(t, f)
        elif not f.rechecked and f.status == "확정" and f.share < self.LOW_SHARE and not self._has_job(f):
            v = f.votes()
            tot = sum(v.values())
            top = sorted(v.items(), key=lambda kv: -kv[1])[:2]
            self._request(t, f, None, "판정이 엇갈림 (" + ", ".join(f"{CLASS_KO[c]} {100 * s / tot:.0f}%" for c, s in top) + ")",
                          "엇갈림")

    def _find_near(self, group, xy, radius, exclude=None):
        best, bd = None, None
        for f in self.findings:
            if f is exclude or f.group != group or f.status in INACTIVE:
                continue
            d = float(np.linalg.norm(f.xy - xy))
            if d < radius and (bd is None or d < bd):
                best, bd = f, d
        return best

    def _place(self, group, xy):
        """도면 제약: 소화기는 거치 자리, 작업대 공구는 작업대 위에만 있다. 가까운 자리에 맞추고, 너무 멀면 그대로."""
        if group == "ext":
            spots = [np.array(m[:2], float) for m in W.EXT_MOUNTS]
            lim = self.EXT_REGROUP_M
        elif group == "tool_stored":
            spots = [c.xy for c in self.checkpoints if c.group == "tool_stored"]
            lim = self.SNAP_M
        else:
            return np.asarray(xy, float)
        sp = min(spots, key=lambda q: np.linalg.norm(q - xy))
        return sp.copy() if np.linalg.norm(sp - xy) < lim else np.asarray(xy, float)

    def _attach(self, t, group, xy, source, rng=0.0, w=1.0, radius=None):
        xy = self._place(group, xy)
        base = self.SAME_PLACE_M + (self.BIG_EXTRA_M if group == "stack" else 0.0)
        best = self._find_near(group, xy, radius if radius is not None else max(base, self.FAR_K * rng))
        if best is not None:
            if group not in ("ext", "tool_stored") and w > 0:
                best.move_to(xy, w)
            return best, False
        f = Finding(f"F{len(self.findings) + 1:02d}", group, np.asarray(xy, float), float(t), source, n_xy=max(w, 1e-6))
        self.findings.append(f)
        return f, True

    def _regroup(self, t, f):
        """판정이 다른 종류로 바뀌면 (예: 소화기 앞을 막은 상자를 처음엔 적재로 봄) 그 종류의 대장 항목과 합친다."""
        c = f.cls
        if c is None or group_of(c) == f.group or f.status in INACTIVE:
            return f
        g = group_of(c)
        xy = f.xy
        if KIND[c] == "ext":
            # 소화기는 도면의 거치 자리에만 있다. 멀리서 본 위치는 몇 m 틀릴 수 있어서 넓게 찾고, 그래도 없으면 현장 확인
            m = min(W.EXT_MOUNTS, key=lambda e: math.hypot(e[0] - xy[0], e[1] - xy[1]))
            if math.hypot(m[0] - xy[0], m[1] - xy[1]) > self.EXT_REGROUP_M:
                if f.status != "현장 확인 필요":
                    f.status = "현장 확인 필요"
                    self.say(t, "현장 확인", f"{f.fid} 도면에 소화기 자리가 없는 곳에서 {CLASS_KO[c]} 로 보임 → 현장 확인 요청")
                return f
            xy = np.array(m[:2], float)
        other = self._find_near(g, xy, self.SAME_PLACE_M, exclude=f)
        if other is None:
            f.group, f.xy = g, np.asarray(xy, float)
            return f
        return self._merge(t, f, other)

    def _merge(self, t, f, other):
        """f 를 other 에 합친다 (같은 물체를 두 번 올린 경우). other 를 돌려준다."""
        for src, dst in ((f.body, other.body), (f.ptz, other.ptz), (f.cctv, other.cctv), (f.types, other.types)):
            for k, v in src.items():
                dst[k] += v
        other.n_body += f.n_body
        other.n_ptz += f.n_ptz
        other.near_miss += f.near_miss
        other.body_confirmed |= f.body_confirmed
        other.rechecked |= f.rechecked
        if f.status == "재확인 완료" and other.status == "재확인 대기":
            other.status = "재확인 완료"
        f.status, f.merged_into = "병합", other.fid
        for tr in self.tracks.values():
            if tr["finding"] is f:
                tr["finding"] = other
        for j in [j for j in self.jobs if j["finding"] is f]:
            self.jobs.remove(j)
        if self.job is not None and self.job["finding"] is f:
            self.job["finding"] = other
        self.say(t, "판정 수정", f"{f.fid} 는 {other.fid} 와 같은 물체 ({verdict(other.cls)}) → 대장에서 합침")
        return other

    def _tick_checkpoints(self, t):
        for cp in self.checkpoints:
            if cp.status != "미확인":
                continue
            for f in self.findings:
                if f.status not in ACTIVE or f.cls is None:
                    continue
                ok = KIND[f.cls] == "ext" if cp.group == "ext" else f.group == "tool_stored"
                if ok and np.linalg.norm(f.xy - cp.xy) < cp.radius:
                    cp.status = "바디캠 확인" if f.body_confirmed else "CCTV 확대 확인"
                    cp.finding, f.checkpoint = f.fid, cp.cid
                    self.say(t, "점검표", f"{cp.cid} {cp.name} 확인: {verdict(f.cls)} (대장 {f.fid})")
                    break

    # ------------------------------------------------------------ 3~5. 재확인
    def _target_of(self, f):
        return np.array([f.xy[0], f.xy[1], AIM_Z[kind_of(f.group)]])

    def _has_job(self, f):
        return any(j["finding"] is f for j in self.jobs) or (self.job is not None and self.job["finding"] is f)

    def _request(self, t, f, cp, reason, jtype):
        target = cp.target if cp is not None else self._target_of(f)
        why = {}
        cams = self.camera_options(target, exclude=f, why=why)
        blocked = "".join(f", {n} 는 {fid} 적재에 가림" for n, fid in why.items())
        job = {"finding": f, "checkpoint": cp, "target": target, "cams": cams, "cam_i": 0, "shot": 0, "dets": [],
               "reason": reason, "type": jtype, "t_req": float(t)}
        name = cp.cid if cp is not None else f.fid
        where = cp.name if cp is not None else W.zone_name(*f.xy)
        self.stats["recheck_requested"] += 1
        if not cams:
            self.say(t, "재확인 계획", f"{name} {where}: {reason} → 볼 수 있는 CCTV 없음{blocked}")
            self._escalate(t, job)
            return
        self.jobs.append(job)
        self.say(t, "재확인 계획", f"{name} {where}: {reason} → {cams[0][0]} 확대로 재확인 예약 ({cams[0][1]:.0f} m{blocked})")

    def _cancel_jobs(self, t, f):
        drop = [j for j in self.jobs if j["finding"] is f]
        for j in drop:
            self.jobs.remove(j)
            self.stats["recheck_canceled"] += 1
            self.say(t, "재확인 취소", f"{f.fid} 기다리는 동안 바디캠이 다시 보고 확정 → CCTV 확대 안 함")

    def _job_still_needed(self, j):
        f, cp = j["finding"], j["checkpoint"]
        if cp is not None:
            return cp.status == "미확인"
        if f.status in INACTIVE:
            return False
        if j["type"] == "잠깐 보임" or j["type"] == "CCTV 발견":
            return f.status == "재확인 대기"
        return not f.rechecked

    def next_ptz(self, t):
        """지금 찍을 확대 촬영: (CCTV 이름, 카메라 자세) 또는 None."""
        while self.job is None:
            ready = [j for j in self.jobs if t - j["t_req"] >= self.JOB_DELAY_S]
            if not ready:
                return None
            self.jobs.remove(ready[0])
            if self._job_still_needed(ready[0]):
                self.job = ready[0]
                self.stats["recheck_run"] += 1
        j = self.job
        name = j["cams"][j["cam_i"]][0]
        j["cam"] = name
        j["pose"] = self.aim(name, j["target"], self.PTZ_WINDOWS[j["shot"]])
        if j["shot"] == 0:
            what = j["checkpoint"].name if j["checkpoint"] is not None else f"{j['finding'].fid} {W.zone_name(*j['finding'].xy)}"
            self.say(t, "CCTV 선택", f"{name} PTZ 를 {what} 쪽으로 돌려 확대 (화각 {j['pose'].vfov:.0f}°, "
                                     f"{j['cams'][j['cam_i']][1]:.0f} m)")
        return name, j["pose"]

    def on_ptz(self, t, dets):
        """확대 촬영 결과. dets: [(클래스, 신뢰도, xyxy, ...)]."""
        j = self.job
        if j is None:
            return
        obs, _ = self._normalize(j["pose"], [(n, c, b, None) for n, c, b, *_ in dets])
        dets = [(n, c, b, tool) for n, c, b, _, tool in obs]
        P = Projector(self.w, self.h)
        P.set_pose(j["pose"])
        uv, _ = P.project(j["target"][None])
        u, v = uv[0]
        want = "ext" if j["checkpoint"] is not None and j["checkpoint"].group == "ext" else (
            "tool" if j["checkpoint"] is not None else kind_of(j["finding"].group))
        best, other = None, None
        for name, conf, xyxy, tool in dets:
            if name == "worker":
                continue
            cx, cy = (xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2
            inside = xyxy[0] <= u <= xyxy[2] and xyxy[1] <= v <= xyxy[3]
            if not (inside or math.hypot(cx - u, cy - v) <= self.MATCH_FRAC * self.h):
                continue
            if KIND[name] == want:
                # 목표점에 가장 가까운 것 (박스 안에 목표점이 있으면 우선). 옆에 있는 다른 물체를 고르지 않게
                rank = (0 if inside else 1, math.hypot(cx - u, cy - v))
                if best is None or rank < best[2]:
                    best = (name, float(conf), rank, xyxy, tool)
            elif other is None or conf > other[1]:
                other = (name, float(conf))
        j["shot"] += 1
        if best:
            j["dets"].append(best[:2])
            if best[4]:
                j.setdefault("types", defaultdict(float))[best[4]] += best[1]
            z = W.TABLE_TOP if best[0] == "tool_stored" else 0.0
            xy = None if KIND[best[0]] == "ext" else floor_point(j["pose"], best[3], best[0], self.w, self.h, max_range=40.0, z=z)
            if xy is not None and self._sanitize(xy) is not None:
                j.setdefault("xys", []).append(self._sanitize(xy))
        if other:
            j.setdefault("others", []).append(other)
        if j["shot"] < len(self.PTZ_WINDOWS):
            return
        more = j["cam_i"] + 1 < len(j["cams"])
        if j["dets"]:
            cls, conf = self._ptz_verdict(j)
            if conf >= self.PTZ_SURE or not more:
                self._apply_recheck(t, j, sure=conf >= self.PTZ_SURE)
                self.job = None
                return
            self.say(t, "재시도", f"{j['cam']} 확대 판정 확신이 낮음 ({CLASS_KO[cls]} {conf * 100:.0f}%) → "
                                 f"{j['cams'][j['cam_i'] + 1][0]} 로 한 번 더 확인")
        elif more:
            self.say(t, "재시도", f"{j['cam']} 확대 화면에서 못 찾음 → {j['cams'][j['cam_i'] + 1][0]} 로 다시")
        else:
            self._escalate(t, j)
            self.job = None
            return
        self.stats["recheck_retry"] += 1
        j["cam_i"] += 1
        j["shot"] = 0

    @staticmethod
    def _ptz_verdict(j):
        """확대 촬영들의 판정: (클래스, 그 클래스 평균 신뢰도)."""
        v = defaultdict(list)
        for c, s in j["dets"]:
            v[c].append(s)
        c = max(v, key=lambda k: sum(v[k]))
        return c, float(np.mean(v[c]))

    def _apply_recheck(self, t, j, sure=True):
        f, cp = j["finding"], j["checkpoint"]
        if f is None:
            # 점검 지점에서 찾은 것은 그 자리 물체: 가까운 다른 물체 항목(예: 작업대 옆 바닥 공구)과 섞지 않게 1 m 안만 같은 것으로
            f, _ = self._attach(t, group_of(self._ptz_verdict(j)[0]), cp.xy.copy(), "CCTV 확대", w=0.0, radius=1.0)
        before = f.cls
        for c, s in j["dets"]:
            f.ptz[c] += s
        for c, s in j.get("types", {}).items():
            f.types[c] += 2.0 * s
        f.n_ptz += len(j["dets"])
        f.rechecked = True
        if j.get("xys") and f.group not in ("ext", "tool_stored") and cp is None:
            moved = np.mean(j["xys"], axis=0)
            if np.linalg.norm(moved - f.xy) > 0.5:
                self.say(t, "위치 보정", f"{f.fid} 확대 화면으로 위치를 ({f.xy[0]:+.1f}, {f.xy[1]:+.1f}) → ({moved[0]:+.1f}, {moved[1]:+.1f}) 로 고침")
            f.move_to(moved, self.PTZ_POS_W * len(j["xys"]))
            other = self._find_near(f.group, f.xy, self.SAME_PLACE_M + (self.BIG_EXTRA_M if f.group == "stack" else 0.0), exclude=f)
            if other is not None:
                f = self._merge(t, f, other)
        _, conf = self._ptz_verdict(j)
        self.stats["recheck_found"] += 1
        f = self._regroup(t, f)
        if not sure and not f.body_confirmed:
            f.status = "현장 확인 필요"
            self.stats["recheck_escalated"] += 1
            self.say(t, "현장 확인", f"{f.fid} 확대 판정 확신이 낮음 ({CLASS_KO[f.cls]} {conf * 100:.0f}%) → 작업자에게 현장 확인 요청")
            if cp is not None:
                cp.status, cp.finding, f.checkpoint = "현장 확인 필요", f.fid, cp.cid
            return
        f.status = "재확인 완료"
        if before and before != f.cls:
            f.changed_by_recheck = True
            self.stats["recheck_changed"] += 1
            self.say(t, "판정 수정", f"{f.fid} {j['cam']} 확대 결과 {CLASS_KO[f.cls]} {conf * 100:.0f}% → "
                                     f"{CLASS_KO[before]} 에서 {verdict(f.cls)} 로 고침")
        else:
            self.say(t, "재확인", f"{f.fid} {j['cam']} 확대 결과 {self._label(f)} {conf * 100:.0f}% 확인")
        if cp is not None:
            cp.status, cp.finding, f.checkpoint = "CCTV 확대 확인", f.fid, cp.cid
            self.say(t, "점검표", f"{cp.cid} {cp.name} 확인: {verdict(f.cls)} (대장 {f.fid})")
        self._tick_checkpoints(t)

    def _escalate(self, t, j):
        f, cp = j["finding"], j["checkpoint"]
        self.stats["recheck_escalated"] += 1
        if cp is not None:
            cp.status = "현장 확인 필요"
            self.say(t, "현장 확인", f"{cp.cid} {cp.name}: CCTV 로 확인 못 함 → 작업자에게 현장 확인 요청")
            return
        others = [o for o in j.get("others", []) if o[1] >= self.OTHER_SURE]
        if f.body_confirmed:
            f.rechecked = True
            self.say(t, "재확인", f"{f.fid} 확대로 못 봄 → 바디캠 판정 유지 ({verdict(f.cls)})")
        elif others and not f.near_miss:
            o = max(others, key=lambda x: x[1])
            f.status = "기각"
            self.stats["recheck_rejected"] += 1
            self.say(t, "재확인", f"{f.fid} 그 자리는 {CLASS_KO[o[0]]} ({o[1] * 100:.0f}%) → 바디캠이 잘못 본 것으로 보고 대장에서 뺌")
        elif sum(f.body.values()) >= 1.0 or f.near_miss:
            f.status = "현장 확인 필요"
            self.say(t, "현장 확인", f"{f.fid} {W.zone_name(*f.xy)} {CLASS_KO[f.cls]}? 카메라로 확정 못 함 → 현장 확인 요청")
        else:
            f.status = "기각"
            self.stats["recheck_rejected"] += 1
            self.say(t, "재확인", f"{f.fid} 확대해도 안 보이고 근거가 약함 → 오검출로 보고 대장에서 뺌")

    # ------------------------------------------------------------ CCTV 접근 경고
    def on_cctv(self, t, name, cam, dets):
        obs, marks = self._normalize(cam, dets)
        for mname, conf, xyxy in marks:
            self._on_marker(t, cam, mname, xyxy, self.MARKER_RANGE)
        dets = [d for d in dets if d[0] == "worker"] + [(n, c, b, tid) for n, c, b, tid, _ in obs]
        events, meas = self.prox.update(t, name, cam, dets)
        for _, cname, cls, d, hp in events:
            cand = [(float(np.linalg.norm(f.xy - hp)), k, f) for k, f in enumerate(self.findings)
                    if f.status not in INACTIVE and f.cls and KIND[f.cls] == KIND[cls]]
            near = min(cand, key=lambda c: c[:2]) if cand else None
            if near and near[0] < self.NEAR_ALERT_M:
                f = near[2]
            else:
                f, new = self._attach(t, group_of(cls), np.asarray(hp, float), "CCTV")
                f.cctv[cls] += 0.5
                if new:
                    f.status = "재확인 대기"
                    self._request(t, f, None, f"CCTV {cname} 접근 경고에서 처음 발견 ({CLASS_KO[cls]}?)", "CCTV 발견")
            f.near_miss += 1
            self.say(t, "접근 경고", f"{cname}: 작업자 ↔ {CLASS_KO[cls]} {d:.1f} m  |  {W.zone_name(*hp)} → 대장 {f.fid} 우선순위 올림")
        return events, meas

    # ------------------------------------------------------------ 6. 위험 영역
    def _on_marker(self, t, cam, name, xyxy, max_range):
        """라바콘, DANGER 표지의 바닥 위치를 모은다 (같은 자리면 하나로)."""
        xy = floor_point(cam, xyxy, name, self.w, self.h, max_range=max_range)
        if xy is None or self._sanitize(xy) is None:
            return
        xy = self._sanitize(xy)
        rng = float(np.linalg.norm(xy - np.asarray(cam.pos[:2], float)))
        w = self._weight(rng)
        same = 0.6 if name == "cone" else 1.0
        best = min((m for m in self.markers if m["kind"] == name), key=lambda m: np.linalg.norm(m["xy"] - xy), default=None)
        if best is not None and np.linalg.norm(best["xy"] - xy) < max(same, 0.12 * rng):
            best["xy"] = (best["xy"] * best["w"] + xy * w) / (best["w"] + w)
            best["w"] += w
            best["n"] += 1
            return
        self.markers.append({"mid": f"{'K' if name == 'cone' else 'S'}{len(self.markers) + 1:02d}", "kind": name,
                             "xy": xy, "w": w, "n": 1, "t0": float(t)})

    def _update_zones(self, t):
        cones = [(m["mid"], m["xy"]) for m in self.markers if m["kind"] == "cone" and m["n"] >= self.MARKER_MIN]
        signs = [(m["mid"], m["xy"]) for m in self.markers if m["kind"] == "danger_sign" and m["n"] >= self.MARKER_MIN]
        hazards = [(f.fid, f.cls, f.xy) for f in self.findings if f.status in ACTIVE and f.cls and HAZARD.get(f.cls)]
        alive = set()
        for key, source, poly, reason, members, n_cones, sign in Z.build(cones, signs, hazards):
            alive.add(key)
            z = self.zones.get(key)
            if z is None:
                z = Z.Zone(f"Z{len(self.zones) + 1}", source, key, poly, reason, members, float(t), n_cones, sign)
                self.zones[key] = z
                self.stats["zones"] += 1
                why = {"cone": "라바콘 표시를 알아봄", "sign": "DANGER 표지를 알아봄", "agent": "에이전트가 영역으로 판단"}[source]
                self.say(t, "위험 영역", f"{z.zid} {reason} ({W.zone_name(*z.center)}, 반지름 약 {z.radius:.1f} m) → {why}, 위험 영역으로 설정")
            else:
                if n_cones > z.n_cones or (sign and not z.has_sign):
                    self.say(t, "위험 영역", f"{z.zid} 갱신: {reason}")
                z.poly, z.reason, z.members, z.n_cones, z.has_sign = poly, reason, members, n_cones, sign
        # 위험물이 정리돼 근거가 사라진 에이전트 영역은 지운다 (라바콘·표지 영역은 한 번 보면 유지)
        for key in [k for k, z in self.zones.items() if z.source == "agent" and k not in alive]:
            self.zones.pop(key)

    # ------------------------------------------------------------ 7. 음성 경고
    def log_worker(self, t, xy):
        """채점용 실제 작업자 위치 (판정에는 안 씀)."""
        self.worker_log.append((float(t), float(xy[0]), float(xy[1])))

    def _check_warnings(self, t, wxy, cam, obs):
        """작업자가 위험물 1 m 안, 위험 영역 0.5 m 안이면 음성 경고. 대장 (기억) 과 지금 화면 둘 다 본다."""
        cands = []
        for f in self.findings:
            if f.status in ACTIVE and f.cls and HAZARD.get(f.cls):
                d = float(np.linalg.norm(f.xy - wxy))
                if d < TOUCH_WARN_M:
                    cands.append((d, f"F:{f.fid}", f"{f.fid} {self._label(f)} {d:.1f} m"))
        for z in self.zones.values():
            d = Z.distance(wxy, z.poly)
            if d < self.ZONE_WARN_M:
                cands.append((d, f"Z:{z.zid}", f"위험 영역 {z.zid} ({z.reason}) " + ("안" if d == 0 else f"{d:.1f} m")))
        for name, conf, xyxy, tid, tool in obs:
            if not HAZARD.get(name) or conf < 0.5:
                continue
            xy, _ = self._locate(cam, xyxy, name)
            if xy is not None and np.linalg.norm(xy - wxy) < TOUCH_WARN_M * 0.8:
                d = float(np.linalg.norm(xy - wxy))
                lab = CLASS_KO[name] + (f" ({TOOL_KO[tool]})" if tool else "")
                cands.append((d, f"D:{name}:{int(xy[0] * 2)}:{int(xy[1] * 2)}", f"화면의 {lab} {d:.1f} m"))
        cands = [c for c in cands if t - self._warned.get(c[1], -1e9) >= self.WARN_REPEAT_S]
        if not cands:
            return
        for _, key, _ in cands:
            self._warned[key] = t
        if t - self._last_voice < self.VOICE_GAP_S:
            return
        d, key, what = min(cands)
        self._last_voice = t
        self.voice_events.append({"t": float(t), "target": key, "dist": round(d, 2), "what": what})
        self.stats["voice"] += 1
        self.say(t, "음성 경고", f"\"경고 경고 위험 요소가 식별되었습니다\" ← 작업자 ↔ {what}")
        if self.on_voice:
            self.on_voice(t)

    # ------------------------------------------------------------ 순찰 끝
    def finish_patrol(self, t):
        todo = [cp for cp in self.checkpoints if cp.status == "미확인"]
        self.say(t, "계획", f"순찰 끝. 점검표 {len(self.checkpoints) - len(todo)}/{len(self.checkpoints)} 확인, "
                            f"남은 {len(todo)}곳과 대기 중인 재확인 {len(self.jobs)}건을 CCTV 확대로 처리")
        for j in self.jobs:
            j["t_req"] = -1e9
        for cp in todo:
            self._request(t, None, cp, "순찰 중 확인 못 한 점검 지점", "점검표")
        for j in self.jobs:
            j["t_req"] = -1e9

    def busy(self):
        return self.job is not None or bool(self.jobs)

    # ------------------------------------------------------------ 6. 조치 지시서
    def report(self):
        items = []
        for f in self.findings:
            c = f.cls
            if f.status in INACTIVE + ("재확인 대기",) or c is None:
                continue
            if f.status == "현장 확인 필요":
                action, sev = CHECK_ACTION
            elif HAZARD[c]:
                action, sev = ACTIONS[c]
            else:
                action, sev = SAFE_NOTES[c], 0
            score = sev + 2 * f.near_miss if sev else 0
            level = "긴급" if score >= 5 else "높음" if score >= 3 else "보통" if score >= 1 else "-"
            state = "확인 필요" if f.status == "현장 확인 필요" else ("위험" if HAZARD[c] else "안전")
            tools = [TOOL_KO[x] for x in f.tool_names] if KIND[c] == "tool" else []
            items.append({"id": f.fid, "class": c, "label": CLASS_KO[c] + (f" ({', '.join(tools)})" if tools else ""),
                          "tools": tools, "tool_types": f.tool_names if KIND[c] == "tool" else [],
                          "state": state, "hazard": bool(HAZARD[c]),
                          "status": f.status, "zone": W.zone_name(*f.xy), "x": round(float(f.xy[0]), 2),
                          "y": round(float(f.xy[1]), 2), "confidence": round(f.share, 2), "n_body": f.n_body,
                          "n_ptz": f.n_ptz, "near_miss": f.near_miss, "source": f.source, "rechecked": f.rechecked,
                          "changed_by_recheck": f.changed_by_recheck, "body_class": f.body_cls,
                          "checkpoint": f.checkpoint, "action": action, "priority": level, "score": score})
        # 카메라로 못 본 점검 지점도 현장 확인 항목으로 (소화기가 없어졌거나 가려졌을 수 있음)
        for cp in self.checkpoints:
            if cp.status == "현장 확인 필요" and cp.finding is None:
                what = "소화기" if cp.group == "ext" else "작업대"
                items.append({"id": cp.cid, "class": None, "label": f"{cp.name} 상태 미확인", "state": "확인 필요", "hazard": False,
                              "status": "현장 확인 필요", "zone": W.zone_name(*cp.xy), "x": round(float(cp.xy[0]), 2),
                              "y": round(float(cp.xy[1]), 2), "confidence": 0.0, "n_body": 0, "n_ptz": 0, "near_miss": 0,
                              "source": "점검표", "rechecked": True, "changed_by_recheck": False, "body_class": None,
                              "checkpoint": cp.cid, "action": f"현장에서 {what} 상태 직접 확인 (CCTV 로 볼 수 없음)",
                              "priority": "보통", "score": CHECK_ACTION[1]})
        items.sort(key=lambda r: (-r["score"], not r["hazard"], r["id"]))
        for k, r in enumerate(items):
            r["rank"] = k + 1
        cps = [{"id": c.cid, "name": c.name, "status": c.status, "finding": c.finding,
                "x": float(c.xy[0]), "y": float(c.xy[1])} for c in self.checkpoints]
        n_h = sum(1 for r in items if r["state"] == "위험")
        summary = {"hazards": n_h, "safe": sum(1 for r in items if r["state"] == "안전"),
                   "need_check": sum(1 for r in items if r["state"] == "확인 필요"),
                   "urgent": sum(1 for r in items if r["priority"] == "긴급"),
                   "checkpoints_done": sum(1 for c in self.checkpoints if c.status in ("바디캠 확인", "CCTV 확대 확인")),
                   "checkpoints": len(self.checkpoints)}
        summary["zones"] = len(self.zones)
        summary["voice"] = len(self.voice_events)
        zones = [{"id": z.zid, "source": z.source, "reason": z.reason, "zone": W.zone_name(*z.center),
                  "x": round(float(z.center[0]), 2), "y": round(float(z.center[1]), 2), "radius": round(z.radius, 2),
                  "poly": [[round(float(a), 2), round(float(b), 2)] for a, b in z.poly], "members": z.members,
                  "n_cones": z.n_cones, "sign": z.has_sign} for z in sorted(self.zones.values(), key=lambda z: z.zid)]
        return {"summary": summary, "findings": items, "checkpoints": cps, "zones": zones, "voice": self.voice_events,
                "stats": dict(self.stats), "timeline": self.timeline}

    # ------------------------------------------------------------ 채점 (정답표, 판정에는 안 씀)
    def evaluate(self, answer_key, match_m=2.0):
        """창고 전체 물체와 비교. before: 바디캠이 확정한 것만 바디캠 판정으로, after: 에이전트 최종 대장."""
        objs = answer_key["objects"]

        def match(entries):
            pairs = sorted((math.hypot(o["x"] - e["x"], o["y"] - e["y"]), i, j)
                           for i, e in enumerate(entries) for j, o in enumerate(objs)
                           if KIND[e["class"]] == KIND[o["class"]])
            ue, uo, out = set(), set(), {}
            for d, i, j in pairs:
                if d < match_m and i not in ue and j not in uo:
                    ue.add(i)
                    uo.add(j)
                    out[j] = i
            return out, [i for i in range(len(entries)) if i not in ue]

        def score(entries, checks):
            m, unmatched = match(entries)
            res = {"hazard_total": 0, "hazard_found": 0, "hazard_as_safe": 0, "hazard_missed": 0,
                   "safe_total": 0, "safe_ok": 0, "safe_as_hazard": 0, "safe_missed": 0, "exact": 0,
                   "false_reports": sum(1 for i in unmatched if HAZARD[entries[i]["class"]]),
                   "need_check": len(checks), "need_check_real": len(match(checks)[0])}
            per = {}
            for j, o in enumerate(objs):
                e = entries[m[j]] if j in m else None
                key = "hazard" if o["hazard"] else "safe"
                res[key + "_total"] += 1
                if e is None:
                    r = "놓침"
                    res[key + "_missed"] += 1
                elif HAZARD[e["class"]] == o["hazard"]:
                    r = "정확" if e["class"] == o["class"] else "위험 여부는 맞음"
                    res["hazard_found" if o["hazard"] else "safe_ok"] += 1
                    res["exact"] += e["class"] == o["class"]
                else:
                    r = "위험을 안전으로 오판" if o["hazard"] else "안전을 위험으로 오판"
                    res["hazard_as_safe" if o["hazard"] else "safe_as_hazard"] += 1
                per[o["id"]] = (e["class"] if e else None, e["id"] if e else None, r)
            return res, per

        def entry(f, cls):
            return {"id": f.fid, "class": cls, "x": float(f.xy[0]), "y": float(f.xy[1])}

        before = [entry(f, f.body_cls) for f in self.findings if f.body_confirmed and f.body_cls]
        after = [entry(f, f.cls) for f in self.findings if f.status in ACTIVE and f.cls]
        checks = [entry(f, f.cls) for f in self.findings if f.status == "현장 확인 필요" and f.cls]
        sb, pb = score(before, [])
        sa, pa = score(after, checks)
        rows = [{"id": o["id"], "class": o["class"], "hazard": o["hazard"], "zone": o["zone"],
                 "before": pb[o["id"]][0], "before_result": pb[o["id"]][2],
                 "after": pa[o["id"]][0], "after_result": pa[o["id"]][2], "finding": pa[o["id"]][1]} for o in objs]
        rc = [f for f in self.findings if f.changed_by_recheck]
        fix_ok = 0
        for f in rc:
            o = next((objs[j] for j, i in match([entry(f, f.cls)])[0].items()), None)
            fix_ok += bool(o and HAZARD[o["class"]] == HAZARD[f.cls])
        return {"before": sb, "after": sa, "rows": rows,
                "recheck": {**{k: self.stats.get(k, 0) for k in ("recheck_requested", "recheck_run", "recheck_found",
                                                                  "recheck_changed", "recheck_retry", "recheck_escalated",
                                                                  "recheck_rejected", "recheck_canceled")},
                            "changed_correct": fix_ok},
                **self._evaluate_extra(answer_key, pa)}

    def _evaluate_extra(self, answer_key, pa):
        """공구 이름, 위험 영역, 음성 경고 채점."""
        from .inspection import _runs
        objs = answer_key["objects"]
        # 공구 이름: 정답 공구 묶음과 짝지어진 대장 항목이 알아본 공구 종류
        t_total = t_hit = 0
        for o in objs:
            if not o.get("tools"):
                continue
            gt = set(o["tools"])
            f = next((x for x in self.findings if x.fid == pa[o["id"]][1]), None)
            t_total += len(gt)
            t_hit += len(gt & set(f.tool_names)) if f is not None else 0
        # 위험 영역: 정답 영역 (라바콘 링, DANGER 표지) 을 알아봤는지, 엉뚱한 곳에 영역을 만들었는지
        gz = answer_key.get("zones", [])
        zl = list(self.zones.values())
        marked = [o for o in objs if o["class"] == "spill_marked"]

        def near(z, x, y, r):
            return float(np.linalg.norm(z.center - np.array([x, y]))) < r
        zone_found = sum(1 for g in gz if any(z.source in ("cone", "sign") and near(z, g["x"], g["y"], g["radius"] + 1.0) for z in zl))
        marked_zone = sum(1 for z in zl if z.source in ("cone", "sign") and not any(near(z, g["x"], g["y"], g["radius"] + 1.0) for g in gz)
                          and any(near(z, m["x"], m["y"], 1.8) for m in marked))
        false_zone = sum(1 for z in zl if z.source in ("cone", "sign") and not any(near(z, g["x"], g["y"], g["radius"] + 1.0) for g in gz)
                         and not any(near(z, m["x"], m["y"], 1.8) for m in marked))
        agent_z = [z for z in zl if z.source == "agent"]
        agent_ok = sum(1 for z in agent_z if any(o["class"] in Z.AGENT_RADIUS and Z.distance((o["x"], o["y"]), z.poly) < 1.0 for o in objs))
        # 음성 경고: 실제 작업자 위치가 정답 위험물 1 m 안, 또는 정답 위험 영역 경계 0.5 m 안이었던 사건
        hz = [(o["x"], o["y"]) for o in objs if o["hazard"]]
        # 정답 위험 영역: 라바콘 링·표지 + 라바콘을 둘러친 조치된 유출 (라바콘 바깥까지)
        zpolys = [Z.circle((g["x"], g["y"]), g["radius"] + 0.3) for g in gz] +                  [Z.circle((m["x"], m["y"]), m.get("radius", 0.6) * 1.3 + 0.5) for m in marked]
        near_t = [t for t, x, y in self.worker_log
                  if any(math.hypot(x - hx, y - hy) < TOUCH_WARN_M for hx, hy in hz) or any(Z.distance((x, y), p) < self.ZONE_WARN_M for p in zpolys)]
        groups = _runs(near_t, 1.0)
        vts = [v["t"] for v in self.voice_events]
        hit = sum(1 for g in groups if any(g[0] - 3.0 <= v <= g[-1] + 0.5 for v in vts))
        useful = sum(1 for v in vts if any(g[0] - 3.0 <= v <= g[-1] + 0.5 for g in groups))
        return {"tools": {"total": t_total, "named": t_hit},
                "zones": {"gt": len(gz), "found": zone_found, "marked_spill": marked_zone, "false": false_zone,
                          "agent": len(agent_z), "agent_real": agent_ok},
                "voice": {"events": len(groups), "warned": hit, "voices": len(vts), "useful": useful}}


def evaluation_summary(ev):
    b, a, r = ev["before"], ev["after"], ev["recheck"]
    return "\n".join([
        f"에이전트 채점 (창고 전체 물체 {b['hazard_total'] + b['safe_total']}개, 정답표와 비교)",
        f"                     바디캠만   에이전트 (재확인, 점검표 포함)",
        f"  위험을 위험으로      {b['hazard_found']:>2}/{b['hazard_total']:<4}   {a['hazard_found']:>2}/{a['hazard_total']}",
        f"  안전을 안전으로      {b['safe_ok']:>2}/{b['safe_total']:<4}   {a['safe_ok']:>2}/{a['safe_total']}",
        f"  위험을 안전으로 오판  {b['hazard_as_safe']:>2}        {a['hazard_as_safe']:>2}",
        f"  안전을 위험으로 오판  {b['safe_as_hazard']:>2}        {a['safe_as_hazard']:>2}",
        f"  없는 위험 보고        {b['false_reports']:>2}        {a['false_reports']:>2}",
        f"  현장 확인 요청        -         {a['need_check']} (실제 물체 {a['need_check_real']})",
        f"  재확인 {r['recheck_run']}건 실행 (요청 {r['recheck_requested']}, 바디캠 확정으로 취소 {r['recheck_canceled']}): "
        f"찾음 {r['recheck_found']}, 판정 고침 {r['recheck_changed']} (맞게 고침 {r['changed_correct']}), "
        f"다른 CCTV 로 재시도 {r['recheck_retry']}, 현장 확인 {r['recheck_escalated']}, 오검출로 뺌 {r['recheck_rejected']}",
    ])


def action_summary(rep, top=8):
    s = rep["summary"]
    lines = [f"조치 지시서: 위험 {s['hazards']}건 (긴급 {s['urgent']}), 현장 확인 {s['need_check']}건, 안전 확인 {s['safe']}건, "
             f"점검표 {s['checkpoints_done']}/{s['checkpoints']}"]
    for r in rep["findings"][:top]:
        if r["state"] == "안전":
            continue
        lines.append(f"  {r['rank']:>2}. [{r['priority']}] {r['state']} {r['label']}  |  {r['zone']} ({r['x']:+.1f}, {r['y']:+.1f})"
                     f"  |  {r['action']}" + (f"  |  접근 경고 {r['near_miss']}회" if r["near_miss"] else ""))
    return "\n".join(lines)
