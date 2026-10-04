"""안전 순찰 에이전트: 점검 계획 -> 바디캠 판정 -> 위험물 대장 -> 재관측 재판단 -> 조치 지시서.

판정에는 정답표를 쓰지 않는다. 에이전트가 아는 것은 도면(경로, 랙, 소화기 자리, 작업대)과 바디캠 화면뿐이다 (CCTV 없음).

  1. 계획      : 도면에서 꼭 확인할 지점(소화기 6곳, 작업대 2곳)으로 점검표를 만든다.
  2. 판정      : 바디캠 YOLO 추적으로 물체마다 판정을 모아 위험물 대장에 올린다 (위치, 상태, 확신도, 본 횟수).
  3. 애매한 판정: 잠깐 보고 지나치거나 판정이 엇갈린 물체는, 후보 중 고위험 클래스가 있으면 즉시 위험으로 확정하고
                 (미탐을 줄이는 쪽), 없으면 "주의" 로 분류해 안전/위험 어느 쪽으로도 단정하지 않는다.
  4. 재관측 재판단: "주의" 물체를 바디캠이 자연스럽게 다시 지나치면, 그때까지 쌓인 전체 증거로 로컬 LLM
                 (recheck_agent.py) 이 위험/안전 중 하나로 재판단한다 (LLM 없으면 규칙: 고위험 후보 재검토).
                 끝까지 다시 안 보이면 "주의" 로 남아 조치 지시서에 "현장 확인 권고" 로 표시된다.
  5. 위험 영역  : 라바콘으로 둘러친 곳, DANGER 표지가 선 곳, 스스로 판단한 위험 주변(유출, 무너질 듯한 적재)을
                 영역으로 설정. 회피가 필요 없는 위험(소화기류)은 영역 근거에서 뺀다.
  6. 경고      : 회피형 위험(유출, 적재, 통로 공구)에 1.5 m 안으로 다가가거나 위험 영역 0.5 m 안에 들어서면 경고
                 (위험은 "멈추세요" + 강한 진동, 주의는 "발밑을 확인하세요" + 짧은 진동). 2 m 안은 경고 없이 근접만 기록.
                 오늘 작업 계획(TBM)의 대상 물체는 경고에서 뺀다. 소화기류는 거리 경고 대상이 아니라 조치 목록에만 오른다.
  7. 보고      : 위험물마다 조치 방법(매뉴얼로 보강 가능), 우선순위, 위치(공구는 이름까지)를 담은 조치 지시서.

채점(evaluate)에서만 정답표를 쓴다: 바디캠 프레임 원시 판정(before)과 에이전트 최종 대장(after)을 창고 전체 물체와 비교.
"""
import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from . import warehouse as W
from . import zones as Z
from .config import CLASS_KO, EQUIPMENT, HAZARD, KIND, PROXIMITY_WARN_M, TOOL_KO, TOOL_TYPES, TOUCH_WARN_M, \
    VOICE_TEXT_CAUTION, VOICE_TEXT_HAZARD
from .geometry import Projector
from .inspection import BodycamInspector, floor_point
from .report import clock

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
CAUTION_ACTION = ("현장 확인 권고: 안전·위험을 아직 가르지 못함. 다시 지나칠 때 재판단", 1)

# 애매한 판정(잠깐 보임/판정 엇갈림) 처리용 심각도: 후보 중 하나라도 "high" 면 즉시 위험 확정, 전부 "low" 면 "주의" 로 분류
HAZARD_SEVERITY = {
    "spill": "high",          # 미끄럼 + 화재 가능성
    "stack_unstable": "high",  # 붕괴 위험
    "ext_fallen": "high",     # 화재 대응 불가 (치명적 결과)
    "ext_blocked": "low",     # 화재 시에만 문제, 당장은 급하지 않음
    "tool_floor": "low",      # 걸려 넘어짐 정도
}
# 거리 경고(음성 1.5 m, 근접 기록 2 m) 대상: 회피가 필요한 위험만. 소화기류는 설비 결함이라 조치 목록에만 올림
AVOIDANCE_HAZARDS = {"spill", "stack_unstable", "tool_floor"}
# 오늘 작업 계획(TBM)의 work 키 -> 거리 경고 제외 대상 (종류, 구역). 구역 매핑이 없는 키는 종류 전체를 제외
TBM_TASK_KIND = {"work_restack": "stack", "work_inspect_ext": "ext", "work_clean": "spill"}
TBM_TASK_ZONE = {"work_restack": "동쪽 통로"}

ACTIVE = ("확정",)
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
    radius: float
    status: str = "미확인"   # 미확인 / 바디캠 확인 / 주의 / 현장 확인 필요
    finding: str = None


def build_checkpoints():
    cps = []
    for i, (x, y, (nx, ny)) in enumerate(W.EXT_MOUNTS):
        side = "서쪽" if x < -4 else ("동쪽" if x > 4 else "가운데")
        end = "북쪽" if ny > 0 else "남쪽"
        cps.append(Checkpoint(f"C{i + 1}", "ext", f"소화기 {i + 1} ({side} 랙 {end} 끝)", np.array([x, y], float), 1.6))
    for j, (x, y, _) in enumerate(W.TABLES):
        cps.append(Checkpoint(f"C{len(W.EXT_MOUNTS) + j + 1}", "tool_stored", f"작업대 {j + 1} ({'서쪽' if x < 0 else '동쪽'})",
                              np.array([x, y], float), 2.0))
    return cps


@dataclass
class Finding:
    """위험물 대장 한 줄. 판정은 바디캠 YOLO 신뢰도 합으로 투표."""
    fid: str
    group: str
    xy: np.ndarray
    t0: float
    source: str
    body: dict = field(default_factory=lambda: defaultdict(float))
    n_body: int = 0
    body_confirmed: bool = False
    forced_cls: str = None   # 애매한 후보 중 고위험이 있어 즉시 위험 확정하거나 재판단으로 바뀌면, 투표와 무관하게 이 클래스로 고정
    status: str = "확정"     # 확정 / 주의 / 현장 확인 필요 / 기각 / 병합
    merged_into: str = None
    near_miss: int = 0
    rechecked: bool = False
    changed_by_recheck: bool = False
    checkpoint: str = None
    n_xy: float = 1.0
    types: dict = field(default_factory=lambda: defaultdict(float))   # 공구 종류별 YOLO 신뢰도 합

    @property
    def cls(self):
        if self.forced_cls:
            return self.forced_cls
        return max(self.body, key=self.body.get) if self.body else None

    @property
    def body_cls(self):
        return max(self.body, key=self.body.get) if self.body else None

    @property
    def share(self):
        tot = sum(self.body.values())
        return max(self.body.values()) / tot if tot > 0 else 0.0

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
    FAR_K = 0.2              # 멀리서 본 것일수록 위치가 부정확해서, 같은 물체로 보는 거리를 거리의 20% 까지 넓힘
    BIG_EXTRA_M = 1.0        # 적재물(팔레트 1.2 m)은 크고 바닥 투영 오차도 커서 같은 물체로 보는 거리를 1 m 더 넓힘
    EXT_REGROUP_M = 6.0      # 다른 종류로 잡혔다가 소화기로 바뀐 항목을 가장 가까운 소화기 자리에 붙이는 거리
    ZONE_WARN_M = 0.5        # 위험 영역 경계에서 이 거리 안이면 경고
    VOICE_GAP_S = 5.5        # 경고끼리 최소 간격 (음성 길이)
    WARN_REPEAT_S = 12.0     # 같은 위험물·영역은 이 간격으로만 다시 경고
    NEAR_MISS_REPEAT_S = 5.0  # 2 m 근접 기록 재알림 간격
    MARKER_MIN = 2           # 라바콘·표지는 이만큼 여러 번 보여야 영역 계산에 씀

    def __init__(self, img_w, img_h, log=print):
        self.w, self.h = img_w, img_h
        self.cctvs = {}          # 호환용 빈 dict (assistant.py/dashboard.py 의 "CCTV 있음?" 체크, 5단계에서 정리)
        self.checkpoints = build_checkpoints()
        self.findings = []
        self.tracks = {}
        self.timeline = []
        self.log = log
        self.stats = defaultdict(int)
        self.scorer = BodycamInspector(img_w, img_h)   # 채점용 (판정에는 안 씀)
        self.markers = []        # 라바콘, DANGER 표지: {"mid", "kind", "xy", "w", "n"}
        self.zones = {}          # key -> Zone
        self.voice_events = []   # 경고 기록
        self.on_voice = None     # 경고 때 부를 함수(t, level). level: "hazard" | "caution"
        self._last_voice = -1e9
        self._warned = {}
        self._prox_warned = {}   # 2 m 근접 기록 재알림 타이머
        self.worker_log = []     # 채점용 실제 작업자 위치 (판정에는 안 씀)
        self.equip_tids = set()  # 운반 카트처럼 장비로 본 추적 번호 (그 번호로는 위험/안전 판정을 안 함)
        self.voice_max = None    # 경고 최대 횟수 (None 이면 제한 없음, 시연은 1)
        self.recheck_judge = None  # 스냅샷 -> LLM 재판단 (assistant_client.AssistantClient.recheck). None 이면 규칙
        self.report_judge = None  # 스냅샷 -> LLM 판단 (assistant_client.AssistantClient.report_action). None 이면 고정 문구만
        self.task_context = []    # [(kind, zone_or_None)] 거리 경고 제외 대상 (set_task_context 로 채움)
        self._t = 0.0

    def set_task_context(self, tbm):
        """오늘 TBM 작업 계획에서 지금 작업 대상인 종류(위치 정보 있으면 구역까지)를 거리 경고 제외 목록에 둔다."""
        self.task_context = [(TBM_TASK_KIND[key], TBM_TASK_ZONE.get(key)) for key in tbm.get("work", []) if key in TBM_TASK_KIND]

    def _in_task_context(self, f):
        kind = KIND.get(f.cls)
        return any(kind == k and (zone is None or W.zone_name(*f.xy) == zone) for k, zone in self.task_context)

    # ------------------------------------------------------------ 기록
    def say(self, t, kind, text):
        self.timeline.append({"t": float(t), "kind": kind, "text": text})
        if self.log:
            self.log(f"[{clock(t)}] <{kind}> {text}")

    # ------------------------------------------------------------ 1. 계획
    def plan(self, path_len):
        self.say(0.0, "계획", f"목표: 순찰 한 바퀴({path_len:.0f} m) 동안 위험물을 찾아 위험/안전을 판정하고, "
                              f"작업자 접근을 경고하고, 조치 지시서를 만든다")
        self.say(0.0, "계획", f"통로 바닥(유출, 공구, 적재)과 점검표 {len(self.checkpoints)}곳 모두 바디캠으로 확인한다. "
                              f"애매하면 고위험 후보가 있는 쪽은 즉시 위험으로, 없으면 주의로 분류해 다시 지나칠 때 재판단한다")

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
        if f is not None and f.status in ACTIVE and f.n_body <= tr["n"] + 1:
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
                if f.forced_cls is None and f.cls != before:
                    self.say(t, "판정 수정", f"{f.fid} 가까이서 다시 보니 {CLASS_KO[before]} → {verdict(f.cls)}")
                    tr["finding"] = self._regroup(t, f)
            elif tr["n"] >= self.CONFIRM and tr["xy"]:
                self._confirm_track(t, tr)
        for tid in [k for k, tr in self.tracks.items() if k not in seen and t - tr["last"] >= self.TRACK_LOST_S]:
            self._end_track(t, self.tracks.pop(tid))
        self._tick_checkpoints(t)
        self._update_zones(t)
        wxy = np.asarray(worker_xy if worker_xy is not None else cam.pos[:2], float)
        self._check_proximity(t, wxy)
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
        was_caution = f.status == "주의"
        prior_body = dict(f.body) if was_caution else None
        self._add_track_votes(f, tr)
        f.body_confirmed = True
        if was_caution:
            f = self._rejudge_caution(t, f, prior_body, dict(tr["votes"]))
        elif f.status == "현장 확인 필요":
            f.status = "확정"
            self.say(t, "판정 수정", f"{f.fid} 현장에서 확인하려던 물체를 바디캠이 다시 보고 확정")
        if new:
            conf = tr["votes"][best] / tr["n"]
            self.say(t, "판정", f"바디캠 {self._label(f)}  |  {W.zone_name(*f.xy)}  |  YOLO {conf * 100:.0f}%  → 대장 {f.fid}")
        elif not was_caution and before and f.cls != before:
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
            was_caution = not new and f.status == "주의"
            prior_body = dict(f.body) if was_caution else None
            self._add_track_votes(f, tr)
            if was_caution:
                self._rejudge_caution(t, f, prior_body, dict(tr["votes"]))
            elif new:
                self._settle_ambiguous(t, f, f"바디캠에 {tr['n']}프레임만 보이고 지나감")
            else:
                self._regroup(t, f)
        elif f.status == "확정" and f.share < self.LOW_SHARE:
            self._settle_ambiguous(t, f, "판정이 엇갈림")

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

    def _attach(self, t, group, xy, source, rng=0.0, w=1.0):
        xy = self._place(group, xy)
        base = self.SAME_PLACE_M + (self.BIG_EXTRA_M if group == "stack" else 0.0)
        best = self._find_near(group, xy, max(base, self.FAR_K * rng))
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
        for c, s in f.body.items():
            other.body[c] += s
        for c, s in f.types.items():
            other.types[c] += s
        other.n_body += f.n_body
        other.near_miss += f.near_miss
        other.rechecked |= f.rechecked
        other.changed_by_recheck |= f.changed_by_recheck
        if f.status == "확정" and other.status == "주의":
            other.status = "확정"
            if other.forced_cls is None and f.forced_cls:
                other.forced_cls = f.forced_cls
        f.status, f.merged_into = "병합", other.fid
        for tr in self.tracks.values():
            if tr["finding"] is f:
                tr["finding"] = other
        self.say(t, "판정 수정", f"{f.fid} 는 {other.fid} 와 같은 물체 ({verdict(other.cls)}) → 대장에서 합침")
        return other

    def _tick_checkpoints(self, t):
        for cp in self.checkpoints:
            if cp.status not in ("미확인", "주의"):
                continue
            for f in self.findings:
                if f.status not in ("확정", "주의") or f.cls is None:
                    continue
                ok = KIND[f.cls] == "ext" if cp.group == "ext" else f.group == "tool_stored"
                if ok and np.linalg.norm(f.xy - cp.xy) < cp.radius:
                    new_status = "바디캠 확인" if f.status == "확정" else "주의"
                    if new_status != cp.status:
                        cp.status, cp.finding, f.checkpoint = new_status, f.fid, cp.cid
                        if new_status == "바디캠 확인":
                            self.say(t, "점검표", f"{cp.cid} {cp.name} 확인: {verdict(f.cls)} (대장 {f.fid})")
                    break

    # ------------------------------------------------------------ 3~4. 애매한 판정과 재관측 재판단
    def _settle_ambiguous(self, t, f, why):
        """잠깐 보임/판정 엇갈림: 후보(f.body) 중 고위험 클래스가 있으면 즉시 위험 확정 (미탐을 줄이는 쪽),
        없으면 '주의' 로 분류해 안전/위험 어느 쪽으로도 단정하지 않는다. 재관측되면 _rejudge_caution 이 재판단.
        강제한 클래스가 원래 그룹과 다른 종류면 _regroup 으로 올바른 그룹에 합친다. 최종 Finding 을 돌려준다."""
        high = {c: v for c, v in f.body.items() if HAZARD_SEVERITY.get(c) == "high"}
        if high:
            cls = max(high, key=high.get)
            if f.forced_cls != cls:     # Finding.status 기본값이 이미 "확정"이라 status 로는 "처음 settle" 을 못 가려냄
                f.forced_cls = cls
                self.stats["ambiguous_hazard"] += 1
                self.say(t, "판정", f"{f.fid} {why} → 고위험 후보 포함, 즉시 {verdict(f.cls)} 로 확정")
            f.status = "확정"
            f = self._regroup(t, f)
        elif f.status != "주의":
            f.forced_cls = None
            f.status = "주의"
            self.stats["ambiguous_caution"] += 1
            self.say(t, "주의", f"{f.fid} {why} → 안전·위험을 가르지 못해 주의로 분류 ({CLASS_KO[f.cls]}), 다시 보이면 재판단")
        return f

    def _rejudge_caution(self, t, f, prior_body, new_votes):
        """'주의' 물체를 바디캠이 자연스럽게 다시 지나침: 이전 관측 + 이번 관측을 합쳐 재판단.
        LLM(recheck_agent) 성공 시 위험/안전 중 하나로 확정. LLM 없음/실패 시 규칙 폴백:
        고위험 후보 있으면 위험 확정, 없으면 다음에 다시 보일 때까지 '주의' 유지. 최종 Finding 을 돌려준다."""
        self.stats["caution_rejudged"] += 1
        if self.recheck_judge is not None:
            snap = {"target_kind": kind_of(f.group), "zone": W.zone_name(*f.xy),
                    "prior_observations": {c: round(s, 2) for c, s in prior_body.items()},
                    "new_observations": {c: round(s, 2) for c, s in new_votes.items()},
                    "elapsed_s": round(t - f.t0, 1)}
            try:
                dec = self.recheck_judge(snap)
            except Exception as e:
                dec = {"llm": False, "error": f"{type(e).__name__}: {e}"}
            if dec.get("llm") and dec.get("verdict") in f.body:
                f.forced_cls = dec["verdict"] if dec["verdict"] != f.cls else None
                f.status = "확정"
                f.rechecked = True
                f.changed_by_recheck = True
                self.stats["caution_rejudged_confirmed"] += 1
                self.say(t, "LLM 판단", f"{f.fid} {dec['model']}: {' → '.join(dec.get('trace', []))} | {dec.get('reason_ko', '')}")
                self.say(t, "판정", f"{f.fid} 재관측 재판단 → {verdict(f.cls)}")
                return self._regroup(t, f)
            self.say(t, "LLM 판단", f"{f.fid} LLM 을 못 써서 규칙으로 정함 ({dec.get('error', '')[:60]})")
        high = {c: v for c, v in f.body.items() if HAZARD_SEVERITY.get(c) == "high"}
        if high:
            cls = max(high, key=high.get)
            f.forced_cls = cls if cls != f.cls else None
            f.status = "확정"
            f.rechecked = True
            self.stats["caution_rejudged_confirmed"] += 1
            self.say(t, "판정", f"{f.fid} 재관측 (규칙) → 고위험 후보 포함, {verdict(f.cls)} 로 확정")
            return self._regroup(t, f)
        self.say(t, "주의", f"{f.fid} 재관측했지만 여전히 못 가름 → 주의 유지")
        return f

    # ------------------------------------------------------------ 5. 위험 영역
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
        # 구역 형성은 회피형 위험(AVOIDANCE_HAZARDS)만 근거로: 소화기류·주의 상태는 지도에만 표시, 구역엔 안 넣음
        hazards = [(f.fid, f.cls, f.xy) for f in self.findings if f.status == "확정" and f.cls in AVOIDANCE_HAZARDS]
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

    # ------------------------------------------------------------ 6. 경고
    def log_worker(self, t, xy):
        """채점용 실제 작업자 위치 (판정에는 안 씀)."""
        self.worker_log.append((float(t), float(xy[0]), float(xy[1])))

    def _check_proximity(self, t, wxy):
        """대장의 회피형 위험물(화면에 안 보여도 포함) 이 PROXIMITY_WARN_M 안이면 근접 기록만 (평가 집계용, 경고 없음).
        오늘 작업 대상인 물체는 제외."""
        for f in self.findings:
            if f.status not in ("확정", "주의") or f.cls not in AVOIDANCE_HAZARDS or self._in_task_context(f):
                continue
            if f.status == "확정" and not HAZARD.get(f.cls):
                continue
            d = float(np.linalg.norm(f.xy - wxy))
            if d < PROXIMITY_WARN_M and t - self._prox_warned.get(f.fid, -1e9) >= self.NEAR_MISS_REPEAT_S:
                self._prox_warned[f.fid] = t
                f.near_miss += 1
                self.say(t, "접근 기록", f"작업자 ↔ {self._label(f)} {d:.1f} m (2 m 안, 경고 없음) → 대장 {f.fid} 우선순위 올림")

    def _check_warnings(self, t, wxy, cam, obs):
        """작업자가 회피형 위험(AVOIDANCE_HAZARDS) 1.5 m 안, 위험 영역 0.5 m 안이면 경고.
        위험(확정)은 강한 경고, 주의는 약한 안내로 톤을 구분한다. 오늘 작업 대상인 물체는 뺀다."""
        cands = []
        for f in self.findings:
            if f.status not in ("확정", "주의") or f.cls not in AVOIDANCE_HAZARDS or self._in_task_context(f):
                continue
            if f.status == "확정" and not HAZARD.get(f.cls):
                continue
            d = float(np.linalg.norm(f.xy - wxy))
            if d < TOUCH_WARN_M:
                level = "hazard" if f.status == "확정" else "caution"
                cands.append((d, f"F:{f.fid}", level, f"{f.fid} {self._label(f)} {d:.1f} m"))
        for z in self.zones.values():
            d = Z.distance(wxy, z.poly)
            if d < self.ZONE_WARN_M:
                cands.append((d, f"Z:{z.zid}", "hazard", f"위험 영역 {z.zid} ({z.reason}) " + ("안" if d == 0 else f"{d:.1f} m")))
        for name, conf, xyxy, tid, tool in obs:
            if name not in AVOIDANCE_HAZARDS or not HAZARD.get(name) or conf < 0.5:
                continue
            xy, _ = self._locate(cam, xyxy, name)
            if xy is not None and np.linalg.norm(xy - wxy) < TOUCH_WARN_M * 0.8:
                d = float(np.linalg.norm(xy - wxy))
                lab = CLASS_KO[name] + (f" ({TOOL_KO[tool]})" if tool else "")
                cands.append((d, f"D:{name}:{int(xy[0] * 2)}:{int(xy[1] * 2)}", "hazard", f"화면의 {lab} {d:.1f} m"))
        cands = [c for c in cands if t - self._warned.get(c[1], -1e9) >= self.WARN_REPEAT_S]
        if not cands or (self.voice_max is not None and len(self.voice_events) >= self.voice_max):
            return
        for _, key, _, _ in cands:
            self._warned[key] = t
        if t - self._last_voice < self.VOICE_GAP_S:
            return
        d, key, level, what = min(cands, key=lambda c: c[0])
        self._last_voice = t
        text = VOICE_TEXT_HAZARD if level == "hazard" else VOICE_TEXT_CAUTION
        vibration = "strong" if level == "hazard" else "short"
        self.voice_events.append({"t": float(t), "target": key, "dist": round(d, 2), "what": what,
                                  "level": level, "vibration": vibration, "text": text})
        self.stats["voice"] += 1
        self.say(t, "음성 경고" if level == "hazard" else "주의 안내", f"\"{text}\" ← 작업자 ↔ {what}")
        if self.on_voice:
            self.on_voice(t, level)

    # ------------------------------------------------------------ 순찰 끝
    def finish_patrol(self, t):
        todo = [cp for cp in self.checkpoints if cp.status == "미확인"]
        caution = [cp for cp in self.checkpoints if cp.status == "주의"]
        for cp in todo:
            cp.status = "현장 확인 필요"
        self.say(t, "계획", f"순찰 끝. 점검표 {len(self.checkpoints) - len(todo) - len(caution)}/{len(self.checkpoints)} 확인, "
                            f"{len(todo)}곳 못 봄(현장 확인 필요), {len(caution)}곳 주의 상태로 남음")

    def _ground_action(self, c, f, standard_action):
        """조치 문구(action)를 매뉴얼 근거로 보강: LLM 이 매뉴얼에 더 구체적인 근거가 있다고 판단하면 그 문구 +
        출처를, 없거나 LLM 이 없으면 고정 문구(standard_action) 그대로 (문구, 출처) 로 돌려준다."""
        if self.report_judge is None:
            return standard_action, None
        try:
            dec = self.report_judge({"class": c, "label_ko": CLASS_KO[c], "standard_action": standard_action,
                                     "zone": W.zone_name(*f.xy)})
        except Exception:
            return standard_action, None
        if dec.get("llm") and dec.get("grounded") and dec.get("action_ko"):
            return dec["action_ko"], dec.get("source") or None
        return standard_action, None

    # ------------------------------------------------------------ 7. 조치 지시서
    def report(self):
        items = []
        for f in self.findings:
            c = f.cls
            if f.status in INACTIVE or c is None:
                continue
            action_source = None
            if f.status == "현장 확인 필요":
                action, sev = CHECK_ACTION
            elif f.status == "주의":
                action, sev = CAUTION_ACTION
            elif HAZARD[c]:
                action, sev = ACTIONS[c]
                action, action_source = self._ground_action(c, f, action)
            else:
                action, sev = SAFE_NOTES[c], 0
            score = sev + 2 * f.near_miss if sev else 0
            level = "긴급" if score >= 5 else "높음" if score >= 3 else "보통" if score >= 1 else "-"
            state = ("확인 필요" if f.status == "현장 확인 필요" else
                     "주의" if f.status == "주의" else ("위험" if HAZARD[c] else "안전"))
            tools = [TOOL_KO[x] for x in f.tool_names] if KIND[c] == "tool" else []
            items.append({"id": f.fid, "class": c, "label": CLASS_KO[c] + (f" ({', '.join(tools)})" if tools else ""),
                          "tools": tools, "tool_types": f.tool_names if KIND[c] == "tool" else [],
                          "state": state, "hazard": bool(HAZARD[c]),
                          "status": f.status, "zone": W.zone_name(*f.xy), "x": round(float(f.xy[0]), 2),
                          "y": round(float(f.xy[1]), 2), "confidence": round(f.share, 2), "n_body": f.n_body,
                          "near_miss": f.near_miss, "source": f.source, "rechecked": f.rechecked,
                          "changed_by_recheck": f.changed_by_recheck, "body_class": f.body_cls,
                          "checkpoint": f.checkpoint, "action": action, "action_source": action_source,
                          "priority": level, "score": score})
        # 카메라로 못 본 점검 지점도 현장 확인 항목으로 (소화기가 없어졌거나 가려졌을 수 있음)
        for cp in self.checkpoints:
            if cp.status == "현장 확인 필요" and cp.finding is None:
                what = "소화기" if cp.group == "ext" else "작업대"
                items.append({"id": cp.cid, "class": None, "label": f"{cp.name} 상태 미확인", "state": "확인 필요", "hazard": False,
                              "status": "현장 확인 필요", "zone": W.zone_name(*cp.xy), "x": round(float(cp.xy[0]), 2),
                              "y": round(float(cp.xy[1]), 2), "confidence": 0.0, "n_body": 0, "near_miss": 0,
                              "source": "점검표", "rechecked": True, "changed_by_recheck": False, "body_class": None,
                              "checkpoint": cp.cid, "action": f"현장에서 {what} 상태 직접 확인 (바디캠으로 확인 못 함)",
                              "priority": "보통", "score": CHECK_ACTION[1]})
        items.sort(key=lambda r: (-r["score"], not r["hazard"], r["id"]))
        for k, r in enumerate(items):
            r["rank"] = k + 1
        cps = [{"id": c.cid, "name": c.name, "status": c.status, "finding": c.finding,
                "x": float(c.xy[0]), "y": float(c.xy[1])} for c in self.checkpoints]
        n_h = sum(1 for r in items if r["state"] == "위험")
        summary = {"hazards": n_h, "safe": sum(1 for r in items if r["state"] == "안전"),
                   "need_check": sum(1 for r in items if r["state"] == "확인 필요"),
                   "caution": sum(1 for r in items if r["state"] == "주의"),
                   "urgent": sum(1 for r in items if r["priority"] == "긴급"),
                   "checkpoints_done": sum(1 for c in self.checkpoints if c.status == "바디캠 확인"),
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

        def score(entries, checks, caution=()):
            m, unmatched = match(entries)
            res = {"hazard_total": 0, "hazard_found": 0, "hazard_as_safe": 0, "hazard_missed": 0,
                   "safe_total": 0, "safe_ok": 0, "safe_as_hazard": 0, "safe_missed": 0, "exact": 0,
                   "false_reports": sum(1 for i in unmatched if HAZARD[entries[i]["class"]]),
                   "need_check": len(checks), "need_check_real": len(match(checks)[0]),
                   "caution": len(caution), "caution_real": len(match(caution)[0])}
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
        caution = [entry(f, f.cls) for f in self.findings if f.status == "주의" and f.cls]
        sb, pb = score(before, [])
        sa, pa = score(after, checks, caution)
        rows = [{"id": o["id"], "class": o["class"], "hazard": o["hazard"], "zone": o["zone"],
                 "before": pb[o["id"]][0], "before_result": pb[o["id"]][2],
                 "after": pa[o["id"]][0], "after_result": pa[o["id"]][2], "finding": pa[o["id"]][1]} for o in objs]
        rc = [f for f in self.findings if f.changed_by_recheck]
        fix_ok = 0
        for f in rc:
            o = next((objs[j] for j, i in match([entry(f, f.cls)])[0].items()), None)
            fix_ok += bool(o and HAZARD[o["class"]] == HAZARD[f.cls])
        return {"before": sb, "after": sa, "rows": rows,
                "rejudge": {"ambiguous_hazard": self.stats.get("ambiguous_hazard", 0),
                            "ambiguous_caution": self.stats.get("ambiguous_caution", 0),
                            "caution_rejudged": self.stats.get("caution_rejudged", 0),
                            "caution_rejudged_confirmed": self.stats.get("caution_rejudged_confirmed", 0),
                            "changed_correct": fix_ok},
                **self._evaluate_extra(answer_key, pa)}

    def _evaluate_extra(self, answer_key, pa):
        """공구 이름, 위험 영역, 경고 채점."""
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
        # 경고: 실제 작업자 위치가 정답 "회피형" 위험물 1.5 m 안, 또는 정답 위험 영역 경계 0.5 m 안이었던 사건
        hz = [(o["x"], o["y"]) for o in objs if o["hazard"] and o["class"] in AVOIDANCE_HAZARDS]
        # 정답 위험 영역: 라바콘 링·표지 + 라바콘을 둘러친 조치된 유출 (라바콘 바깥까지)
        zpolys = ([Z.circle((g["x"], g["y"]), g["radius"] + 0.3) for g in gz]
                 + [Z.circle((m["x"], m["y"]), m.get("radius", 0.6) * 1.3 + 0.5) for m in marked])
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
    b, a, r = ev["before"], ev["after"], ev["rejudge"]
    return "\n".join([
        f"에이전트 채점 (창고 전체 물체 {b['hazard_total'] + b['safe_total']}개, 정답표와 비교)",
        f"                     바디캠만   에이전트 (재판단, 점검표 포함)",
        f"  위험을 위험으로      {b['hazard_found']:>2}/{b['hazard_total']:<4}   {a['hazard_found']:>2}/{a['hazard_total']}",
        f"  안전을 안전으로      {b['safe_ok']:>2}/{b['safe_total']:<4}   {a['safe_ok']:>2}/{a['safe_total']}",
        f"  위험을 안전으로 오판  {b['hazard_as_safe']:>2}        {a['hazard_as_safe']:>2}",
        f"  안전을 위험으로 오판  {b['safe_as_hazard']:>2}        {a['safe_as_hazard']:>2}",
        f"  없는 위험 보고        {b['false_reports']:>2}        {a['false_reports']:>2}",
        f"  현장 확인 요청        -         {a['need_check']} (실제 물체 {a['need_check_real']})",
        f"  주의 (안전·위험 미확정) -        {a['caution']} (실제 물체 {a['caution_real']})",
        f"  애매한 판정 {r['ambiguous_hazard'] + r['ambiguous_caution']}건 (즉시 위험 확정 {r['ambiguous_hazard']}, 주의 분류 {r['ambiguous_caution']}): "
        f"주의 재관측 재판단 {r['caution_rejudged']}건 중 확정 전환 {r['caution_rejudged_confirmed']} (맞게 고침 {r['changed_correct']})",
    ])


def action_summary(rep, top=8):
    s = rep["summary"]
    lines = [f"조치 지시서: 위험 {s['hazards']}건 (긴급 {s['urgent']}), 현장 확인 {s['need_check']}건, 주의 {s['caution']}건, "
             f"안전 확인 {s['safe']}건, 점검표 {s['checkpoints_done']}/{s['checkpoints']}"]
    for r in rep["findings"][:top]:
        if r["state"] == "안전":
            continue
        lines.append(f"  {r['rank']:>2}. [{r['priority']}] {r['state']} {r['label']}  |  {r['zone']} ({r['x']:+.1f}, {r['y']:+.1f})"
                     f"  |  {r['action']}" + (f"  |  접근 경고 {r['near_miss']}회" if r["near_miss"] else ""))
    return "\n".join(lines)
