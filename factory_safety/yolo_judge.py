"""YOLO 화면 검출 결과로 위험도를 판단한다.

SimDetector 와 달리 위험 요소의 정답 위치를 쓰지 않는다. 쓰는 것은 실제 로봇도 아는 정보뿐:
카메라 화면 (YOLO 박스), 자기 위치와 카메라 자세, 공장 지도 (벽, 랙, 기계의 3D 박스).

  1) 클래스 -> 위험도   puddle, unstable_stack, gauge_low 위험 / power_tool 주의 / gauge_normal 정상
  2) 위치 추정          박스 아래쪽 가운데 픽셀에서 광선을 쏴서 처음 닿는 면(바닥, 선반, 작업대, 벽)에 있다고 본다.
                       물체는 그 아래쪽 끝이 무언가에 놓여 있거나 붙어 있으니까. 낮은 설비 너머를 내려다보면
                       광선이 그 위로 지나가서 뒤쪽 면에 닿는다. 웅덩이처럼 납작한 물체는 박스 가운데를 바닥에 투영.
  3) 같은 물체 묶기      같은 종류가 공장 좌표로 MERGE_DIST 안이면 같은 물체 (화면 밖으로 나갔다 다시 보여도 유지)
  4) 확정              CONFIRM_HITS 번 이상 보이면 확정해서 로그에 한 번만 남김

압력계 정상/부족은 한 물체(gauge)로 묶는다. zoom=True 면 줌 카메라 판독(gauge_reader.py)으로 확정하고,
아니면 YOLO 정상/부족 신뢰도 합이 큰 쪽으로 판정한다 (작은 압력계라 부정확).
소화기 몸통(extinguisher)은 판정하지 않고, 압력계를 아직 못 읽은 소화기 쪽으로 로봇이 고개를
돌리게 하는 데만 쓴다 (focus_view).
"""
import math
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np

from . import layout as L
from .config import CLASS_KO, IMG_H, IMG_W
from .geometry import Projector

RISK = {"puddle": "high", "power_tool": "mid", "unstable_stack": "high", "gauge_low": "high", "gauge_normal": "ok"}
DETAIL = {
    "puddle": "바닥에 물기나 기름이 있어 미끄럼 사고 위험",
    "power_tool": "전동공구가 방치되어 걸림, 베임 위험",
    "unstable_stack": "적재물이 기울어져 낙하 위험",
    "gauge_low": "소화기 압력이 부족해 화재 때 쓸 수 없음",
    "gauge_normal": "소화기 압력 정상",
}
# 묶음 단위: 압력계 정상/부족은 같은 물체, 소화기 몸통은 판단에 안 씀 (압력계로 점검)
GROUP = {"puddle": "puddle", "power_tool": "power_tool", "unstable_stack": "unstable_stack",
         "gauge_normal": "gauge", "gauge_low": "gauge", "extinguisher": "ext_body"}
# 같은 물체로 묶는 거리 (m). 같은 종류 자리끼리는 최소 7.8 m 떨어져 있다 (layout.py 슬롯 기준)
MERGE_DIST = {"puddle": 4.0, "power_tool": 3.0, "unstable_stack": 3.5, "gauge": 3.0, "ext_body": 3.0}
# 로봇이 고개를 돌릴 때 바라볼 높이 (m), 압력계를 읽으려면 이 거리 안으로 가야 함
FOCUS_Z = {"puddle": 0.0, "power_tool": 0.3, "unstable_stack": 1.5, "gauge": 1.4, "ext_body": 1.3}
GAUGE_READ_RANGE = 6.5
# 바닥에 납작하게 깔린 물체: 박스 아래쪽 대신 가운데를 바닥에 투영 (아래쪽은 보는 방향 따라 앞뒤 가장자리가 바뀜)
FLAT = {"puddle"}
# 벽, 기둥에 붙은 물체 (위치 추정 때 좌우 부채꼴 광선 사용)
WALL_MOUNTED = {"gauge", "ext_body"}
# 정답 위험 요소 종류 -> 묶음 (시뮬레이션 채점용)
GT_GROUP = {"puddle": "puddle", "tool": "power_tool", "stack": "unstable_stack", "ext": "gauge"}


@dataclass
class Track:
    group: str
    pos: np.ndarray
    hits: int = 0
    conf_max: float = 0.0
    votes: dict = field(default_factory=dict)   # 클래스별 신뢰도 합
    first_t: float = 0.0
    last_t: float = 0.0
    confirmed: bool = False
    verdict: str = None                         # 확정 당시 클래스
    zoom_votes: dict = field(default_factory=dict)   # 압력계 줌 판독 결과별 횟수

    @property
    def cls_name(self):
        if self.zoom_votes:
            return max(self.zoom_votes, key=self.zoom_votes.get)
        return max(self.votes, key=self.votes.get)


class YoloJudge:
    CONFIRM_HITS = 3
    EDGE_PX = 2
    MIN_BOX_PX = 3
    FAN_RAD = 0.05
    MAX_RANGE = 18.0
    # 이보다 멀리 추정되면 위치를 믿지 않는다 (멀면 몇 픽셀 차이로 수 m 가 틀어져 같은 물체가 두 번 잡힘).
    # 로봇이 다가가서 다시 보면 그때 잡힌다.
    LOC_RANGE = 12.0
    # 줌 판독: 이 거리 안의 압력계만 겨누고, 같은 판정이 ZOOM_CONFIRM 번 나오면 확정
    ZOOM_RANGE = 7.0
    ZOOM_CONFIRM = 2
    ZOOM_MAX_READS = 4
    ZOOM_MIN_PX = 4

    def __init__(self, map_mins, map_maxs, img_w=IMG_W, img_h=IMG_H, zoom=False):
        """map_mins/maxs: 공장 지도 3D 박스 (layout.map_boxes_3d).
        zoom=True: 압력계 정상/부족을 YOLO 클래스 대신 줌 카메라 판독(add_zoom_reading)으로 확정한다."""
        self.map_mins, self.map_maxs = np.asarray(map_mins).reshape(-1, 3), np.asarray(map_maxs).reshape(-1, 3)
        self.proj = Projector(img_w, img_h)
        self.zoom = zoom
        self.tracks = []

    def _first_hit(self, origin, dirs):
        """단위 광선들(K,3)이 지도 박스에 처음 닿는 거리. 안 닿으면 MAX_RANGE."""
        if len(self.map_mins) == 0:
            return np.full(len(dirs), self.MAX_RANGE)
        d = np.where(np.abs(dirs) < 1e-12, 1e-12, dirs)
        inv = 1.0 / d
        t1 = (self.map_mins[None] - origin) * inv[:, None, :]
        t2 = (self.map_maxs[None] - origin) * inv[:, None, :]
        tn = np.max(np.minimum(t1, t2), axis=2)
        tf = np.min(np.maximum(t1, t2), axis=2)
        hit = (tf >= tn) & (tn > 0.05)          # 카메라를 감싼 박스는 뺌
        return np.where(hit, tn, self.MAX_RANGE).min(axis=1)

    def locate(self, cam, xyxy, flat=False, wall_mounted=False):
        """화면 박스 -> 공장 좌표 (x, y). 거리를 정할 수 없거나 LOC_RANGE 보다 멀면 None."""
        self.proj.set_pose(cam)
        p = self.proj
        u = (xyxy[0] + xyxy[2]) / 2.0
        h = xyxy[3] - xyxy[1]
        if flat:
            rows = [(xyxy[1] + xyxy[3]) / 2.0]
        else:
            # 아래쪽 끝 + 조금 위 두 줄. 작업대 끝에 걸쳐 튀어나온 공구는 아래쪽 끝 광선이 상판 밑으로 빠지는데,
            # 조금 위 광선은 상판에 닿는다. 셋 중 가장 가까이 닿는 면을 받침면으로 본다.
            rows = [xyxy[3], xyxy[3] - 0.15 * h, xyxy[3] - 0.3 * h]
        origin = np.asarray(cam.pos, dtype=float)
        fan = np.linspace(-self.FAN_RAD, self.FAN_RAD, 5) if wall_mounted else np.zeros(1)
        c, s = np.cos(fan), np.sin(fan)
        best = None
        for v in rows:
            d = p.f + p.r * (u - p.w / 2.0) / p.fx + p.u * (p.h / 2.0 - v) / p.fy
            d = d / np.linalg.norm(d)
            horiz = math.hypot(d[0], d[1])
            if horiz < 1e-6:
                continue
            t = origin[2] / -d[2] if d[2] < -1e-6 else self.MAX_RANGE      # 바닥
            if not flat:
                # 벽, 기둥에 붙은 물체는 표면보다 조금 앞에 있어서 광선 하나는 모서리를 비껴갈 수 있으니
                # 좌우로 조금 벌린 부채꼴에서 가장 가까운 면을 쓴다 (다른 물체는 멀리서 옆 장애물에 걸려서 안 씀).
                dirs = np.stack([c * d[0] - s * d[1], s * d[0] + c * d[1], np.full_like(fan, d[2])], axis=1)
                t = min(t, float(self._first_hit(origin, dirs).min()))
            if best is None or t * horiz < best[0]:
                best = (t * horiz, origin[:2] + t * d[:2])
        if best is None or best[0] >= self.LOC_RANGE:
            return None
        return best[1]

    def _box_ok(self, group, xyxy):
        if xyxy[2] - xyxy[0] < self.MIN_BOX_PX or xyxy[3] - xyxy[1] < self.MIN_BOX_PX:
            return False
        # 화면 좌우 끝에 걸려 잘린 박스는 가운데가 틀어져 방향이 어긋난다 (납작한 물체는 넓어서 예외)
        return group in FLAT or (xyxy[0] > self.EDGE_PX and xyxy[2] < self.proj.w - self.EDGE_PX)

    def _observe(self, t, cam, group, xyxy):
        """박스 하나를 공장 좌표로 옮겨 가까운 Track 에 합친다 (없으면 새로). 위치를 못 정하면 None."""
        if not self._box_ok(group, xyxy):
            return None
        xy = self.locate(cam, xyxy, flat=group in FLAT, wall_mounted=group in WALL_MOUNTED)
        if xy is None:
            return None
        tr = self._nearest(group, xy)
        if tr is None:
            tr = Track(group=group, pos=xy.copy(), first_t=t)
            self.tracks.append(tr)
        else:
            tr.pos = tr.pos + (xy - tr.pos) / (tr.hits + 1)
        tr.hits += 1
        tr.last_t = t
        return tr

    def update(self, t, cam, dets):
        """dets: [(class_name, conf, xyxy), ...]. 새로 확정되거나 재판정된 Track 목록을 반환."""
        events = []
        for name, conf, xyxy in dets:
            group = GROUP.get(name)
            if group is None:
                continue
            tr = self._observe(t, cam, group, xyxy)
            if tr is None:
                continue
            tr.conf_max = max(tr.conf_max, conf)
            tr.votes[name] = tr.votes.get(name, 0.0) + conf
            if group == "ext_body" or (group == "gauge" and self.zoom):
                continue   # 소화기 몸통은 판정 안 함, 줌을 쓰면 압력계는 줌 판독으로만 확정
            if not tr.confirmed and tr.hits >= self.CONFIRM_HITS:
                tr.confirmed, tr.verdict = True, tr.cls_name
                events.append(tr)
            elif tr.confirmed and tr.verdict != tr.cls_name and RISK[tr.cls_name] == "high":
                tr.verdict = tr.cls_name
                events.append(tr)
        return events

    def zoom_target(self, cam, dets):
        """줌으로 읽을 압력계 박스 (아직 확정 못 했거나 판독이 적은 것 중 가장 큰 것). 없으면 None."""
        best, best_size = None, self.ZOOM_MIN_PX - 1e-6
        for name, conf, xyxy in dets:
            if GROUP.get(name) != "gauge" or not self._box_ok("gauge", xyxy):
                continue
            size = max(xyxy[2] - xyxy[0], xyxy[3] - xyxy[1])
            if size <= best_size:
                continue
            xy = self.locate(cam, xyxy, wall_mounted=True)
            if xy is None or np.linalg.norm(xy - np.asarray(cam.pos[:2])) > self.ZOOM_RANGE:
                continue
            tr = self._nearest("gauge", xy)
            if tr is not None and sum(tr.zoom_votes.values()) >= self.ZOOM_MAX_READS:
                continue
            best, best_size = list(xyxy), size
        return best

    def add_zoom_reading(self, t, cam, xyxy, name):
        """줌 판독 결과 (gauge_normal | gauge_low) 를 그 압력계 Track 에 반영. 확정/재판정된 Track 목록 반환."""
        tr = self._observe(t, cam, "gauge", xyxy)
        if tr is None:
            return []
        tr.zoom_votes[name] = tr.zoom_votes.get(name, 0) + 1
        best = tr.cls_name
        if not tr.confirmed and tr.zoom_votes[best] >= self.ZOOM_CONFIRM:
            tr.confirmed, tr.verdict = True, best
            return [tr]
        if tr.confirmed and tr.zoom_votes[best] > tr.zoom_votes.get(tr.verdict, 0):
            tr.verdict = best
            return [tr]
        return []

    def _nearest(self, group, xy):
        best, best_d = None, MERGE_DIST[group]
        for tr in self.tracks:
            if tr.group != group:
                continue
            dd = float(np.linalg.norm(tr.pos - xy))
            if dd < best_d:
                best, best_d = tr, dd
        return best

    @property
    def confirmed(self):
        return [tr for tr in self.tracks if tr.confirmed]

    def focus_view(self, t, recent=0.6):
        """RobotPatrol 능동 추적이 쓰는 검출기 모양 (hazards, detected, in_view, conf).
        아직 확정 안 된 물체와, 압력계를 못 읽은 소화기 몸통을 최근에 봤으면 그쪽을 보게 한다."""
        gauges = [tr.pos for tr in self.confirmed if tr.group == "gauge"]
        items, detected, in_view, conf = [], [], [], []
        for tr in self.tracks:
            done = tr.confirmed
            if tr.group == "ext_body":
                done = any(np.linalg.norm(tr.pos - g) < MERGE_DIST["gauge"] for g in gauges)
            items.append(SimpleNamespace(center=np.array([tr.pos[0], tr.pos[1], FOCUS_Z[tr.group]]),
                                         read_range=GAUGE_READ_RANGE if tr.group in ("gauge", "ext_body") else None))
            detected.append(done)
            in_view.append(t - tr.last_t < recent)
            conf.append(tr.conf_max)
        return SimpleNamespace(hazards=items, detected=np.array(detected, dtype=bool),
                               in_view=np.array(in_view, dtype=bool), conf=np.array(conf))

    def score(self, hazards, match_dist=4.0):
        """시뮬레이션 채점: 확정한 물체를 정답과 맞춰본다 (판단에는 정답을 안 씀).
        반환: dict(found, total, ext_checked, ext_total, false_alarms, missed=[Hazard])"""
        used = set()
        found = ext_checked = wrong = 0
        missed = []
        for h in hazards:
            group = GT_GROUP[h.type]
            want = None if h.type != "ext" else ("gauge_normal" if h.ok else "gauge_low")
            best, best_d = None, match_dist
            for j, tr in enumerate(self.confirmed):
                if j in used or tr.group != group:
                    continue
                dd = float(np.linalg.norm(tr.pos - h.center[:2]))
                if dd < best_d:
                    best, best_d = j, dd
            ok = best is not None and (want is None or self.confirmed[best].verdict == want)
            if best is not None:
                used.add(best)
            if h.type == "ext":
                ext_checked += ok
                if not h.ok and not ok:
                    missed.append(h)
                if not h.ok:
                    found += ok
                elif best is not None and not ok:
                    wrong += 1          # 정상 소화기를 압력 부족으로 판정
            else:
                found += ok
                if not ok:
                    missed.append(h)
        # 정답과 안 맞는 위험 판정 (정상 압력계는 위험 알림이 아니라 제외)
        false_alarms = wrong + sum(1 for j, tr in enumerate(self.confirmed)
                                   if j not in used and RISK[tr.verdict] != "ok")
        return dict(found=found, total=sum(1 for h in hazards if h.is_hazard),
                    ext_checked=ext_checked, ext_total=sum(1 for h in hazards if h.type == "ext"),
                    false_alarms=false_alarms, missed=missed)


def judge_line(tr, t, color=True):
    from .report import _COLOR, _RESET, _TAG, clock
    name = tr.verdict
    risk = RISK[name]
    head = f"[{clock(t)}] {_TAG[risk]} {CLASS_KO[name]}"
    if color:
        head = _COLOR[risk] + head + _RESET
    zone = L.zone_name(float(tr.pos[0]), float(tr.pos[1]))
    if tr.zoom_votes:
        basis = f"줌 판독 {tr.zoom_votes.get(name, 0)}/{sum(tr.zoom_votes.values())}회"
    else:
        basis = f"YOLO 신뢰도 {tr.conf_max * 100:.0f}%"
    return f"{head}  |  {zone} (추정 {tr.pos[0]:+.1f}, {tr.pos[1]:+.1f} m)  |  {DETAIL[name]}  |  {basis}"


def judge_summary(judge, t, hazards=None):
    from .report import clock
    conf = judge.confirmed
    n_risk = sum(1 for tr in conf if RISK[tr.verdict] != "ok")
    n_gauge = sum(1 for tr in conf if tr.group == "gauge")
    lines = [f"순찰 시간 {clock(t)}  |  위험 판정 {n_risk}건  |  소화기 점검 {n_gauge}개 (화면 판단만 사용)"]
    if hazards is not None:
        s = judge.score(hazards)
        lines.append(f"  [채점] 위험 요소 {s['found']}/{s['total']}  |  소화기 점검 {s['ext_checked']}/{s['ext_total']}  |  "
                     f"오탐 {s['false_alarms']}건")
        for h in s["missed"]:
            lines.append(f"  못 찾음: {h.label} ({h.zone})")
    return "\n".join(lines)
