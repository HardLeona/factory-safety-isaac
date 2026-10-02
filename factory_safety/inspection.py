"""바디캠 YOLO 판정, 정답표 채점, CCTV 작업자-위험물 거리 경고.

판정(실시간 출력)에는 정답을 쓰지 않는다:
  바디캠: YOLO 추적 번호로 같은 물체를 묶고, CONFIRM 프레임 이상 같은 판정이면 위험/안전을 한 번 알린다.
          위치(구역)는 작업자 위치 + 박스 아래쪽을 바닥에 투영해서 추정.
  CCTV  : YOLO 로 작업자와 위험 물체를 찾고, 박스 아래쪽을 바닥에 투영해서 거리를 잰다. 가까우면 경고.
채점(끝나고)에만 정답을 쓴다:
  바디캠: 매 프레임 Replicator 정답 박스(물체 경로까지)와 YOLO 박스를 겹침으로 맞춰, 물체마다 판정을 모아
          정답표(위험/안전)와 비교.
  CCTV  : 실제 작업자 위치와 물체 위치로 진짜 거리를 구해 경고가 맞았는지, 거리 오차가 얼마인지.
"""
import math
import re
from collections import defaultdict

import numpy as np

from . import warehouse as W
from .config import CLASS_KO, HAZARD, KIND, KIND_KO, PROXIMITY_WARN_M
from .geometry import Projector

FLAT = {"spill", "spill_marked"}


def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def floor_point(cam, xyxy, cls, img_w, img_h, max_range=30.0, z=0.0):
    """박스 -> 바닥(높이 z 평면) 위 공장 좌표. 납작한 유출은 박스 가운데, 나머지는 아래쪽 가운데 (발, 바닥에 닿은 곳).
    작업대 위 물체처럼 바닥이 아닌 곳에 놓인 것은 z 에 그 높이를 준다."""
    p = Projector(img_w, img_h)
    p.set_pose(cam)
    u = (xyxy[0] + xyxy[2]) / 2.0
    # 표지된 유출은 표지판·라바콘이 박스 위쪽을 키워서 가운데보다 아래(3/4) 를 바닥 위치로 씀
    v = {"spill": 0.5, "spill_marked": 0.75}.get(cls, 1.0) * (xyxy[3] - xyxy[1]) + xyxy[1]
    d = p.f + p.r * (u - p.w / 2.0) / p.fx + p.u * (p.h / 2.0 - v) / p.fy
    if d[2] >= -1e-6 or cam.pos[2] <= z:
        return None
    t = (z - cam.pos[2]) / d[2]
    if t * math.hypot(d[0], d[1]) > max_range:
        return None
    return np.asarray(cam.pos[:2], float) + t * d[:2]


def object_id_from_path(path):
    m = re.search(r"/(O\d+)_", path or "")
    return m.group(1) if m else None


# ---------------------------------------------------------------------------- 바디캠
class BodycamInspector:
    CONFIRM = 3          # 같은 추적 번호가 이만큼 같은 판정이면 알림
    MIN_GT_PX = 12       # 정답 박스가 이보다 작으면 "보였다" 로 안 셈
    MATCH_IOU = 0.3

    SAME_PLACE_M = 2.5   # 화면에서 나갔다 들어와 추적 번호가 바뀌어도, 같은 판정이 이 거리 안이면 다시 안 알림

    def __init__(self, img_w, img_h):
        self.w, self.h = img_w, img_h
        self.announced = []                # 이미 알린 (클래스, 바닥 위치)
        self.tracks = {}                   # 추적 번호 -> {votes, n, announced, cls, conf}
        self.obs = defaultdict(list)       # 물체 id -> [(예측 클래스, 신뢰도)]
        self.visible = defaultdict(int)    # 물체 id -> 보인 프레임 수
        self.false_boxes = defaultdict(int)  # 정답 없는 곳의 위험 판정 박스 (클래스별)
        self.frames = 0

    def update(self, t, cam, dets):
        """실시간 판정. dets: [(클래스, 신뢰도, xyxy, 추적 번호 또는 None)]. 새로 알릴 것 [(t, 클래스, 신뢰도, 구역)]."""
        events = []
        for name, conf, xyxy, tid in dets:
            if name == "worker" or tid is None:
                continue
            tr = self.tracks.setdefault(tid, {"votes": defaultdict(float), "n": 0, "announced": None})
            tr["votes"][name] += conf
            tr["n"] += 1
            best = max(tr["votes"], key=tr["votes"].get)
            if tr["n"] >= self.CONFIRM and tr["announced"] != best:
                # 처음 확정되거나, 더 가까이서 보고 판정이 바뀌면 다시 알림
                xy = floor_point(cam, xyxy, best, self.w, self.h)
                tr["announced"] = best
                if xy is not None and any(c == best and np.linalg.norm(p - xy) < self.SAME_PLACE_M for c, p in self.announced):
                    continue
                if xy is not None:
                    self.announced.append((best, xy))
                zone = W.zone_name(*xy) if xy is not None else "?"
                events.append((t, best, conf, zone))
        return events

    def score_frame(self, dets, gt):
        """채점용. gt: [(클래스 번호, x0, y0, x1, y1, 가림, 경로)] (Replicator). dets 는 update 와 같음."""
        from .config import CLASSES
        self.frames += 1
        gts = []
        for c, x0, y0, x1, y1, occ, path in gt:
            oid = object_id_from_path(path)
            if oid is None or CLASSES[c] == "worker":
                continue
            if min(x1 - x0, y1 - y0) >= self.MIN_GT_PX and occ < 0.7:
                self.visible[oid] += 1
            gts.append((oid, (x0, y0, x1, y1)))
        pairs = sorted(((iou(d[2], g[1]), i, j) for i, d in enumerate(dets) for j, g in enumerate(gts)
                        if dets[i][0] != "worker"), reverse=True)
        used_d, used_g = set(), set()
        for v, i, j in pairs:
            if v < self.MATCH_IOU or i in used_d or j in used_g:
                continue
            used_d.add(i)
            used_g.add(j)
            self.obs[gts[j][0]].append((dets[i][0], dets[i][1]))
        for i, d in enumerate(dets):
            if i in used_d or d[0] == "worker" or not HAZARD.get(d[0], False):
                continue
            if all(iou(d[2], g[1]) < 0.1 for g in gts):
                self.false_boxes[d[0]] += 1

    def report(self, answer_key, min_frames=2):
        """정답표와 맞춘 결과: (행 목록, 요약 dict)."""
        rows = []
        for o in answer_key["objects"]:
            oid, gt = o["id"], o["class"]
            obs = self.obs.get(oid, [])
            vis = self.visible.get(oid, 0)
            if len(obs) >= min_frames:
                votes = defaultdict(float)
                for c, conf in obs:
                    votes[c] += conf
                pred = max(votes, key=votes.get)
                if pred == gt:
                    res = "정확"
                elif HAZARD[pred] == o["hazard"]:
                    res = "위험 여부는 맞음"
                elif o["hazard"]:
                    res = "위험을 안전으로 오판"
                else:
                    res = "안전을 위험으로 오판"
            else:
                pred = None
                res = "놓침" if vis >= min_frames else "안 보임"
            rows.append({**o, "pred": pred, "matched": len(obs), "visible": vis, "result": res})
        seen = [r for r in rows if r["result"] != "안 보임"]
        hz = [r for r in seen if r["hazard"]]
        sf = [r for r in seen if not r["hazard"]]
        summ = {
            "objects": len(rows), "seen": len(seen),
            "hazard_total": len(hz),
            "hazard_found": sum(1 for r in hz if r["pred"] and HAZARD[r["pred"]]),
            "hazard_as_safe": sum(1 for r in hz if r["result"] == "위험을 안전으로 오판"),
            "hazard_missed": sum(1 for r in hz if r["result"] == "놓침"),
            "safe_total": len(sf),
            "safe_ok": sum(1 for r in sf if r["pred"] and not HAZARD[r["pred"]]),
            "safe_as_hazard": sum(1 for r in sf if r["result"] == "안전을 위험으로 오판"),
            "safe_missed": sum(1 for r in sf if r["result"] == "놓침"),
            "exact": sum(1 for r in seen if r["result"] == "정확"),
            "false_hazard_boxes": int(sum(self.false_boxes.values())),
            "frames": self.frames,
        }
        return rows, summ


def bodycam_line(t, cls, conf, zone):
    from .report import clock, tag
    return f"[{clock(t)}] 바디캠  {tag(HAZARD[cls])} {CLASS_KO[cls]}  |  {zone}  |  YOLO {conf * 100:.0f}%"


def bodycam_summary(rows, s):
    lines = [f"바디캠 판정 채점 (정답표 물체 {s['objects']}개 중 경로에서 보인 것 {s['seen']}개)",
             f"  위험 물체 {s['hazard_total']}개: 위험으로 판정 {s['hazard_found']}  |  안전으로 오판 {s['hazard_as_safe']}  |  놓침 {s['hazard_missed']}",
             f"  안전 물체 {s['safe_total']}개: 안전으로 판정 {s['safe_ok']}  |  위험으로 오판 {s['safe_as_hazard']}  |  놓침 {s['safe_missed']}",
             f"  상태까지 정확히 맞힘 {s['exact']}/{s['seen']}  |  정답 없는 곳의 위험 박스 {s['false_hazard_boxes']}개 ({s['frames']}프레임)"]
    for r in rows:
        if r["result"] not in ("정확", "안 보임"):
            pred = CLASS_KO[r["pred"]] if r["pred"] else "-"
            lines.append(f"  · {r['id']} {r['label']} ({r['zone']}): {r['result']} (판정 {pred}, {r['matched']}프레임)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------- CCTV
def _inter(a, b):
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


class CCTVProximity:
    REPEAT_S = 5.0       # 같은 위험물 경고는 이 간격으로만 다시
    MAX_RANGE = 20.0     # 카메라에서 이보다 먼 바닥 추정은 안 씀
    INSIDE_WORKER = 0.5  # 위험물 박스의 이 비율 이상이 작업자 박스 안이면 뺌

    def __init__(self, img_w, img_h, warn=PROXIMITY_WARN_M):
        self.w, self.h, self.warn = img_w, img_h, warn
        self.last_alert = {}
        self.records = []        # 채점용: 프레임마다 (추정 경고, 실제 경고, 위치 오차들)
        self.worker_err = []
        self.dist_err = []

    def update(self, t, name, cam, dets):
        """실시간. 반환: (경고 [(t, cctv, 클래스, 거리, 위치)], 측정 결과 dict).
        - 카메라에서 MAX_RANGE 보다 먼 추정은 안 쓴다 (몇 픽셀 차이로 수 m 가 틀어짐, 그 구역은 다른 CCTV 가 맡음)
        - 유출 박스가 작업자 박스 안에 대부분 들어 있으면 작업자 발밑 그림자를 잘못 본 것으로 보고 뺀다
          (평가에서 오경보의 절반이 이것. 작업자가 밟고 지나가는 넓은 웅덩이는 작업자 박스보다 커서 안 빠짐.
          공구, 적재, 소화기는 작업자 뒤에 겹쳐 보여도 진짜일 수 있어서 그대로 둠)"""
        wboxes = [b for n, c, b, *_ in dets if n == "worker"]
        workers = [floor_point(cam, b, "worker", self.w, self.h, self.MAX_RANGE) for b in wboxes]
        workers = [p for p in workers if p is not None]
        hazards = []
        for n, c, b, *_ in dets:
            if n == "worker" or not HAZARD.get(n, False):
                continue
            area = max((b[2] - b[0]) * (b[3] - b[1]), 1e-6)
            if n in FLAT and any(_inter(b, wb) / area > self.INSIDE_WORKER for wb in wboxes):
                continue
            p = floor_point(cam, b, n, self.w, self.h, self.MAX_RANGE)
            if p is not None:
                hazards.append((n, c, p))
        events, pairs = [], []
        for wp in workers:
            for n, c, hp in hazards:
                d = float(np.linalg.norm(wp - hp))
                pairs.append((wp, n, hp, d))
                if d < self.warn:
                    key = (name, n, int(hp[0] // 2), int(hp[1] // 2))
                    if t - self.last_alert.get(key, -99) >= self.REPEAT_S:
                        self.last_alert[key] = t
                        events.append((t, name, n, d, hp))
        return events, {"workers": workers, "pairs": pairs}

    def score_frame(self, t, cam_name, meas, worker_gt, hazard_objs, visible_ids):
        """채점용 기록 (판정에는 안 씀). worker_gt: 실제 작업자 (x, y). hazard_objs: 정답 위험 물체 [SceneObject].
        visible_ids: 이 CCTV 정답 박스에 보인 물체 id."""
        if not hasattr(self, "objs"):
            self.objs = [(o.id, KIND[o.cls], float(o.x), float(o.y)) for o in hazard_objs]
            self.log = []
        self.log.append({"t": float(t), "cam": cam_name, "worker_gt": [float(v) for v in worker_gt],
                         "workers": [[float(v) for v in w] for w in meas["workers"]],
                         "pairs": [[n, [float(hp[0]), float(hp[1])], float(d)] for wp, n, hp, d in meas["pairs"]],
                         "visible": sorted(i for i in visible_ids if i)})

    def report(self, match_m=1.5, false_margin=1.0, gap_s=0.6):
        """사건 단위 채점.
        다가간 사건: 작업자가 어떤 정답 위험물에 경고 거리 안으로 들어간 구간 (CCTV 를 하나로 묶어서).
        잡음: 그 구간(앞뒤 1초)에 그 물체로 경고가 났으면. 사각지대: 그 구간에 어느 CCTV 에도 그 물체가 안 보임.
        오경보: 경고한 물체의 실제 거리가 경고 거리 + false_margin 보다 멀거나, 근처에 그런 정답 물체가 없음."""
        log = getattr(self, "log", [])
        objs = getattr(self, "objs", [])
        if not log:
            return None
        times = sorted({e["t"] for e in log})
        wgt = {e["t"]: np.array(e["worker_gt"]) for e in log}
        vis_any = defaultdict(set)
        alerts = defaultdict(list)                 # t -> [(물체 id 또는 None, 추정 거리, 실제 거리)]
        worker_err, dist_err = [], []
        for e in log:
            t, w = e["t"], wgt[e["t"]]
            vis_any[t] |= set(e["visible"])
            if e["workers"]:
                worker_err.append(min(float(np.linalg.norm(np.array(p) - w)) for p in e["workers"]))
            for n, hp, d in e["pairs"]:
                cand = [(math.hypot(x - hp[0], y - hp[1]), oid, x, y) for oid, kind, x, y in objs if kind == KIND[n]]
                best = min(cand) if cand else None
                if best and best[0] < match_m:
                    true_d = math.hypot(best[2] - w[0], best[3] - w[1])
                    dist_err.append(abs(d - true_d))
                    if d < self.warn:
                        alerts[t].append((best[1], d, true_d))
                elif d < self.warn:
                    alerts[t].append((None, d, None))
        events = []
        for oid, kind, x, y in objs:
            near = [t for t in times if math.hypot(x - wgt[t][0], y - wgt[t][1]) < self.warn]
            for grp in _runs(near, gap_s):
                t0, t1 = grp[0], grp[-1]
                win = [t for t in times if t0 - 1.0 <= t <= t1 + 1.0]
                events.append({"id": oid, "t0": t0, "t1": t1,
                               "visible": any(oid in vis_any[t] for t in grp),
                               "detected": any(a[0] == oid for t in win for a in alerts[t])})
        false_t = [t for t in times if any(a[0] is None or a[2] > self.warn + false_margin for a in alerts[t])]
        vis_ev = [e for e in events if e["visible"]]
        return {
            "frames": len(times), "events": len(events), "events_visible": len(vis_ev),
            "events_detected": sum(1 for e in vis_ev if e["detected"]), "events_blind": len(events) - len(vis_ev),
            "false_alert_episodes": len(_runs(false_t, 1.0)), "false_alert_frames": len(false_t),
            "alert_frames": sum(1 for t in times if alerts[t]),
            "worker_err_median": float(np.median(worker_err)) if worker_err else None,
            "dist_err_median": float(np.median(dist_err)) if dist_err else None,
            "dist_err_p90": float(np.percentile(dist_err, 90)) if dist_err else None,
            "event_list": events, "log": log,
        }


def _runs(ts, gap):
    """정렬된 시각 목록을 gap 보다 가까운 것끼리 묶음."""
    out = []
    for t in ts:
        if out and t - out[-1][-1] <= gap:
            out[-1].append(t)
        else:
            out.append([t])
    return out


def cctv_line(t, name, cls, d, hp):
    from .report import clock
    return (f"[{clock(t)}] CCTV {name}  ⚠ 접근 경고: 작업자 ↔ {CLASS_KO[cls]} {d:.1f} m  |  "
            f"{W.zone_name(*hp)} ({hp[0]:+.1f}, {hp[1]:+.1f})")


def cctv_summary(s):
    def f(v):
        return "-" if v is None else f"{v:.2f} m"
    if not s:
        return "CCTV 접근 경고 채점: 기록 없음"
    return (f"CCTV 접근 경고 채점 (경고 거리 {PROXIMITY_WARN_M} m, CCTV {len({e['cam'] for e in s['log']})}대를 묶어서)\n"
            f"  작업자가 위험물에 다가간 사건 {s['events']}건: CCTV 에 보인 {s['events_visible']}건 중 경고 {s['events_detected']}건"
            f"  |  사각지대 {s['events_blind']}건\n"
            f"  오경보 {s['false_alert_episodes']}번 (실제 {PROXIMITY_WARN_M + 1.0:.0f} m 넘는데 경고, {s['false_alert_frames']}프레임)\n"
            f"  작업자 위치 오차 중앙값 {f(s['worker_err_median'])}  |  거리 오차 중앙값 {f(s['dist_err_median'])}, 90% {f(s['dist_err_p90'])}")


__all__ = ["BodycamInspector", "CCTVProximity", "bodycam_line", "bodycam_summary", "cctv_line", "cctv_summary",
           "floor_point", "iou", "object_id_from_path", "KIND_KO"]
