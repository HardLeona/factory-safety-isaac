"""바디캠 YOLO 판정, 정답표 채점.

판정(실시간 출력)에는 정답을 쓰지 않는다:
  바디캠: YOLO 추적 번호로 같은 물체를 묶고, CONFIRM 프레임 이상 같은 판정이면 위험/안전을 한 번 알린다.
          위치(구역)는 작업자 위치 + 박스 아래쪽을 바닥에 투영해서 추정.
채점(끝나고)에만 정답을 쓴다:
  매 프레임 Replicator 정답 박스(물체 경로까지)와 YOLO 박스를 겹침으로 맞춰, 물체마다 판정을 모아
  정답표(위험/안전)와 비교.
"""
import math
import re
from collections import defaultdict

import numpy as np

from . import warehouse as W
from .config import CLASS_KO, HAZARD, KIND, KIND_KO
from .geometry import Projector


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
            if oid is None or CLASSES[c] in ("worker", "cone", "danger_sign", "cart"):
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


def _runs(ts, gap):
    """정렬된 시각 목록을 gap 보다 가까운 것끼리 묶음."""
    out = []
    for t in ts:
        if out and t - out[-1][-1] <= gap:
            out[-1].append(t)
        else:
            out.append([t])
    return out


__all__ = ["BodycamInspector", "bodycam_line", "bodycam_summary", "floor_point", "iou", "object_id_from_path", "KIND_KO"]
