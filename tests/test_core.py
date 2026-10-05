"""Isaac Sim 없이 돌아가는 핵심 로직 테스트.

    python -m pytest tests -q        (또는 python tests/test_core.py)
"""
import math
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from factory_safety import warehouse as W  # noqa: E402
from factory_safety.config import CLASSES, HAZARD, KIND, STATES, TOOL_TYPES  # noqa: E402
from factory_safety.geometry import CameraPose, Projector  # noqa: E402
from factory_safety.scenario import sample_scenario  # noqa: E402
from factory_safety.walker import PathWalker  # noqa: E402


def test_scenario_and_answer_key():
    kinds_seen, tools_seen = set(), set()
    for seed in range(30):
        sc = sample_scenario(seed)
        key = sc.answer_key()
        ids = [o["id"] for o in key["objects"]]
        assert len(ids) == len(set(ids))
        cnt = {}
        for o in sc.objects:
            cnt[o.kind] = cnt.get(o.kind, 0) + 1
            kinds_seen.add(o.cls)
            assert -10.5 < o.x < 10.0 and -12.0 < o.y < 18.0
            assert o.cls in STATES and HAZARD[o.cls] == o.hazard
            if o.kind == "tool":
                assert o.tools and all(t in TOOL_TYPES for t in o.tools)
                tools_seen.update(o.tools)
        assert cnt["spill"] == 3 and cnt["stack"] == 4 and cnt["ext"] == len(W.EXT_MOUNTS)
        assert 2 + len(W.TABLES) <= cnt["tool"] <= 3 + len(W.TABLES)
        assert all(set(o) >= {"id", "class", "hazard", "x", "y", "zone"} for o in key["objects"])
        # 위험 영역 2곳 (라바콘 또는 DANGER 표지), 다른 물체와 겹치지 않게
        assert len(key["zones"]) == 2 and all(z["n_cones"] >= 4 or z["sign"] for z in key["zones"])
        for z in sc.zones:
            assert all(math.hypot(o.x - z.x, o.y - z.y) > z.radius + 0.3 for o in sc.objects if o.kind != "ext")
        # 경로 바로 옆 (1 m 안) 위험물이 하나 이상 (작업자가 닿기 직전까지 다가감)
        path = W.PatrolPath()
        assert any(np.min(np.hypot(path.pts[:, 0] - o.x, path.pts[:, 1] - o.y)) < 1.0 for o in sc.hazards)
    # 30 개 시나리오에서 위험/안전 상태와 공구 종류가 모두 나와야 한다
    assert kinds_seen == set(STATES) and tools_seen == set(TOOL_TYPES)


def test_patrol_path():
    p = W.PatrolPath()
    assert 50 < p.length < 70
    assert np.linalg.norm(p.point_at(0.0) - p.point_at(p.length)) < 1e-6
    pts = np.array([p.point_at(s) for s in np.linspace(0, p.length, 400)])
    assert np.max(np.linalg.norm(np.diff(pts, axis=0), axis=1)) < 0.3          # 끊긴 곳 없음
    # 경로는 랙과 벽을 피한다
    for x, y in pts:
        assert not (-9.8 < x < -8.2 and -3.8 < y < 12.4) and not (-0.8 < x < 0.8 and -3.8 < y < 12.4)
        assert -10 < x < 9.5 and -11.5 < y < 17.5


def test_walker():
    w = PathWalker(seed=1)
    for _ in range(int(w.path.length / w.speed / 0.05) + 5):
        cam = w.step(0.05)
        assert 1.3 < cam.pos[2] < 1.45
        x, y, yaw, at = w.base_pose
        assert abs(math.hypot(cam.pos[0] - x, cam.pos[1] - y) - w.cam_ahead) < 1e-6
    assert w.laps >= 1.0


def test_walk_animation_targets():
    from factory_safety.walk_anim import PERIOD, _targets
    a, b = _targets(0.25 * PERIOD), _targets(0.75 * PERIOD)
    # 반 주기 차이면 왼다리와 오른다리가 서로 바뀐다
    assert np.allclose(a["L_Thigh"][1:], b["R_Thigh"][1:], atol=1e-6)
    assert a["L_Thigh"][1] < 0 < a["R_Thigh"][1]                # 왼다리 앞(-Y), 오른다리 뒤
    assert a["L_Upperarm"][1] > 0                                # 팔은 다리와 반대로


def test_floor_point_roundtrip():
    from factory_safety.inspection import floor_point
    cam = CameraPose(pos=np.array([-9.0, -2.0, 1.38]), yaw=0.0, pitch=-0.2, vfov=70)
    P = Projector(960, 540)
    P.set_pose(cam)
    target = np.array([-4.0, -2.0, 0.0])
    uv, z = P.project(target[None])
    u, v = uv[0]
    xy = floor_point(cam, [u - 10, v - 40, u + 10, v], "worker", 960, 540)
    assert np.linalg.norm(xy - target[:2]) < 0.05


def test_bodycam_scoring():
    from factory_safety.inspection import BodycamInspector
    sc = sample_scenario(0)
    key = sc.answer_key()
    ins = BodycamInspector(960, 540)
    hz = next(o for o in sc.objects if o.hazard and o.kind != "tool")
    sf = next(o for o in sc.objects if not o.hazard and o.kind != "tool")
    path = lambda o: f"/World/Scn000/{o.id}_{o.cls}"   # noqa: E731
    gt = [(CLASSES.index(hz.cls), 100, 100, 200, 200, 0.0, path(hz)), (CLASSES.index(sf.cls), 400, 100, 500, 220, 0.0, path(sf))]
    for k in range(4):
        dets = [(hz.cls, 0.8, [102, 98, 199, 203], 1), (sf.cls, 0.7, [398, 104, 502, 218], 2)]
        if k == 0:
            dets.append(("spill", 0.6, [700, 300, 760, 340], 3))   # 정답 없는 곳의 위험 박스
        ins.score_frame(dets, gt)
    rows, s = ins.report(key)
    r = {x["id"]: x for x in rows}
    assert r[hz.id]["result"] == "정확" and r[sf.id]["result"] == "정확"
    assert s["hazard_found"] == 1 and s["safe_ok"] == 1 and s["false_hazard_boxes"] == 1
    # 같은 종류의 다른 상태로 판정하면 위험 여부 기준으로 채점
    ins2 = BodycamInspector(960, 540)
    safe_twin = next(c for c in STATES if KIND.get(c) == hz.kind and not HAZARD[c])
    for _ in range(3):
        ins2.score_frame([(safe_twin, 0.9, [100, 100, 200, 200], 1)], gt[:1])
    rows2, s2 = ins2.report(key)
    assert next(x for x in rows2 if x["id"] == hz.id)["result"] == "위험을 안전으로 오판"
    assert s2["hazard_as_safe"] == 1


def test_bodycam_live_confirm():
    from factory_safety.inspection import BodycamInspector
    ins = BodycamInspector(960, 540)
    cam = CameraPose(pos=np.array([-4.5, 0.0, 1.38]), yaw=math.pi / 2, pitch=-0.2, vfov=70)
    events = []
    for k in range(5):
        events += ins.update(k * 0.2, cam, [("spill", 0.8, [400, 300, 520, 360], 7)])
    assert len(events) == 1 and events[0][1] == "spill"


def _agent():
    from factory_safety.agent import SafetyAgent
    return SafetyAgent(960, 540, log=None)


def _box_at(cam, xyz, w=60, h=30, flat=True):
    P = Projector(960, 540)
    P.set_pose(cam)
    uv, _ = P.project(np.array([xyz], float))
    u, v = uv[0]
    return [u - w / 2, v - h / 2, u + w / 2, v + h / 2] if flat else [u - w / 2, v - h, u + w / 2, v]


def test_agent_ambiguous_hazard_immediate():
    """잠깐 보고 지나친 물체라도 후보 중 고위험 클래스(stack_unstable)가 있으면 즉시 위험으로 확정한다."""
    a = _agent()
    cam = CameraPose(pos=np.array([7.2, -4.0, 1.38]), yaw=-math.pi / 2, pitch=-0.2, vfov=70)
    box = _box_at(cam, [7.2, -5.6, 0.0])
    a.on_bodycam(1.0, cam, [("stack_unstable", 0.6, box, 7)])
    a.on_bodycam(2.0, cam, [])     # 1프레임만 보이고 사라짐 (CONFIRM 미달)
    f = a.findings[0]
    assert f.status == "확정" and f.cls == "stack_unstable" and a.stats["ambiguous_hazard"] == 1
    rep = a.report()
    assert rep["findings"][0]["state"] == "위험"


def test_agent_ambiguous_caution_classification():
    """저위험 후보(tool_floor)만 있으면 안전·위험을 단정하지 않고 '주의' 로 분류하고, 위험구역에는 안 들어간다."""
    a = _agent()
    cam = CameraPose(pos=np.array([-4.5, -2.0, 1.38]), yaw=math.pi / 2, pitch=-0.2, vfov=70)
    box = _box_at(cam, [-5.6, 3.0, 0.0], 40, 20, flat=False)
    a.on_bodycam(1.0, cam, [("tool_floor", 0.5, box, 11)])
    a.on_bodycam(2.0, cam, [])                                   # 1프레임만 보이고 사라짐
    f = a.findings[0]
    assert f.status == "주의" and a.stats["ambiguous_caution"] == 1
    rep = a.report()
    row = next(r for r in rep["findings"] if r["id"] == f.fid)
    assert row["state"] == "주의" and "현장 확인 권고" in row["action"]
    assert a.zones == {}       # 위험구역에는 포함 안 됨 (지도에는 report() 로 표시됨)


def test_agent_caution_reobserved_llm_and_fallback():
    """'주의' 물체를 바디캠이 다시 지나치면 recheck_judge(가짜 LLM)가 누적 증거로 재판단해 확정한다.
    LLM 이 없거나 실패하면 규칙(고위험 후보 재검토)으로 폴백 — 고위험 없으면 주의를 유지한다."""
    a = _agent()
    cam = CameraPose(pos=np.array([-4.5, -2.0, 1.38]), yaw=math.pi / 2, pitch=-0.2, vfov=70)
    box = _box_at(cam, [-5.6, 3.0, 0.0], 40, 20, flat=False)
    a.on_bodycam(1.0, cam, [("tool_floor", 0.5, box, 11)])
    a.on_bodycam(2.0, cam, [])
    f = a.findings[0]
    assert f.status == "주의"
    seen = []

    def judge(snap):
        seen.append(snap)
        return {"llm": True, "model": "fake", "verdict": "tool_stored", "reason_ko": "작업대 위로 옮겨짐", "trace": ["evidence()"]}
    a.recheck_judge = judge
    for k in range(3):   # 다시 지나칠 때 투표가 갈림 (tool_floor/tool_stored 비슷) -> LLM 이 가른다
        a.on_bodycam(10.0 + 0.1 * k, cam, [("tool_floor", 0.4, box, 12), ("tool_stored", 0.4, box, 12)])
    assert f.status == "확정" and f.cls == "tool_stored" and a.stats["caution_rejudged"] == 1
    assert seen and seen[0]["target_kind"] == "tool" and "tool_floor" in seen[0]["prior_observations"]
    assert any(e["kind"] == "LLM 판단" for e in a.timeline)
    # LLM 없이(규칙 폴백): 저위험 후보만 다시 보이면 주의 유지
    a2 = _agent()
    a2.on_bodycam(1.0, cam, [("tool_floor", 0.5, box, 21)])
    a2.on_bodycam(2.0, cam, [])
    f2 = a2.findings[0]
    assert f2.status == "주의"
    for k in range(3):
        a2.on_bodycam(10.0 + 0.1 * k, cam, [("tool_floor", 0.6, box, 22)])
    assert f2.status == "주의" and a2.stats["caution_rejudged"] == 1 and a2.stats["caution_rejudged_confirmed"] == 0


def test_agent_ambiguous_mixed_candidates_immediate_hazard():
    """잠깐 보임에서 저위험(tool_floor) + 고위험(stack_unstable) 후보가 섞이면, 고위험 후보 쪽으로 즉시 위험 확정한다."""
    a = _agent()
    cam = CameraPose(pos=np.array([7.2, -4.0, 1.38]), yaw=-math.pi / 2, pitch=-0.2, vfov=70)
    box = _box_at(cam, [7.2, -5.6, 0.0])
    a.on_bodycam(1.0, cam, [("tool_floor", 0.5, box, 31), ("stack_unstable", 0.4, box, 31)])
    a.on_bodycam(2.0, cam, [])
    f = a.findings[0]
    assert f.status == "확정" and f.cls == "stack_unstable" and a.stats["ambiguous_hazard"] == 1


def test_agent_report_action_grounding():
    """report_judge(가짜 LLM)가 매뉴얼 근거(grounded=true)를 주면 조치 문구를 보강하고 출처를 남긴다.
    grounded=false 거나 LLM 이 없거나 실패하면 고정 문구(ACTIONS) 그대로 쓴다."""
    from factory_safety.agent import ACTIONS
    a = _agent()
    cam = CameraPose(pos=np.array([4.5, 8.0, 1.38]), yaw=-math.pi / 2, pitch=-0.2, vfov=70)
    box = _box_at(cam, [3.1, 2.2, 0.0])
    for k in range(3):
        a.on_bodycam(0.1 * k, cam, [("spill", 0.8, box, 1)], worker_xy=(4.5, 8.0))
    seen = []

    def judge(snap):
        seen.append(snap)
        return {"llm": True, "grounded": True, "action_ko": "유출물 제거 + 미끄럼 방지 패드 설치, 바닥 균열도 점검",
               "source": "spill / 청소·정리정돈"}
    a.report_judge = judge
    rep = a.report()
    row = next(r for r in rep["findings"] if r["class"] == "spill")
    assert "미끄럼 방지 패드" in row["action"] and row["action_source"] == "spill / 청소·정리정돈"
    assert seen and seen[0]["class"] == "spill" and seen[0]["standard_action"] == ACTIONS["spill"][0]
    # grounded=false 면 고정 문구 그대로
    a.report_judge = lambda snap: {"llm": True, "grounded": False, "action_ko": "", "source": ""}
    rep2 = a.report()
    row2 = next(r for r in rep2["findings"] if r["class"] == "spill")
    assert row2["action"] == ACTIONS["spill"][0] and row2["action_source"] is None
    # LLM 실패해도(예외) 고정 문구로 안전하게 폴백
    a.report_judge = lambda snap: (_ for _ in ()).throw(RuntimeError("꺼짐"))
    rep3 = a.report()
    row3 = next(r for r in rep3["findings"] if r["class"] == "spill")
    assert row3["action"] == ACTIONS["spill"][0] and row3["action_source"] is None


def test_agent_checkpoints_and_finish_patrol():
    """바디캠이 확인한 소화기 점검 지점은 '바디캠 확인' 으로, 끝내 못 본 지점은 순찰이 끝날 때 바로
    '현장 확인 필요' 로 넘어간다 (재시도 없이)."""
    from factory_safety.agent import Finding
    a = _agent()
    mx, my, _ = W.EXT_MOUNTS[1]
    f = Finding("F01", "ext", np.array([mx, my], float), 0.0, "바디캠")
    f.body["ext_ok"] += 3.0
    a.findings.append(f)
    a._tick_checkpoints(1.0)
    c2 = next(c for c in a.checkpoints if c.cid == "C2")
    assert c2.status == "바디캠 확인" and c2.finding == "F01"
    a.finish_patrol(60.0)
    c1 = next(c for c in a.checkpoints if c.cid == "C1")
    assert c1.status == "현장 확인 필요" and c2.status == "바디캠 확인"   # 확인된 지점은 그대로, 못 본 지점만 넘어감
    rep = a.report()
    row = next(r for r in rep["findings"] if r["id"] == "C1")
    assert row["state"] == "확인 필요"


def test_agent_near_miss_priority_and_eval():
    from factory_safety.agent import Finding
    a = _agent()
    for fid, cls, xy in (("F01", "tool_floor", (-7.3, 4.6)), ("F02", "spill", (3.1, 2.2)), ("F03", "stack_stable", (7.2, -5.6))):
        f = Finding(fid, cls if KIND[cls] == "tool" else KIND[cls], np.array(xy, float), 0.0, "바디캠")
        f.body[cls] += 3.0
        f.n_body = 3
        a.findings.append(f)
    a.findings[0].near_miss = 2
    rep = a.report()
    assert [r["id"] for r in rep["findings"]] == ["F01", "F02", "F03"] and rep["findings"][0]["priority"] == "긴급"
    key = {"objects": [
        {"id": "O1", "class": "tool_floor", "hazard": True, "x": -7.2, "y": 4.4, "zone": ""},
        {"id": "O2", "class": "spill_marked", "hazard": False, "x": 3.0, "y": 2.0, "zone": ""},   # 안전을 위험으로
        {"id": "O3", "class": "stack_stable", "hazard": False, "x": 7.0, "y": -5.5, "zone": ""},
        {"id": "O4", "class": "ext_fallen", "hazard": True, "x": 9.1, "y": -4.3, "zone": ""}]}    # 놓침
    ev = a.evaluate(key)["after"]
    assert (ev["hazard_found"], ev["hazard_missed"], ev["safe_ok"], ev["safe_as_hazard"]) == (1, 1, 1, 1)
    assert ev["false_reports"] == 0


def test_agent_evaluate_before_is_frame_raw_after_is_ledger():
    """before = BodycamInspector 프레임 단위 YOLO 원시 판정, after = 에이전트 최종 대장. 단위가 달라 필드 구성도 다르다."""
    from factory_safety.agent import Finding
    a = _agent()
    gt = [(CLASSES.index("ext_fallen"), 10, 10, 50, 50, 0.0, "/World/O1_ext/Mesh")]
    for _ in range(3):      # 프레임 단위 YOLO 는 (잘못) "안전하게 정리된 공구" 로 봄
        a.scorer.score_frame([("tool_stored", 0.9, (10, 10, 50, 50), 1)], gt)
    f = Finding("F01", "ext", np.array([0.0, 0.0]), 0.0, "바디캠")   # 대장은 고위험 후보로 즉시 위험 확정 (규칙)
    f.body["ext_fallen"] += 5.0
    f.n_body = 3
    a.findings.append(f)
    key = {"objects": [{"id": "O1", "class": "ext_fallen", "hazard": True, "x": 0.0, "y": 0.0, "zone": ""}]}
    ev = a.evaluate(key)
    assert ev["before"]["hazard_found"] == 0 and ev["before"]["hazard_as_safe"] == 1
    assert ev["after"]["hazard_found"] == 1 and ev["after"]["hazard_as_safe"] == 0
    assert "frames" in ev["before"] and "frames" not in ev["after"]        # 프레임 단위만의 필드
    assert "need_check" in ev["after"] and "need_check" not in ev["before"]  # 물체 단위만의 필드


def test_agent_regroup_and_low_confidence():
    from factory_safety.agent import Finding, SafetyAgent
    a = _agent()
    ext = Finding("F01", "ext", np.array([0.09, -3.81]), 0.0, "바디캠")
    ext.body["ext_blocked"] += 5.0
    ext.body_confirmed = True
    stack = Finding("F02", "stack", np.array([-0.4, -3.9]), 0.0, "바디캠")
    stack.body["stack_unstable"] += 1.0
    stack.body["ext_blocked"] += 2.0                       # 소화기 앞을 막은 상자를 처음엔 적재로 봄
    a.findings += [ext, stack]
    assert a._regroup(1.0, stack) is ext and stack.status == "병합" and ext.n_body == 0
    assert [r["id"] for r in a.report()["findings"]] == ["F01"]
    a.findings = [ext]
    # 랙 안이나 벽 밖으로 떨어진 위치는 안으로 끌어옴
    assert SafetyAgent._sanitize(np.array([9.0, 5.0]))[0] in (8.0, 10.0) and SafetyAgent._sanitize(np.array([2.0, -12.5]))[1] > -12
    assert SafetyAgent._sanitize(np.array([2.0, -14.0])) is None          # 벽 밖 1 m 넘게 나간 추정은 버림
    # 멀리서 봐서 위치가 틀어진 "가로막힌 소화기" 도 가장 가까운 소화기 자리 항목과 합침
    far = Finding("F03", "stack", np.array([-5.14, -7.12]), 0.0, "바디캠")
    far.body["ext_blocked"] += 2.0
    lone = Finding("F04", "ext", np.array([-8.91, -3.81]), 0.0, "바디캠")
    lone.body["ext_blocked"] += 2.0
    a.findings += [far, lone]
    assert a._regroup(2.0, far) is lone and far.status == "병합"
    odd = Finding("F05", "stack", np.array([4.5, 4.0]), 0.0, "바디캠")       # 소화기 자리에서 8 m 넘게 떨어짐
    odd.body["ext_fallen"] += 2.0
    a.findings.append(odd)
    assert a._regroup(3.0, odd) is odd and odd.status == "현장 확인 필요"


def test_zones_build():
    from factory_safety import zones as Z
    ring = [(f"K{k}", (2.0 + 1.2 * math.cos(a), 1.0 + 1.2 * math.sin(a))) for k, a in enumerate(np.linspace(0, 6, 5))]
    out = Z.build(ring, [("S1", (2.3, 1.1))], [("F1", "spill", (2.1, 0.9)), ("F2", "stack_unstable", (-5.0, 3.0)),
                                                 ("F3", "spill", (-4.0, 4.5)), ("F4", "tool_floor", (8.0, 8.0))])
    src = sorted(o[1] for o in out)
    assert src == ["agent", "cone"]                          # 표지는 라바콘 영역에 합쳐지고, 링 안 유출은 따로 안 만듦
    cone = next(o for o in out if o[1] == "cone")
    assert Z.inside((2.0, 1.0), cone[2]) and cone[6] and cone[5] == 5
    agent = next(o for o in out if o[1] == "agent")
    assert agent[4] == ["F2", "F3"] and "2개" in agent[3]   # 3 m 안에 모인 위험물은 한 영역
    assert Z.distance((2.0, 1.0), cone[2]) == 0.0 and Z.distance((6.0, 1.0), cone[2]) > 2.0
    lone = Z.build([], [("S1", (0.0, 0.0))], [])
    assert lone[0][1] == "sign" and Z.inside((0.5, 0.5), lone[0][2])


def test_agent_zone_voice_and_tools():
    """화면의 라바콘을 영역으로 묶고, 작업자가 영역에 다가가면 음성 경고. 공구는 놓인 자리로 위험/안전."""
    a = _agent()
    heard = []
    a.on_voice = lambda t, level: heard.append((t, level))
    cam = CameraPose(pos=np.array([-4.5, -2.0, 1.38]), yaw=math.pi / 2, pitch=-0.2, vfov=70)
    ring = [(-6.2 + 1.2 * math.cos(q), 2.6 + 1.2 * math.sin(q)) for q in np.linspace(0, 5.5, 5)]
    for k in range(3):
        dets = [("cone", 0.9, _box_at(cam, [x, y, 0.0], 30, 50, flat=False), 100 + i) for i, (x, y) in enumerate(ring)]
        a.on_bodycam(0.2 * k, cam, dets, worker_xy=(-4.5, -2.0))
    zl = list(a.zones.values())
    assert len(zl) == 1 and zl[0].source == "cone" and zl[0].n_cones == 5 and not heard
    a.on_bodycam(1.0, cam, [], worker_xy=(-4.85, 2.6))           # 링 경계 0.5 m 안
    assert len(heard) == 1 and a.voice_events[0]["target"] == "Z:Z1"
    a.on_bodycam(2.0, cam, [], worker_xy=(-4.9, 2.6))
    assert len(heard) == 1                                       # 같은 영역은 바로 다시 안 울림
    # 공구 종류: 작업대 위 / 통로 바닥
    top = CameraPose(pos=np.array([-6.0, -9.5, 1.6]), yaw=-math.pi / 2, pitch=-0.6, vfov=60)
    assert a._tool_state(top, _box_at(top, [-6.0, -11.4, W.TABLE_TOP], 40, 20, flat=False)) == "tool_stored"
    assert a._tool_state(top, _box_at(top, [-5.0, -10.4, 0.0], 40, 20, flat=False)) == "tool_floor"
    for k in range(3):
        a.on_bodycam(5.0 + 0.2 * k, cam, [("hammer", 0.8, _box_at(cam, [-3.8, 1.5, 0.0], 40, 20, flat=False), 7)])
    f = next(f for f in a.findings if f.group == "tool_floor")
    assert f.tool_names == ["hammer"] and "망치" in a.report()["findings"][0]["label"] or any("망치" in r["label"] for r in a.report()["findings"])


def test_dashboard():
    import tempfile
    from factory_safety.dashboard import write_dashboard
    a = _agent()
    a.plan(59.0)
    cam = CameraPose(pos=np.array([-4.5, -2.0, 1.38]), yaw=math.pi / 2, pitch=-0.2, vfov=70)
    box = _box_at(cam, [-5.6, 3.0, 0.0], 40, 20, flat=False)
    for k in range(3):
        a.on_bodycam(0.1 * k, cam, [("tool_floor", 0.8, box, 1)])
    key = {"objects": [{"id": "O1", "class": "tool_floor", "hazard": True, "x": -5.6, "y": 3.0, "zone": ""}]}
    path = os.path.join(tempfile.mkdtemp(), "d.html")
    write_dashboard(path, a.report(), evaluation=a.evaluate(key), seed=1)
    html = open(path, encoding="utf-8").read()
    assert "조치 목록" in html and "정답표 비교" in html


def test_usd_build():
    try:
        from pxr import Usd
    except ImportError:
        print("  (pxr 없음: USD 테스트 건너뜀, pip install usd-core 로 가능)")
        return
    from factory_safety.scene import WarehouseScene
    stage = Usd.Stage.CreateInMemory()
    scene = WarehouseScene(stage, load_env=False).build()
    sc1, sc2 = sample_scenario(1), sample_scenario(2)
    scene.add_scenario(sc1)
    i2 = scene.add_scenario(sc2)
    scene.show_scenario(i2)
    def label(prim):
        v = [a.Get() for a in prim.GetAttributes() if a.GetName().startswith("semantics:labels")]
        return list(v[0]) if v else None
    for o in sc2.objects:
        prim = stage.GetPrimAtPath(scene.object_path(o, i2))
        assert prim, o.id
        if o.kind == "tool":                     # 공구는 묶음이 아니라 하나씩 종류 라벨
            assert label(prim) is None
            assert sorted(label(c)[0] for c in prim.GetChildren()) == sorted(o.tools)
        else:
            assert label(prim) == [o.cls]
    root = scene.object_path(sc2.objects[0], i2).rsplit("/", 1)[0]
    for z in sc2.zones:
        zp = stage.GetPrimAtPath(f"{root}/{z.id}_zone")
        names = sorted(label(c)[0] for c in zp.GetChildren() if label(c))
        assert names.count("cone") == len(z.params["cones"]) and names.count("danger_sign") == int(z.params["sign"])
    assert stage.GetPrimAtPath(scene.cam_path)


def _hand_pts(fingers, thumb):
    """가로로 내민 오른손 (손가락이 화면 왼쪽) 21점. fingers: 검지~새끼 편 여부, thumb: 엄지 편 여부."""
    p = np.zeros((21, 2))
    p[0] = (0, 0)
    for k, (y, on) in enumerate(zip((-30, -10, 10, 30), fingers)):
        m = 5 + 4 * k
        p[m] = (-80, y)
        p[m + 1:m + 4] = [(-120, y), (-145, y), (-165, y)] if on else [(-108, y + 5), (-96, y + 18), (-86, y + 22)]
    p[1:4] = [(-15, -25), (-35, -45), (-50, -60)]
    p[4] = (-60, -80) if thumb else (-76, 6)
    return p


def test_hand_count():
    from factory_safety.hand_count import GestureFilter, count_fingers
    cases = {1: ((1, 0, 0, 0), False), 2: ((1, 1, 0, 0), False), 3: ((1, 1, 1, 0), False)}
    for want, (fingers, thumb) in cases.items():
        assert count_fingers(_hand_pts(fingers, thumb)) == want, want
    assert count_fingers(_hand_pts((1, 0, 0, 1), False)) == 0          # 정해진 모양 아님
    assert count_fingers(_hand_pts((0, 0, 0, 0), False)) == 0          # 주먹
    assert count_fingers(_hand_pts((1, 1, 1, 1), False)) == 0          # 손가락 4개는 명령 아님
    assert count_fingers(_hand_pts((1, 1, 1, 1), True)) == 0           # 손가락 5개(엄지 폄)도 명령 아님
    # 화면에서 돌려도 (손을 세워도) 같은 수
    rot = np.array([[0, -1], [1, 0]])
    assert count_fingers(_hand_pts((1, 1, 0, 0), False) @ rot.T) == 2
    f = GestureFilter(need=3, cooldown=4.0)
    assert [f.update(0.1 * k, c) for k, c in enumerate([2, 2, 3, 3, 3, 3, 3])] == [0, 0, 0, 0, 3, 0, 0]
    assert [f.update(5.0 + 0.1 * k, 3) for k in range(4)] == [0, 0, 0, 0]        # 손을 계속 들고 있으면 다시 안 함
    assert f.update(5.5, 0) == 0 and [f.update(5.6 + 0.1 * k, 3) for k in range(3)] == [0, 0, 3]   # 내렸다 다시 올리면
    assert [f.update(6.0 + 0.1 * k, 4) for k in range(3)] == [0, 0, 0]          # 4초 안에는 다른 명령도 안 받음
    # 손이 움직이는 중 (올리는 중) 에는 같은 수가 이어져도 안 셈, 멈추면 셈
    f = GestureFilter(need=3)
    base = _hand_pts((1, 1, 0, 0), False)
    moving = [f.update(0.1 * k, 2, base + [0, 30 * k]) for k in range(5)]
    still = [f.update(0.5 + 0.1 * k, 2, base + [0, 150]) for k in range(4)]
    assert moving == [0] * 5 and still == [0, 0, 0, 2]           # 멈춘 뒤 3번 연속


def test_i18n_templates():
    from factory_safety import i18n
    langs = ("ko", "en", "zh", "ja")
    for table in (i18n.NAMES, i18n.INFO, i18n.TBM_ITEMS, i18n.ACTIONS, i18n.COMMANDS, i18n.DIRS, i18n.ZONE_KIND, i18n.T):
        for k, v in table.items():
            assert all(v.get(lg) for lg in langs), (k, v)
    for v in i18n.ZONES.values():
        assert all(v.get(lg) for lg in ("en", "zh", "ja"))
    for t in TOOL_TYPES:
        assert t in i18n.NAMES and t in i18n.INFO
    for c in STATES:
        assert c in i18n.NAMES and (c in i18n.INFO or KIND[c] == "tool")
    assert i18n.direction(math.pi / 2, (0, 0), (0, 3)) == "front" and i18n.direction(math.pi / 2, (0, 0), (-3, 0.5)) == "left"
    assert i18n.direction(math.pi / 2, (0, 0), (3, 0.5)) == "right" and i18n.direction(math.pi / 2, (0, 0), (0, -3)) == "back"
    assert i18n.date_text("2026-10-03", "en") == "October 3" and i18n.date_text("2026-10-03", "zh") == "10月3日"


def test_assistant_commands():
    from factory_safety.assistant import DemoScript, SiteAssistant, gesture_eval
    a = _agent()
    tbm = {"date": "2026-10-03", "work": ["work_receive"], "risks": ["risk_slip"], "rules": ["rule_ppe"],
           "todo": [{"cls": "spill", "zone": "남쪽 작업 구역"}]}
    s = SiteAssistant(a, langs=["zh", "en", "ja"], tbm=tbm)
    cam = CameraPose(pos=np.array([-4.5, -2.0, 1.38]), yaw=math.pi / 2, pitch=-0.2, vfov=70)
    dets = [("hammer", 0.85, _box_at(cam, [-4.5, 0.2, 0.0], 50, 22, flat=False), 1),
            ("spill", 0.8, _box_at(cam, [-3.6, 2.5, 0.0], 160, 40), 2)]
    for k in range(4):
        s.observe(0.1 * k, cam, dets)
        a.on_bodycam(0.1 * k, cam, dets, worker_xy=(-4.5, -2.3))
    ev = s.run(1.0, 1, cam, (-4.5, -2.3), math.pi / 2)            # 중국어: 장비 설명
    assert ev["lang"] == "zh" and "锤子" in ev["text"] and "망치" in ev["text_ko"] and ev["items"][0]["tool"] == "hammer"
    ev = s.run(1.1, 2, cam, (-4.5, -2.3), math.pi / 2, lang="en")    # 영어: 오늘의 TBM
    assert "south work area" in ev["text"] and "slippery floors" in ev["text"]
    ev = s.run(1.2, 2, cam, (-4.5, -2.3), math.pi / 2, lang="ja")    # 일본어: 오늘의 TBM
    assert ev["lang"] == "ja" and "南側作業エリア" in ev["text"] and "남쪽 작업 구역" in ev["text_ko"]
    ev = s.run(1.3, 3, cam, (-4.5, -2.3), math.pi / 2, lang="ko")    # 관리자 호출·SOS (안전팀 구분 없이 동급, 관리자에게만)
    assert "서쪽 통로" in ev["text"] and any(e["kind"] == "관리자 호출·SOS" for e in a.timeline)
    ev = s.run(2.0, 3, cam, (-4.5, -2.3), math.pi / 2, lang="en")
    assert "location" in ev["text"].lower() and "safety team" not in ev["text"].lower()
    assert any(e["kind"] == "관리자 호출·SOS" for e in a.timeline)
    s2 = SiteAssistant(_agent(), langs=["en"], tbm=tbm)
    # 운반 카트: 장비를 먼저 고름, 에이전트 대장에는 안 들어감
    cart = [("cart", 0.9, _box_at(cam, [-4.2, 0.5, 0.0], 90, 160, flat=False), 9)] + dets
    for k in range(3):
        s2.observe(5.0 + 0.1 * k, cam, cart)
        s2.agent.on_bodycam(5.0 + 0.1 * k, cam, cart, worker_xy=(-4.5, -2.3))
    s2.observe(4.9, cam, [("stack_unstable", 0.6, cart[0][2], 9)])       # 같은 추적 번호가 잠깐 적재로 보였어도 투표로 카트
    ev = s2.run(5.5, 1, cam, (-4.5, -2.3), math.pi / 2)
    assert "hand cart" in ev["text"] and ev["items"][0]["cls"] == "cart" and all(f.group != "cart" for f in s2.agent.findings)
    # 시연 순서와 인식 채점
    d = DemoScript()
    assert d.next(0.5, 0.0, a, cam, [], (-4.5, -2.3), True) is None and d.next(1.0, 0.0, a, cam, [], (-4.5, -2.3), True) == 2
    assert d.next(2.0, 0.1, a, cam, [], (-4.5, -2.3), True) is None       # 안내가 끝나기 전
    d.said(2.0, 3.0, 2)
    assert d.next(6.5, 0.1, a, cam, dets, (-4.5, -2.3), True) == 1          # 망치가 화면 가운데 2 m
    d.missed(7.0)
    assert d.next(8.5, 0.1, a, cam, dets, (-4.5, -2.3), True) == 1          # 인식이 안 되면 한 번 더
    d.missed(9.0)
    assert d.next(10.5, 0.7, a, cam, [], (-4.5, -2.3), True) == 3           # 두 번 안 되면 다음으로 (관리자 호출·SOS)
    g = gesture_eval([(1.0, 3), (6.5, 1), (20.0, 2)], [{"t": 1.5, "count": 3}, {"t": 7.0, "count": 2}, {"t": 40.0, "count": 1}])
    assert g == {"shown": 3, "recognized": 1, "wrong": 1, "missed": 1, "extra": 1}


def test_walker_gesture():
    w = PathWalker(seed=0)
    for _ in range(30):
        w.step(1 / 30)
    s0, ahead0 = w.s, np.linalg.norm(w.camera().pos[:2] - np.array(w.base_pose[:2]))
    w.start_gesture(3, hold=1.0)
    for _ in range(25):
        w.step(1 / 30)
    assert w.s == s0 and w.gesture_count == 3 and w.base_pose[3] > 30          # 멈춰 서서 손동작 클립
    cam = w.camera()
    assert np.linalg.norm(cam.pos[:2] - np.array(w.base_pose[:2])) < ahead0 - 0.1 and abs(cam.yaw - w.base_pose[2]) < 1e-6
    for _ in range(55):
        w.step(1 / 30)
    assert w.gesture is None and w.s > s0


def _arm_skeleton():
    """오른팔 + 손가락만 있는 T 자세 뼈대 (회전 없음, 부모 기준 위치)."""
    j = [("Hips", (0, 0, 0.95)), ("Hips/Spine", (0, 0, 0.45)), ("Hips/Spine/R_Clavicle", (-0.07, 0.03, 0.02)),
         ("Hips/Spine/R_Clavicle/R_Upperarm", (-0.14, 0.045, 0)), ("Hips/Spine/R_Clavicle/R_Upperarm/R_Forearm", (-0.284, 0, 0)),
         ("Hips/Spine/R_Clavicle/R_Upperarm/R_Forearm/R_Hand", (-0.216, 0, 0))]
    hand = "Hips/Spine/R_Clavicle/R_Upperarm/R_Forearm/R_Hand"
    for name, y in (("Index", -0.045), ("Mid", -0.02), ("Ring", 0.0), ("Pinky", 0.02)):
        j += [(f"{hand}/R_{name}1", (-0.10, y, 0)), (f"{hand}/R_{name}1/R_{name}2", (-0.045, 0, 0)),
              (f"{hand}/R_{name}1/R_{name}2/R_{name}3", (-0.03, 0, 0))]
    j += [(f"{hand}/R_Thumb1", (-0.02, -0.03, -0.005)), (f"{hand}/R_Thumb1/R_Thumb2", (-0.05, -0.05, 0)),
          (f"{hand}/R_Thumb1/R_Thumb2/R_Thumb3", (-0.03, -0.01, 0))]
    rest = []
    for _, t in j:
        m = np.eye(4)
        m[:3, 3] = t
        rest.append(m)
    return [n for n, _ in j], rest


def test_gesture_rig():
    from factory_safety.walk_anim import GESTURE_CAM_AHEAD, Rig, _quat, _rot_between
    r = _rot_between(np.array([1.0, 0, 0]), np.array([-1.0, 0, 0]))       # 정반대 방향도 제대로
    assert np.allclose(r @ [1, 0, 0], [-1, 0, 0]) and np.isclose(np.linalg.det(r), 1.0)
    assert np.allclose(_quat(np.eye(3)), [1, 0, 0, 0])
    joints, rest = _arm_skeleton()
    rig = Rig(joints, rest)
    for c in range(1, 6):
        _, world = rig.solve(*rig.gesture(c))
        pos = {n: world[i][:3, 3] for i, n in enumerate(rig.names)}
        assert pos["R_Hand"][1] < -GESTURE_CAM_AHEAD - 0.1                  # 손은 바디캠 앞
        assert pos["R_Index3"][2] - pos["R_Index1"][2] > 0.04               # 검지는 펴서 위로
        for k, name in enumerate(("Mid", "Ring", "Pinky")):
            reach = np.linalg.norm(pos[f"R_{name}3"] - pos["R_Hand"])
            assert (reach > 0.16) == (c >= k + 2 or c == 5), (c, name, reach)
        thumb_out = np.linalg.norm(pos["R_Thumb3"] - pos["R_Ring1"])
        assert (thumb_out > 0.07) == (c == 5), (c, thumb_out)


def test_story():
    """시연 이야기: TBM(2) → 카트 보고 설명(1) → 상자 4개 싣고 끌기 → 한 바퀴 뒤 관리자 호출·SOS(3) → 끝."""
    from factory_safety.story import FOLLOW_M, N_BOXES, Story

    class FakeScene:
        @staticmethod
        def cart_slot(k, box="box_c"):
            return (0.0, -0.30, 0.03 + 0.25 * k)

    w = PathWalker(seed=0)
    st = Story(w)
    st.scene = FakeScene()
    t, dt, states = 0.0, 1 / 30, []
    while not st.done and t < 200:
        t += dt
        w.step(dt)
        st.update(t)
        if w.gesture_count and (not st.pending or st.pending[0] not in st.heard) and t - w.gesture["t0"] > 1.0:
            st.said(t, 6.0, w.gesture["count"])          # 비서가 알아듣고 6초 안내했다고 침
        if not states or states[-1] != st.state:
            states.append(st.state)
        if st.state == "pull" and "attach" in states:
            x, y = w.path.point_at(w.s)
            assert abs(math.hypot(st.cart[0] - x, st.cart[1] - y) - FOLLOW_M) < 0.3      # 카트는 작업자 뒤
    assert st.done and st.lap_done and st.loaded == N_BOXES and w.pulling
    assert [c for _, c in st.shown] == [2, 1, 3]
    assert states[:8] == ["start", "tbm", "walk", "look", "look_ask", "load", "attach", "pull"]
    assert 60 < t < 110          # 시연 길이 (초)


def test_agent_equipment_track():
    """카트를 돌아보다 잠깐 적재로 잡혀 대장에 오른 것은, 같은 추적 번호가 운반 카트로 보이면 뺀다."""
    a = _agent()
    cam = CameraPose(pos=np.array([-4.5, -2.0, 1.38]), yaw=math.pi / 2, pitch=-0.2, vfov=70)
    box = _box_at(cam, [-4.2, 0.5, 0.0], 90, 160, flat=False)
    for k in range(3):
        a.on_bodycam(0.1 * k, cam, [("stack_unstable", 0.7, box, 5)], worker_xy=(-4.5, -2.3))
    f = next(f for f in a.findings if f.group == "stack")
    assert f.status == "확정"
    for k in range(3):
        a.on_bodycam(0.5 + 0.1 * k, cam, [("cart", 0.9, box, 5)], worker_xy=(-4.5, -2.3))
    assert f.status == "기각" and 5 in a.equip_tids
    a.on_bodycam(1.0, cam, [("stack_unstable", 0.7, box, 5)], worker_xy=(-4.5, -2.3))
    assert all(x.status == "기각" for x in a.findings if x.group == "stack")


def test_assistant_llm_planner():
    """LLM 결정 (가짜 planner) 을 받아 말할 물체를 고르고, 관리자 메시지에 사실 + AI 요약, 기록에 LLM 판단을 남긴다."""
    from factory_safety.assistant import SiteAssistant
    a = _agent()
    tbm = {"date": "2026-10-03", "work": ["work_move_boxes"], "risks": ["risk_slip"], "rules": ["rule_ppe"], "todo": []}
    s = SiteAssistant(a, langs=["en"], tbm=tbm)
    seen = []

    def planner(snap):
        seen.append(snap)
        if snap["count"] == 1:      # 규칙이라면 카트를 고르지만 LLM 은 망치를 고름
            hammer = next(o["id"] for o in snap["view"] if o["name"] == "a hammer")
            return {"llm": True, "model": "fake", "say_ids": [hammer], "reason_ko": "망치가 바닥에 있어 위험", "trace": ["look_around()"]}
        return {"llm": True, "model": "fake", "say_ids": [], "reason_ko": "호출", "manager_ko": "서쪽 통로 작업자 호출", "trace": []}
    s.planner = planner
    cam = CameraPose(pos=np.array([-4.5, -2.0, 1.38]), yaw=math.pi / 2, pitch=-0.2, vfov=70)
    dets = [("cart", 0.9, _box_at(cam, [-4.4, 0.0, 0.0], 90, 160, flat=False), 9),
            ("hammer", 0.85, _box_at(cam, [-3.9, 0.6, 0.0], 50, 22, flat=False), 1)]
    for k in range(3):
        s.observe(0.1 * k, cam, dets)
        a.on_bodycam(0.1 * k, cam, dets, worker_xy=(-4.5, -2.3))
    ev = s.run(1.0, 1, cam, (-4.5, -2.3), math.pi / 2)
    assert "hammer" in ev["text"] and ev["llm"]["llm"] and {o["kind"] for o in seen[0]["view"]} >= {"equipment", "tool"}
    ev = s.run(2.0, 3, cam, (-4.5, -2.3), math.pi / 2)
    assert "작업자 위치 서쪽 통로" in ev["manager_ko"] and "AI 요약: 서쪽 통로 작업자 호출" in ev["manager_ko"]
    assert sum(e["kind"] == "LLM 판단" for e in a.timeline) == 2
    s.planner = lambda snap: {"llm": False, "error": "꺼짐"}             # LLM 이 안 되면 규칙
    ev = s.run(3.0, 1, cam, (-4.5, -2.3), math.pi / 2)
    assert "hand cart" in ev["text"] and not ev["llm"]["llm"]


def test_assistant_equip_refuse():
    """손동작 1 에서 LLM 이 매뉴얼 근거를 못 찾으면(refuse=true) 추측하지 않고 거부 + 관리자 호출로 넘긴다."""
    from factory_safety.assistant import SiteAssistant
    a = _agent()
    s = SiteAssistant(a, langs=["en"])
    cam = CameraPose(pos=np.array([-4.5, -2.0, 1.38]), yaw=math.pi / 2, pitch=-0.2, vfov=70)
    dets = [("hammer", 0.85, _box_at(cam, [-3.9, 0.6, 0.0], 50, 22, flat=False), 1)]
    for k in range(3):
        s.observe(0.1 * k, cam, dets)
        a.on_bodycam(0.1 * k, cam, dets, worker_xy=(-4.5, -2.3))

    def planner(snap):
        return {"llm": True, "model": "fake", "say_ids": [], "reason_ko": "매뉴얼 근거 없음", "manager_ko": "", "refuse": True,
               "trace": ["retrieve_manual(query=hammer)"]}
    s.planner = planner
    ev = s.run(1.0, 1, cam, (-4.5, -2.3), math.pi / 2)
    assert ev["llm"]["refuse"] and "관리자" in ev["notify"]
    assert "매뉴얼" in ev["text_ko"] or "검증된" in ev["text_ko"]
    assert "작업자 위치" in ev["manager_ko"]


def test_manuals_load_chunks():
    """data/manuals/*.md 가 프런트매터 + ## 섹션으로 쪼개지는지 (임베딩·벡터스토어 없이, 순수 파싱만)."""
    from factory_safety.config import CLASSES, EQUIPMENT
    from factory_safety.manuals import load_chunks
    chunks = load_chunks()
    assert len(chunks) >= 6                 # 문서 6개 이상 (컨베이어 추가로 7개), 섹션은 여러 개
    doc_ids = {c["doc_id"] for c in chunks}
    assert doc_ids == {"hand_tools", "hand_cart", "stacking", "spill", "fire_extinguisher", "danger_zone", "conveyor"}
    for c in chunks:
        assert c["text"] and c["title"] and c["source"]
        assert isinstance(c["applies_to"], list) and c["applies_to"]
        # machine_conveyor 는 YOLO 클래스가 아니라 끼임 위험 기계용 의사(pseudo) 클래스 (19클래스 모델은 재학습 안 함)
        assert all(t in CLASSES or t in EQUIPMENT or t == "machine_conveyor" for t in c["applies_to"])
        assert c["id"] == f"{c['doc_id']}#{c['section']}"
    # 공구 8종, 카트, 유출/적재/소화기/위험구역 상태가 모두 어느 문서에 걸려 있어야 한다
    covered = {t for c in chunks for t in c["applies_to"]}
    assert covered >= {"hammer", "screwdriver", "saw", "power_saw", "pickaxe", "shovel", "wrench", "drill", "cart",
                       "spill", "spill_marked", "stack_unstable", "stack_stable", "ext_ok", "ext_fallen", "ext_blocked",
                       "cone", "danger_sign"}


def test_pinch_unproject_roundtrip():
    """Projector.unproject 는 project() 의 정확한 역변환이어야 한다 (depth = 카메라 정면 축 거리)."""
    from factory_safety import pinch_detect as PD
    cam_pos = np.array([5.0, 10.0, 1.4])
    true_pt = np.array([5.3, 13.2, 0.8])
    d = true_pt - cam_pos
    yaw = math.atan2(d[1], d[0])
    pitch = math.atan2(d[2], math.hypot(d[0], d[1]))
    cam = CameraPose(pos=cam_pos, yaw=yaw, pitch=pitch, vfov=66.0)
    P = Projector(960, 540)
    P.set_pose(cam)
    uv, z = P.project(true_pt[None])
    u, v = uv[0]
    depth = np.full((540, 960), np.nan, dtype=np.float32)
    depth[int(round(v)), int(round(u))] = z[0]
    pts21 = [[0, 0]] * 8 + [[u, v]] + [[0, 0]] * 12       # 인덱스 8 = 검지 끝
    got = PD.fingertip_xyz(P, pts21, depth)
    assert np.linalg.norm(got - true_pt) < 1e-4
    xyxy = [u - 5, v - 5, u + 5, v + 5]
    got2 = PD.pinch_point_xyz(P, xyxy, depth)
    assert np.linalg.norm(got2 - true_pt) < 1e-4
    assert PD.fingertip_xyz(P, None, depth) is None                 # 손 없음
    assert PD._sample(P, 2, 2, depth) is None                       # 그 픽셀에 깊이 없음


def test_machine_zone_on_off():
    """기계가 on 이면 끼임점 반경 MACHINE_ZONE_M 위험구역, off 면 즉시 해제."""
    from factory_safety.config import MACHINE_ZONE_M
    a = _agent()
    assert a.zones == {}
    a.machines.set("M1", True)
    assert list(a.zones.keys()) == ["machine:M1"]
    z = a.zones["machine:M1"]
    assert z.source == "machine" and abs(z.radius - MACHINE_ZONE_M) < 1e-6
    pinch = a.machines.pinch_xyz("M1")
    assert np.linalg.norm(z.center - pinch[:2]) < 1e-6
    a.machines.set("M1", False)
    assert a.zones == {}                                             # 꺼지면 바로 해제
    a.machines.set("M1", False)                                      # 같은 상태 재호출은 아무 일 없음
    assert a.zones == {}


def test_machine_zone_voice_warning():
    """작업자가 작동 중인 기계의 1.5 m 위험구역 안에 들어오면 기존 음성 경고 규칙대로 강한 경고."""
    a = _agent()
    heard = []
    a.on_voice = lambda t, level: heard.append((t, level))
    a.machines.set("M1", True)
    pinch = a.machines.pinch_xyz("M1")
    cam = CameraPose(pos=np.array([pinch[0], pinch[1] - 3.0, 1.4]), yaw=math.pi / 2, pitch=0.0, vfov=66)
    a.on_bodycam(1.0, cam, [], worker_xy=(pinch[0], pinch[1] - 0.5))   # 영역(반경 1.5 m) 안
    assert heard and a.voice_events[-1]["target"] == "Z:" + a.zones["machine:M1"].zid
    assert a.machine_report()[0]["zone_alerts"] == 1
    a.on_bodycam(2.0, cam, [], worker_xy=(pinch[0], pinch[1] - 10.0))  # 멀리 있으면 경고 없음
    assert a.machine_report()[0]["zone_alerts"] == 1


def test_pinch_alert_requires_on_and_10cm():
    """손가락 끝 - 끼임점 10 cm 이내 + 작동 중이면 최고 등급 경보. 꺼져 있으면 아무리 가까워도 경보 없음."""
    a = _agent()
    pinch = a.machines.pinch_xyz("M1")
    near = pinch + np.array([0.03, 0.0, 0.02])    # 3.6 cm
    far = pinch + np.array([0.5, 0.0, 0.0])       # 50 cm

    assert a.check_pinch(0.0, "M1", near) is None          # 기계가 꺼져 있음 -> 경보 없음
    a.machines.set("M1", True)
    assert a.check_pinch(1.0, "M1", None) is None          # 손 없음
    assert a.check_pinch(1.0, "M1", far) is None            # 멀리 있음
    ev = a.check_pinch(2.0, "M1", near)
    assert ev is not None and ev["level"] == "critical" and ev["say"] == "손 빼세요" and ev["dist_cm"] < 10.0
    assert a.pinch_events == [ev]
    assert a.machine_report()[0]["pinch_alerts"] == 1
    a.machines.set("M1", False)
    assert a.check_pinch(3.0, "M1", near) is None           # 꺼지면 아무리 가까워도 경보 없음
    assert a.machine_report()[0]["pinch_alerts"] == 1       # 늘지 않음


def test_pinch_fallback_logs_and_counts():
    """끼임점 탐지 실패(detected_pinch_xyz=None) 시 등록된 위치로 폴백 계산하고, 폴백 사용을 로그와 집계에 남긴다."""
    a = _agent()
    a.machines.set("M1", True)
    pinch = a.machines.pinch_xyz("M1")
    near = pinch + np.array([0.0, 0.03, 0.0])
    ev = a.check_pinch(1.0, "M1", near, detected_pinch_xyz=None)      # 탐지 실패 -> 폴백
    assert ev is not None and ev["fallback"] is True
    assert any("폴백" in e["text"] for e in a.timeline if e["kind"] == "끼임 경보")
    assert a.machine_report()[0]["pinch_fallback"] == 1
    # 탐지가 됐으면(위치를 직접 줌) 폴백이 아님
    ev2 = a.check_pinch(2.0, "M1", near, detected_pinch_xyz=pinch)
    assert ev2["fallback"] is False
    assert a.machine_report()[0]["pinch_fallback"] == 1                # 더 안 늘어남


def test_forklift_zone_tracks_moving_position():
    """지게차는 기계처럼 on/off 가 아니라 계속 움직인다: update_forklift 가 매번 위험구역 중심을 다시 그리고,
    active=False 면 즉시 해제한다."""
    from factory_safety.config import FORKLIFT_ZONE_M
    a = _agent()
    assert a.zones == {}
    a.update_forklift(1.0, "F1", (3.0, 4.0), True)
    assert list(a.zones.keys()) == ["forklift:F1"]
    z = a.zones["forklift:F1"]
    assert z.source == "forklift" and abs(z.radius - FORKLIFT_ZONE_M) < 1e-6
    assert np.linalg.norm(z.center - np.array([3.0, 4.0])) < 1e-6
    a.update_forklift(1.2, "F1", (3.5, 4.0), True)          # 움직임: 같은 영역, 중심만 갱신
    assert list(a.zones.keys()) == ["forklift:F1"]
    assert np.linalg.norm(a.zones["forklift:F1"].center - np.array([3.5, 4.0])) < 1e-6
    a.update_forklift(1.4, "F1", (10.0, 4.0), False)        # 지나감: 즉시 해제
    assert a.zones == {}


def test_forklift_proximity_warns_and_notifies_manager():
    """작업자가 지게차 위험구역에 들어오면 기존 경고 규칙대로 강한 경고가 나고, 관리자 알림 피드에도 쌓인다."""
    a = _agent()
    heard = []
    a.on_voice = lambda t, level: heard.append((t, level))
    a.update_forklift(1.0, "F1", (0.0, 0.0), True)
    cam = CameraPose(pos=np.array([0.0, -2.0, 1.4]), yaw=math.pi / 2, pitch=0.0, vfov=66)
    a.on_bodycam(1.0, cam, [], worker_xy=(0.0, -0.5))        # 위험구역 안
    assert heard and heard[-1][1] == "hazard"
    assert any(n["source"] == "F1" and n["level"] == "critical" for n in a.manager_notifications)
    assert a.forklift_report() == [{"id": "F1", "zone_alerts": 1}]


def test_manager_notifications_feed_pinch_and_caution():
    """끼임 경보와 '주의' 분류는 관리자 알림 피드에도 자동으로 쌓인다 (작업자가 직접 호출하지 않아도)."""
    a = _agent()
    a.machines.set("M1", True)
    pinch = a.machines.pinch_xyz("M1")
    a.check_pinch(1.0, "M1", pinch + np.array([0.02, 0.0, 0.0]))
    assert any(n["level"] == "critical" and n["source"] == "M1" for n in a.manager_notifications)

    cam = CameraPose(pos=np.array([-4.5, -2.0, 1.38]), yaw=math.pi / 2, pitch=-0.2, vfov=70)
    box = _box_at(cam, [-5.6, 3.0, 0.0], 40, 20, flat=False)
    a.on_bodycam(3.0, cam, [("tool_floor", 0.5, box, 11)])
    a.on_bodycam(4.0, cam, [])                                # 1프레임만 보이고 사라짐 -> 주의
    assert any(n["level"] == "caution" for n in a.manager_notifications)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"통과  {name}")
