"""Isaac Sim 없이 돌아가는 핵심 로직 테스트.

    python -m pytest tests -q        (또는 python tests/test_core.py)
"""
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from factory_safety import warehouse as W  # noqa: E402
from factory_safety.config import CLASSES, HAZARD, KIND, STATES, TOOL_TYPES  # noqa: E402
from factory_safety.geometry import CameraPose, Projector  # noqa: E402
from factory_safety.scenario import sample_scenario  # noqa: E402
from factory_safety.walker import PathWalker, cctv_pose  # noqa: E402


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
    cam = cctv_pose("cctv_west")
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


def test_cctv_proximity():
    from factory_safety.inspection import CCTVProximity
    cam = cctv_pose("cctv_west")
    P = Projector(960, 540)
    P.set_pose(cam)

    def box_at(x, y, h=40, w=16):
        (u, v) = P.project(np.array([[x, y, 0.0]]))[0][0]
        return [u - w / 2, v - h, u + w / 2, v]

    prox = CCTVProximity(960, 540)
    dets = [("worker", 0.9, box_at(-4.5, 0.0, 120, 40), None), ("tool_floor", 0.8, box_at(-4.5, 1.2), None)]
    events, meas = prox.update(10.0, "cctv_west", cam, dets)
    assert len(events) == 1 and abs(events[0][3] - 1.2) < 0.1
    events2, _ = prox.update(11.0, "cctv_west", cam, dets)
    assert events2 == []                                      # 5초 안 반복 경고 없음
    far = [("worker", 0.9, box_at(-4.5, -5.0, 120, 40), None), ("tool_floor", 0.8, box_at(-4.5, 1.2), None)]
    assert prox.update(30.0, "cctv_west", cam, far)[0] == []
    # 작업자 박스 안에 들어간 작은 "유출" 은 발밑 그림자 오인으로 보고 뺀다
    wb = box_at(-4.5, 0.0, 120, 40)
    shadow = [("worker", 0.9, wb, None), ("spill", 0.7, [wb[0] + 5, wb[3] - 12, wb[2] - 5, wb[3]], None)]
    assert prox.update(60.0, "cctv_west", cam, shadow)[1]["pairs"] == []


def test_cctv_event_scoring():
    """작업자가 위험물 옆을 지나가는 동안 경고가 나면 사건 1건 잡음, 먼 곳 경고는 오경보."""
    from types import SimpleNamespace
    from factory_safety.inspection import CCTVProximity
    hz = SimpleNamespace(id="O01", cls="spill", x=0.0, y=0.0)
    prox = CCTVProximity(960, 540)
    for k in range(40):
        t = k * 0.2
        wy = -4.0 + 0.2 * k                      # 작업자가 (0.5, -4) -> (0.5, 4) 로 지나감
        d = math.hypot(0.5, wy)
        pairs = [((0.5, wy), "spill", (0.1, 0.0), d)] if d < 3.0 else []
        if k == 2:
            pairs.append(((0.5, wy), "spill", (6.0, 6.0), 1.0))   # 엉뚱한 곳 경고 (오경보)
        meas = {"workers": [np.array([0.5, wy])], "pairs": [(np.array(a), n, np.array(b), dd) for a, n, b, dd in pairs]}
        prox.score_frame(t, "cctv_west", meas, (0.5, wy), [hz], {"O01"})
    s = prox.report()
    assert s["events"] == 1 and s["events_visible"] == 1 and s["events_detected"] == 1
    assert s["false_alert_episodes"] == 1
    assert s["worker_err_median"] < 1e-6


def _agent():
    from factory_safety.agent import SafetyAgent
    return SafetyAgent(960, 540, [c[0] for c in W.CCTVS], log=None)


def _box_at(cam, xyz, w=60, h=30, flat=True):
    P = Projector(960, 540)
    P.set_pose(cam)
    uv, _ = P.project(np.array([xyz], float))
    u, v = uv[0]
    return [u - w / 2, v - h / 2, u + w / 2, v + h / 2] if flat else [u - w / 2, v - h, u + w / 2, v]


def _shoot(agent, t, cls=None, conf=0.8):
    """확대 촬영 한 장: 목표점 자리에 cls 박스를 돌려준다 (cls 가 None 이면 아무것도 못 찾음)."""
    name, pose = agent.next_ptz(t)
    dets = [(cls, conf, _box_at(pose, agent.job["target"], 120, 80), None)] if cls else []
    agent.on_ptz(t, dets)
    return name


def test_agent_plan_and_cameras():
    a = _agent()
    a.plan(59.0)
    opts = {cp.cid: [n for n, _ in a.camera_options(cp.target)] for cp in a.checkpoints}
    assert "cctv_east" in opts["C1"] and "cctv_west" in opts["C2"]      # 서쪽 랙 북쪽 끝 / 남쪽 끝 소화기
    assert all(opts[c] for c in ("C7", "C8"))                          # 작업대 2곳은 남쪽 CCTV 로 보임
    # 랙(6 m)이 가리면 후보에서 빠진다: 서쪽 통로 바닥은 동쪽 CCTV 에서 안 보임
    assert "cctv_east" not in [n for n, _ in a.camera_options(np.array([-5.9, 0.5, 0.0]))]
    # 대장에 있는 적재물이 시야를 가리면 그 CCTV 도 뺀다 (동쪽 랙 남쪽 끝 소화기를 서쪽 CCTV 에서 볼 때)
    from factory_safety.agent import Finding
    c6 = next(c for c in a.checkpoints if c.cid == "C6")
    assert "cctv_west" in [n for n, _ in a.camera_options(c6.target)]
    st = Finding("F09", "stack", np.array([6.85, -5.22]), 0.0, "바디캠")
    st.body["stack_stable"] += 3.0
    a.findings.append(st)
    why = {}
    assert "cctv_west" not in [n for n, _ in a.camera_options(c6.target, why=why)] and why == {"cctv_west": "F09"}
    pose = a.aim("cctv_west", np.array([-5.9, 0.5, 0.0]), 4.0)
    P = Projector(960, 540)
    P.set_pose(pose)
    (u, v), = P.project(np.array([[-5.9, 0.5, 0.0]]))[0]
    assert abs(u - 480) < 1 and abs(v - 270) < 1 and pose.vfov < 20


def test_agent_tentative_recheck():
    a = _agent()
    cam = CameraPose(pos=np.array([-4.5, -2.0, 1.38]), yaw=math.pi / 2, pitch=-0.2, vfov=70)
    a.on_bodycam(1.0, cam, [("tool_floor", 0.5, _box_at(cam, [-5.6, 3.0, 0.0], 40, 20, flat=False), 7)])
    a.on_bodycam(2.0, cam, [])                                   # 1프레임만 보이고 사라짐
    f = a.findings[0]
    assert f.status == "재확인 대기" and len(a.jobs) == 1 and abs(f.xy[1] - 3.0) < 0.3
    assert a.next_ptz(2.5) is None                                # 잠깐 기다림 (바디캠이 다시 볼 수도 있음)
    _shoot(a, 4.5, "tool_floor", 0.9)
    _shoot(a, 4.7, "tool_floor", 0.85)
    assert f.status == "재확인 완료" and f.cls == "tool_floor" and a.job is None
    rep = a.report()
    assert rep["findings"][0]["state"] == "위험" and "작업대로 회수" in rep["findings"][0]["action"]


def test_agent_retry_and_escalate():
    a = _agent()
    cam = CameraPose(pos=np.array([4.5, 8.0, 1.38]), yaw=-math.pi / 2, pitch=-0.2, vfov=70)
    a.on_bodycam(1.0, cam, [("spill", 0.35, _box_at(cam, [3.1, 2.2, 0.0]), 3)])          # 근거 약함
    a.on_bodycam(1.2, cam, [("spill", 0.7, _box_at(cam, [5.9, 9.4 - 6.0, 0.0]), 4),
                            ("spill", 0.7, _box_at(cam, [5.9, 9.4 - 6.0, 0.0]), 4)])     # 2프레임 0.7
    a.on_bodycam(3.0, cam, [])
    weak, strong = a.findings
    shots = []
    while a.busy() and len(shots) < 20:
        shots.append(_shoot(a, 10.0 + len(shots)))
    assert weak.status == "기각" and strong.status == "현장 확인 필요"
    # 나중에 바디캠이 다시 보고 확정하면 현장 확인은 취소
    box = _box_at(cam, [5.9, 9.4 - 6.0, 0.0])
    for k in range(3):
        a.on_bodycam(40.0 + 0.2 * k, cam, [("spill", 0.8, box, 9)])
    assert strong.status == "확정"
    assert a.stats["recheck_retry"] >= 1 or len(set(shots)) == 1     # 볼 수 있는 CCTV 가 여럿이면 다음 CCTV 로
    rep = a.report()
    assert [r["state"] for r in rep["findings"]] == ["위험"]


def test_agent_cancel_and_checkpoints():
    a = _agent()
    cam = CameraPose(pos=np.array([-4.5, -6.0, 1.38]), yaw=math.pi / 2, pitch=-0.2, vfov=70)
    box = _box_at(cam, [-3.1, 7.8, 0.0])
    a.on_bodycam(1.0, cam, [("spill_marked", 0.6, box, 1)])
    a.on_bodycam(2.0, cam, [])                                    # 잠깐 보임 -> 재확인 예약
    for k in range(3):                                           # 다른 추적 번호로 다시 보고 확정
        a.on_bodycam(2.5 + 0.2 * k, cam, [("spill_marked", 0.8, box, 2)])
    assert a.findings[0].status == "확정" and not a.jobs and a.stats["recheck_canceled"] == 1
    a.finish_patrol(60.0)
    assert len(a.jobs) == len(a.checkpoints)                     # 순찰 중 점검 지점을 하나도 못 봄
    n = 0
    while a.busy() and n < 60:
        job = a.job or a.jobs[0]
        want = "ext_ok" if (job["checkpoint"] and job["checkpoint"].group == "ext") else "tool_stored"
        _shoot(a, 61.0 + n, want)
        n += 1
    rep = a.report()
    assert rep["summary"]["checkpoints_done"] == len(a.checkpoints)
    assert sum(1 for r in rep["findings"] if r["class"] == "ext_ok") == len(W.EXT_MOUNTS)


def test_agent_table_check_picks_target():
    """작업대 확대 화면에 옆 바닥 공구가 같이 보여도 작업대 위 공구를 고르고, 바닥 공구 항목은 안 건드림."""
    from factory_safety.agent import Finding
    a = _agent()
    floor = Finding("F01", "tool_floor", np.array([-6.16, -9.71]), 0.0, "바디캠")
    floor.body["tool_floor"] += 3.0
    floor.body_confirmed = True
    a.findings.append(floor)
    a.finish_patrol(50.0)
    a.jobs = [j for j in a.jobs if j["checkpoint"].cid == "C7"]
    for k in range(2):
        name, pose = a.next_ptz(51.0 + k)
        P = Projector(960, 540)
        P.set_pose(pose)
        (uf, vf), = P.project(np.array([[-6.16, -9.71, 0.1]]))[0]
        (ut, vt), = P.project(np.array([a.job["target"]]))[0]
        a.on_ptz(51.0 + k, [("tool_floor", 0.95, [uf - 30, vf - 20, uf + 30, vf + 20], None),
                            ("tool_stored", 0.7, [ut - 80, vt - 40, ut + 80, vt + 40], None)])
    table = [f for f in a.findings if f.group == "tool_stored"]
    assert len(table) == 1 and table[0].cls == "tool_stored" and np.allclose(floor.xy, [-6.16, -9.71])


def test_agent_ptz_relocates_and_merges():
    """바디캠 위치가 3 m 넘게 틀린 적재물도 확대 화면에서 구한 위치로 고쳐 원래 항목과 합친다."""
    from factory_safety.agent import Finding
    a = _agent()
    true = Finding("F01", "stack", np.array([7.2, -5.6]), 0.0, "바디캠")
    true.body["stack_unstable"] += 3.0
    true.body_confirmed = True
    off = Finding("F02", "stack", np.array([4.0, -7.6]), 0.0, "바디캠", status="재확인 대기")
    off.body["stack_unstable"] += 0.9
    a.findings += [true, off]
    a.PTZ_WINDOWS, a.MATCH_FRAC = (9.0, 7.0), 0.9    # 진짜 자리까지 화면에 들어오고 짝지어지게 넓게
    a._request(1.0, off, None, "잠깐 보임", "잠깐 보임")
    for k in range(2):
        name, pose = a.next_ptz(5.0 + k)
        P = Projector(960, 540)
        P.set_pose(pose)
        (u, v), = P.project(np.array([[7.2, -5.6, 0.0]]))[0]
        assert 0 < u < 960 and 0 < v < 540
        a.on_ptz(5.0 + k, [("stack_unstable", 0.9, [u - 60, v - 120, u + 60, v], None)])
    assert off.status == "병합" and off.merged_into == "F01" and true.n_ptz == 2


def test_agent_near_miss_priority_and_eval():
    from factory_safety.agent import Finding
    a = _agent()
    for fid, cls, xy in (("F01", "tool_floor", (-7.3, 4.6)), ("F02", "spill", (3.1, 2.2)), ("F03", "stack_stable", (7.2, -5.6))):
        f = Finding(fid, cls if KIND[cls] == "tool" else KIND[cls], np.array(xy, float), 0.0, "바디캠")
        f.body[cls] += 3.0
        f.n_body, f.body_confirmed = 3, True
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
    # 확대 판정 확신이 낮으면 다른 CCTV 로 한 번 더, 마지막까지 낮으면 현장 확인
    a = _agent()
    a.finish_patrol(60.0)
    job = a.jobs[[j["checkpoint"].cid for j in a.jobs].index("C4")]
    a.jobs = [job]
    n_cams = len(job["cams"])
    for k in range(2 * n_cams):
        _shoot(a, 61.0 + k, "ext_blocked", 0.34)
    cp = next(c for c in a.checkpoints if c.cid == "C4")
    assert n_cams >= 2 and cp.status == "현장 확인 필요" and a.stats["recheck_retry"] == n_cams - 1


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
    a.on_voice = heard.append
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
    test_agent_tentative_recheck()
    path = os.path.join(tempfile.mkdtemp(), "d.html")
    write_dashboard(path, a.report(), seed=1)
    assert "조치 목록" in open(path, encoding="utf-8").read()


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
    assert stage.GetPrimAtPath(scene.cam_path) and len(scene.cctv_paths) == len(W.CCTVS)


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
    cases = {1: ((1, 0, 0, 0), False), 2: ((1, 1, 0, 0), False), 3: ((1, 1, 1, 0), False), 4: ((1, 1, 1, 1), False),
             5: ((1, 1, 1, 1), True)}
    for want, (fingers, thumb) in cases.items():
        assert count_fingers(_hand_pts(fingers, thumb)) == want, want
    assert count_fingers(_hand_pts((1, 0, 0, 1), False)) == 0          # 정해진 모양 아님
    assert count_fingers(_hand_pts((0, 0, 0, 0), False)) == 0          # 주먹
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
    for table in (i18n.NAMES, i18n.INFO, i18n.TBM_ITEMS, i18n.ACTIONS, i18n.COMMANDS, i18n.DIRS, i18n.ZONE_KIND, i18n.CAMS, i18n.T):
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
    ev = s.run(1.0, 1, cam, (-4.5, -2.3), math.pi / 2)            # 중국어
    assert ev["lang"] == "zh" and "锤子" in ev["text"] and "망치" in ev["text_ko"] and ev["items"][0]["tool"] == "hammer"
    ev = s.run(1.1, 2, cam, (-4.5, -2.3), math.pi / 2)            # 영어: 공장 전체 스캔 (위험물 대장)
    assert ev["lang"] == "en" and ev["n_hazards"] == 2 and "spill" in ev["text"] and "hammer" in ev["text"] and "west aisle" in ev["text"]
    assert s.ptz_view(1.2)[2].startswith("스캔") and s.ptz_view(30.0) is None
    ev = s.run(1.2, 3, cam, (-4.5, -2.3), math.pi / 2)            # 일본어
    assert ev["lang"] == "ja" and "TBM" in ev["text"] and "南側作業エリア" in ev["text"] and "남쪽 작업 구역" in ev["text_ko"]
    ev = s.run(1.3, 4, cam, (-4.5, -2.3), math.pi / 2, lang="ko")
    assert "서쪽 통로" in ev["text"] and any(e["kind"] == "관리자 호출" for e in a.timeline)
    ev = s.run(2.0, 5, cam, (-4.5, -2.3), math.pi / 2, lang="en")
    assert ev["cctv"] and s.sos_view(3.0)[0] == ev["cctv"] and s.sos_view(20.0) is None
    assert any(e["kind"] == "SOS" for e in a.timeline)
    s2 = SiteAssistant(_agent(), langs=["en"], tbm=tbm)
    assert "No hazards" in s2.run(0.0, 2, cam, (-4.5, -2.3), math.pi / 2)["text"]
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
    assert d.next(0.5, 0.0, a, cam, [], (-4.5, -2.3), True) is None and d.next(1.0, 0.0, a, cam, [], (-4.5, -2.3), True) == 3
    assert d.next(2.0, 0.1, a, cam, [], (-4.5, -2.3), True) is None       # 안내가 끝나기 전
    d.said(2.0, 3.0, 3)
    assert d.next(6.5, 0.1, a, cam, dets, (-4.5, -2.3), True) == 1          # 망치가 화면 가운데 2 m
    d.missed(7.0)
    assert d.next(8.5, 0.1, a, cam, dets, (-4.5, -2.3), True) == 1          # 인식이 안 되면 한 번 더
    d.missed(9.0)
    assert d.next(10.5, 0.7, a, cam, [], (-4.5, -2.3), True) == 2           # 두 번 안 되면 다음으로
    g = gesture_eval([(1.0, 3), (6.5, 1), (20.0, 2)], [{"t": 1.5, "count": 3}, {"t": 7.0, "count": 4}, {"t": 40.0, "count": 5}])
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
    """시연 이야기: TBM(3) → 카트 보고 설명(1) → 상자 4개 싣고 끌기 → 공장 스캔(2) → 한 바퀴 뒤 관리자 호출(4) → 끝."""
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
    assert [c for _, c in st.shown] == [3, 1, 2, 4]
    assert states[:8] == ["start", "tbm", "walk", "look", "look_ask", "load", "attach", "pull"] and "scan" in states
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
    ev = s.run(2.0, 4, cam, (-4.5, -2.3), math.pi / 2)
    assert "작업자 위치 서쪽 통로" in ev["manager_ko"] and "AI 요약: 서쪽 통로 작업자 호출" in ev["manager_ko"]
    assert sum(e["kind"] == "LLM 판단" for e in a.timeline) == 2
    s.planner = lambda snap: {"llm": False, "error": "꺼짐"}             # LLM 이 안 되면 규칙
    ev = s.run(3.0, 1, cam, (-4.5, -2.3), math.pi / 2)
    assert "hand cart" in ev["text"] and not ev["llm"]["llm"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"통과  {name}")
