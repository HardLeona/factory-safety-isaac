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
from factory_safety.config import CLASSES, HAZARD, KIND  # noqa: E402
from factory_safety.geometry import CameraPose, Projector  # noqa: E402
from factory_safety.scenario import sample_scenario  # noqa: E402
from factory_safety.walker import PathWalker, cctv_pose  # noqa: E402


def test_scenario_and_answer_key():
    kinds_seen = set()
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
            assert o.cls in CLASSES and HAZARD[o.cls] == o.hazard
        assert cnt["spill"] == 3 and cnt["stack"] == 4 and cnt["ext"] == len(W.EXT_MOUNTS)
        assert 2 + len(W.TABLES) <= cnt["tool"] <= 3 + len(W.TABLES)
        assert all(set(o) >= {"id", "class", "hazard", "x", "y", "zone"} for o in key["objects"])
    # 30 개 시나리오에서 위험/안전 상태가 모두 나와야 한다
    assert kinds_seen == set(CLASSES) - {"worker"}


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
    hz = next(o for o in sc.objects if o.hazard)
    sf = next(o for o in sc.objects if not o.hazard)
    path = lambda o: f"/World/Scn000/{o.id}_{o.cls}"   # noqa: E731
    gt = [(hz.cls_id, 100, 100, 200, 200, 0.0, path(hz)), (sf.cls_id, 400, 100, 500, 220, 0.0, path(sf))]
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
    safe_twin = next(c for c in CLASSES if KIND.get(c) == hz.kind and c != "worker" and not HAZARD[c])
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
    for o in sc2.objects:
        prim = stage.GetPrimAtPath(scene.object_path(o, i2))
        assert prim, o.id
        labels = [a.Get() for a in prim.GetAttributes() if a.GetName().startswith("semantics:labels")]
        assert labels and list(labels[0]) == [o.cls]
    assert stage.GetPrimAtPath(scene.cam_path) and len(scene.cctv_paths) == len(W.CCTVS)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"통과  {name}")
