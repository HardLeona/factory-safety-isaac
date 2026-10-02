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
