"""Isaac Sim 없이 돌아가는 핵심 로직 테스트.

    python -m pytest tests -q        (또는 python tests/test_core.py)
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from factory_safety import layout as L  # noqa: E402
from factory_safety.detector import SimDetector  # noqa: E402
from factory_safety.patrol import BodycamWalker, ClosedPath, RobotPatrol  # noqa: E402
from factory_safety.scenario import sample_scenario  # noqa: E402


def _detector(sc):
    mins, maxs = L.occluder_arrays(L.build_static(0))
    ex = sc.occluder_boxes()
    if ex:
        mins = np.vstack([mins, [b[0] for b in ex]])
        maxs = np.vstack([maxs, [b[1] for b in ex]])
    return SimDetector(sc.hazards, mins, maxs)


def test_layout_and_scenario():
    models = L.build_static(0)
    assert len(L.occluder_arrays(models)[0]) > 100
    for seed in range(20):
        sc = sample_scenario(seed)
        kinds = [h.type for h in sc.hazards]
        assert kinds.count("puddle") == 2 and kinds.count("tool") == 2 and kinds.count("stack") == 2
        assert kinds.count("ext") == 8
        assert 1 <= sum(1 for h in sc.hazards if h.type == "ext" and not h.ok) <= 2
        for h in sc.hazards:
            assert np.all(h.aabb_max > h.aabb_min)


def test_path_closed():
    p = ClosedPath()
    assert 100 < p.length < 120
    assert np.linalg.norm(p.point_at(0.0) - p.point_at(1.0)) < 1e-6


def test_robot_patrol_finds_everything():
    """능동 추적 로봇은 두 바퀴 안에 모든 위험 요소와 소화기를 찾아야 한다."""
    for seed in (0, 1, 2):
        sc = sample_scenario(seed)
        det = _detector(sc)
        robot = RobotPatrol()
        dt = 1 / 30
        for _ in range(int(2 * robot.path.length / robot.speed / dt)):
            det.update(dt, robot.step(dt, det))
        assert det.detected.all(), [h.label for i, h in enumerate(sc.hazards) if not det.detected[i]]


def test_bodycam_runs():
    sc = sample_scenario(5)
    det = _detector(sc)
    w = BodycamWalker(seed=5)
    for _ in range(600):
        cam = w.step(1 / 30)
        det.update(1 / 30, cam)
        assert 1.2 < cam.pos[2] < 1.5


def test_rl_env():
    from gymnasium.utils.env_checker import check_env
    from factory_safety.rl_env import FactoryPatrolEnv, PathFollower
    env = FactoryPatrolEnv(max_steps=1500)
    check_env(env, skip_render_check=True)
    obs, info = env.reset(seed=3)
    pol = PathFollower(env)
    done = False
    while not done:
        obs, r, term, trunc, info = env.step(pol(obs))
        done = term or trunc
    assert term, "기준 정책이 150초 안에 전부 찾아야 한다"
    assert env.collisions == 0


def test_usd_build(tmp_path=None):
    try:
        from pxr import Usd
    except ImportError:
        print("  (pxr 없음: USD 테스트 건너뜀, pip install usd-core 로 가능)")
        return
    from factory_safety.usd_scene import FactoryStage
    tex = str(tmp_path) if tmp_path else os.path.join(ROOT, "outputs", "textures")
    stage = Usd.Stage.CreateInMemory()
    scene = FactoryStage(stage, tex).build()
    scene.set_scenario(sample_scenario(1))
    scene.set_scenario(sample_scenario(2))
    assert stage.GetPrimAtPath("/World/Hazards/H00_puddle/main")
    assert stage.GetPrimAtPath(scene.cam_path)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"통과  {name}")
