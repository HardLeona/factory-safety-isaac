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


def _perfect_yolo(det, sc, cam):
    """완벽한 YOLO 흉내: 화면에 보이고 가려지지 않은 물체의 박스. 압력계는 가까이, 정면에서만 읽힌다."""
    from factory_safety.config import CLASSES
    dets = []
    for i, h in enumerate(sc.hazards):
        if det.rects[i] is None or not det.los[i]:
            continue
        r = det.rects[i][:4]
        if h.type == "ext":
            dets.append(("extinguisher", 0.7, r))
            if det.in_view[i] and float(np.linalg.norm(h.center - cam.pos)) < h.read_range:
                dets.append((CLASSES[h.gauge_cls], 0.8, r))
        elif det.in_view[i]:
            dets.append((CLASSES[h.cls], 0.8, r))
    return dets


def _run_judge(seed, laps=2, focus=False):
    """focus=True 면 로봇이 정답(SimDetector) 대신 YOLO 판단 결과만 보고 고개를 돌린다."""
    from factory_safety.yolo_judge import YoloJudge
    sc = sample_scenario(seed)
    det = _detector(sc)
    judge = YoloJudge(*L.map_boxes_3d(L.build_static(0)))
    robot = RobotPatrol()
    dt, t = 1 / 30, 0.0
    for k in range(int(laps * robot.path.length / robot.speed / dt)):
        cam = robot.step(dt, judge.focus_view(t) if focus else det)
        det.update(dt, cam)
        t += dt
        if k % 6 == 0:
            judge.update(t, cam, _perfect_yolo(det, sc, cam))
    return sc, judge


def test_yolo_judge_with_perfect_boxes():
    """화면에 보이는 박스만 넣어주면 위치 추정과 중복 제거로 전부 한 번씩 찾아야 한다."""
    for seed in (0, 1, 2):
        sc, judge = _run_judge(seed)
        s = judge.score(sc.hazards)
        assert s["found"] == s["total"], [h.label for h in s["missed"]]
        assert s["ext_checked"] >= s["ext_total"] - 1
        assert s["false_alarms"] == 0
        # 같은 물체를 여러 번 세지 않아야 한다
        assert len([tr for tr in judge.confirmed if tr.group != "gauge"]) == s["total"] - sum(
            1 for h in sc.hazards if h.type == "ext" and not h.ok)


def test_yolo_focus_closed_loop():
    """정답 없이 화면 판단 결과로만 고개를 돌려도 두 바퀴 안에 전부 찾아야 한다."""
    for seed in (0, 1, 2):
        sc, judge = _run_judge(seed, focus=True)
        s = judge.score(sc.hazards)
        assert s["found"] == s["total"], [h.label for h in s["missed"]]
        assert s["ext_checked"] == s["ext_total"]
        assert s["false_alarms"] == 0


def test_gauge_reader():
    """줌 화면 압력계 판독: 비스듬히, 회전, 반전, 흐림, 노이즈, 흰 벽/빨간 몸통 배경에서도 틀리면 안 된다 (보류는 허용)."""
    import tempfile
    from PIL import Image, ImageDraw, ImageFilter
    from factory_safety.gauge_reader import read_gauge
    from factory_safety.textures import gauge_texture
    rng = np.random.default_rng(0)
    tmp = tempfile.mkdtemp()
    backgrounds = [(222, 224, 226), (205, 40, 40), (50, 54, 60), (240, 190, 20), (150, 150, 150)]
    right = wrong = 0
    n = 120
    for k in range(n):
        ok = bool(k % 2)
        g = Image.open(gauge_texture(os.path.join(tmp, f"g{k}.png"), ok, seed=k)).convert("RGB")
        size = int(rng.uniform(70, 140))
        g = g.resize((max(8, int(size * rng.uniform(0.35, 1.0))), size), Image.BILINEAR)
        g = g.rotate(rng.uniform(-12, 12), expand=True, fillcolor=(43, 47, 52))
        if rng.random() < 0.5:
            g = g.transpose(Image.FLIP_LEFT_RIGHT)
        bg = Image.new("RGB", (256, 256), backgrounds[int(rng.integers(len(backgrounds)))])
        if rng.random() < 0.7:   # 소화기 몸통
            x0 = int(rng.uniform(40, 140))
            ImageDraw.Draw(bg).rectangle([x0, 0, x0 + int(rng.uniform(60, 120)), 256], fill=(205, 40, 40))
        bg.paste(g, (128 - g.width // 2 + int(rng.uniform(-25, 25)), 128 - g.height // 2 + int(rng.uniform(-25, 25))))
        a = np.asarray(bg.filter(ImageFilter.GaussianBlur(rng.uniform(0, 1.5)))).astype(float)
        a = np.clip(a * rng.uniform(0.7, 1.15) + rng.normal(0, rng.uniform(0, 6), a.shape), 0, 255).astype(np.uint8)
        name, _ = read_gauge(a)
        right += name == ("gauge_normal" if ok else "gauge_low")
        wrong += name not in (None, "gauge_normal" if ok else "gauge_low")
    assert wrong == 0
    assert right >= 0.85 * n


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
    assert stage.GetPrimAtPath(scene.hazard_root + "/H00_puddle/main")
    assert not stage.GetPrimAtPath("/World/Hazards")      # 이전 배치는 지워짐
    assert stage.GetPrimAtPath(scene.cam_path)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"통과  {name}")
