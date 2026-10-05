"""Isaac Sim 버전 차이를 흡수하는 도우미.

Isaac Sim은 버전마다 모듈 이름이 바뀌어 왔다 (omni.isaac.* -> isaacsim.* -> isaacsim.core.experimental.*).
여기서 여러 이름을 차례로 시도해서 6.x 기준으로 맞추고, 4.5 / 5.x 에서도 최대한 돌아가게 했다.
이 파일의 함수들은 반드시 SimulationApp을 만든 뒤에 호출해야 한다.
"""
import math

import numpy as np


def make_app(headless=False, width=1600, height=900, renderer=None):
    """SimulationApp 생성. 다른 omni / isaacsim 모듈보다 먼저 호출해야 한다."""
    try:
        from isaacsim import SimulationApp
    except ImportError:
        from omni.isaac.kit import SimulationApp  # Isaac Sim 4.x 이전
    cfg = {"headless": headless, "width": width, "height": height}
    if renderer:
        cfg["renderer"] = renderer
    return SimulationApp(cfg)


def new_stage():
    import omni.usd
    ctx = omni.usd.get_context()
    ctx.new_stage()
    return ctx.get_stage()


def isaac_labeler(prim, cls_name):
    """Isaac Sim 버전에 맞는 방식으로 의미 라벨을 붙인다."""
    try:
        from isaacsim.core.experimental.utils.semantics import add_labels  # 6.x
        add_labels(prim, labels=[cls_name], taxonomy="class")
        return True
    except Exception:
        pass
    try:
        from isaacsim.core.utils.semantics import add_labels  # 5.x
        add_labels(prim, labels=[cls_name], instance_name="class")
        return True
    except Exception:
        pass
    try:
        from isaacsim.core.utils.semantics import add_update_semantics  # 4.5
        add_update_semantics(prim, semantic_label=cls_name, type_label="class")
        return True
    except Exception:
        pass
    from .scene import default_labeler
    return default_labeler(prim, cls_name)


def set_viewport_camera(camera_path):
    try:
        from omni.kit.viewport.utility import get_active_viewport
        vp = get_active_viewport()
        if vp is not None:
            vp.camera_path = camera_path
            return True
    except Exception as e:
        print(f"[안내] 뷰포트 카메라를 바꾸지 못했어요: {e}")
    return False


def set_viewport_top_view(stage):
    """관제용으로 창고를 위에서 비스듬히 내려다보는 카메라를 만들고 뷰포트에 연결."""
    from pxr import Gf, UsdGeom
    from .geometry import CameraPose
    path = "/World/OverviewCam"
    cam = UsdGeom.Camera.Define(stage, path)
    cam.CreateFocalLengthAttr(18.0)
    cam.CreateHorizontalApertureAttr(20.955)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.5, 400.0))
    # 천장(약 9 m) 아래, 남쪽 벽 위에서 북쪽을 내려다봄
    pose = CameraPose(pos=np.array([0.0, -11.5, 8.2]), yaw=math.pi / 2, pitch=-math.radians(38), vfov=62)
    m = pose.usd_matrix()
    xf = UsdGeom.Xformable(cam)
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(Gf.Matrix4d(*[float(v) for v in m.flatten()]))
    set_viewport_camera(path)
    return path


class DebugBoxes:
    """뷰포트에 3D 선으로 탐지 박스를 그린다 (디버그 드로잉 확장이 없으면 조용히 건너뜀)."""

    def __init__(self):
        self.draw = None
        for mod in ("isaacsim.util.debug_draw", "omni.isaac.debug_draw"):
            try:
                m = __import__(mod, fromlist=["_debug_draw"])
                self.draw = m._debug_draw.acquire_debug_draw_interface()
                break
            except Exception:
                continue
        if self.draw is None:
            _enable_extension("isaacsim.util.debug_draw")
            try:
                from isaacsim.util.debug_draw import _debug_draw
                self.draw = _debug_draw.acquire_debug_draw_interface()
            except Exception:
                print("[안내] 디버그 드로잉을 쓸 수 없어서 3D 박스 표시는 생략해요.")

    def show(self, boxes):
        """boxes: [(mn, mx, rgba), ...]"""
        if self.draw is None:
            return
        self.draw.clear_lines()
        starts, ends, colors, sizes = [], [], [], []
        for mn, mx, rgba in boxes:
            c = [(x, y, z) for x in (mn[0], mx[0]) for y in (mn[1], mx[1]) for z in (mn[2], mx[2])]
            for a, b in [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]:
                starts.append(c[a])
                ends.append(c[b])
                colors.append(tuple(rgba))
                sizes.append(3.0)
        if starts:
            self.draw.draw_lines(starts, ends, colors, sizes)


def _enable_extension(name):
    try:
        import omni.kit.app
        omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate(name, True)
    except Exception:
        pass


def get_annotator(name):
    import omni.replicator.core as rep
    try:
        return rep.annotators.get(name)
    except Exception:
        return rep.AnnotatorRegistry.get_annotator(name)


def attach(annotator, render_product):
    try:
        annotator.attach(render_product)
    except Exception:
        annotator.attach([render_product])


def disable_capture_on_play():
    import omni.replicator.core as rep
    try:
        rep.orchestrator.set_capture_on_play(False)
    except Exception:
        import carb.settings
        carb.settings.get_settings().set("/omni/replicator/captureOnPlay", False)


def parse_bboxes(data, class_names, with_paths=False):
    """bounding_box_2d_tight 결과 -> [(class_id, x0, y0, x1, y1, occlusion)].
    with_paths=True 면 끝에 라벨이 붙은 물체 경로(prim path)도 붙인다 (정답표와 맞출 때)."""
    if data is None:
        return []
    arr = data.get("data") if isinstance(data, dict) else data
    info = data.get("info", {}) if isinstance(data, dict) else {}
    id_to_labels = info.get("idToLabels", {}) or {}
    paths = list(info.get("primPaths", []) or [])
    out = []
    if arr is None or len(arr) == 0:
        return out
    for k, row in enumerate(arr):
        sid = str(int(row["semanticId"]))
        lab = id_to_labels.get(sid, id_to_labels.get(int(sid), ""))
        if isinstance(lab, dict):
            lab = lab.get("class", "")
        names = [s.strip() for s in str(lab).split(",")]
        cls = next((class_names.index(n) for n in names if n in class_names), None)
        if cls is None:
            continue
        occ = float(row["occlusionRatio"]) if "occlusionRatio" in row.dtype.names else 0.0
        box = (cls, float(row["x_min"]), float(row["y_min"]), float(row["x_max"]), float(row["y_max"]), occ)
        if with_paths:
            box = box + (str(paths[k]) if k < len(paths) else "",)
        out.append(box)
    return out


def rgb_array(data):
    """rgb 어노테이터 결과 -> (H, W, 3) uint8 numpy."""
    if isinstance(data, dict):
        data = data.get("data")
    a = np.asarray(data)
    if a.ndim == 3 and a.shape[2] >= 3:
        return a[..., :3].astype(np.uint8)
    return None


def depth_array(data):
    """distance_to_image_plane 어노테이터 결과 -> (H, W) float32 (카메라 정면 축 방향 미터 거리).
    geometry.Projector.unproject 와 짝 (같은 깊이 정의라 좌표가 바로 맞음)."""
    if isinstance(data, dict):
        data = data.get("data")
    if data is None:
        return None
    return np.asarray(data, dtype=np.float32).reshape(np.asarray(data).shape[:2])


def timeline_setter():
    """걷기 애니메이션용: 타임라인 시각(초)을 바꾸는 함수. 타임라인은 재생하지 않고 시각만 맞춘다."""
    import omni.timeline
    tl = omni.timeline.get_timeline_interface()
    return tl.set_current_time
