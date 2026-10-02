"""USD 장면: NVIDIA 실사 창고 + 위험/안전 물체 + 걷는 작업자 + 바디캠 + CCTV.

OpenUSD(pxr)만 쓴다. Isaac Sim 안에서도, usd-core 만 깔린 일반 파이썬에서도 만들 수 있다
(일반 파이썬에서는 원격 에셋을 못 불러와서 창고 안 소화기 위치는 warehouse.EXT_MOUNTS 값을 씀).

물체 하나 = Xform 묶음 하나, 그 묶음에 의미 라벨(클래스)을 붙인다.
Replicator 의 bounding_box_2d 는 라벨이 붙은 묶음 전체(유출+표지판, 소화기+앞을 막은 상자 등)를 감싼다.
"""
import math

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, UsdSkel

from . import warehouse as W
from .config import ASSETS, IMG_H, IMG_W, WAREHOUSE_URL
from .walk_anim import FPS, build_walk_animation

H_APERTURE = 20.955  # mm (USD 기본값)
# 에셋 원점 높이 보정 (원점이 가운데인 YCB 공구는 반 높이만큼 올려야 바닥에 놓임)
TOOL_LIFT = {"drill": 0.03, "clamp": 0.019, "scissors": 0.009, "wood": 0.045}
BOX_H = {"box_a": 0.5, "box_b": 0.5}
BOX_GRID = {"box_b": [(-0.27, -0.25), (0.27, -0.25), (-0.27, 0.25), (0.27, 0.25)],
            "box_a": [(0.0, -0.25), (0.0, 0.25)]}


def default_labeler(prim, cls_name):
    """의미 라벨 작성 (Isaac Sim 5.0 이상 UsdSemantics)."""
    try:
        from pxr import UsdSemantics
        api = UsdSemantics.LabelsAPI.Apply(prim, "class")
        api.CreateLabelsAttr().Set([cls_name])
        return True
    except Exception:
        return False


def strip_semantics(root_prim):
    """root 아래 (참조 안쪽 포함) 의미 라벨을 지운다: 적용된 라벨 스키마를 빼고 라벨 속성 값을 막는다."""
    for p in Usd.PrimRange(root_prim):
        for schema in p.GetAppliedSchemas():
            if "Semantics" in schema:
                p.RemoveAppliedSchema(schema)
        for a in p.GetAttributes():
            if a.GetName().startswith("semantic"):
                a.Block()


class WarehouseScene:
    def __init__(self, stage, labeler=None, set_time=None, load_env=True):
        """set_time: 걷기 애니메이션 시각을 바꾸는 함수 (Isaac 에서는 타임라인). 없으면 애니메이션 정지."""
        self.stage = stage
        self.labeler = labeler or default_labeler
        self.set_time = set_time
        self.load_env = load_env
        self._ops = {}
        self._bank = []
        self.scenario = None
        self.cam_path = "/World/BodyCam"
        self.cctv_paths = {}
        self.ext_mats = []

    # ------------------------------------------------------------ 기본 구조
    def build(self):
        st = self.stage
        UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(st, 1.0)
        st.SetTimeCodesPerSecond(FPS)
        st.SetStartTimeCode(0)
        st.SetEndTimeCode(1_000_000)
        UsdGeom.Xform.Define(st, "/World")
        if self.load_env:
            st.DefinePrim("/World/Env", "Xform").GetReferences().AddReference(WAREHOUSE_URL)
        self._read_env_extinguishers()
        fill = UsdLux.DistantLight.Define(st, "/World/Fill")
        fill.CreateIntensityAttr(0.0)
        fill.CreateAngleAttr(2.0)
        UsdGeom.Xformable(fill).AddRotateXYZOp().Set(Gf.Vec3f(40, 0, 30))
        self.fill = fill
        self._materials()
        for i, (x, y, yaw) in enumerate(W.TABLES):
            t = self._ref(f"/World/Fixed/Table{i}", ASSETS["table"], (x, y, 0.0), yaw=yaw)
            # 작업대 위에 원래 놓인 부품 상자는 치운다 (공구 자리)
            for p in Usd.PrimRange(t.GetPrim()):
                if p.GetName().lower().startswith("container"):
                    p.SetActive(False)
        self._build_worker()
        self.add_camera(self.cam_path)
        for name, *_ in W.CCTVS:
            self.cctv_paths[name] = self.add_camera(f"/World/CCTV/{name}")
        return self

    def _read_env_extinguishers(self):
        """창고에 원래 있던 소화기의 자리(행렬)를 읽고 끈다. 못 읽으면 EXT_MOUNTS 로 만든다."""
        st = self.stage
        self.ext_mats = []
        for (x, y, (nx, ny)), p in zip(W.EXT_MOUNTS, W.ENV_EXTINGUISHERS):
            prim = st.GetPrimAtPath("/World/Env" + p[len("/Root"):]) if self.load_env else None
            if prim and prim.IsValid():
                m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                prim.SetActive(False)
            else:
                yaw = math.degrees(math.atan2(ny, nx)) - 90.0
                m = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), yaw)) * Gf.Matrix4d().SetTranslate(Gf.Vec3d(x, y, 1.0))
            self.ext_mats.append(Gf.Matrix4d(m))

    def _materials(self):
        self.mat = {
            "oil": self._preview("/World/Looks/SpillOil", (0.018, 0.015, 0.010), 0.04, 1.0),
            "water": self._preview("/World/Looks/SpillWater", (0.33, 0.40, 0.46), 0.02, 0.45),
        }

    def _preview(self, path, color, rough, opacity=1.0):
        m = UsdShade.Material.Define(self.stage, path)
        s = UsdShade.Shader.Define(self.stage, path + "/PBR")
        s.CreateIdAttr("UsdPreviewSurface")
        s.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        s.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(rough))
        s.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        s.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(opacity))
        m.CreateSurfaceOutput().ConnectToSource(s.ConnectableAPI(), "surface")
        return m

    def _ref(self, path, url, pos=(0, 0, 0), yaw=0.0, rot=(0, 0, 0), matrix=None, strip_labels=True):
        """에셋을 참조로 놓는다. yaw 는 라디안 (Z 축), rot 은 도 (X, Y, Z 순서, yaw 보다 먼저 적용).
        strip_labels: NVIDIA 창고 소품에는 예전 형식 라벨(box, pallet, sign 등)이 들어 있어서, 그대로 두면
        묶음 라벨과 섞여 부품마다 정답 박스가 따로 생긴다. 물체로 쓰는 에셋은 그 라벨을 지운다."""
        x = UsdGeom.Xform.Define(self.stage, path)
        if matrix is not None:
            x.AddTransformOp().Set(matrix)
        else:
            x.AddTranslateOp().Set(Gf.Vec3d(*map(float, pos)))
            x.AddRotateZOp().Set(math.degrees(yaw))
            if any(rot):
                x.AddRotateXYZOp().Set(Gf.Vec3f(*map(float, rot)))
        ref = self.stage.DefinePrim(path + "/a", "Xform")
        ref.GetReferences().AddReference(url)
        if strip_labels:
            strip_semantics(ref)
        return x

    # ------------------------------------------------------------ 작업자
    def _build_worker(self):
        st = self.stage
        w = UsdGeom.Xform.Define(st, "/World/Worker")
        self._ops["worker"] = (w.AddTranslateOp(), w.AddRotateZOp())
        st.DefinePrim("/World/Worker/char", "Xform").GetReferences().AddReference(ASSETS["worker"])
        self.labeler(w.GetPrim(), "worker")
        skel = [p for p in Usd.PrimRange(w.GetPrim()) if p.IsA(UsdSkel.Skeleton)]
        self.has_walk = False
        if skel:
            build_walk_animation(st, skel[0], "/World/Worker/WalkAnim")
            self.has_walk = True
        self.set_worker(0.0, -7.6, math.pi / 2, 0.0)

    def set_worker(self, x, y, yaw, anim_t=0.0, visible=True):
        t_op, r_op = self._ops["worker"]
        t_op.Set(Gf.Vec3d(float(x), float(y), 0.0))
        r_op.Set(math.degrees(yaw) + 90.0)          # 캐릭터는 원래 -Y 를 봄
        self._set_visible("/World/Worker", visible)
        if self.set_time and self.has_walk:
            self.set_time(anim_t)

    # ------------------------------------------------------------ 카메라
    def add_camera(self, path):
        cam = UsdGeom.Camera.Define(self.stage, path)
        cam.CreateHorizontalApertureAttr(H_APERTURE)
        cam.CreateVerticalApertureAttr(H_APERTURE * IMG_H / IMG_W)
        cam.CreateFocalLengthAttr(15.0)
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 300.0))
        self._ops[path] = UsdGeom.Xformable(cam).AddTransformOp()
        return path

    def set_camera(self, pose, width=IMG_W, height=IMG_H, path=None):
        path = path or self.cam_path
        cam = UsdGeom.Camera(self.stage.GetPrimAtPath(path))
        v_ap = H_APERTURE * height / width
        cam.GetVerticalApertureAttr().Set(v_ap)
        cam.GetFocalLengthAttr().Set((v_ap / 2.0) / math.tan(math.radians(pose.vfov) / 2.0))
        self._ops[path].Set(Gf.Matrix4d(*[float(v) for v in pose.usd_matrix().flatten()]))

    def randomize_lighting(self, rng):
        """보조 조명 세기와 색 (학습 데이터 다양화). 창고 자체 조명은 그대로."""
        self.fill.GetIntensityAttr().Set(float(rng.uniform(0, 2.5)))
        k = rng.uniform(0.85, 1.0)
        self.fill.CreateColorAttr(Gf.Vec3f(1.0, float(k), float(k * rng.uniform(0.85, 1.0))))

    # ------------------------------------------------------------ 시나리오
    def add_scenario(self, scenario):
        """배치를 하나 더 만들어 쌓아 둔다. Isaac Sim 6.0 Replicator 는 첫 렌더 뒤에 새로 만든 라벨 물체를
        제대로 못 읽어서, 배치는 첫 렌더 전에 전부 만들고 show_scenario 로 보이기/숨기기만 바꾼다."""
        root = f"/World/Scn{len(self._bank):03d}"
        UsdGeom.Xform.Define(self.stage, root)
        for o in scenario.objects:
            self._build_object(root, o)
        self._bank.append((scenario, root))
        return len(self._bank) - 1

    def set_scenario(self, scenario):
        """배치 하나만 쓸 때 (첫 렌더 전에 부를 것)."""
        self.show_scenario(self.add_scenario(scenario))

    def show_scenario(self, idx):
        for j, (_, root) in enumerate(self._bank):
            self._set_visible(root, j == idx)
        self.scenario = self._bank[idx][0]

    def object_path(self, o, idx=None):
        root = self._bank[idx if idx is not None else self._current()][1]
        return f"{root}/{o.id}_{o.cls}"

    def _current(self):
        return next(i for i, (sc, _) in enumerate(self._bank) if sc is self.scenario)

    def _build_object(self, root, o):
        g = UsdGeom.Xform.Define(self.stage, f"{root}/{o.id}_{o.cls}")
        if o.kind != "ext":
            g.AddTranslateOp().Set(Gf.Vec3d(o.x, o.y, 0.0))
            g.AddRotateZOp().Set(math.degrees(o.yaw))
        base = str(g.GetPath())
        getattr(self, f"_build_{o.kind}")(base, o)
        self.labeler(g.GetPrim(), o.cls)

    def _build_spill(self, base, o):
        p = o.params
        r = p["radius"]
        self._puddle(base + "/Puddle", r, p["fluid"], p["shape_seed"])
        src = p["source"]
        if o.cls == "spill":
            # 쏟아진 통이 옆에 쓰러져 있음
            if src != "none":
                self._ref(base + "/Source", ASSETS[src], (r * 1.25, 0.1, {"bucket": 0.31, "bottle": 0.08, "barrel": 0.3}[src]),
                          rot=(90, 0, 70))
        else:
            # 조치됨: 미끄럼 주의 표지판 + 라바콘, 통은 세워 둠
            self._ref(base + "/Sign", ASSETS["wet_sign"], (-r - 0.25, 0.35, 0.0), yaw=0.4)
            for k in range(p["n_cones"]):
                a = math.pi * (0.2 + 0.9 * k)
                self._ref(base + f"/Cone{k}", ASSETS["cone"], ((r + 0.35) * math.cos(a) * 1.3, (r + 0.35) * math.sin(a), 0.0))
            if src != "none":
                self._ref(base + "/Source", ASSETS[src], (r * 1.3, -0.4, 0.0))

    def _puddle(self, path, radius, fluid, seed):
        rng = np.random.default_rng(seed)
        n = 56
        ang = np.linspace(0, 2 * math.pi, n, endpoint=False)
        rr = radius * (1 + 0.32 * np.sin(ang * 3 + rng.uniform(0, 6)) * rng.uniform(0.4, 1)
                       + 0.18 * np.sin(ang * 5 + rng.uniform(0, 6)) + 0.08 * np.sin(ang * 9 + rng.uniform(0, 6)))
        rr *= np.where(rng.random(n) < 0.12, rng.uniform(1.1, 1.45, n), 1.0)
        stretch = rng.uniform(1.0, 1.6)
        pts = [Gf.Vec3f(0, 0, 0.004)] + [Gf.Vec3f(float(r * math.cos(a) * stretch), float(r * math.sin(a)), 0.004)
                                          for a, r in zip(ang, rr)]
        idx = []
        for i in range(n):
            idx += [0, 1 + i, 1 + (i + 1) % n]
        mesh = UsdGeom.Mesh.Define(self.stage, path)
        mesh.CreatePointsAttr(pts)
        mesh.CreateFaceVertexCountsAttr([3] * n)
        mesh.CreateFaceVertexIndicesAttr(idx)
        mesh.CreateNormalsAttr([Gf.Vec3f(0, 0, 1)] * len(pts))
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(self.mat[fluid])

    def _build_tool(self, base, o):
        z0 = W.TABLE_TOP if o.cls == "tool_stored" else 0.0
        for k, (kind, dx, dy, yaw) in enumerate(o.params["items"]):
            rot = (0, 0, 0) if kind != "wood" else (90, 0, 0)
            lift = TOOL_LIFT[kind] if kind != "wood" else 0.1
            self._ref(f"{base}/T{k}_{kind}", ASSETS[kind], (dx, dy, z0 + lift), yaw=yaw, rot=rot)

    def _build_stack(self, base, o):
        p = o.params
        self._ref(base + "/Pallet", ASSETS["pallet"], (0, 0, 0))
        rng = np.random.default_rng(p["tilt_seed"])
        box, h = p["box"], BOX_H[p["box"]]
        for L in range(p["layers"]):
            z = 0.21 + L * h
            sx = p["shift"] * L
            for k, (bx, by) in enumerate(BOX_GRID[box]):
                if o.cls == "stack_unstable" and L == p["layers"] - 1:
                    # 맨 위 층: 한쪽으로 밀려나 기울어짐 (팔레트 밖으로 삐져나옴)
                    tilt = p["tilt"] * rng.uniform(0.6, 1.0)
                    self._ref(f"{base}/B{L}{k}", ASSETS[box], (bx + sx + 0.05 * rng.uniform(-1, 1), by + 0.05 * rng.uniform(-1, 1), z + 0.04),
                              yaw=rng.uniform(-0.25, 0.25), rot=(rng.uniform(-3, 3), tilt, 0))
                else:
                    self._ref(f"{base}/B{L}{k}", ASSETS[box], (bx + sx, by, z), yaw=rng.uniform(-0.03, 0.03) if o.cls == "stack_stable" else rng.uniform(-0.15, 0.15))
        if p.get("fallen_box"):
            self._ref(base + "/Fallen", ASSETS[box], (0.95, rng.uniform(-0.3, 0.3), 0.25), rot=(90, 0, rng.uniform(0, 90)))

    def _build_ext(self, base, o):
        p = o.params
        m = self.ext_mats[p["mount"]]
        nx, ny = p["out"]
        mx, my = W.EXT_MOUNTS[p["mount"]][:2]
        if o.cls == "ext_fallen":
            x, y = mx + nx * W.EXT_FALL_OUT, my + ny * W.EXT_FALL_OUT
            self._ref(base + "/Ext", ASSETS["extinguisher"], (x, y, 0.09), yaw=math.atan2(ny, nx) + p["fall_yaw"], rot=(90, 0, 0))
            return
        self._ref(base + "/Ext", ASSETS["extinguisher"], matrix=m)
        if o.cls == "ext_blocked":
            bx, by = mx + nx * W.EXT_BLOCK_OUT, my + ny * W.EXT_BLOCK_OUT
            box = p["block_box"]
            for L in range(p["block_layers"]):
                self._ref(f"{base}/Block{L}", ASSETS[box], (bx, by, L * BOX_H[box]), yaw=math.atan2(ny, nx) + 0.1 * (L % 2))

    def _set_visible(self, path, visible):
        prim = self.stage.GetPrimAtPath(path)
        if prim:
            im = UsdGeom.Imageable(prim)
            im.MakeVisible() if visible else im.MakeInvisible()

    def save(self, path):
        self.stage.GetRootLayer().Export(path)
