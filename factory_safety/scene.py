"""USD 장면: NVIDIA 실사 창고 + 위험/안전 물체 + 걷는 작업자 + 바디캠.

OpenUSD(pxr)만 쓴다. Isaac Sim 안에서도, usd-core 만 깔린 일반 파이썬에서도 만들 수 있다
(일반 파이썬에서는 원격 에셋을 못 불러와서 창고 안 소화기 위치는 warehouse.EXT_MOUNTS 값을 씀).

물체 하나 = Xform 묶음 하나, 그 묶음에 의미 라벨(클래스)을 붙인다.
Replicator 의 bounding_box_2d 는 라벨이 붙은 묶음 전체(유출+표지판, 소화기+앞을 막은 상자 등)를 감싼다.
공구는 하나씩 종류(망치, 삽 ...) 라벨, 라바콘과 DANGER 표지도 따로 라벨을 붙인다.
"""
import json
import math
import os

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, UsdSkel

from . import warehouse as W
from .config import ASSETS, IMG_H, IMG_W, WAREHOUSE_URL
from .walk_anim import FPS, build_walk_animation

H_APERTURE = 20.955  # mm (USD 기본값)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POLYHAVEN_DIR = os.path.join(ROOT_DIR, "assets", "polyhaven")
GENERATED_DIR = os.path.join(ROOT_DIR, "assets", "generated")
YCB_DRILL_LIFT = 0.03     # YCB 드릴은 원점이 가운데라 반 높이만큼 올림


def _layouts():
    path = os.path.join(POLYHAVEN_DIR, "models.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def danger_texture(path):
    """DANGER 표지 그림 (위: 빨간 타원 DANGER, 아래: 위험 구역 / 출입 금지)."""
    if os.path.exists(path):
        return path
    from PIL import Image, ImageDraw, ImageFont
    os.makedirs(os.path.dirname(path), exist_ok=True)
    w, h = 600, 900
    im = Image.new("RGB", (w, h), (250, 250, 248))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, w, 300], fill=(15, 15, 15))
    d.ellipse([40, 40, w - 40, 260], fill=(210, 20, 30), outline=(250, 250, 250), width=10)

    def font(sz):
        for name in ("malgunbd.ttf", "arialbd.ttf"):
            try:
                return ImageFont.truetype(os.path.join("C:/Windows/Fonts", name), sz)
            except OSError:
                continue
        return ImageFont.load_default()
    d.text((w / 2, 150), "DANGER", font=font(120), fill=(255, 255, 255), anchor="mm")
    d.text((w / 2, 450), "위험 구역", font=font(120), fill=(15, 15, 15), anchor="mm")
    d.text((w / 2, 640), "출입 금지", font=font(110), fill=(200, 20, 30), anchor="mm")
    d.text((w / 2, 800), "KEEP OUT", font=font(70), fill=(15, 15, 15), anchor="mm")
    d.rectangle([0, 0, w - 1, h - 1], outline=(15, 15, 15), width=14)
    im.save(path)
    return path
BOX_H = {"box_a": 0.5, "box_b": 0.5, "box_c": 0.25, "box_d": 0.15}
# 핸드트럭 (Poly Haven hand_truck): 손잡이가 +Y 쪽, 짐 받침이 -Y 쪽. 끌 때는 바퀴 축을 중심으로 손잡이 쪽(+Y)으로 기움
CART_MODEL = "hand_truck"
CART_AXLE = (0.0, -0.03, 0.13)       # 바퀴 축 (모델 좌표)
CART_LOAD_Y = -0.30                  # 상자를 올리는 자리 (짐 받침 위, 프레임 앞)
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
        self.ext_mats = []
        self.machine_paths = {}
        self._machine_lamps = {}

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
        for m in W.MACHINES:
            self.machine_paths[m["id"]] = self.add_machine(f"/World/Fixed/Machine_{m['id']}", m)
        self.forklift_path = self.add_forklift("/World/Forklift")
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
        for z in getattr(scenario, "zones", []):
            self._build_zone(root, z)
        for e in getattr(scenario, "equipment", []):
            self.add_cart(f"{root}/{e.id}_cart", e.x, e.y, e.yaw, e.params.get("tilt", 0.0), e.params.get("boxes", 0),
                          e.params.get("box", "box_c"))
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
        if o.kind != "tool":                   # 공구는 하나씩 종류 라벨 (_build_tool)
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
                c = self._ref(base + f"/Cone{k}", ASSETS["cone"], ((r + 0.35) * math.cos(a) * 1.3, (r + 0.35) * math.sin(a), 0.0))
                self.labeler(c.GetPrim(), "cone")
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
        lay = _layouts()
        for k, (kind, model, dx, dy, yaw) in enumerate(o.params["items"]):
            path = f"{base}/T{k}_{kind}"
            if model == "procedural":
                x = UsdGeom.Xform.Define(self.stage, path)
                x.AddTranslateOp().Set(Gf.Vec3d(dx, dy, z0))
                x.AddRotateZOp().Set(math.degrees(yaw))
                self._power_saw(path + "/a")
            elif model == "ycb_drill":
                x = self._ref(path, ASSETS["ycb_drill"], (dx, dy, z0 + YCB_DRILL_LIFT), yaw=yaw)
            else:
                L = lay.get(model, {"rot": [0, 0, 0], "z": 0.0})
                url = os.path.join(POLYHAVEN_DIR, model, f"{model}.usdc").replace(os.sep, "/")
                x = self._ref(path, url, (dx, dy, z0 + L["z"]), yaw=yaw, rot=L["rot"])
            self.labeler(x.GetPrim(), kind)

    def _power_saw(self, path):
        """원형 전동톱 (Poly Haven 에 없어서 기본 도형으로): 노란 몸체, 검은 손잡이, 금속 날과 덮개, 받침판."""
        st = self.stage
        UsdGeom.Xform.Define(st, path)
        yellow = self._preview("/World/Looks/SawYellow", (0.85, 0.62, 0.05), 0.45)
        black = self._preview("/World/Looks/SawBlack", (0.03, 0.03, 0.03), 0.6)
        steel = self._metal("/World/Looks/SawSteel", (0.62, 0.63, 0.66), 0.3)

        def part(name, kind, pos, scale=(1, 1, 1), mat=None, r=0.05, h=0.1, axis="Y"):
            g = (UsdGeom.Cylinder if kind == "cyl" else UsdGeom.Cube).Define(st, f"{path}/{name}")
            if kind == "cyl":
                g.CreateRadiusAttr(r)
                g.CreateHeightAttr(h)
                g.CreateAxisAttr(axis)
            else:
                g.CreateSizeAttr(1.0)
            xf = UsdGeom.Xformable(g)
            xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
            if kind != "cyl":
                xf.AddScaleOp().Set(Gf.Vec3f(*scale))
            UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(mat)
        part("Base", "cube", (0.0, 0.0, 0.006), (0.30, 0.19, 0.012), mat=steel)
        part("Blade", "cyl", (0.0, -0.035, 0.085), mat=steel, r=0.085, h=0.003)
        part("Guard", "cyl", (0.0, -0.035, 0.10), mat=steel, r=0.095, h=0.045)
        part("Motor", "cyl", (0.02, 0.055, 0.10), mat=yellow, r=0.055, h=0.11)
        part("Body", "cube", (-0.02, 0.01, 0.15), (0.20, 0.07, 0.08), mat=yellow)
        part("Grip", "cyl", (-0.02, 0.01, 0.235), mat=black, r=0.017, h=0.17, axis="X")
        part("GripPostA", "cube", (-0.10, 0.01, 0.205), (0.025, 0.03, 0.06), mat=black)
        part("GripPostB", "cube", (0.06, 0.01, 0.205), (0.025, 0.03, 0.06), mat=black)
        part("Knob", "cyl", (0.12, 0.01, 0.16), mat=black, r=0.02, h=0.05, axis="Z")

    def _metal(self, path, color, rough):
        if self.stage.GetPrimAtPath(path):
            return UsdShade.Material(self.stage.GetPrimAtPath(path))
        m = self._preview(path, color, rough)
        UsdShade.Shader(self.stage.GetPrimAtPath(path + "/PBR")).GetInput("metallic").Set(1.0)
        return m

    # ------------------------------------------------------------ 위험 영역
    def _build_zone(self, root, z):
        """라바콘 링 (+ DANGER 표지) 또는 표지만. 안쪽에 뚜껑 열린 바닥 구멍이 있기도 함. 묶음에는 라벨 없음."""
        base = f"{root}/{z.id}_zone"
        UsdGeom.Xform.Define(self.stage, base)
        p = z.params
        for k, (cx, cy) in enumerate(p.get("cones", [])):
            c = self._ref(f"{base}/Cone{k}", ASSETS["cone"], (cx, cy, 0.0), yaw=k * 0.7)
            self.labeler(c.GetPrim(), "cone")
        a = p.get("sign_yaw", 0.0)
        if p.get("inner") == "pit":
            px, py = (z.x, z.y) if z.type == "cone" else (z.x - 0.7 * math.cos(a), z.y - 0.7 * math.sin(a))
            self._pit(f"{base}/Pit", px, py, a)
        if p.get("sign"):
            r = 0.55 * z.radius if z.type == "cone" else 0.0
            sx, sy = z.x + r * math.cos(a), z.y + r * math.sin(a)
            sg = self._danger_sign(f"{base}/Sign", sx, sy, a)
            self.labeler(sg.GetPrim(), "danger_sign")

    def _danger_sign(self, path, x, y, yaw):
        """A 자형 세움 표지 (높이 0.85 m, 폭 0.55 m), 양면에 DANGER 그림."""
        st = self.stage
        g = UsdGeom.Xform.Define(st, path)
        g.AddTranslateOp().Set(Gf.Vec3d(x, y, 0.0))
        g.AddRotateZOp().Set(math.degrees(yaw))
        mat = self._textured("/World/Looks/DangerSign", danger_texture(os.path.join(GENERATED_DIR, "danger_sign.png")))
        w, h, lean = 0.55, 0.85, math.radians(14)
        top = h * math.cos(lean)
        off = h * math.sin(lean)
        pts, idx, uv = [], [], []
        for side in (1, -1):
            b = len(pts)
            # 아래 두 점은 바깥으로 벌어지고, 위 두 점은 가운데에서 만남
            pts += [Gf.Vec3f(side * off, -side * w / 2, 0.0), Gf.Vec3f(side * off, side * w / 2, 0.0),
                    Gf.Vec3f(0.0, side * w / 2, top), Gf.Vec3f(0.0, -side * w / 2, top)]
            idx += [b, b + 1, b + 2, b + 3]
            uv += [Gf.Vec2f(0, 0), Gf.Vec2f(1, 0), Gf.Vec2f(1, 1), Gf.Vec2f(0, 1)]
        mesh = UsdGeom.Mesh.Define(st, path + "/Board")
        mesh.CreatePointsAttr(pts)
        mesh.CreateFaceVertexCountsAttr([4, 4])
        mesh.CreateFaceVertexIndicesAttr(idx)
        mesh.CreateDoubleSidedAttr(True)
        pv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
        pv.Set(uv)
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
        return g

    def _textured(self, path, image):
        if self.stage.GetPrimAtPath(path):
            return UsdShade.Material(self.stage.GetPrimAtPath(path))
        m = UsdShade.Material.Define(self.stage, path)
        s = UsdShade.Shader.Define(self.stage, path + "/PBR")
        s.CreateIdAttr("UsdPreviewSurface")
        s.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
        reader = UsdShade.Shader.Define(self.stage, path + "/st")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
        tex = UsdShade.Shader.Define(self.stage, path + "/Tex")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(image.replace(os.sep, "/"))
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
        tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        s.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
        m.CreateSurfaceOutput().ConnectToSource(s.ConnectableAPI(), "surface")
        return m

    def _pit(self, path, x, y, yaw):
        """뚜껑이 열린 바닥 구멍: 검은 구멍 + 노랑 테두리 + 옆에 놓인 철판 뚜껑."""
        st = self.stage
        g = UsdGeom.Xform.Define(st, path)
        g.AddTranslateOp().Set(Gf.Vec3d(x, y, 0.0))
        g.AddRotateZOp().Set(math.degrees(yaw))
        dark = self._preview("/World/Looks/PitDark", (0.005, 0.005, 0.005), 0.9)
        rim = self._preview("/World/Looks/PitRim", (0.9, 0.7, 0.05), 0.5)
        steel = self._metal("/World/Looks/PitLid", (0.45, 0.46, 0.48), 0.45)

        def box(name, pos, scale, mat, rot=(0, 0, 0)):
            c = UsdGeom.Cube.Define(st, f"{path}/{name}")
            c.CreateSizeAttr(1.0)
            xf = UsdGeom.Xformable(c)
            xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
            if any(rot):
                xf.AddRotateXYZOp().Set(Gf.Vec3f(*rot))
            xf.AddScaleOp().Set(Gf.Vec3f(*scale))
            UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(mat)
        box("Hole", (0, 0, 0.002), (0.8, 0.8, 0.004), dark)
        for k, (dx, dy, sx, sy) in enumerate([(0, 0.43, 0.92, 0.06), (0, -0.43, 0.92, 0.06), (0.43, 0, 0.06, 0.92), (-0.43, 0, 0.06, 0.92)]):
            box(f"Rim{k}", (dx, dy, 0.012), (sx, sy, 0.024), rim)
        box("Lid", (0.95, 0.1, 0.03), (0.8, 0.8, 0.02), steel, rot=(0, -6, 12))

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

    # ------------------------------------------------------------ 끼임 위험 기계 (컨베이어 롤러)
    MACHINE_ON_COLOR = (0.95, 0.15, 0.08)    # 표시등: 작동 중 (빨강)
    MACHINE_OFF_COLOR = (0.12, 0.80, 0.25)   # 표시등: 정지 (초록)

    def add_machine(self, path, m):
        """컨베이어 롤러 1대. 진입 롤러(작업자가 접근하는 쪽)에 pinch_point 라벨을 붙인다 (롤러 닙 = 끼임점).
        표시등 색으로 작동 상태를 보여준다 (set_machine_state 로 바꿈). 고정 설비라 묶음 라벨은 없음."""
        st = self.stage
        g = UsdGeom.Xform.Define(st, path)
        g.AddTranslateOp().Set(Gf.Vec3d(float(m["x"]), float(m["y"]), 0.0))
        g.AddRotateZOp().Set(math.degrees(m["yaw"]))
        L, Wd, H, R = m["length"], m["width"], m["leg_h"], m["roller_r"]
        steel = self._metal(f"{path}/Looks/Frame", (0.5, 0.52, 0.55), 0.4)
        rubber = self._preview(f"{path}/Looks/Belt", (0.16, 0.16, 0.17), 0.7)
        # 끼임점(진입 롤러)은 현장 안전색(노랑)으로 칠해 다른 부품과 뚜렷이 구분되게 한다 (실제 닙 가드 관행과도 맞음)
        hazard_yellow = self._preview(f"{path}/Looks/HazardYellow", (0.95, 0.72, 0.05), 0.35)
        lamp_mat = self._preview(f"{path}/Looks/Lamp", self.MACHINE_OFF_COLOR, 0.3)
        UsdShade.Shader(st.GetPrimAtPath(f"{path}/Looks/Lamp/PBR")).CreateInput(
            "emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*self.MACHINE_OFF_COLOR))

        def box(name, pos, scale, mat):
            c = UsdGeom.Cube.Define(st, f"{path}/{name}")
            c.CreateSizeAttr(1.0)
            xf = UsdGeom.Xformable(c)
            xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
            xf.AddScaleOp().Set(Gf.Vec3f(*scale))
            UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(mat)

        for k, (sx, sy) in enumerate(((0.15, Wd / 2 - 0.05), (0.15, -Wd / 2 + 0.05),
                                      (L - 0.15, Wd / 2 - 0.05), (L - 0.15, -Wd / 2 + 0.05))):
            box(f"Leg{k}", (sx, sy, H / 2), (0.05, 0.05, H), steel)
        box("Top", (L / 2, 0.0, H + 0.02), (L, Wd, 0.04), steel)
        box("Belt", (L / 2, 0.0, H + 0.045), (L - 2 * R, Wd - 0.05, 0.01), rubber)
        entry = UsdGeom.Cylinder.Define(st, f"{path}/RollerEntry")
        entry.CreateRadiusAttr(R)
        entry.CreateHeightAttr(Wd)
        entry.CreateAxisAttr("Y")
        UsdGeom.Xformable(entry).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, H + R))
        UsdShade.MaterialBindingAPI.Apply(entry.GetPrim()).Bind(hazard_yellow)
        self.labeler(entry.GetPrim(), "pinch_point")      # 진입 롤러 닙 = 끼임점 (개별 라벨, 19클래스 모델과 별도)
        exit_r = UsdGeom.Cylinder.Define(st, f"{path}/RollerExit")
        exit_r.CreateRadiusAttr(R)
        exit_r.CreateHeightAttr(Wd)
        exit_r.CreateAxisAttr("Y")
        UsdGeom.Xformable(exit_r).AddTranslateOp().Set(Gf.Vec3d(L, 0.0, H + R))
        UsdShade.MaterialBindingAPI.Apply(exit_r.GetPrim()).Bind(steel)
        lamp = UsdGeom.Sphere.Define(st, f"{path}/Lamp")
        lamp.CreateRadiusAttr(0.035)
        UsdGeom.Xformable(lamp).AddTranslateOp().Set(Gf.Vec3d(L / 2, 0.0, H + 0.18))
        UsdShade.MaterialBindingAPI.Apply(lamp.GetPrim()).Bind(lamp_mat)
        self._machine_lamps[path] = lamp_mat
        return path

    def set_machine_state(self, path, on):
        """표시등 색을 작동 상태에 맞게 바꾼다 (렌더·시연에서 눈에 보이는 on/off 표시)."""
        mat = self._machine_lamps.get(path)
        if mat is None:
            return
        color = Gf.Vec3f(*(self.MACHINE_ON_COLOR if on else self.MACHINE_OFF_COLOR))
        shader = UsdShade.Shader(self.stage.GetPrimAtPath(str(mat.GetPath()) + "/PBR"))
        shader.GetInput("diffuseColor").Set(color)
        shader.GetInput("emissiveColor").Set(color)

    # ------------------------------------------------------------ 운반 카트
    @staticmethod
    def cart_matrix(x, y, yaw, tilt_deg):
        """핸드트럭 놓는 행렬: 바퀴 축을 중심으로 tilt 만큼 손잡이 쪽으로 기울이고, yaw 로 돌려 (x, y) 에."""
        ax = Gf.Vec3d(*CART_AXLE)
        return (Gf.Matrix4d().SetTranslate(-ax) * Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1, 0, 0), -float(tilt_deg)))
                * Gf.Matrix4d().SetTranslate(ax) * Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), math.degrees(yaw)))
                * Gf.Matrix4d().SetTranslate(Gf.Vec3d(float(x), float(y), 0.0)))

    @staticmethod
    def cart_slot(k, box="box_c"):
        """카트에 실은 k 번째 상자 자리 (카트 좌표, 상자 원점은 바닥 가운데)."""
        return (0.0, CART_LOAD_Y, 0.03 + k * BOX_H[box])

    def add_cart(self, path, x, y, yaw, tilt=0.0, boxes=0, box="box_c"):
        """핸드트럭 (라벨 cart) + 실은 상자 (라벨 없음, 카트 박스에 안 들어감). 반환: 움직일 때 쓰는 변환 op."""
        g = UsdGeom.Xform.Define(self.stage, path)
        op = g.AddTransformOp()
        op.Set(self.cart_matrix(x, y, yaw, tilt))
        url = os.path.join(POLYHAVEN_DIR, CART_MODEL, f"{CART_MODEL}.usdc").replace(os.sep, "/")
        t = self._ref(path + "/Truck", url)
        self.labeler(t.GetPrim(), "cart")
        for k in range(boxes):
            bx, by, bz = self.cart_slot(k, box)
            self._ref(f"{path}/Load{k}", ASSETS[box], (bx, by, bz), yaw=0.04 * ((k * 7) % 3 - 1))
        self._ops[path] = op
        return op

    def set_cart(self, path, x, y, yaw, tilt=0.0):
        self._ops[path].Set(self.cart_matrix(x, y, yaw, tilt))

    # ------------------------------------------------------------ 지게차 (위치는 실제로는 RTLS/UWB 위치 추적 태그 개념, YOLO 라벨 없음)
    @staticmethod
    def forklift_matrix(x, y, yaw):
        return (Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), math.degrees(yaw)))
                * Gf.Matrix4d().SetTranslate(Gf.Vec3d(float(x), float(y), 0.0)))

    def add_forklift(self, path, x=0.0, y=-100.0, yaw=0.0):
        """시연 전에는 장면 밖(y=-100)에 둔다. set_forklift 로 매 프레임 위치를 바꿈."""
        g = UsdGeom.Xform.Define(self.stage, path)
        op = g.AddTransformOp()
        op.Set(self.forklift_matrix(x, y, yaw))
        self._ref(path + "/Body", ASSETS["forklift"])
        self._ops[path] = op
        return path

    def set_forklift(self, path, x, y, yaw):
        self._ops[path].Set(self.forklift_matrix(x, y, yaw))

    def add_box(self, path, box="box_c"):
        """따로 움직이는 상자 (시연에서 팔레트 → 카트로 옮겨 실음)."""
        g = UsdGeom.Xform.Define(self.stage, path)
        self._ops[path] = g.AddTransformOp()
        self.stage.DefinePrim(path + "/a", "Xform").GetReferences().AddReference(ASSETS[box])
        strip_semantics(self.stage.GetPrimAtPath(path + "/a"))
        return path

    def set_matrix(self, path, m):
        self._ops[path].Set(Gf.Matrix4d(m))

    def _set_visible(self, path, visible):
        prim = self.stage.GetPrimAtPath(path)
        if prim:
            im = UsdGeom.Imageable(prim)
            im.MakeVisible() if visible else im.MakeInvisible()

    def save(self, path):
        self.stage.GetRootLayer().Export(path)
