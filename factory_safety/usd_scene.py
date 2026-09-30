"""USD 장면 생성. pxr(OpenUSD)만 사용해서 Isaac Sim 버전이 바뀌어도 거의 그대로 동작한다.

    stage = omni.usd.get_context().get_stage()     # Isaac Sim 안에서
    scene = FactoryStage(stage, "outputs/textures")
    scene.build()
    scene.set_scenario(sample_scenario(seed))
"""
import math

import numpy as np
from pxr import Gf, Sdf, Tf, UsdGeom, UsdLux, UsdShade

from . import layout as L
from . import textures
from .config import CAM_H_ROBOT, CLASSES, H, IMG_H, IMG_W, srgb
from .models import Model, box, patrol_robot, worker

H_APERTURE = 20.955  # mm (USD 기본값)


def default_labeler(prim, cls_name):
    """의미 라벨(semantic label) 작성. Isaac Sim 5.0 이상은 UsdSemantics, 그 이전은 Semantics 스키마."""
    try:
        from pxr import UsdSemantics
        api = UsdSemantics.LabelsAPI.Apply(prim, "class")
        api.CreateLabelsAttr().Set([cls_name])
        return True
    except Exception:
        pass
    try:
        import Semantics  # Isaac Sim 4.x
        sem = Semantics.SemanticsAPI.Apply(prim, "Semantics")
        sem.CreateSemanticTypeAttr().Set("class")
        sem.CreateSemanticDataAttr().Set(cls_name)
        return True
    except Exception:
        return False


class FactoryStage:
    def __init__(self, stage, tex_dir="outputs/textures", static_seed=0, labeler=None, lamp_lights=True):
        self.stage = stage
        self.static_seed = static_seed
        self.labeler = labeler or default_labeler
        self.lamp_lights = lamp_lights
        self.tex = textures.make_all(tex_dir, seed=static_seed)
        self.static_models = L.build_static(static_seed)
        self._mat_cache = {}
        self._mat_count = 0
        self._ops = {}
        self.scenario = None
        self.cam_path = "/World/PatrolCam"
        self.lights = {}

    # ------------------------------------------------------------ 기본 구조
    def build(self):
        st = self.stage
        UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(st, 1.0)
        world = UsdGeom.Xform.Define(st, "/World")
        st.SetDefaultPrim(world.GetPrim())
        for g in ("Looks", "Static", "Hazards", "Extras", "Agents", "Dynamic", "Lights"):
            UsdGeom.Xform.Define(st, f"/World/{g}")
        for m in self.static_models:
            self._add_model(m, "/World/Static")
        self._build_lights()
        self._build_dynamic()
        self._build_agents()
        self._build_camera()
        return self

    def _build_lights(self):
        st = self.stage
        sun = UsdLux.DistantLight.Define(st, "/World/Lights/Sun")
        sun.CreateIntensityAttr(1600.0)
        sun.CreateAngleAttr(1.0)
        UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(35.0, 0.0, 30.0))
        dome = UsdLux.DomeLight.Define(st, "/World/Lights/Dome")
        dome.CreateIntensityAttr(450.0)
        dome.CreateColorAttr(Gf.Vec3f(0.85, 0.88, 0.92))
        self.lights = {"sun": sun, "dome": dome, "lamps": []}
        if self.lamp_lights:
            for i, (x, y) in enumerate(L.LAMPS):
                rl = UsdLux.RectLight.Define(st, f"/World/Lights/Lamp{i:02d}")
                rl.CreateWidthAttr(2.6)
                rl.CreateHeightAttr(0.34)
                rl.CreateIntensityAttr(9000.0)
                rl.CreateColorAttr(Gf.Vec3f(1.0, 0.96, 0.88))
                UsdGeom.Xformable(rl).AddTranslateOp().Set(Gf.Vec3d(x, y, H - 0.62))
                self.lights["lamps"].append(rl)

    def _build_camera(self):
        cam = UsdGeom.Camera.Define(self.stage, self.cam_path)
        cam.CreateHorizontalApertureAttr(H_APERTURE)
        cam.CreateVerticalApertureAttr(H_APERTURE * IMG_H / IMG_W)
        cam.CreateFocalLengthAttr(15.0)
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 200.0))
        self._ops["cam"] = UsdGeom.Xformable(cam).AddTransformOp()

    def _build_agents(self):
        r = Model("Robot", patrol_robot())
        self._add_model(r, "/World/Agents", dynamic=True, role_offsets={"head": (0.0, 0.0, CAM_H_ROBOT)})
        w = Model("Worker", worker())
        self._add_model(w, "/World/Agents", dynamic=True)
        self.set_worker_visible(False)

    def _build_dynamic(self):
        rng = np.random.default_rng(self.static_seed + 7)
        self.parcels = []
        for i in range(6):
            s, h = rng.uniform(0.32, 0.46), rng.uniform(0.2, 0.34)
            m = Model(f"Parcel{i}", [box((0, 0, h / 2), (s, s * rng.uniform(0.8, 1.1), h),
                                        srgb(int(rng.choice([0xB58A55, 0xC89F69, 0xA47A49]))), rough=0.9)])
            self._add_model(m, "/World/Dynamic", dynamic=True)
            self.parcels.append((m.name, 1.4 + i * 2.25))
        # 로봇팔 (관절마다 Xform 중첩)
        st, org, dk = self.stage, srgb(0xE8590C), srgb(0x2C3238)
        base = "/World/Dynamic/Arm"
        UsdGeom.Xform.Define(st, base).AddTranslateOp().Set(Gf.Vec3d(8.5, 3.1, 0.5))
        turret = UsdGeom.Xform.Define(st, base + "/Turret")
        self._ops["arm_turret"] = turret.AddRotateZOp()
        self._make_part(base + "/Turret/Ring", "cyl", pos=(0, 0, 0.15), radius=0.3, height=0.3, color=org)
        sh = UsdGeom.Xform.Define(st, base + "/Turret/Shoulder")
        sh.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.32))
        self._ops["arm_shoulder"] = sh.AddRotateYOp()
        self._make_part(base + "/Turret/Shoulder/Link1", "box", pos=(0, 0, 0.62), size=(0.24, 0.24, 1.25), color=org)
        el = UsdGeom.Xform.Define(st, base + "/Turret/Shoulder/Elbow")
        el.AddTranslateOp().Set(Gf.Vec3d(0, 0, 1.25))
        self._ops["arm_elbow"] = el.AddRotateYOp()
        self._make_part(base + "/Turret/Shoulder/Elbow/Joint", "cyl", pos=(0, 0, 0), radius=0.15, height=0.3, axis="Y", color=dk)
        self._make_part(base + "/Turret/Shoulder/Elbow/Link2", "box", pos=(0, 0, 0.5), size=(0.19, 0.19, 1.0), color=org)
        self._make_part(base + "/Turret/Shoulder/Elbow/Gripper", "box", pos=(0, 0, 1.03), size=(0.32, 0.22, 0.12), color=dk)

    # ------------------------------------------------------------ 시나리오
    def set_scenario(self, scenario):
        st = self.stage
        for g in ("/World/Hazards", "/World/Extras"):
            if st.GetPrimAtPath(g):
                st.RemovePrim(g)
            UsdGeom.Xform.Define(st, g)
        self.scenario = scenario
        for h in scenario.hazards:
            labels = {"main": CLASSES[h.cls]}
            if h.gauge_cls is not None:
                labels["gauge"] = CLASSES[h.gauge_cls]
            self._add_model(h.model, "/World/Hazards", labels=labels)
        for m in scenario.extras:
            self._add_model(m, "/World/Extras")

    # ------------------------------------------------------------ 매 프레임 갱신
    def set_camera(self, pose, width=IMG_W, height=IMG_H):
        cam = UsdGeom.Camera(self.stage.GetPrimAtPath(self.cam_path))
        v_ap = H_APERTURE * height / width
        cam.GetVerticalApertureAttr().Set(v_ap)
        cam.GetFocalLengthAttr().Set((v_ap / 2.0) / math.tan(math.radians(pose.vfov) / 2.0))
        m = pose.usd_matrix()
        self._ops["cam"].Set(Gf.Matrix4d(*[float(v) for v in m.flatten()]))

    def set_robot(self, x, y, yaw, head_yaw=0.0, visible=True):
        self._ops["Robot"][0].Set(Gf.Vec3d(x, y, 0.0))
        self._ops["Robot"][1].Set(math.degrees(yaw))
        self._ops["Robot/head"].Set(math.degrees(head_yaw))
        self._set_visible("/World/Agents/Robot", visible)

    def set_worker(self, x, y, yaw, bob=0.0, visible=True):
        self._ops["Worker"][0].Set(Gf.Vec3d(x, y, bob))
        self._ops["Worker"][1].Set(math.degrees(yaw))
        self._set_visible("/World/Agents/Worker", visible)

    def set_worker_visible(self, visible):
        self._set_visible("/World/Agents/Worker", visible)

    def set_robot_visible(self, visible):
        self._set_visible("/World/Agents/Robot", visible)

    def animate(self, t):
        self._ops["arm_turret"].Set(math.degrees(math.sin(t * 0.6) * 1.1))
        self._ops["arm_shoulder"].Set(math.degrees(0.45 + math.sin(t * 0.9) * 0.2))
        self._ops["arm_elbow"].Set(math.degrees(1.5 + math.sin(t * 0.9 + 1) * 0.3))
        for name, x0 in self.parcels:
            x = 1.4 + ((x0 - 1.4 + t * 0.6) % 13.2)
            self._ops[name][0].Set(Gf.Vec3d(x, 1.0, 0.89))

    def set_lighting(self, sun=1600.0, dome=450.0, lamps=9000.0, tint=(1.0, 1.0, 1.0), floor=1.0):
        self.lights["sun"].GetIntensityAttr().Set(float(sun))
        self.lights["dome"].GetIntensityAttr().Set(float(dome))
        self.lights["sun"].CreateColorAttr().Set(Gf.Vec3f(*[float(c) for c in tint]))
        for rl in self.lights["lamps"]:
            rl.GetIntensityAttr().Set(float(lamps))
        tex_node = self._mat_cache.get(("texnode", "floor"))
        if tex_node is not None:
            tex_node.GetInput("scale").Set(Gf.Vec4f(floor, floor, floor, 1.0))

    def randomize_lighting(self, rng):
        k = rng.uniform(0, 1)   # 0 = 따뜻한 조명, 1 = 차가운 조명
        tint = tuple((1 - k) * np.array([1.0, 0.93, 0.82]) + k * np.array([0.85, 0.92, 1.0]))
        self.set_lighting(sun=rng.uniform(500, 2400), dome=rng.uniform(150, 800), lamps=rng.uniform(3000, 14000),
                          tint=tint, floor=rng.uniform(0.7, 1.05))

    def save(self, path):
        self.stage.GetRootLayer().Export(path)
        return path

    # ------------------------------------------------------------ 내부 도우미
    def _set_visible(self, path, visible):
        img = UsdGeom.Imageable(self.stage.GetPrimAtPath(path))
        img.MakeVisible() if visible else img.MakeInvisible()

    def _add_model(self, model, parent, dynamic=False, labels=None, role_offsets=None):
        st = self.stage
        mpath = f"{parent}/{Tf.MakeValidIdentifier(model.name)}"
        xf = UsdGeom.Xform.Define(st, mpath)
        x, y, z, yaw = model.pose
        t_op = xf.AddTranslateOp()
        t_op.Set(Gf.Vec3d(x, y, z))
        r_op = xf.AddRotateZOp()
        r_op.Set(math.degrees(yaw))
        if dynamic:
            self._ops[model.name] = (t_op, r_op)
        used = set()
        roles = {}
        for i, p in enumerate(model.parts):
            role = p.role or "main"
            if role not in roles:
                rpath = f"{mpath}/{role}"
                rx = UsdGeom.Xform.Define(st, rpath)
                off = (role_offsets or {}).get(role)
                if off is not None:
                    rx.AddTranslateOp().Set(Gf.Vec3d(*off))
                    self._ops[f"{model.name}/{role}"] = rx.AddRotateZOp()
                if labels and role in labels:
                    self.labeler(rx.GetPrim(), labels[role])
                roles[role] = rpath
            name = Tf.MakeValidIdentifier(p.name or f"{p.kind}{i}")
            while name in used:
                name += "_"
            used.add(name)
            self._make_part(f"{roles[role]}/{name}", p.kind, part=p)
        return mpath

    def _make_part(self, path, kind, part=None, **kw):
        if part is None:
            from .models import Part
            part = Part(kind, **kw)
        st, p = self.stage, part
        if p.kind == "box":
            g = UsdGeom.Cube.Define(st, path)
            g.CreateSizeAttr(1.0)
            g.CreateExtentAttr([(-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)])
        elif p.kind == "cyl":
            g = UsdGeom.Cylinder.Define(st, path)
            g.CreateRadiusAttr(p.radius)
            g.CreateHeightAttr(p.height)
            g.CreateAxisAttr(p.axis)
            r, hh = p.radius, p.height / 2
            e = {"X": (hh, r, r), "Y": (r, hh, r), "Z": (r, r, hh)}[p.axis]
            g.CreateExtentAttr([tuple(-v for v in e), e])
        elif p.kind == "sphere":
            g = UsdGeom.Sphere.Define(st, path)
            g.CreateRadiusAttr(p.radius)
            g.CreateExtentAttr([(-p.radius,) * 3, (p.radius,) * 3])
        elif p.kind == "mesh":
            g = self._polygon_mesh(path, p.points)
        elif p.kind == "quad":
            g = self._quad_mesh(path, p.size, p.facing)
        else:
            raise ValueError(p.kind)
        xf = UsdGeom.Xformable(g)
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in p.pos]))
        if any(p.rot):
            xf.AddRotateXYZOp().Set(Gf.Vec3f(*[float(v) for v in p.rot]))
        if p.kind == "box":
            xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in p.size]))
        g.CreateDisplayColorAttr([Gf.Vec3f(*p.color)])
        if not p.shadow:
            UsdGeom.PrimvarsAPI(g).CreatePrimvar("doNotCastShadows", Sdf.ValueTypeNames.Bool).Set(True)
        mat = self._material(p)
        try:
            UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(mat)
        except AttributeError:
            UsdShade.MaterialBindingAPI(g.GetPrim()).Bind(mat)
        return g

    def _polygon_mesh(self, path, pts2):
        pts = [Gf.Vec3f(float(x), float(y), 0.0) for x, y in pts2]
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        pts = [Gf.Vec3f(cx, cy, 0.0)] + pts
        n = len(pts2)
        idx = []
        for i in range(n):
            idx += [0, 1 + i, 1 + (i + 1) % n]
        m = UsdGeom.Mesh.Define(self.stage, path)
        m.CreatePointsAttr(pts)
        m.CreateFaceVertexCountsAttr([3] * n)
        m.CreateFaceVertexIndicesAttr(idx)
        m.CreateNormalsAttr([Gf.Vec3f(0, 0, 1)] * len(pts))
        m.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        m.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        m.CreateExtentAttr([(min(xs), min(ys), 0.0), (max(xs), max(ys), 0.0)])
        return m

    def _quad_mesh(self, path, size, facing):
        a, b = size[0] / 2.0, size[1] / 2.0
        if facing == "+X":      # 가로 = Y, 세로 = Z, 앞면 +X (앞에서 보면 +Y가 오른쪽)
            pts = [(0, -a, -b), (0, a, -b), (0, a, b), (0, -a, b)]
            nrm = (1, 0, 0)
        elif facing == "+Z":    # 바닥
            pts = [(-a, -b, 0), (a, -b, 0), (a, b, 0), (-a, b, 0)]
            nrm = (0, 0, 1)
        else:                   # -Z, 천장
            pts = [(-a, b, 0), (a, b, 0), (a, -b, 0), (-a, -b, 0)]
            nrm = (0, 0, -1)
        m = UsdGeom.Mesh.Define(self.stage, path)
        m.CreatePointsAttr([Gf.Vec3f(*p) for p in pts])
        m.CreateFaceVertexCountsAttr([4])
        m.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        m.CreateNormalsAttr([Gf.Vec3f(*nrm)] * 4)
        m.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        m.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        st_uv = [(0, 0), (1, 0), (1, 1), (0, 1)] if facing != "-Z" else [(0, 1), (1, 1), (1, 0), (0, 0)]
        pv = UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
        pv.Set([Gf.Vec2f(*uv) for uv in st_uv])
        arr = np.array(pts, dtype=float)
        m.CreateExtentAttr([tuple(arr.min(axis=0)), tuple(arr.max(axis=0))])
        return m

    def _material(self, p):
        key = (tuple(round(c, 4) for c in p.color), round(p.rough, 3), round(p.metal, 3),
               tuple(round(c, 4) for c in p.emissive) if p.emissive else None, round(p.opacity, 3), p.texture)
        if key in self._mat_cache:
            return self._mat_cache[key]
        st = self.stage
        path = f"/World/Looks/M{self._mat_count:03d}"
        self._mat_count += 1
        mat = UsdShade.Material.Define(st, path)
        sh = UsdShade.Shader.Define(st, path + "/PBR")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(p.rough))
        sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(p.metal))
        if p.opacity < 1.0:
            sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(p.opacity))
        if p.emissive:
            sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*p.emissive))
        if p.texture and p.texture in self.tex:
            reader = UsdShade.Shader.Define(st, path + "/stReader")
            reader.CreateIdAttr("UsdPrimvarReader_float2")
            reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
            reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
            tex = UsdShade.Shader.Define(st, path + "/Tex")
            tex.CreateIdAttr("UsdUVTexture")
            tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(self.tex[p.texture]))
            tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
            tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
            tex.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(1, 1, 1, 1))
            tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
            sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
            if p.texture == "floor":
                self._mat_cache[("texnode", "floor")] = tex
        else:
            sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*p.color))
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        self._mat_cache[key] = mat
        return mat
