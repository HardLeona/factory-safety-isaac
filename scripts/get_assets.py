"""Poly Haven (CC0) 공구 모델 내려받기 (일반 파이썬). USD 와 텍스처를 assets/polyhaven/<이름>/ 에.

    python scripts/get_assets.py

Poly Haven 모델은 CC0 (출처 표시 의무 없음, 상업 이용 가능). https://polyhaven.com
"""
import json
import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from factory_safety.config import POLYHAVEN_EXTRA, POLYHAVEN_MODELS  # noqa: E402

OUT = os.path.join(ROOT, "assets", "polyhaven")


def fetch(url, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        return
    req = urllib.request.Request(url, headers={"User-Agent": "factory-safety-isaac"})
    with urllib.request.urlopen(req, timeout=60) as r, open(path, "wb") as f:
        f.write(r.read())


def main(res="1k"):
    names = sorted({n for v in POLYHAVEN_MODELS.values() for n in v} | set(POLYHAVEN_EXTRA))
    for name in names:
        req = urllib.request.Request(f"https://api.polyhaven.com/files/{name}", headers={"User-Agent": "factory-safety-isaac"})
        files = json.load(urllib.request.urlopen(req, timeout=60))
        usd = files["usd"][res]["usd"]
        base = os.path.join(OUT, name)
        fetch(usd["url"], os.path.join(base, f"{name}.usdc"))
        for rel, info in usd.get("include", {}).items():
            fetch(info["url"], os.path.join(base, rel))
        print(f"  {name}: {len(usd.get('include', {}))} 텍스처")
    fix_texture_paths(names)
    write_layouts(names)
    print(f"[완료] {OUT}")


def fix_texture_paths(names):
    """Poly Haven USD 안의 텍스처 경로가 그쪽 빌드 서버의 절대 경로라서, 받은 textures/ 폴더를 가리키게 고친다."""
    from pxr import Sdf, Usd, UsdShade
    for name in names:
        path = os.path.join(OUT, name, f"{name}.usdc")
        st = Usd.Stage.Open(path)
        n = 0
        for prim in st.Traverse():
            sh = UsdShade.Shader(prim)
            if not sh:
                continue
            for inp in sh.GetInputs():
                if inp.GetTypeName() != Sdf.ValueTypeNames.Asset:
                    continue
                v = inp.Get()
                if not v or not v.path:
                    continue
                base = os.path.basename(v.path.replace("\\", "/"))
                if not os.path.exists(os.path.join(OUT, name, "textures", base)):
                    # USD 는 .exr 을 가리키는데 받은 파일은 .jpg 인 경우 (Drill_01 거칠기)
                    stem = os.path.splitext(base)[0]
                    alt = [f for f in os.listdir(os.path.join(OUT, name, "textures")) if os.path.splitext(f)[0] == stem]
                    base = alt[0] if alt else base
                if os.path.exists(os.path.join(OUT, name, "textures", base)) and v.path != f"./textures/{base}":
                    inp.Set(Sdf.AssetPath(f"./textures/{base}"))
                    n += 1
        st.GetRootLayer().Save()
        print(f"  {name}: 텍스처 경로 {n}개 고침")


def write_layouts(names):
    """모델마다 바닥에 눕히는 회전 (가장 얇은 축을 위로) 과 바닥에 닿게 하는 높이, 눕힌 뒤 크기를 models.json 에."""
    from pxr import Usd, UsdGeom
    out = {}
    for name in names:
        st = Usd.Stage.Open(os.path.join(OUT, name, f"{name}.usdc"))
        b = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]).ComputeWorldBound(st.GetPseudoRoot()).ComputeAlignedRange()
        mn, mx = [list(b.GetMin()), list(b.GetMax())]
        size = [mx[i] - mn[i] for i in range(3)]
        thin = size.index(min(size))
        # 눕히기: 얇은 축이 X 면 Y 축으로 90도, Y 면 X 축으로 90도 (rotateXYZ 각도, 도)
        rot = {0: [0.0, 90.0, 0.0], 1: [90.0, 0.0, 0.0], 2: [0.0, 0.0, 0.0]}[thin]
        # 돌린 뒤 바닥 높이: 얇은 축 방향의 최소값이 아래로 감
        lo = {0: -mx[0], 1: mn[1], 2: mn[2]}[thin]
        flat = {0: [size[2], size[1], size[0]], 1: [size[0], size[2], size[1]], 2: size}[thin]
        out[name] = {"rot": rot, "z": -lo, "size": [round(v, 3) for v in flat],
                     "center": [round((mn[i] + mx[i]) / 2, 3) for i in range(3)],
                     "min": [round(v, 3) for v in mn], "max": [round(v, 3) for v in mx]}
    with open(os.path.join(OUT, "models.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
