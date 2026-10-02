"""보고서·발표용 장면 고르기: run_patrol.py --record 결과에서 바디캠, 확대, 접근 경고 장면과 조치 지시서 화면을 docs/ 로.

    python scripts/pick_figures.py --record outputs/record/seed1
    python scripts/pick_figures.py --record outputs/record/seed1 --body 120 --ptz 41      # 장면 번호 직접 지정
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, "docs")
EDGE = "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"


def frame_at(states, t, need=None):
    """시각 t 이후 처음으로 need 파일이 있는 장면 번호."""
    for s in states:
        if s["t"] >= t and (need is None or need(s)):
            return s["k"]
    return states[-1]["k"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--record", default=os.path.join(ROOT, "outputs", "record", "seed1"))
    p.add_argument("--body", type=int, default=None)
    p.add_argument("--ptz", type=int, default=None)
    p.add_argument("--cctv", default=None, help="이름:장면번호 (예 cctv_east:150)")
    p.add_argument("--video", type=int, default=None)
    a = p.parse_args()
    rec = a.record
    states = [json.loads(ln) for ln in open(os.path.join(rec, "state.jsonl"), encoding="utf-8") if ln.strip()]
    final = json.load(open(os.path.join(rec, "final.json"), encoding="utf-8"))
    tl = final["report"]["timeline"]
    seed = final["seed"]
    has = lambda name: (lambda s: os.path.exists(os.path.join(rec, f"{name}_{s['k']:05d}.jpg")))  # noqa: E731

    # 바디캠: 위험 판정이 처음 확정된 순간 (바로 앞 장면이 박스가 더 많음)
    if a.body is None:
        ev = next(e for e in tl if e["kind"] == "판정" and "위험" in e["text"] and e["t"] > 3)
        a.body = frame_at(states, ev["t"] - 0.3, has("body"))
    shutil.copy(os.path.join(rec, f"body_{a.body:05d}.jpg"), os.path.join(DOCS, "fig_body.jpg"))
    # 확대: 순찰 중 재확인이 성공한 첫 작업의 두 번째(좁은) 장면
    if a.ptz is None:
        ok = next(e for e in tl if e["kind"] in ("재확인", "판정 수정") and "확대 결과" in e["text"])
        ks = [s["k"] for s in states if s.get("ptz") and s["t"] <= ok["t"] + 1e-6 and has("ptz")(s)]
        a.ptz = ks[-1]
    shutil.copy(os.path.join(rec, f"ptz_{a.ptz:05d}.jpg"), os.path.join(DOCS, "fig_ptz.jpg"))
    # 접근 경고: 첫 경고를 낸 CCTV 의 그 순간
    if a.cctv is None:
        ev = next((e for e in tl if e["kind"] == "접근 경고"), None)
        cam = ev["text"].split(":")[0] if ev else "cctv_west"
        k = frame_at(states, ev["t"], has(cam)) if ev else states[len(states) // 2]["k"]
    else:
        cam, k = a.cctv.split(":")
        k = int(k)
    shutil.copy(os.path.join(rec, f"{cam}_{k:05d}.jpg"), os.path.join(DOCS, "fig_cctv.jpg"))
    # 창고 전경 (서쪽 CCTV 첫 화면)
    k0 = next(s["k"] for s in states if has("cctv_west")(s))
    shutil.copy(os.path.join(rec, f"cctv_west_{k0:05d}.jpg"), os.path.join(DOCS, "fig_cctv_raw.jpg"))
    # 조치 지시서 화면 (녹화한 순찰의 대장으로 다시 만들어서 영상과 같은 내용)
    sys.path.insert(0, ROOT)
    from factory_safety.dashboard import write_dashboard
    html = os.path.join(ROOT, "outputs", "agent", f"dashboard_seed{seed}.html")
    write_dashboard(html, final["report"], final.get("evaluation"), seed=seed)
    png = os.path.join(DOCS, "fig_dashboard.png")
    if os.path.exists(EDGE) and os.path.exists(html):
        subprocess.run([EDGE, "--headless", "--disable-gpu", "--hide-scrollbars", f"--screenshot={png}", "--window-size=1600,1250",
                        "file:///" + os.path.abspath(html).replace("\\", "/")], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
        im = Image.open(png).convert("RGB")
        im.crop((0, 0, 1600, 760)).save(os.path.join(DOCS, "fig_dashboard_crop.png"))
    # 영상 한 장면 (재확인 중인 순간)
    if a.video is None:
        a.video = a.ptz
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "make_video.py"), "--record", rec, "--out",
                    os.path.join(DOCS, "fig_video.mp4"), "--preview", str(a.video)], check=True, stdout=subprocess.DEVNULL)
    shutil.move(os.path.join(DOCS, f"fig_video_preview_{a.video}.png"), os.path.join(DOCS, "fig_video.png"))
    print(f"바디캠 {a.body}, 확대 {a.ptz}, 접근 경고 {cam}:{k}, 영상 {a.video} → docs/fig_*.jpg|png")


if __name__ == "__main__":
    main()
