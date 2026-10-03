"""보고서·발표용 장면 고르기: run_patrol.py --record 결과에서 바디캠, 확대, 접근 경고 장면과 조치 지시서 화면을 docs/ 로.
CCTV 없이 녹화한 순찰 (--story) 이면 CCTV 그림 (fig_ptz, fig_cctv, fig_cctv_raw) 은 건드리지 않는다.

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
    p.add_argument("--record", default=os.path.join(ROOT, "outputs", "record", "seed5"))
    p.add_argument("--body", type=int, default=None)
    p.add_argument("--ptz", type=int, default=None)
    p.add_argument("--cctv", default=None, help="이름:장면번호 (예 cctv_east:150)")
    p.add_argument("--video", type=int, default=None)
    p.add_argument("--zone", type=int, default=None, help="위험 영역 장면 번호")
    a = p.parse_args()
    rec = a.record
    states = [json.loads(ln) for ln in open(os.path.join(rec, "state.jsonl"), encoding="utf-8") if ln.strip()]
    final = json.load(open(os.path.join(rec, "final.json"), encoding="utf-8"))
    tl = final["report"]["timeline"]
    seed = final["seed"]
    has = lambda name: (lambda s: os.path.exists(os.path.join(rec, f"{name}_{s['k']:05d}.jpg")))  # noqa: E731
    solo = final.get("cctv") is None          # CCTV 없이 바디캠만
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import make_video as MV
    MV.LAYOUT.update(solo=solo, body=(0, 70, 1280, 790) if solo else (0, 70, 960, 610))
    compose_fn = MV.compose_solo if solo else MV.compose
    cam, k = None, None

    # 바디캠: 위험 판정이 처음 확정된 순간 (바로 앞 장면이 박스가 더 많음)
    ready = [s["k"] for s in states if s.get("story") == "준비" and not (s.get("hand") or {}).get("alpha") and has("body")(s)]
    if a.body is None and solo and ready:
        a.body = ready[-1]          # 시연 이야기: 출발점에서 손을 들기 직전 (가까운 위험물이 한 화면에)
    elif a.body is None:
        ev = next(e for e in tl if e["kind"] == "판정" and "위험" in e["text"] and e["t"] > 3)
        a.body = frame_at(states, ev["t"] - 0.3, has("body"))
    shutil.copy(os.path.join(rec, f"body_{a.body:05d}.jpg"), os.path.join(DOCS, "fig_body.jpg"))
    # 확대: 순찰 중 재확인이 성공한 첫 작업의 두 번째(좁은) 장면
    if a.ptz is None and not solo:
        ok = next(e for e in tl if e["kind"] in ("재확인", "판정 수정") and "확대 결과" in e["text"])
        ks = [s["k"] for s in states if s.get("ptz") and s["t"] <= ok["t"] + 1e-6 and has("ptz")(s)]
        a.ptz = ks[-1]
    if a.ptz is not None:
        shutil.copy(os.path.join(rec, f"ptz_{a.ptz:05d}.jpg"), os.path.join(DOCS, "fig_ptz.jpg"))
    # 접근 경고: 첫 경고를 낸 CCTV 의 그 순간
    if solo:
        pass
    elif a.cctv is None:
        ev = next((e for e in tl if e["kind"] == "접근 경고"), None)
        cam = ev["text"].split(":")[0] if ev else "cctv_west"
        k = frame_at(states, ev["t"], has(cam)) if ev else states[len(states) // 2]["k"]
    else:
        cam, k = a.cctv.split(":")
        k = int(k)
    if not solo:
        shutil.copy(os.path.join(rec, f"{cam}_{k:05d}.jpg"), os.path.join(DOCS, "fig_cctv.jpg"))
    # 위험 영역: 라바콘·표지 영역을 처음 알아본 순간의 바디캠 (없으면 아무 영역)
    if a.zone is None:
        ev = next((e for e in tl if e["kind"] == "위험 영역" and ("라바콘" in e["text"] or "DANGER" in e["text"])), None) or             next((e for e in tl if e["kind"] == "위험 영역"), None)
        a.zone = frame_at(states, ev["t"] - 0.2, has("body")) if ev else a.body
    shutil.copy(os.path.join(rec, f"body_{a.zone:05d}.jpg"), os.path.join(DOCS, "fig_zone.jpg"))
    # 음성 경고 순간의 영상 화면 (빨간 띠 포함)
    vst = next((s for s in states if s.get("voice")), None)
    if vst is not None:
        im = compose_fn(rec, vst, tl, [None, None], MV.W.PatrolPath().pts[::3], {})
        MV.voice_banner(im)
        im.save(os.path.join(DOCS, "fig_zone_map.png"))
    # 손동작 명령: 명령을 알아본 순간의 바디캠 (손 관절 + 명령) 과 작업자 언어 자막 (장비 설명을 먼저)
    assist = final["report"].get("assistant", [])
    if assist:
        i = next((k for k, e in enumerate(assist) if e["count"] == 1), 0)
        gst = next(s for s in states if s.get("n_assist", 0) >= i + 1 and has("body")(s))
        bx0, by0, bx1, by1 = MV.LAYOUT["body"]
        im = Image.new("RGB", (MV.WIDTH, MV.HEIGHT))
        im.paste(Image.open(os.path.join(rec, f"body_{gst['k']:05d}.jpg")).convert("RGB").resize((bx1 - bx0, by1 - by0)), (bx0, by0))
        MV.assist_caption(im, assist[i], 0.2)
        im.crop((bx0, by0, bx1, by1)).resize((960, 540)).save(os.path.join(DOCS, "fig_gesture.jpg"), quality=92)
    # 손가락 1~5 와 명령 (scripts/test_gestures.py --save 로 찍은 화면)
    gdir = os.path.join(ROOT, "outputs", "eval", "gesture_frames")
    names = {1: "장비 설명", 2: "공장 위험 스캔", 3: "오늘의 TBM", 4: "관리자 호출", 5: "SOS 신고"}
    # 손가락 수마다 명령이 맞게 나온 첫 지점의 화면 (명령이 확정된 순간)
    gt = os.path.join(ROOT, "outputs", "eval", "gesture_test.json")
    rows = json.load(open(gt, encoding="utf-8")).get("rows", []) if os.path.exists(gt) else []
    spot = {c: next((r["spot"] for r in rows if r["shown"] == c and r["ok"]), 0) for c in range(1, 6)}
    shots = [os.path.join(gdir, f"spot{spot[c]}_g{c}.jpg") for c in range(1, 6)]
    if all(os.path.exists(f) for f in shots):
        from PIL import ImageDraw
        w, h = 480, 270
        sheet = Image.new("RGB", (w * 5, h + 56), (18, 24, 34))
        d = ImageDraw.Draw(sheet)
        for c, f in enumerate(shots, 1):
            sheet.paste(Image.open(f).convert("RGB").resize((w, h)), ((c - 1) * w, 0))
            MV.text(d, ((c - 1) * w + w // 2, h + 28), f"손가락 {c} · {names[c]}", 26, (255, 255, 255), True, anchor="mm")
        sheet.save(os.path.join(DOCS, "fig_gestures.jpg"), quality=90)
    # 창고 전경 (서쪽 CCTV 첫 화면)
    k0 = next((s["k"] for s in states if has("cctv_west")(s)), None)
    if k0 is not None:
        shutil.copy(os.path.join(rec, f"cctv_west_{k0:05d}.jpg"), os.path.join(DOCS, "fig_cctv_raw.jpg"))
    # 조치 지시서 화면 (녹화한 순찰의 대장으로 다시 만들어서 영상과 같은 내용)
    sys.path.insert(0, ROOT)
    from factory_safety.dashboard import write_dashboard
    html = os.path.join(ROOT, "outputs", "agent", f"dashboard_{MV.dash_tag(final, states)}.html")
    write_dashboard(html, final["report"], final.get("evaluation"), seed=seed)
    png = os.path.join(DOCS, "fig_dashboard.png")
    if os.path.exists(EDGE) and os.path.exists(html):
        subprocess.run([EDGE, "--headless", "--disable-gpu", "--hide-scrollbars", f"--screenshot={png}", "--window-size=1600,1250",
                        "file:///" + os.path.abspath(html).replace("\\", "/")], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
        im = Image.open(png).convert("RGB")
        im.crop((0, 0, 1600, 760)).save(os.path.join(DOCS, "fig_dashboard_crop.png"))
    # 영상 한 장면 (재확인 중인 순간, CCTV 가 없으면 카트를 끌며 순찰하는 순간)
    if a.video is None:
        pull = [s for s in states if s.get("story") in ("pull", "pull2")]
        a.video = a.ptz if a.ptz is not None else (pull[len(pull) // 3]["k"] if pull else states[len(states) // 2]["k"])
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "make_video.py"), "--record", rec, "--out",
                    os.path.join(DOCS, "fig_video.mp4"), "--preview", str(a.video)], check=True, stdout=subprocess.DEVNULL)
    shutil.move(os.path.join(DOCS, f"fig_video_preview_{a.video}.png"), os.path.join(DOCS, "fig_video.png"))
    print(f"바디캠 {a.body}, 확대 {a.ptz}, 접근 경고 {cam}:{k}, 영상 {a.video} → docs/fig_*.jpg|png")


if __name__ == "__main__":
    main()
