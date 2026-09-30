"""터미널 로그 출력 도우미."""
import os

from .config import RISK_RGBA

if os.name == "nt":
    os.system("")  # 윈도우 콘솔에서 색상 코드(ANSI) 켜기

_TAG = {"high": "위험", "mid": "주의", "ok": "정상"}
_COLOR = {"high": "\033[91m", "mid": "\033[93m", "ok": "\033[92m"}
_RESET = "\033[0m"


def clock(t):
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


def event_line(h, t, conf, color=True):
    tag = _TAG.get(h.risk, "")
    head = f"[{clock(t)}] {tag} {h.label}"
    if color:
        head = _COLOR.get(h.risk, "") + head + _RESET
    return f"{head}  |  {h.zone}  |  {h.detail}  |  신뢰도 {conf * 100:.0f}%"


def summary(detector, t):
    hz = [h for h in detector.hazards if h.is_hazard]
    ex = [h for h in detector.hazards if h.type == "ext"]
    found = sum(1 for i, h in enumerate(detector.hazards) if h.is_hazard and detector.detected[i])
    ext = sum(1 for i, h in enumerate(detector.hazards) if h.type == "ext" and detector.detected[i])
    lines = [f"순찰 시간 {clock(t)}  |  위험 요소 {found}/{len(hz)}  |  소화기 점검 {ext}/{len(ex)}"]
    missed = [h for i, h in enumerate(detector.hazards) if h.is_hazard and not detector.detected[i]]
    for h in missed:
        lines.append(f"  못 찾음: {h.label} ({h.zone})")
    return "\n".join(lines)


def box_color(h, detected, conf):
    if detected:
        return RISK_RGBA.get(h.risk, RISK_RGBA["info"])
    r, g, b, _ = RISK_RGBA["info"]
    return (r, g, b, min(1.0, 0.3 + conf))
