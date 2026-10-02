"""터미널 로그 출력 도우미."""
import os

if os.name == "nt":
    os.system("")  # 윈도우 콘솔에서 색상 코드(ANSI) 켜기

_RED, _GREEN, _RESET = "\033[91m", "\033[92m", "\033[0m"


def clock(t):
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


def tag(hazard, color=True):
    s = "위험" if hazard else "안전"
    if not color:
        return s
    return (_RED if hazard else _GREEN) + s + _RESET
