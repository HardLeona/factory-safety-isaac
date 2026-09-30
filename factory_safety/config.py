"""프로젝트 전체에서 같이 쓰는 상수.

좌표계 (Isaac Sim 기본과 동일)
  X: 동(+) / 서(-)
  Y: 북(+) / 남(-)
  Z: 위(+)
  단위: 미터, 각도는 따로 표시가 없으면 라디안
  yaw: +X 방향이 0, 위에서 봤을 때 반시계 방향이 +
"""

# 공장 크기 (내부 기준)
W = 44.0   # 동서 길이
D = 32.0   # 남북 길이
H = 7.0    # 천장 높이

# 카메라 높이
CAM_H_ROBOT = 1.55   # 순찰 로봇 카메라
BODY_H = 1.38        # 작업자 가슴 바디캠

# 이미지 크기 (검출기 투영, 데이터셋 기본값)
IMG_W = 960
IMG_H = 540

# YOLO 클래스 (순서가 곧 클래스 번호)
CLASSES = ["puddle", "power_tool", "unstable_stack", "extinguisher", "gauge_normal", "gauge_low"]
CLASS_KO = {
    "puddle": "미끄러운 바닥",
    "power_tool": "방치된 전동공구",
    "unstable_stack": "불안정 적재물",
    "extinguisher": "소화기",
    "gauge_normal": "압력계 정상",
    "gauge_low": "압력계 부족",
}
# 위험 요소 종류 -> 기본 YOLO 클래스 번호
HAZARD_CLASS = {"puddle": 0, "tool": 1, "stack": 2, "ext": 3}

# 로그 색상 (터미널, 디버그 드로잉)
RISK_RGBA = {
    "high": (1.0, 0.30, 0.31, 1.0),
    "mid": (1.0, 0.65, 0.15, 1.0),
    "ok": (0.24, 0.86, 0.52, 1.0),
    "info": (0.30, 0.67, 0.97, 1.0),
}


def srgb(hex_value):
    """0xRRGGBB (sRGB) -> USD 머티리얼용 선형 RGB 튜플."""
    r = ((hex_value >> 16) & 255) / 255.0
    g = ((hex_value >> 8) & 255) / 255.0
    b = (hex_value & 255) / 255.0

    def lin(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    return (lin(r), lin(g), lin(b))
