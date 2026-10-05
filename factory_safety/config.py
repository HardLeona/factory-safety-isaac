"""프로젝트 전체에서 같이 쓰는 상수.

좌표계 (Isaac Sim 기본과 동일, NVIDIA 창고 warehouse_multiple_shelves.usd 그대로)
  X: 동(+) / 서(-), Y: 북(+) / 남(-), Z: 위(+), 단위 미터
  yaw: +X 방향이 0, 위에서 봤을 때 반시계 방향이 +
"""

# NVIDIA Isaac Sim 6.0 에셋 서버
ASSET_ROOT = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/"
WAREHOUSE_URL = ASSET_ROOT + "Environments/Simple_Warehouse/warehouse_multiple_shelves.usd"
PROPS = ASSET_ROOT + "Environments/Simple_Warehouse/Props/"
ASSETS = {
    "extinguisher": PROPS + "SM_FireExtinguisher_02.usd",
    "wet_sign": PROPS + "S_WetFloorSign.usd",
    "cone": PROPS + "S_TrafficCone.usd",
    "bucket": PROPS + "SM_BucketPlastic_B.usd",
    "barrel": PROPS + "SM_BarelPlastic_A_01.usd",
    "bottle": PROPS + "SM_BottlePlasticA_01.usd",
    "box_a": PROPS + "SM_CardBoxA_01.usd",       # 0.7 x 0.5 x 0.5
    "box_b": PROPS + "SM_CardBoxB_01.usd",       # 0.5 x 0.5 x 0.5
    "box_c": PROPS + "SM_CardBoxC_01.usd",       # 0.5 x 0.5 x 0.25
    "box_d": PROPS + "SM_CardBoxD_01.usd",       # 0.38 x 0.25 x 0.15
    "pallet": PROPS + "SM_PaletteA_01.usd",      # 1.2 x 1.0 x 0.21
    "ycb_drill": ASSET_ROOT + "Props/YCB/Axis_Aligned/035_power_drill.usd",
    "table": ASSET_ROOT + "Props/PackingTable/packing_table.usd",
    "worker": ASSET_ROOT + "People/Characters/original_male_adult_construction_05/male_adult_construction_05.usd",
}

# 바디캠 화면 크기 (YOLO 입력과 데이터셋 기본값)
IMG_W = 960
IMG_H = 540

# 공구 종류: YOLO 가 무슨 공구인지까지 알아본다. 위험(통로 바닥에 방치) / 안전(작업대 위) 은 에이전트가 놓인 자리로 판단
TOOL_TYPES = ["hammer", "screwdriver", "saw", "power_saw", "pickaxe", "shovel", "wrench", "drill"]
TOOL_KO = {"hammer": "망치", "screwdriver": "드라이버", "saw": "톱", "power_saw": "전동톱", "pickaxe": "곡괭이",
           "shovel": "삽", "wrench": "렌치", "drill": "전동 드릴"}
# 공구 3D 모델: Poly Haven (CC0, scripts/get_assets.py 로 assets/polyhaven/ 에 받음). 전동톱(원형 톱)은 직접 모델링, 드릴은 YCB 도 씀
POLYHAVEN_MODELS = {
    "hammer": ["cross_pein_hammer", "wooden_hammer_01", "sledgehammer_01"],
    "screwdriver": ["screwdriver", "flathead_screwdriver"],
    "saw": ["handsaw_wood", "rusted_hacksaw"],
    "pickaxe": ["picke_dirty_01"],
    "shovel": ["rusted_spade_01"],
    "wrench": ["adjustable_wrench", "combination_wrench", "pipe_wrench"],
    "drill": ["Drill_01"],
}
# 장비 (공구 외): 핸드트럭 (Poly Haven hand_truck, CC0). 박스를 싣고 끄는 카트
POLYHAVEN_EXTRA = ["hand_truck"]

# YOLO 클래스 (순서가 곧 클래스 번호). 위험한 상태와 안전한 상태를 따로 둬서 YOLO 가 직접 가린다.
# 공구는 종류별, 라바콘과 DANGER 표지는 위험 영역 표시용
# 장비 (위험/안전 판정 대상이 아님): 운반 카트 (핸드트럭). 작업자가 손동작으로 물으면 쓰는 법을 안내
EQUIPMENT = ["cart"]
CLASSES = ["spill", "spill_marked", "stack_unstable", "stack_stable", "ext_fallen", "ext_blocked", "ext_ok",
           "worker", "cone", "danger_sign"] + TOOL_TYPES + EQUIPMENT
# 상태 (정답표와 에이전트 판정의 단위): 공구는 YOLO 종류 + 놓인 자리 -> tool_floor / tool_stored
STATES = ["spill", "spill_marked", "tool_floor", "tool_stored", "stack_unstable", "stack_stable",
          "ext_fallen", "ext_blocked", "ext_ok"]
CLASS_KO = {
    "spill": "방치된 유출 (기름/물)",
    "spill_marked": "조치된 유출 (표지판, 라바콘)",
    "tool_floor": "통로에 방치된 공구",
    "tool_stored": "작업대에 정리된 공구",
    "stack_unstable": "무너질 듯한 적재",
    "stack_stable": "반듯한 적재",
    "ext_fallen": "쓰러진 소화기",
    "ext_blocked": "앞이 가로막힌 소화기",
    "ext_ok": "정상 비치 소화기",
    "worker": "작업자",
    "cone": "라바콘",
    "danger_sign": "DANGER 표지",
    "cart": "운반 카트",
    **TOOL_KO,
}
# 위험(True) / 안전(False). worker, 라바콘, 표지, 공구 종류는 판정 단위가 아님 (공구는 놓인 자리로)
HAZARD = {"spill": True, "spill_marked": False, "tool_floor": True, "tool_stored": False,
          "stack_unstable": True, "stack_stable": False, "ext_fallen": True, "ext_blocked": True, "ext_ok": False}
# 물체 종류 (위험/안전 짝): 같은 종류끼리는 YOLO 가 상태를 헷갈려도 "종류는 맞힘" 으로 본다
KIND = {"spill": "spill", "spill_marked": "spill", "tool_floor": "tool", "tool_stored": "tool",
        "stack_unstable": "stack", "stack_stable": "stack", "ext_fallen": "ext", "ext_blocked": "ext", "ext_ok": "ext",
        "worker": "worker", "cone": "marker", "danger_sign": "marker", "cart": "equipment", **{t: "tool" for t in TOOL_TYPES}}
KIND_KO = {"spill": "바닥 유출", "tool": "공구", "stack": "적재", "ext": "소화기", "worker": "작업자", "marker": "영역 표시",
           "equipment": "장비"}
DETAIL = {
    "spill": "바닥에 기름이나 물이 퍼져 있는데 아무 조치가 없어 미끄럼 사고 위험",
    "spill_marked": "유출이 있지만 미끄럼 주의 표지판과 라바콘으로 조치됨",
    "tool_floor": "공구가 통로 바닥에 방치되어 걸려 넘어질 위험",
    "tool_stored": "공구가 작업대 위에 정리되어 있음",
    "stack_unstable": "적재물이 기울거나 삐져나와 무너질 위험",
    "stack_stable": "적재물이 반듯하게 쌓여 있음",
    "ext_fallen": "소화기가 바닥에 쓰러져 있어 화재 때 바로 쓸 수 없음",
    "ext_blocked": "소화기 앞을 적재물이 가로막아 화재 때 꺼낼 수 없음",
    "ext_ok": "소화기가 제자리에 보이게 비치됨",
}

# 경고: 작업자가 위험물에 이 거리 안으로 다가가거나 위험 영역에 들어가면 (닿기 직전)
# 작업자 보행속도 1.25 m/s 기준, 1.0 m 는 도달까지 0.8 초라 "1초 이상" 여유를 못 채움 -> 1.5 m 로 올림 (1.5 / 1.25 = 1.2 초)
TOUCH_WARN_M = 1.5
# 위험(확정)은 강하게, 주의(안전·위험 미확정)는 약하게 — 거리는 같고(1.5 m) 문구·진동 세기로만 구분
VOICE_TEXT_HAZARD = "경고! 경고! 멈추세요! 위험 요소가 식별되었습니다."
VOICE_TEXT_CAUTION = "주의하세요. 발밑을 확인하세요."

# 작업자-위험물 근접 기록 거리 (음성 경고 없이 기록만, 평가 집계용). 바디캠 위치(=작업자 위치) 기준
PROXIMITY_WARN_M = 2.0

# 로그 색상 (터미널, 디버그 드로잉)
RISK_RGBA = {
    "high": (1.0, 0.30, 0.31, 1.0),
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
