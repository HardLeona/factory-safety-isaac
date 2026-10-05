"""작업자 언어별 안내 문장 (한국어, 영어, 중국어, 일본어 + 베트남어 일부).

안전 안내는 잘못 번역되면 위험해서, 정해진 문장 틀과 현장 용어집(사람이 검수한 번역)으로 만든다.
번역 모델(NLLB) 시험에서 "곡괭이 -> gravel pit", "안전화 -> seat belt", "지게차 -> traffic zone" 처럼
현장 용어를 틀리는 경우가 많았기 때문. 틀에 없는 자유 문장(관리자 메모)만 번역 모델을 쓰고 "자동 번역" 으로 표시한다.
베트남어(vi)는 시연(손가락 1, DANGER 표지 설명) 범위만 넣었다 — 전체 TBM/위험 안내 문구까지는 아직 현장 용어집 검수가
안 끝나서, 실제 배포 전에는 반드시 원어민 검수가 필요하다.
"""
import math

LANGS = {"ko": "한국어", "en": "English", "zh": "中文", "ja": "日本語", "vi": "Tiếng Việt"}

# 근접 위험 경보 음성 (config.VOICE_TEXT_HAZARD 와 ko 문구를 맞춤)
VOICE = {
    "ko": "경고! 경고! 멈추세요! 위험 요소가 식별되었습니다.",
    "en": "Warning! Warning! Stop! A hazard has been identified.",
    "zh": "警告！警告！请停下！已识别到危险因素。",
    "ja": "警告！警告！止まってください！危険要素が検出されました。",
}

# 근접 주의 경보 음성 (config.VOICE_TEXT_CAUTION 과 ko 문구를 맞춤)
VOICE_CAUTION = {
    "ko": "주의하세요. 발밑을 확인하세요.",
    "en": "Caution. Watch your step.",
    "zh": "请注意，留意脚下。",
    "ja": "注意してください。足元を確認してください。",
}

# 물체 이름
NAMES = {
    "spill": {"ko": "방치된 기름·물 유출", "en": "an untreated oil or water spill", "zh": "未处理的油水泄漏", "ja": "放置された油・水の漏れ"},
    "spill_marked": {"ko": "표지판과 라바콘으로 조치된 유출", "en": "a spill marked with signs and cones", "zh": "已设置警示牌和锥桶的泄漏",
                     "ja": "標識とカラーコーンで対処済みの漏れ"},
    "tool_floor": {"ko": "통로 바닥에 방치된 공구", "en": "tools left on the aisle floor", "zh": "遗留在通道地面上的工具", "ja": "通路の床に放置された工具"},
    "tool_stored": {"ko": "작업대에 정리된 공구", "en": "tools stored on the workbench", "zh": "整齐放在工作台上的工具", "ja": "作業台に片付けられた工具"},
    "stack_unstable": {"ko": "무너질 듯한 적재물", "en": "an unstable stack that may collapse", "zh": "可能倒塌的不稳定堆垛", "ja": "崩れそうな積み荷"},
    "stack_stable": {"ko": "반듯하게 쌓인 적재물", "en": "a neatly stacked load", "zh": "码放整齐的堆垛", "ja": "きちんと積まれた積み荷"},
    "ext_fallen": {"ko": "쓰러진 소화기", "en": "a fallen fire extinguisher", "zh": "倒地的灭火器", "ja": "倒れた消火器"},
    "ext_blocked": {"ko": "앞이 가로막힌 소화기", "en": "a fire extinguisher blocked by boxes", "zh": "被箱子挡住的灭火器", "ja": "前を箱でふさがれた消火器"},
    "ext_ok": {"ko": "제자리에 있는 소화기", "en": "a fire extinguisher in its proper place", "zh": "放在规定位置的灭火器", "ja": "所定の位置にある消火器"},
    "cone": {"ko": "라바콘", "en": "a traffic cone", "zh": "交通锥", "ja": "カラーコーン"},
    "danger_sign": {"ko": "DANGER 표지", "en": "a DANGER sign", "zh": "“危险”警示牌", "ja": "DANGER標識", "vi": "biển báo NGUY HIỂM"},
    "hammer": {"ko": "망치", "en": "a hammer", "zh": "锤子", "ja": "ハンマー"},
    "screwdriver": {"ko": "드라이버", "en": "a screwdriver", "zh": "螺丝刀", "ja": "ドライバー"},
    "saw": {"ko": "톱", "en": "a hand saw", "zh": "手锯", "ja": "のこぎり"},
    "power_saw": {"ko": "전동톱", "en": "a power saw", "zh": "电锯", "ja": "電動のこぎり"},
    "pickaxe": {"ko": "곡괭이", "en": "a pickaxe", "zh": "镐", "ja": "つるはし"},
    "shovel": {"ko": "삽", "en": "a shovel", "zh": "铁锹", "ja": "スコップ"},
    "wrench": {"ko": "렌치", "en": "a wrench", "zh": "扳手", "ja": "レンチ"},
    "drill": {"ko": "전동 드릴", "en": "a power drill", "zh": "电钻", "ja": "電動ドリル"},
    "cart": {"ko": "운반 카트", "en": "a hand cart", "zh": "手推运货车", "ja": "台車"},
    "machine_conveyor": {"ko": "컨베이어", "en": "a conveyor", "zh": "传送带", "ja": "コンベア"},
}

# 장비 설명 (손가락 하나)
INFO = {
    "hammer": {"ko": "못을 박거나 물건을 두드리는 공구입니다. 쓸 때는 손가락을 조심하고 보안경을 쓰세요.",
               "en": "It is a hand tool for driving nails and striking objects. Watch your fingers and wear safety glasses.",
               "zh": "这是用来钉钉子和敲击物体的工具。使用时注意手指，并佩戴护目镜。",
               "ja": "釘を打ったり物をたたいたりする工具です。指に注意し、保護メガネを着用してください。"},
    "screwdriver": {"ko": "나사를 조이거나 푸는 공구입니다. 끝이 날카로우니 몸 쪽으로 힘을 주지 마세요.",
                    "en": "It is a tool for tightening and loosening screws. The tip is sharp, so never push it toward your body.",
                    "zh": "这是用来拧紧和拧松螺丝的工具。刀头锋利，不要朝自己身体方向用力。",
                    "ja": "ねじを締めたり緩めたりする工具です。先端が鋭いので、体の方向に力を入れないでください。"},
    "saw": {"ko": "나무나 금속을 자르는 손톱입니다. 장갑을 끼고, 쓰지 않을 때는 날에 덮개를 씌우세요.",
            "en": "It is a hand saw for cutting wood or metal. Wear gloves and cover the blade when it is not in use.",
            "zh": "这是用来切割木材或金属的手锯。请戴好手套，不用时给锯片套上护套。",
            "ja": "木材や金属を切る手のこぎりです。手袋を着用し、使わないときは刃にカバーをかけてください。"},
    "power_saw": {"ko": "회전하는 날로 자재를 자르는 전동톱입니다. 쓰기 전에 날 덮개를 확인하고 보안경과 귀마개를 착용하세요. 전원을 끈 뒤에만 옮기세요.",
                  "en": "It is a power saw that cuts with a spinning blade. Check the blade guard, wear safety glasses and ear protection, and move it only with the power off.",
                  "zh": "这是用旋转锯片切割材料的电锯。使用前检查锯片护罩，佩戴护目镜和耳塞，断电后才能搬动。",
                  "ja": "回転する刃で材料を切る電動のこぎりです。使う前に刃のカバーを確認し、保護メガネと耳栓を着用してください。電源を切ってから運んでください。"},
    "pickaxe": {"ko": "땅을 파거나 단단한 것을 깨는 곡괭이입니다. 휘두르기 전에 주변 2미터 안에 사람이 없는지 확인하세요.",
                "en": "It is a pickaxe for digging and breaking hard material. Make sure no one is within two meters before swinging it.",
                "zh": "这是用来挖掘和破碎坚硬物体的镐。挥动前请确认周围两米内没有人。",
                "ja": "地面を掘ったり硬いものを砕いたりするつるはしです。振る前に周囲2メートル以内に人がいないことを確認してください。"},
    "shovel": {"ko": "흙이나 자재를 퍼서 옮기는 삽입니다. 허리가 아니라 무릎을 굽혀 들어 올리세요.",
               "en": "It is a shovel for moving soil or materials. Lift with your knees, not your back.",
               "zh": "这是用来铲运泥土或材料的铁锹。请弯曲膝盖抬起，不要弯腰。",
               "ja": "土や資材をすくって運ぶスコップです。腰ではなく膝を曲げて持ち上げてください。"},
    "wrench": {"ko": "볼트와 너트를 조이거나 푸는 렌치입니다. 크기에 맞는 것을 쓰고, 미끄러지지 않게 몸 쪽으로 당기세요.",
               "en": "It is a wrench for tightening and loosening bolts and nuts. Use the right size and pull it toward you so it does not slip.",
               "zh": "这是用来拧紧和拧松螺栓螺母的扳手。请使用合适的尺寸，并朝自己方向拉动以防打滑。",
               "ja": "ボルトやナットを締めたり緩めたりするレンチです。合ったサイズを使い、滑らないよう手前に引いてください。"},
    "drill": {"ko": "구멍을 뚫는 전동 드릴입니다. 회전부에 장갑이나 옷이 말려 들어가지 않게 조심하고 보안경을 착용하세요.",
              "en": "It is a power drill for making holes. Keep gloves and sleeves away from the rotating part and wear safety glasses.",
              "zh": "这是用来钻孔的电钻。注意不要让手套或衣袖卷入旋转部件，并佩戴护目镜。",
              "ja": "穴をあける電動ドリルです。回転部に手袋や袖が巻き込まれないよう注意し、保護メガネを着用してください。"},
    "cart": {"ko": "상자를 실어 옮기는 운반 카트입니다. 무거운 상자를 아래에, 앞이 보이는 높이까지만 싣고, 손잡이를 두 손으로 잡아 천천히 끄세요. "
                   "모퉁이에서는 멈춰 사람을 확인하고, 카트에 올라타지 마세요.",
             "en": "It is a hand cart for moving boxes. Put heavy boxes at the bottom and load only as high as you can see over. "
                   "Hold the handle firmly, pull slowly, stop at corners to check for people, and never ride on it.",
             "zh": "这是搬运箱子的手推运货车。重的箱子放在下面，堆放高度不要挡住视线。握紧把手慢慢拉动，在拐角处停下确认行人，不要站在车上。",
             "ja": "箱を運ぶ台車です。重い箱を下にして、前が見える高さまで積んでください。取っ手をしっかり持ってゆっくり引き、"
                   "曲がり角では止まって人を確認し、台車に乗らないでください。"},
    "ext": {"ko": "작은 불을 초기에 끄는 소화기입니다. 안전핀을 뽑고 호스를 불 쪽으로 향한 뒤 손잡이를 누르세요.",
            "en": "It is a fire extinguisher for putting out small fires. Pull the pin, aim the hose at the fire, and squeeze the handle.",
            "zh": "这是用于扑灭初期小火的灭火器。拔出安全销，将喷管对准火焰，然后按压把手。",
            "ja": "初期の小さな火を消す消火器です。安全ピンを抜き、ホースを火に向けてレバーを握ってください。"},
    "ext_fallen": {"ko": "지금 바닥에 쓰러져 있어 제자리에 다시 걸어야 합니다.", "en": "It has fallen to the floor and must be put back on its mount.",
                   "zh": "它现在倒在地上，需要挂回原位。", "ja": "今は床に倒れているので、元の位置に戻す必要があります。"},
    "ext_blocked": {"ko": "지금 앞이 상자로 막혀 있어 급할 때 꺼내기 어렵습니다. 앞을 비워 주세요.",
                    "en": "Boxes are blocking it, so it is hard to reach in an emergency. Please clear the space in front.",
                    "zh": "现在前面被箱子挡住，紧急时难以取用。请清空前方。", "ja": "今は前が箱でふさがれていて、緊急時に取り出しにくいです。前を空けてください。"},
    "ext_ok": {"ko": "제자리에 잘 비치되어 있습니다.", "en": "It is in its proper place.", "zh": "它放在规定的位置。", "ja": "所定の位置にきちんと設置されています。"},
    "stack": {"ko": "팔레트에 쌓인 적재물입니다.", "en": "It is a load stacked on a pallet.", "zh": "这是码放在托盘上的货物。", "ja": "パレットに積まれた積み荷です。"},
    "stack_unstable": {"ko": "위쪽 상자가 기울어 무너질 수 있으니 가까이 가지 마세요.", "en": "The top boxes are leaning and may fall, so keep away.",
                       "zh": "上层箱子倾斜，可能倒塌，请不要靠近。", "ja": "上の箱が傾いていて崩れるおそれがあるので、近づかないでください。"},
    "stack_stable": {"ko": "반듯하게 쌓여 있습니다.", "en": "It is stacked neatly.", "zh": "码放整齐。", "ja": "きちんと積まれています。"},
    "spill": {"ko": "바닥에 흘린 기름이나 물입니다. 미끄러질 수 있으니 밟지 말고 돌아가세요.",
              "en": "It is oil or water spilled on the floor. You may slip, so walk around it.",
              "zh": "这是洒在地上的油或水。可能会滑倒，请不要踩踏，绕行通过。", "ja": "床にこぼれた油や水です。滑るおそれがあるので、踏まずに迂回してください。"},
    "spill_marked": {"ko": "표지판과 라바콘으로 표시된 유출입니다. 표시된 곳 안으로 들어가지 마세요.",
                     "en": "This spill is marked with signs and cones. Do not enter the marked area.",
                     "zh": "这是已用警示牌和锥桶标出的泄漏区域。请不要进入标记区域。", "ja": "標識とカラーコーンで示された漏れです。表示された場所に入らないでください。"},
    "cone": {"ko": "위험 영역을 표시하는 라바콘입니다. 라바콘으로 둘러친 곳 안으로 들어가지 마세요.",
             "en": "It is a traffic cone marking a danger zone. Do not enter the area enclosed by cones.",
             "zh": "这是标示危险区域的交通锥。请不要进入锥桶围起的区域。", "ja": "危険エリアを示すカラーコーンです。コーンで囲まれた場所に入らないでください。"},
    "danger_sign": {"ko": "출입 금지 구역을 알리는 DANGER 표지입니다. 표지 너머로 들어가지 마세요.",
                    "en": "It is a DANGER sign for a no-entry zone. Do not go past the sign.",
                    "zh": "这是表示禁止进入区域的“危险”警示牌。请不要越过警示牌。", "ja": "立入禁止区域を示すDANGER標識です。標識の先に入らないでください。",
                    "vi": "Đây là biển báo NGUY HIỂM, cho biết khu vực cấm vào. Không được đi qua biển báo này."},
    "machine_conveyor": {"ko": "상자를 옮기는 컨베이어입니다. 작동 중일 때는 롤러가 맞물리는 진입부에 손이나 옷이 끼일 수 있으니 "
                               "손을 넣지 마세요. 걸린 물건을 뺄 때는 반드시 먼저 기계를 끄세요.",
                         "en": "It is a conveyor that moves boxes. While it is running, your hand or clothing can get caught "
                               "at the entry roller nip, so never reach in. Always turn it off before clearing a jam.",
                         "zh": "这是搬运箱子的传送带。运转时手或衣物可能被入口滚轮的咬合处夹住，请勿将手伸入。清理卡阻物前务必先关闭机器。",
                         "ja": "箱を運ぶコンベアです。稼働中は入口ローラーの噛み合わせ部に手や衣服が巻き込まれるおそれがあるので、"
                               "手を入れないでください。詰まりを取り除く前には必ず機械を止めてください。"},
}

DIRS = {"front": {"ko": "정면", "en": "ahead", "zh": "正前方", "ja": "正面", "vi": "phía trước"},
        "left": {"ko": "왼쪽", "en": "on your left", "zh": "左侧", "ja": "左側", "vi": "bên trái"},
        "right": {"ko": "오른쪽", "en": "on your right", "zh": "右侧", "ja": "右側", "vi": "bên phải"},
        "back": {"ko": "뒤쪽", "en": "behind you", "zh": "后方", "ja": "後方", "vi": "phía sau"}}
ZONES = {"남쪽 작업 구역": {"en": "the south work area", "zh": "南侧作业区", "ja": "南側作業エリア"},
         "북쪽 보관 구역": {"en": "the north storage area", "zh": "北侧存放区", "ja": "北側保管エリア"},
         "북쪽 구역": {"en": "the north area", "zh": "北侧区域", "ja": "北側エリア"},
         "서쪽 통로": {"en": "the west aisle", "zh": "西侧通道", "ja": "西側通路"},
         "동쪽 통로": {"en": "the east aisle", "zh": "东侧通道", "ja": "東側通路"},
         "가운데 랙": {"en": "the center rack", "zh": "中间货架", "ja": "中央ラック"}}
ZONE_KIND = {"cone": {"ko": "라바콘으로 둘러친 위험 영역", "en": "a danger zone enclosed by cones", "zh": "锥桶围起的危险区域", "ja": "カラーコーンで囲まれた危険エリア"},
             "sign": {"ko": "DANGER 표지가 선 위험 영역", "en": "a danger zone with a DANGER sign", "zh": "设有危险警示牌的危险区域", "ja": "DANGER標識のある危険エリア"},
             "agent": {"ko": "위험물 주변 위험 영역", "en": "a danger zone around a hazard", "zh": "危险物周边的危险区域", "ja": "危険物の周りの危険エリア"}}
ACTIONS = {"spill": {"ko": "유출물 제거, 표지판·라바콘 설치", "en": "remove the spill and set up signs and cones", "zh": "清除泄漏物并设置警示牌和锥桶",
                     "ja": "漏れを除去し、標識とカラーコーンを設置"},
           "stack_unstable": {"ko": "기울어진 상자 다시 쌓기", "en": "restack the leaning boxes", "zh": "重新码放倾斜的箱子", "ja": "傾いた箱を積み直す"},
           "ext_blocked": {"ko": "소화기 앞 적재물 치우기", "en": "clear the boxes in front of the extinguisher", "zh": "清理灭火器前的箱子", "ja": "消火器の前の荷物を片付ける"},
           "ext_fallen": {"ko": "소화기 제자리에 걸기", "en": "put the extinguisher back on its mount", "zh": "将灭火器挂回原位", "ja": "消火器を元の位置に戻す"},
           "tool_floor": {"ko": "바닥 공구 작업대로 회수", "en": "return the tools on the floor to the workbench", "zh": "将地面工具收回工作台", "ja": "床の工具を作業台に戻す"}}

# 손동작 명령
COMMANDS = {1: {"ko": "장비 설명", "en": "Equipment info", "zh": "设备说明", "ja": "設備の説明", "vi": "Thông tin thiết bị"},
            2: {"ko": "오늘의 TBM", "en": "Today's TBM", "zh": "今日班前会", "ja": "本日のTBM", "vi": "TBM hôm nay"},
            3: {"ko": "관리자 호출·SOS", "en": "Call manager / SOS", "zh": "呼叫管理人员／SOS", "ja": "管理者呼び出し・SOS", "vi": "Gọi quản lý"}}

T = {
    "ack": {"ko": "{n}번, {cmd}.", "en": "{n}: {cmd}.", "zh": "{n}号：{cmd}。", "ja": "{n}番、{cmd}。", "vi": "Số {n}: {cmd}."},
    "equip": {"ko": "{dir} {d}미터, {name}. {info}", "en": "{name_cap}, {d} meters {dir}. {info}",
              "zh": "{dir}{d}米处是{name}。{info}", "ja": "{dir}{d}メートル先に{name}があります。{info}",
              "vi": "{name_cap}, cách {d} mét {dir}. {info}"},
    "equip_none": {"ko": "지금 화면에서 알아본 장비가 없습니다. 장비를 화면 가운데에 비춰 주세요.",
                   "en": "I cannot recognize any equipment right now. Please point the camera at it.",
                   "zh": "现在画面中没有识别到设备。请把设备对准画面中央。", "ja": "今の画面では設備を認識できません。設備を画面の中央に映してください。",
                   "vi": "Hiện tại không nhận diện được thiết bị nào trong khung hình. Vui lòng hướng camera vào thiết bị."},
    "equip_refuse": {"ko": "검증된 매뉴얼에서 이 장비에 대한 근거를 찾지 못해 답변하지 않습니다. 관리자를 불렀으니 직접 확인해 주세요.",
                     "en": "I could not find a verified manual passage for this, so I will not guess. A manager has been called to help.",
                     "zh": "未在已验证的手册中找到相关依据，因此不作回答。已呼叫管理人员，请您直接确认。",
                     "ja": "検証済みのマニュアルに根拠が見つからないため回答しません。管理者を呼びましたので直接確認してください。",
                     "vi": "Không tìm thấy căn cứ trong tài liệu đã được kiểm chứng cho thiết bị này nên tôi sẽ không đoán. "
                           "Đã gọi quản lý, vui lòng xác nhận trực tiếp."},
    "tool_floor_note": {"ko": "바닥에 있으니 작업대로 옮겨 주세요.", "en": "It is on the floor, so please put it on the workbench.",
                        "zh": "它在地面上，请放回工作台。", "ja": "床にあるので作業台に戻してください。"},
    "tool_stored_note": {"ko": "작업대에 잘 정리되어 있습니다.", "en": "It is stored properly on the workbench.", "zh": "它已整齐地放在工作台上。",
                         "ja": "作業台にきちんと片付けられています。"},
    "haz_head": {"ko": "지금 화면에서 위험 요소 {n}개를 찾았습니다.", "en": "I found {n} hazards in view.",
                 "zh": "当前画面中发现{n}个危险因素。", "ja": "今の画面で危険要素を{n}つ見つけました。"},
    "haz_item": {"ko": "{dir} {d}미터, {name}.", "en": "{name_cap}, {d} meters {dir}.", "zh": "{dir}{d}米处有{name}。", "ja": "{dir}{d}メートル先に{name}。"},
    "haz_zone": {"ko": "{dir} {d}미터에 {kind}이 있습니다. 들어가지 마세요.", "en": "There is {kind}, {d} meters {dir}. Do not enter.",
                 "zh": "{dir}{d}米处有{kind}，请不要进入。", "ja": "{dir}{d}メートル先に{kind}があります。入らないでください。"},
    "haz_inside": {"ko": "지금 {kind} 안에 있습니다. 바로 밖으로 나오세요.", "en": "You are inside {kind}. Leave it now.",
                   "zh": "您现在位于{kind}内，请立即离开。", "ja": "今、{kind}の中にいます。すぐに外に出てください。"},
    "haz_none": {"ko": "지금 화면에서 위험 요소를 찾지 못했습니다. 주의해서 이동하세요.", "en": "No hazards found in view. Keep moving carefully.",
                 "zh": "当前画面中没有发现危险因素。请小心行走。", "ja": "今の画面では危険要素は見つかりませんでした。注意して移動してください。"},
    "tbm_head": {"ko": "오늘 {date} TBM입니다.", "en": "Today's TBM, {date}.", "zh": "今天{date}的班前会内容。", "ja": "本日{date}のTBMです。"},
    "tbm_work": {"ko": "오늘 작업: {items}.", "en": "Today's work: {items}.", "zh": "今日作业：{items}。", "ja": "本日の作業：{items}。"},
    "tbm_risk": {"ko": "주의할 위험: {items}.", "en": "Watch out for: {items}.", "zh": "注意危险：{items}。", "ja": "注意すべき危険：{items}。"},
    "tbm_rule": {"ko": "지킬 것: {items}.", "en": "Rules: {items}.", "zh": "必须遵守：{items}。", "ja": "守ること：{items}。"},
    "tbm_todo": {"ko": "지난 순찰 조치 {n}건: {items}.", "en": "{n} actions from the last patrol: {items}.", "zh": "上次巡检的{n}项整改：{items}。",
                 "ja": "前回の巡回の対応{n}件：{items}。"},
    "todo_item": {"ko": "{zone} {action}", "en": "{action} in {zone}", "zh": "{zone}{action}", "ja": "{zone}で{action}"},
    "manager_sos": {"ko": "관리자 호출·SOS 신고 접수. 위치({zone})와 바디캠 화면을 관리자에게 보냈습니다. 움직이지 말고 기다리세요.",
                    "en": "Manager called / SOS sent. Your manager has your location, {zone}, and bodycam view. Stay still and wait for help.",
                    "zh": "已呼叫管理人员／发送SOS。您的位置（{zone}）和随身摄像头画面已发送给管理人员。请不要移动，等待帮助。",
                    "ja": "管理者呼び出し・SOS通報を送信しました。現在地（{zone}）とボディカメラの映像を管理者に送りました。動かずに待ってください。"},
}

# TBM (작업 전 안전 회의) 항목: 관리자가 고르는 목록
TBM_ITEMS = {
    "work_move_boxes": {"ko": "운반 카트로 북쪽 보관 구역의 상자 4개를 남쪽 작업 구역으로 옮기기",
                        "en": "moving 4 boxes from the north storage area to the south work area with the hand cart",
                        "zh": "用手推运货车把北侧存放区的4个箱子搬到南侧作业区", "ja": "台車で北側保管エリアの箱4つを南側作業エリアへ運ぶ"},
    "work_restack": {"ko": "동쪽 통로 팔레트 다시 쌓기", "en": "restacking pallets in the east aisle", "zh": "重新码放东侧通道的托盘", "ja": "東側通路のパレットの積み直し"},
    "work_inspect_ext": {"ko": "소화기 점검", "en": "fire extinguisher inspection", "zh": "灭火器检查", "ja": "消火器の点検"},
    "work_clean": {"ko": "바닥 청소와 유출 정리", "en": "floor cleaning and spill cleanup", "zh": "地面清洁和泄漏清理", "ja": "床の清掃と漏れの片付け"},
    "work_receive": {"ko": "남쪽 작업 구역 입고 작업", "en": "receiving goods in the south work area", "zh": "南侧作业区收货作业", "ja": "南側作業エリアでの入荷作業"},
    "risk_forklift": {"ko": "지게차 이동", "en": "forklift traffic", "zh": "叉车行驶", "ja": "フォークリフトの走行"},
    "risk_slip": {"ko": "미끄러운 바닥", "en": "slippery floors", "zh": "地面湿滑", "ja": "滑りやすい床"},
    "risk_falling": {"ko": "떨어지는 적재물", "en": "falling loads", "zh": "坠落的货物", "ja": "落下する積み荷"},
    "risk_trip": {"ko": "통로의 공구에 걸려 넘어짐", "en": "tripping over tools in the aisle", "zh": "被通道上的工具绊倒", "ja": "通路の工具につまずく"},
    "rule_ppe": {"ko": "안전모와 안전화 착용", "en": "wear a hard hat and safety shoes", "zh": "佩戴安全帽、穿安全鞋", "ja": "ヘルメットと安全靴の着用"},
    "rule_zone": {"ko": "라바콘과 DANGER 표지 안쪽 출입 금지", "en": "do not enter areas marked by cones or DANGER signs", "zh": "禁止进入锥桶和危险警示牌内侧",
                  "ja": "カラーコーンとDANGER標識の内側は立入禁止"},
    "rule_report": {"ko": "유출이나 위험을 보면 바로 보고", "en": "report spills or hazards immediately", "zh": "发现泄漏或危险立即报告", "ja": "漏れや危険を見つけたらすぐに報告"},
    "rule_lift": {"ko": "무거운 물건은 두 명이 들기", "en": "lift heavy items with two people", "zh": "重物须两人搬运", "ja": "重い物は二人で持つ"},
}

SEP = {"ko": ", ", "en": ", ", "zh": "、", "ja": "、"}


MONTHS_EN = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December")


def date_text(iso, lang):
    """'2026-10-03' -> 10월 3일 / October 3 / 10月3日."""
    y, m, d = (int(v) for v in iso.split("-"))
    return {"ko": f"{m}월 {d}일", "en": f"{MONTHS_EN[m - 1]} {d}", "zh": f"{m}月{d}日", "ja": f"{m}月{d}日"}[lang]


def name(cls, lang):
    return NAMES.get(cls, {}).get(lang) or NAMES.get(cls, {}).get("ko", cls)


def zone(z, lang):
    return z if lang == "ko" else ZONES.get(z, {}).get(lang, z)


def direction(cam_yaw, cam_xy, target_xy):
    """작업자가 바라보는 방향 기준 정면/왼쪽/오른쪽/뒤쪽."""
    a = math.atan2(target_xy[1] - cam_xy[1], target_xy[0] - cam_xy[0]) - cam_yaw
    a = math.degrees(math.atan2(math.sin(a), math.cos(a)))
    if abs(a) <= 30:
        return "front"
    if abs(a) >= 135:
        return "back"
    return "left" if a > 0 else "right"


def dist_text(d):
    return f"{d:.0f}" if d >= 2 else f"{d:.1f}"


def fmt(key, lang, **kw):
    s = T[key].get(lang) or T[key]["ko"]
    if "name" in kw and "name_cap" not in kw:
        kw["name_cap"] = kw["name"][:1].upper() + kw["name"][1:]
    return s.format(**kw)


def join(items, lang):
    return SEP[lang].join(items)
