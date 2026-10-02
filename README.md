# 🏭 창고 안전 순찰 AI 에이전트 · Isaac Sim

> 작업자가 정해진 경로를 걸으며 **가슴 바디캠**으로 찍으면, **YOLO** 가 물체마다 **진짜 위험한지 아닌지** 판정하고,
> **안전 에이전트**가 판정을 위험물 대장에 모아 애매한 것은 **볼 수 있는 CCTV 를 골라 확대(PTZ)해서 다시 확인**합니다.
> CCTV 3대는 작업자와 위험물 사이 거리를 재서 가까워지면 경고하고, 순찰이 끝나면 **조치 지시서**(우선순위, 위치, 조치 방법)를 만듭니다.
> 물체마다 위험/안전 **정답표를 미리 만들어 두고** 판정 결과를 채점합니다 (판정에는 안 씀).
> 환경과 물체는 **NVIDIA 실사 에셋** (창고, 상자, 팔레트, 소화기, 표지판, 공구 실물 스캔, 작업자) 을 씁니다.

<p align="center"><img src="docs/architecture.png" width="900" alt="에이전트 구조"></p>

제4회 경남AI·SW경진대회 (③ 제조·피지컬 AI Agent) 출품작 · IBDP 팀 (문상균, 이승혜, 조현준)

---

## ✨ 할 수 있는 것

| | 기능 | 스크립트 | 실행 환경 |
|---|---|---|---|
| 🏗 | **장면 만들기**: NVIDIA 창고 + 위험/안전 물체 + 걷는 작업자, 정답표 저장 | `build_scene.py` | Isaac Sim |
| 📸 | **학습 데이터 자동 생성**: 바디캠, CCTV, 자유 시점에서 촬영 + 정답 박스 자동 계산 | `generate_dataset.py` | Isaac Sim |
| 🧠 | **YOLO 학습**: 위험한 상태와 안전한 상태를 따로 가르쳐서 YOLO 가 직접 구분 | `train_yolo.py` | 일반 파이썬 |
| 🤖 | **순찰 + 에이전트**: 바디캠 판정, CCTV 접근 경고, CCTV 확대 재확인, 조치 지시서, 정답표 채점 | `run_patrol.py` | Isaac Sim |
| 📊 | **여러 시나리오 평가**: 순찰을 시드별로 돌려 채점 표 (바디캠만 vs 에이전트) | `eval_patrol.py` | 일반 파이썬 (내부에서 Isaac) |
| 🎬 | **시연 영상**: 녹화한 화면과 에이전트 기록을 1920x1080 영상으로 | `make_video.py` | 일반 파이썬 |
| 📝 | **제출 문서**: 개발완료보고서, 기술설명서 (수치는 채점 결과에서) | `make_docs.py` | 일반 파이썬 |
| 🖥 | **전체 시연**: Isaac 창을 단계별로 띄워 보여주고 조치 지시서를 엶 | `demo_all.py` | 일반 파이썬 (내부에서 Isaac) |

---

## 🔁 전체 흐름

```
 ① 장면 + 정답표 ──► ② 학습 데이터 (4000장) ──► ③ YOLO 학습 ──► ④ 순찰 + 에이전트 ──► ⑤ 조치 지시서 + 정답표 채점
   build_scene        generate_dataset            train_yolo       run_patrol              eval_patrol
```

---

## 🤖 에이전트가 하는 일 (`factory_safety/agent.py`)

<p align="center"><img src="docs/workflow.png" width="900" alt="에이전트 흐름"></p>

| 요소 | 구현 |
|---|---|
| **목표 (Goal)** | 순찰 한 바퀴 동안 위험물을 찾아 위험/안전을 판정하고, 작업자 접근을 경고하고, 조치 지시서를 만든다 |
| **계획 (Planning)** | 순찰 전에 도면(소화기 6곳, 작업대 2곳)으로 **점검표**를 만들고 지점마다 볼 수 있는 CCTV 를 계산. 순찰이 끝나면 못 본 지점을 CCTV 확대 일정으로 바꿈 |
| **판단 (Reasoning)** | 재확인이 필요한 물체를 고름: 바디캠에 1~2프레임만 보이고 지나감, 판정 표 1등이 70% 미만, CCTV 경고로 처음 알게 된 물체. 2초 기다렸다가 그사이 바디캠이 확정하면 취소 |
| **도구 (Tool)** | 바디캠, CCTV 3대, **CCTV PTZ** (방향·화각 명령), YOLO, 바닥 투영 거리 계산, 조치 지시서(HTML) |
| **기억 (Memory)** | **위험물 대장**: 위치, 출처별 판정 표, 확신도, 본 횟수, 접근 경고 횟수, 상태 (확정 / 재확인 대기 / 재확인 완료 / 현장 확인 필요 / 기각 / 병합) |
| **피드백 (Feedback)** | 확대 판정을 대장에 합쳐 판정 수정, 확대 화면(높이 5 m)에서 구한 위치로 대장 위치를 고치고 중복이면 병합. 못 찾거나 확신 50% 미만이면 **다른 CCTV 로 재시도**, 끝까지 안 되면 **현장 확인 요청** (나중에 바디캠이 확정하면 취소). 같은 물체가 다른 종류로 잡히면 병합 |

**도면 제약**: 소화기는 도면의 거치 자리에만, 작업대 공구는 작업대 위에만 있다고 보고 위치를 맞춥니다. 벽 밖으로 1 m 넘게 나간 바닥 추정은 버리고, 랙 안에 떨어진 추정은 랙 면으로 밀어냅니다.

**CCTV 고르기**: 재확인할 점과 CCTV 사이 선분이 랙(높이 6 m, CCTV 5 m 보다 높음)·기둥이나 대장에 있는 적재물(약 1.7 m)에 막히는지, 거리가 32 m 안인지 계산해서 가까운 순으로 시도합니다. PTZ 는 목표점을 향하고, 화면 세로가 4 m → 2.6 m 가 되게 두 장 찍습니다.

```
[00:01] <재확인 계획> F03 남쪽 작업 구역: 바디캠에 1프레임만 보이고 지나감 (정상 비치 소화기?) → cctv_west 확대로 재확인 예약 (10 m, cctv_south 는 F01 적재에 가림)
[00:03] <CCTV 선택> cctv_west PTZ 를 F03 남쪽 작업 구역 쪽으로 돌려 확대 (화각 23°, 10 m)
[00:03] <재확인> F03 cctv_west 확대 결과 안전 정상 비치 소화기 89% 확인
[00:08] <위치 보정> F06 확대 화면으로 위치를 (+7.2, +16.4) → (+6.8, +17.2) 로 고침
[00:08] <판정 수정> F06 cctv_east 확대 결과 반듯한 적재 88% → 무너질 듯한 적재 에서 안전 반듯한 적재 로 고침
[00:08] <접근 경고> cctv_south: 작업자 ↔ 방치된 유출 (기름/물) 1.4 m  |  서쪽 통로 → 대장 F09 우선순위 올림
[00:25] <재확인 취소> F12 기다리는 동안 바디캠이 다시 보고 확정 → CCTV 확대 안 함
[00:47] <현장 확인> C6 소화기 6 (동쪽 랙 남쪽 끝): CCTV 로 확인 못 함 → 작업자에게 현장 확인 요청
[00:47] <재확인> F17 cctv_east 확대 결과 위험 쓰러진 소화기 91% 확인
```

**조치 지시서** (`outputs/agent/dashboard_seed<시드>.html`): 평면도(번호 = 우선순위), 조치 목록 (긴급/높음/보통, 위치, 조치 방법, 근거), 점검표, 에이전트 기록. 우선순위 점수 = 위험 종류별 심각도 + 접근 경고 횟수 × 2.

---

## 🚧 위험한 상태 vs 안전한 상태 (YOLO 클래스)

같은 종류의 물체를 **위험한 상태와 안전한 상태로 함께** 놓고, YOLO 가 상태까지 구분합니다.

| 종류 | 위험 (🔴) | 안전 (🟢) | 실사 에셋 |
|---|---|---|---|
| 바닥 유출 | `spill` 기름/물 웅덩이 + 쓰러진 통·양동이, 아무 조치 없음 | `spill_marked` 같은 유출 + **미끄럼 주의 표지판, 라바콘** | 웅덩이는 직접 만든 광택 재질, 통·표지판·라바콘은 NVIDIA |
| 공구·자재 | `tool_floor` 통로 바닥에 흩어진 드릴·클램프·가위·나무토막 | `tool_stored` **작업대 위**에 정리 | YCB 실물 스캔 공구, NVIDIA 작업대 |
| 적재 | `stack_unstable` 맨 위 층이 밀려나 기울고 팔레트 밖으로 삐져나옴 (상자가 떨어져 있기도) | `stack_stable` 반듯하게 쌓인 상자 | NVIDIA 팔레트, 골판지 상자 |
| 소화기 | `ext_fallen` 바닥에 쓰러짐 · `ext_blocked` 앞을 상자가 가로막음 | `ext_ok` 랙 끝 제자리에 보이게 비치 | NVIDIA 소화기 |
| 작업자 | `worker` (CCTV 거리 측정용) | | NVIDIA 건설 작업자 + 직접 만든 걷기 동작 |

**배치** (시나리오마다 무작위): 유출 7자리 중 3곳 (45% 조치됨), 통로 공구 6자리 중 2~3곳, 작업대 2개, 팔레트 6자리 중 4곳 (절반 불안정), 소화기 6곳 (정상 60%, 쓰러짐 20%, 가로막힘 20%). 한 시나리오에 물체 약 18개.

---

## 🧑‍🏭 판정과 채점

**판정** (실시간, 정답 안 씀, `factory_safety/inspection.py` + `agent.py`)

| | 방법 |
|---|---|
| 바디캠 | YOLO 추적 (ByteTrack) 으로 같은 물체에 번호를 붙이고, 3프레임 이상 같은 판정이면 **위험/안전** 을 한 번 알림. 가까이서 판정이 바뀌면 다시 알림 |
| CCTV 3대 | YOLO 로 작업자와 위험 물체를 찾고, 박스 아래쪽 (유출은 가운데) 을 바닥에 투영해 공장 좌표를 구한 뒤 **거리가 2 m 안이면 접근 경고** (같은 물체는 5초에 한 번) |

**채점** (끝나고, 정답표와 비교)

- 정답표: 시나리오를 만들 때 물체마다 `id, 클래스, 위험 여부, 위치, 구역` 을 `outputs/eval/answer_key_seed<시드>.json` 에 미리 저장
- 바디캠: 매 프레임 Replicator 정답 박스 (어느 물체인지 경로까지) 와 YOLO 박스를 겹침으로 맞추고, 물체마다 모인 판정을 정답과 비교 → **위험을 위험으로 / 위험을 안전으로 오판 / 안전을 위험으로 오판 / 놓침 / 경로에서 안 보임**
- CCTV: 실제 작업자 위치와 물체 위치로 진짜 거리를 구해 경고가 맞았는지, 위치·거리 오차가 얼마인지
- 에이전트: 최종 위험물 대장을 **창고 전체 물체**(경로에서 안 보이는 것 포함)와 위치·종류로 맞춰 비교. **바디캠만**(바디캠이 확정한 것만, 바디캠 판정 그대로) 과 **에이전트**(재확인, 점검표 포함) 를 같은 기준으로 비교

---

## 📊 결과

### YOLO (YOLO26s, 960 px, 80 epoch, 합성 데이터 4000장, 학습 약 2.2시간)

데이터: 학습 3429장 + 검증 571장 (바디캠 2208, CCTV 1182, 자유 시점 610, 물체 없는 장면 263).

| 클래스 | 위험 | mAP50 | | 클래스 | 위험 | mAP50 |
|---|:-:|:-:|---|---|:-:|:-:|
| `spill` | 🔴 | 0.884 | | `spill_marked` | 🟢 | 0.962 |
| `tool_floor` | 🔴 | 0.865 | | `tool_stored` | 🟢 | 0.906 |
| `stack_unstable` | 🔴 | 0.959 | | `stack_stable` | 🟢 | 0.954 |
| `ext_fallen` | 🔴 | 0.912 | | `ext_ok` | 🟢 | 0.926 |
| `ext_blocked` | 🔴 | 0.898 | | `worker` | | 0.965 |

**전체 mAP50 0.923, mAP50-95 0.733.** 혼동 행렬에서 위험/안전 짝끼리 헷갈린 비율은 1~4% 이고, 틀린 것은 대부분 상태 혼동이 아니라 배경으로 놓친 경우입니다.

<p align="center"><img src="docs/yolo_confusion.png" width="440" alt="정규화 혼동 행렬"> <img src="docs/yolo_training.png" width="440" alt="학습 곡선"></p>

학습된 가중치는 `outputs/yolo/warehouse/weights/best.pt` 에 들어 있어서 데이터 생성과 학습 없이 바로 순찰을 돌릴 수 있습니다.

### 순찰 채점 (학습에 안 쓴 배치 10개, 한 바퀴씩, `eval_patrol.py --seeds 0 1 2 3 4 5 6 7 8 9`)

**에이전트 최종 위험물 대장** (창고 전체 물체 174개, 경로에서 안 보이는 물체 포함)

| 항목 | 바디캠만 | 에이전트 (재확인 + 점검표) |
|---|:-:|:-:|
| 위험 물체를 위험으로 | 43/72 (60%) | **58/72 (81%)** |
| 안전 물체를 안전으로 | 50/102 (49%) | **98/102 (96%)** |
| 위험을 안전으로 / 안전을 위험으로 오판 | 0 / 0 | **0 / 0** |
| 상태까지 정확 | 93/174 | **156/174** |
| 없는 위험 보고 | 4 | 2 |
| 현장 확인 요청 (그중 실제 물체) | - | 9 (5) |

- **바디캠만**: 바디캠이 3프레임 이상 확정한 물체만, 바디캠 판정 그대로. **에이전트**: CCTV 확대 재확인, 점검표, 위치 보정·병합까지 거친 최종 대장
- 재확인 105건 실행 (요청 136건 중 24건은 기다리는 동안 바디캠이 확정해서 취소): 다시 찾음 96, 판정 고침 2 (둘 다 정답과 맞게), 다른 CCTV 로 재시도 5, 현장 확인 15, 오검출로 뺌 5
- 끝내 못 찾은 위험 14개 중 **12개가 통로 바닥의 작은 공구** (YOLO 에서 가장 약한 클래스). 바디캠에 한 번도 검출되지 않으면 재확인 대상에도 못 오름
- 없는 위험 보고 2건은 남쪽 작업 구역에 붙어 있는 유출·적재물 사이에서 추적이 섞이거나 위치가 틀어져 적재물이 한 번 더 올라간 경우 (시드 5)
- 동쪽 랙 남쪽 끝 소화기는 기둥과 적재물에 가려 CCTV 로 볼 수 없는 경우가 많아 현장 확인으로 넘어감 (정답표 확인 결과 실제 물체)

**바디캠 YOLO 판정** (경로에서 보인 물체 151개, 프레임 단위 채점)

| 항목 | 결과 |
|---|:-:|
| 위험 물체를 위험으로 판정 | **61/66 (92%)** |
| 안전 물체를 안전으로 판정 | **82/85 (96%)** |
| 위험을 안전으로 / 안전을 위험으로 오판 | **0 / 0** |
| 정답 없는 곳에 위험 박스 | 11 / 2340 프레임 |

**CCTV 접근 경고** (2 m, CCTV 3대)

| 항목 | 결과 |
|---|:-:|
| 작업자가 위험물 2 m 안으로 다가간 사건 | 12 |
| 경고 성공 | **8/12** |
| 오경보 | 4번 |
| 작업자 위치 오차 / 거리 오차 (중앙값) | **0.30 m / 0.29 m** |

- 놓친 접근 경고 4건은 모두 북쪽 끝 유출 자리: 그곳을 보는 CCTV 2대가 약 29 m 떨어져 작업자가 너무 작게 찍힘 (거리 계산은 20 m 안만). 북쪽 CCTV 를 더 두면 해결되는 배치 문제

시나리오별 표는 [`outputs/eval/patrol_results.md`](outputs/eval/patrol_results.md), 물체별 판정은 `outputs/eval/inspection_seed<시드>.json`, 조치 지시서는 `outputs/agent/dashboard_seed<시드>.html`.
시드마다 약 4.3분 (Isaac 안 YOLO 는 CPU). RTX 렌더링이 매번 조금씩 달라서 같은 시드라도 결과가 한두 개 달라질 수 있습니다.

---

## 📁 폴더 구조

```
factory-safety-isaac/
├── README.md
├── requirements.txt          일반 파이썬 패키지 (YOLO 학습, 평가, 테스트)
├── docs/                     구조도, 평면도, YOLO 혼동 행렬과 학습 곡선, 보고서 그림
├── factory_safety/           ── 핵심 패키지 ──
│   ├── config.py             클래스 (위험/안전), 에셋 주소, 경고 거리
│   ├── warehouse.py          ★ 창고 배치: 순찰 경로, 물체 자리, 작업대, CCTV 위치, 구역 이름
│   ├── scenario.py           ★ 위험/안전 물체 무작위 배치 + 정답표
│   ├── scene.py              USD 장면: 창고 참조, 물체 묶음 + 의미 라벨, 작업자, 카메라
│   ├── walk_anim.py          작업자 걷기 동작 (뼈대에 직접 만듦)
│   ├── walker.py             정해진 경로 걷기 + 가슴 바디캠 흔들림
│   ├── agent.py              ★ 안전 에이전트: 점검표 계획, 위험물 대장, CCTV 선택·확대 재확인, 조치 지시서, 채점
│   ├── dashboard.py          조치 지시서 HTML (평면도, 조치 목록, 점검표, 기록)
│   ├── inspection.py         바닥 투영, 바디캠 프레임 채점, CCTV 거리 경고
│   ├── detector.py           YOLO 래퍼 (추적 포함)
│   ├── dataset.py            학습 데이터 촬영 시점, 후처리, YOLO 형식
│   ├── geometry.py           카메라 투영, 회전
│   ├── isaac_utils.py        Isaac Sim 버전 차이 흡수, Replicator 도우미
│   └── report.py             터미널 로그
├── scripts/
│   ├── build_scene.py        [Isaac] 장면 + 정답표
│   ├── generate_dataset.py   [Isaac] YOLO 합성 데이터
│   ├── run_patrol.py         [Isaac] 순찰 + 에이전트, 조치 지시서, 채점
│   ├── train_yolo.py         [파이썬] YOLO 학습
│   ├── compare_yolo.py       [파이썬] YOLO 가중치 비교
│   ├── eval_patrol.py        [파이썬] 여러 시나리오 평가 (내부에서 Isaac)
│   ├── demo_all.py           [파이썬] 전체 시연 (내부에서 Isaac)
│   ├── make_video.py         [파이썬] 시연 영상 (run_patrol --record 결과로)
│   ├── make_figures.py       [파이썬] 구조도, 흐름도
│   ├── make_docs.py          [파이썬] 개발완료보고서, 기술설명서 (docx)
│   └── plot_layout.py        [파이썬] 평면도
├── tests/test_core.py        Isaac 없이 도는 테스트
└── outputs/                  결과물 (저장소에는 eval/ 채점 결과와 yolo/warehouse/weights/best.pt 만)
```

`★` 두 파일을 고치면 경로, 물체 자리, 배치 확률이 장면, 학습 데이터, 순찰, 채점에 한꺼번에 반영됩니다.

---

## 🛠 준비물과 설치

| 항목 | 내용 |
|---|---|
| Isaac Sim | **6.0.1** (pip 설치) · [공식 문서](https://docs.isaacsim.omniverse.nvidia.com/) |
| GPU | RTX 계열 NVIDIA GPU (확인한 환경: RTX 4060 Ti 16 GB) |
| 인터넷 | NVIDIA 에셋 서버에서 창고와 소품을 불러옴 (처음 한 번은 느림) |
| 일반 파이썬 | 3.12, YOLO 학습과 평가용 |

```powershell
cd C:\dev\factory-safety-isaac          # 한글, 공백 없는 경로 추천

# 1) Isaac Sim 6.x 가상환경 (Python 3.12 필수, 용량 큼)
py -3.12 -m venv .venv-isaac
.venv-isaac\Scripts\python -m pip install --upgrade pip
.venv-isaac\Scripts\python -m pip install "isaacsim[all,extscache]==6.0.1.0" --extra-index-url https://pypi.nvidia.com
setx ISAACSIM_PYTHON "C:\dev\factory-safety-isaac\.venv-isaac\Scripts\python.exe"
# Isaac 안에서 YOLO 와 추적을 돌리려면 (Isaac 의 torch 는 건드리지 않게 --no-deps)
.venv-isaac\Scripts\python -m pip install --no-deps ultralytics==8.4.166 ultralytics-thop polars lap

# 2) 학습용 가상환경
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
# Windows 에서는 위 명령이 CPU 전용 torch 를 깔아요. GPU 학습용으로 CUDA 빌드로 바꿔주기
.venv\Scripts\python -m pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu126 --force-reinstall --no-deps
```

> Isaac Sim 첫 실행 때 NVIDIA Omniverse 라이선스(EULA) 동의를 물어봐요. 터미널에서 `Yes` 를 입력하거나 환경 변수 `OMNI_KIT_ACCEPT_EULA=YES`.
> `setx` 후에는 **VSCode를 완전히 껐다 켜야** 환경 변수가 적용돼요.

**확인** (Isaac 없이 1초): `.venv\Scripts\python -m pytest tests -q` → `19 passed`

---

## 🚀 빠른 시작

```powershell
# 1) 장면 확인 (위에서 본 창고, 정답표 저장)
& $env:ISAACSIM_PYTHON scripts/build_scene.py

# 2) 학습 데이터 4000장 (약 30분) → YOLO 학습 (약 2시간). 학습된 가중치가 들어 있으니 건너뛰어도 됨
& $env:ISAACSIM_PYTHON scripts/generate_dataset.py --num 4000 --scenario-every 40
.venv\Scripts\python scripts/train_yolo.py --epochs 80

# 3) 순찰 (바디캠 화면 / 관제 화면 / CCTV 화면)
& $env:ISAACSIM_PYTHON scripts/run_patrol.py
& $env:ISAACSIM_PYTHON scripts/run_patrol.py --view top
& $env:ISAACSIM_PYTHON scripts/run_patrol.py --view cctv_west

# 4) 여러 시나리오 채점, 전체 시연
.venv\Scripts\python scripts/eval_patrol.py --seeds 0 1 2 3 4
.venv\Scripts\python scripts/demo_all.py
```

VSCode 에서는 `Ctrl+Shift+P` → **Tasks: Run Task** 에 전부 들어 있어요.

---

## 📖 스크립트별 사용법

<details>
<summary><b>run_patrol.py</b> · 순찰과 채점</summary>

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--weights` | `outputs/yolo/warehouse/weights/best.pt` | YOLO 가중치 |
| `--seed` | 무작위 | 위험 요소 배치 시드 (같으면 같은 배치) |
| `--laps` | `1` | 몇 바퀴 (한 바퀴 59 m, 걸음 1.25 m/s 로 약 47초) |
| `--view` | `bodycam` | `bodycam` / `top` (위에서) / `ptz` (CCTV 확대 재확인) / `cctv_west` `cctv_east` `cctv_south` |
| `--sim-dt` | `0` | 0보다 크면 프레임마다 고정 시간 (평가 재현용, 예 `0.0333`) |
| `--yolo-every` | `6` | 몇 프레임마다 YOLO (바디캠 + CCTV 3대) |
| `--no-cctv` | | CCTV 거리 측정과 확대 재확인 끄기 |
| `--no-recheck` | | 에이전트 재확인(CCTV 확대) 끄기 (비교용) |
| `--record` | | 영상용: 바디캠, CCTV, 확대 화면과 에이전트 상태를 저장할 폴더 (`make_video.py` 입력) |
| `--result` | `outputs/eval/inspection_seed<시드>.json` | 채점 결과 |

관제/CCTV 화면에서는 접근 경고가 나면 작업자와 위험물 사이에 빨간 선이 그려집니다.
끝나면 `outputs/agent/dashboard_seed<시드>.html` (조치 지시서), `outputs/agent/report_seed<시드>.json` (대장과 기록) 이 생깁니다.
</details>

<details>
<summary><b>generate_dataset.py</b> · YOLO 합성 데이터</summary>

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--num` | `500` | 이미지 수 |
| `--scenario-every` | `40` | 몇 장마다 배치를 바꿀지 (배치는 처음에 전부 만들어 둠) |
| `--rt-subframes` | `8` | 프레임당 렌더링 반복 |
| `--val-every` | `7` | 7장 중 1장을 검증용 |
| `--no-light-random` `--no-post` | | 조명 무작위화 / 흔들림·노이즈 끄기 |
| `--gui` | | 창을 띄워 찍히는 장면 보기 |

**촬영 시점**: 바디캠 55% (경로 위 가슴 높이 1.2~1.55 m, 절반은 물체 쪽을 봄), CCTV 30% (3대 근처에서 위치·각도를 흔들고 작업자를 시야에 둠), 자유 15% (물체 주변 1.0~2.2 m).
**출력**: `outputs/dataset/{images,labels}/{train,val}`, `data.yaml`, `README.txt` (클래스별 라벨 수)
</details>

<details>
<summary><b>train_yolo.py</b> · YOLO 학습</summary>

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--model` | `yolo26s.pt` | 시작 가중치 (YOLO26: NMS 없는 출력, 작은 물체용 라벨 할당) |
| `--imgsz` | `960` | 데이터 해상도 그대로 |
| `--epochs` | `100` | 4000장 기준 80이면 충분 |
| `--batch` | `16` | YOLO26s, 960 에서 약 11 GB |
| `--name` | `warehouse` | 결과 `outputs/yolo/<name>/weights/best.pt` |
</details>

---

## 🧭 좌표계

NVIDIA `warehouse_multiple_shelves.usd` 그대로. 단위 미터, +Z 위, yaw 0 = +X, 반시계 +.
벽 y = -12 (남), 18 (북). 랙 3줄 x = -9 / 0 / +9, y = -3.8 ~ 12.4. 통로 2개 (서쪽 x≈-4.5, 동쪽 x≈+4.5).

---

## 🔧 고치고 싶을 때

| 하고 싶은 것 | 고칠 곳 |
|---|---|
| 순찰 경로 | `warehouse.py` 의 `PATH_WAYPOINTS` |
| 물체가 놓일 자리 | `warehouse.py` 의 `SPILL_SLOTS`, `TOOL_FLOOR_SLOTS`, `STACK_SLOTS`, `TABLES`, `EXT_MOUNTS` |
| CCTV 위치와 각도 | `warehouse.py` 의 `CCTVS` |
| 몇 개씩, 위험 확률 | `scenario.py` 의 `sample_scenario()` |
| 물체 모양 (기울기, 상자 수, 공구 종류) | `scene.py` 의 `_build_spill`, `_build_tool`, `_build_stack`, `_build_ext` |
| 접근 경고 거리 | `config.py` 의 `PROXIMITY_WARN_M` |
| 걷는 속도, 바디캠 높이 | `walker.py` 의 `PathWalker` |
| 재확인 조건, CCTV 확대 화각, 확신 기준 | `agent.py` 의 `SafetyAgent` 상수 (`LOW_SHARE`, `PTZ_WINDOWS`, `PTZ_SURE` 등) |
| 조치 방법, 심각도 | `agent.py` 의 `ACTIONS`, `SAFE_NOTES` |

**새 위험 요소 추가**: `config.py` 의 `CLASSES`, `HAZARD`, `KIND` 에 위험/안전 두 클래스 추가 → `warehouse.py` 에 자리 → `scenario.py` 에서 배치 → `scene.py` 에 `_build_<종류>` → `pytest`.

---

## ❓ Isaac Sim 6.0 에서 겪은 문제와 해결

| 증상 | 원인과 해결 |
|---|---|
| 데이터셋 라벨이 비거나 일부 물체 라벨이 빠짐 | Replicator 는 **첫 렌더 뒤에 새로 만든 라벨 물체를 제대로 못 읽어요** (지웠다 다시 만들기, 숨기기, render product 다시 만들기 모두 불안정). 배치를 **첫 렌더 전에 전부 만들고** (`WarehouseScene.add_scenario`) 보이기/숨기기로만 바꿉니다. 숨긴 물체는 정답 박스에 안 나와요 |
| 물체 하나에 정답 박스가 여러 개 | NVIDIA 창고 소품 안에 예전 형식 라벨 (`box`, `pallet`, `sign` 등) 이 들어 있어서 묶음 라벨과 섞여요. `scene.strip_semantics()` 로 지웁니다 |
| 렌더 중 화면(어노테이터)이 비어 있음 | `app.update()` 만으로는 rgb 어노테이터가 비어서 나와요 (크기 0). 화면이 필요한 프레임은 `rep.orchestrator.step()` 으로 렌더 |
| 카메라를 돌렸는데 이전 화면이 찍힘 | `rt_subframes=1` 은 한 단계 전 화면이 나와요. 2 이상으로 렌더 |
| 작업자가 T 자세로 굳어 있음 | NVIDIA 사람 모델(Reallusion 뼈대)과 NVIDIA 걷기 애니메이션은 관절 이름이 달라 그대로 안 붙어요. `walk_anim.py` 가 허벅지, 정강이, 팔 방향을 매 프레임 정해 걷기 동작을 직접 만듭니다. 재생은 타임라인 시각만 맞춤 |
| 데이터 생성이 점점 느려지다 GPU 메모리 부족 | render product 를 지웠다 다시 만들면 메모리가 다 안 풀려요. 지금은 render product 를 하나만 씁니다 |
| `Invalid CUDA 'device=0'` | `.venv` 에 CPU 전용 torch 가 깔린 경우. 설치 섹션의 CUDA 빌드 명령 |
| Isaac 안 YOLO 가 느림 | Isaac 6.0.1 의 torch 는 CPU 전용이라 YOLO 도 CPU 로 돌아요 (`--sim-dt` 로 평가하면 결과는 같음) |
| 표준 입력으로 넘긴 파이썬이 멈춤 (Windows) | 데이터 로더 작업자 프로세스가 스크립트를 다시 못 불러와요. 파일로 저장해서 실행 |
| `No module named 'isaacsim'` | 일반 파이썬으로 실행한 경우. `$ISAACSIM_PYTHON` 으로 실행 |

---

## ✅ 검증 상태

| 부분 | 상태 |
|---|---|
| 배치, 정답표, 경로, 걷기, 판정·채점 로직, 바닥 투영, CCTV 거리, USD 장면, 에이전트 (재확인·재시도·병합·위치 보정·조치 지시서) | 테스트 19개 통과 (`tests/test_core.py`) |
| 실제 Isaac Sim 6.0.1 (RTX 4060 Ti 16 GB, Windows 11) 장면, 라벨, 작업자 걷기 | ✅ |
| 학습 데이터 4000장, YOLO 학습, 시나리오 10개 순찰 + 에이전트 채점 | ✅ (위 결과 표) |
| 실제 사진, 실제 CCTV | ❌ 아직 안 함 (합성 데이터만으로 학습) |

---

## 🗺 다음 단계

- **실사 테스트**: 휴대폰으로 찍은 실제 창고·실습실 사진에서 합성 데이터만으로 학습한 YOLO 시험
- **실제 CCTV 보정**: 카메라 내부·외부 파라미터를 재서 바닥 투영 거리 정확도 확인
- **동적 사고**: Isaac Sim 6.1 사고 이벤트 확장(넘어짐, 유출, 화재) 으로 "적재물이 쓰러지는 순간" 데이터
- **VR 체험**: 바디캠 시점을 XR 로 연결해 작업자 시점 안전 교육

---

## 📦 사용한 것

- NVIDIA Isaac Sim 6.0, OpenUSD, Omniverse Replicator
- NVIDIA Isaac Sim 에셋 (창고 `Simple_Warehouse`, 소품, 사람 모델) · YCB 물체 (공구 실물 스캔). 에셋은 NVIDIA 에셋 서버에서 참조로 불러오고 저장소에 포함하지 않음
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (AGPL-3.0, 상업적으로 쓸 계획이면 라이선스 확인 필요)
- numpy, Pillow, matplotlib

<sub>제조 피지컬 AI Agent 공모전 (국립창원대 앵커사업단) 출품용 프로젝트</sub>
