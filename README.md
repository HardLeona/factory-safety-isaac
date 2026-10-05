# 🦺 외국인 근로자를 위한 바디캠 안전관리 에이전트 · Isaac Sim

> 작업자가 **가슴 바디캠 하나**를 차고 평소처럼 일하면, **YOLO** 가 화면에 보이는 물체마다 **진짜 위험한지 아닌지** 실시간으로 판정합니다.
> (지금 Isaac Sim 시뮬레이션은 재현 가능한 평가를 위해 정해진 경로를 걷게 했지만, 실제 배치 시에는 작업자가 평소 업무 중 착용한
> 상태로 상시 작동하는 것을 목표로 합니다. 한국어가 서툰 **외국인 근로자**도 모국어 음성으로 위험을 안내받을 수 있도록
> 손동작 명령·다국어 응답을 핵심으로 설계했습니다.)
> **안전 에이전트**는 판정을 위험물 대장에 모으고, 후보 중 **고위험 클래스가 하나라도 있으면 그 자리에서 즉시 위험으로 확정**합니다(미탐을 줄이는 쪽).
> 저위험 후보만 있어 안전·위험을 가르지 못하면 **"주의"로 분류**해 두고, 작업자가 순찰 중 **같은 물체를 바디캠으로 자연스럽게 다시 지나치면**
> **로컬 LLM(LangGraph + Qwen2.5-7B)** 이 그동안 쌓인 증거로 위험/안전을 재판단합니다 (LLM이 없거나 실패하면 규칙으로 대신 판단).
> **라바콘으로 둘러친 곳, DANGER 표지가 선 곳, 스스로 위험하다고 판단한 주변(회피형 위험만)은 위험 영역**으로 잡고,
> 작업자가 위험물에 1.5 m 안으로 다가가거나 영역에 들어서면 음성으로 경고합니다 — **위험은 "멈추세요! 위험 요소가 식별되었습니다"**,
> **주의는 "발밑을 확인하세요"** 로 세기를 구분합니다. 오늘 작업 계획(TBM)에 적힌 작업 대상 물체는 경고에서 제외합니다.
> 공구는 **망치, 드라이버, 톱, 전동톱, 곡괭이, 삽, 렌치, 전동 드릴** 중 무엇인지까지 알아봅니다.
> 작업자가 바디캠 앞에 **손가락 1~5개**를 보이면 **로컬 LLM (Qwen2.5-7B, LangGraph)** 이 도구로 상황을 보고 무엇을 알려 줄지 정해,
> 장비 설명(검증된 안전 매뉴얼 RAG 근거), 공장 위험 스캔, 오늘의 TBM, 관리자 호출, SOS 를 **작업자 언어 (중국어·영어·일본어·한국어)** 음성으로 처리합니다.
> 순찰이 끝나면 **조치 지시서**(우선순위, 위치, 조치 방법, 매뉴얼 근거)를 만듭니다.
> 물체마다 위험/안전 **정답표를 미리 만들어 두고** 판정 결과를 채점합니다 (판정에는 안 씀).
> 환경과 물체는 **NVIDIA 실사 에셋** (창고, 상자, 팔레트, 소화기, 표지판, 라바콘, 작업자) 과 **Poly Haven 공구 실물 스캔** (CC0) 을 씁니다.

<p align="center"><img src="docs/architecture.png" width="900" alt="에이전트 구조"></p>

제4회 경남AI·SW경진대회 (③ 제조·피지컬 AI Agent) 출품작 · IBDP 팀 (문상균, 이승혜, 조현준)

---

## ✨ 할 수 있는 것

| | 기능 | 스크립트 | 실행 환경 |
|---|---|---|---|
| 📥 | **공구 모델 받기**: Poly Haven 실물 스캔 공구 (망치, 드라이버, 톱, 곡괭이, 삽, 렌치) | `get_assets.py` | 일반 파이썬 |
| 📥 | **학습된 YOLO 가중치 받기**: Hugging Face ([IBDPLab/factory-safety-isaac-yolo](https://huggingface.co/IBDPLab/factory-safety-isaac-yolo))에서 `warehouse_v3`·`pinch_v1` 내려받기 | `get_model.py` | 일반 파이썬 |
| 🏗 | **장면 만들기**: NVIDIA 창고 + 위험/안전 물체 + 걷는 작업자, 정답표 저장 | `build_scene.py` | Isaac Sim |
| 📸 | **학습 데이터 자동 생성**: 바디캠·자유 시점에서 촬영 + 정답 박스 자동 계산 | `generate_dataset.py` | Isaac Sim |
| 🧠 | **YOLO 학습**: 위험한 상태와 안전한 상태를 따로 가르쳐서 YOLO 가 직접 구분, 공구는 종류별로 | `train_yolo.py`, `val_yolo.py` | 일반 파이썬 |
| 🤖 | **순찰 + 에이전트**: 바디캠 판정, 2단계 분류(즉시 확정/주의), 재관측 재판단, 위험 영역, 음성 경고, 조치 지시서, 정답표 채점 | `run_patrol.py` | Isaac Sim |
| ✋ | **손동작 명령**: 손가락 1~5 → 장비 설명, 공장 위험 스캔, TBM, 관리자 호출, SOS (작업자 언어 음성) | `run_patrol.py --story`, `test_gestures.py` | Isaac Sim + `.venv-assistant` |
| 💬 | **LLM 에이전트**: LangGraph + 로컬 Qwen2.5-7B (Ollama) 가 손동작 요청마다, 그리고 주의 물체 재관측마다 도구로 상황을 보고 판단 | `llm_agent.py`, `recheck_agent.py` | `.venv-assistant` + Ollama |
| 📚 | **매뉴얼 근거 (RAG)**: 검증된 안전 매뉴얼에서 찾은 근거로만 장비 설명·조치 문구를 답함, 근거 없으면 추측 대신 거부 | `manuals.py`, `report_agent.py` | `.venv-assistant` + Ollama |
| 🧷 | **끼임 위험 경보**: 작동 중인 기계(컨베이어 롤러)의 끼임점 1.5 m 위험구역 + 손끝 10 cm 이내 최고 등급 경보, 규칙 기반 | `run_patrol.py --pinch-demo` | Isaac Sim |
| 📊 | **여러 시나리오 평가**: 순찰을 시드별로 돌려 3열 비교표 (바디캠 프레임 원시판정 / 규칙 에이전트 / LangGraph 에이전트) | `eval_patrol.py` | 일반 파이썬 (내부에서 Isaac) |
| 🎬 | **시연 영상**: 녹화한 화면과 에이전트 기록을 1920x1080 영상으로 | `make_video.py` | 일반 파이썬 |
| 📝 | **제출 문서**: 개발완료보고서, 기술설명서, 발표자료 (수치는 채점 결과에서) | `make_docs.py`, `make_slides.py` | 일반 파이썬 |
| 🖥 | **전체 시연**: Isaac 창을 단계별로 띄워 보여주고 조치 지시서를 엶 | `demo_all.py` | 일반 파이썬 (내부에서 Isaac) |

---

## 🔁 전체 흐름

```
 ① 장면 + 정답표 ──► ② 학습 데이터 (7700장) ──► ③ YOLO 학습 ──► ④ 순찰 + 에이전트 ──► ⑤ 조치 지시서 + 정답표 채점
   build_scene        generate_dataset            train_yolo       run_patrol              eval_patrol
```

---

## 🤖 에이전트가 하는 일 (`factory_safety/agent.py`)

<p align="center"><img src="docs/workflow.png" width="900" alt="에이전트 흐름"></p>

**CCTV 는 쓰지 않습니다.** 카메라는 작업자 가슴에 달린 바디캠 하나뿐이고, 위험 판정·접근 경고·재판단 전부 이 화면만으로 합니다.

| 요소 | 구현 |
|---|---|
| **목표 (Goal)** | 순찰 한 바퀴 동안 위험물을 찾아 위험/안전을 판정하고, 작업자 접근을 경고하고, 조치 지시서를 만든다 |
| **계획 (Planning)** | 순찰 전에 도면(소화기 6곳, 작업대 2곳)으로 **점검표**를 만듦. 오늘 TBM(`data/tbm_today.json`)을 읽어 **오늘 작업 대상 물체는 거리 경고에서 제외**하도록 설정 |
| **판단 (Reasoning)** | 바디캠에 잠깐 보이고 지나가거나 판정이 엇갈린 물체는, 후보 클래스 중 **고위험(`HAZARD_SEVERITY`="high")이 하나라도 있으면 그 자리에서 즉시 위험으로 확정**(미탐을 줄이는 쪽). 전부 저위험이면 **"주의"로 분류**해 안전·위험 어느 쪽으로도 단정하지 않음 |
| **도구 (Tool)** | 바디캠, YOLO, 바닥 투영 거리 계산, 음성 경고(위험/주의 톤 구분), 조치 지시서(HTML), 매뉴얼 RAG. 손동작 요청과 주의 물체 재판단은 LLM 이 도구를 골라 부름 (아래) |
| **기억 (Memory)** | **위험물 대장**: 위치, 출처별 판정 표, 확신도, 본 횟수, 접근 경고 횟수, 상태 (확정 / 주의 / 현장 확인 필요 / 기각 / 병합) |
| **피드백 (Feedback)** | "주의" 물체를 작업자가 순찰 중 **같은 자리에서 바디캠으로 다시 포착**하면, 이전 증거 + 새 증거를 합쳐 `recheck_agent.py`(LangGraph + 로컬 LLM) 가 위험/안전 중 하나로 재판단. LLM 이 없거나 실패하면 규칙(합쳐진 후보에 고위험이 있으면 위험 확정, 없으면 주의 유지)으로 대신 정함. 끝까지 재관측되지 않으면 "주의"로 남아 조치 지시서에 "현장 확인 권고"로 표시 |

**도면 제약**: 소화기는 도면의 거치 자리에만, 작업대 공구는 작업대 위에만 있다고 보고 위치를 맞춥니다. 벽 밖으로 1 m 넘게 나간 바닥 추정은 버리고, 랙 안에 떨어진 추정은 랙 면으로 밀어냅니다.

**애매한 판정 처리 — 클래스 심각도 표** (`HAZARD_SEVERITY`, `agent.py`)

| 클래스 | 심각도 | 애매하면 |
|---|:-:|---|
| `spill` (유출), `stack_unstable` (불안정 적재) | high | 즉시 위험 확정 (미끄럼·붕괴는 결과가 치명적) |
| `ext_fallen` (쓰러진 소화기) | high | 즉시 위험 확정 (화재 대응 불가) |
| `ext_blocked` (가로막힌 소화기), `tool_floor` (방치된 공구) | low | 후보가 이것뿐이면 "주의"로 분류, 재관측 때 재판단 |

```
[00:01] <계획> 목표: 순찰 한 바퀴(59 m) 동안 위험물을 찾아 위험/안전을 판정하고, ...
[00:12] <판정> 바디캠 유출(방치)  |  서쪽 통로  |  YOLO 91%  → 대장 F03
[00:19] <주의> F07 잠깐 보고 지나침 → 안전·위험을 가르지 못해 주의로 분류 (가로막힌 소화기), 다시 보이면 재판단
[00:19] <위험 영역> Z02 F03 주변 (서쪽 통로, 반지름 약 1.3 m) → 미끄럼 위험, 위험 영역으로 설정
[00:24] <음성 경고> "경고! 경고! 멈추세요! 위험 요소가 식별되었습니다." ← 작업자 ↔ F03 방치된 유출 1.2 m
[00:41] <LLM 판단> qwen2.5:7b: evidence → retrieve_manual → finish | 이전엔 가려서 못 봤지만 지금은 소화기 앞이 뚜렷이 막혀 있음
[00:41] <판정> F07 재관측 재판단 → 위험 가로막힌 소화기
[00:47] <현장 확인> C6 소화기 6 (동쪽 랙 남쪽 끝): 바디캠으로 끝까지 확인 못 함 → 작업자에게 현장 확인 요청
```

**위험 영역** (`factory_safety/zones.py`) — **회피형 위험(`AVOIDANCE_HAZARDS` = 유출·불안정 적재·방치된 공구)만** 영역 근거가 됩니다. 소화기류는 "그 자리에 가까이 가는 것"이 아니라 "화재 시 못 쓴다"는 설비 결함이라 위험 영역·거리 경고 대상이 아니고, 조치 지시서 항목으로만 올라갑니다.

| 근거 | 영역 |
|---|---|
| 라바콘 | 바닥 위치가 3 m 안으로 이어진 라바콘 2개 이상을 한 영역 (볼록 다각형 + 0.3 m). 라바콘을 둘러친 조치된 유출도 영역이 됨 |
| DANGER 표지 | 표지 주변 1.2 m (라바콘 영역 근처면 그 영역에 합침) |
| 에이전트 판단 | 회피형 위험만: 방치된 유출 (미끄럼 1.3 m), 무너질 듯한 적재 (붕괴 1.6 m), 통로에 방치된 공구 (걸려 넘어짐). 3 m 안에 모인 위험물은 한 영역. 라바콘·표지 영역 안이면 따로 안 만듦 |

**음성 경고** (`factory_safety/voice.py`): 회피형 위험에 **1.5 m** 안으로 다가가거나 위험 영역 경계 0.5 m 안에 들어서면 경보음 + 음성.
작업자 보행속도(1.25 m/s) 기준 1.5 m 는 도착까지 1.2초라 "1초 이상" 여유를 확보합니다 (1 m 는 0.8초라 모자람).
위험과 주의는 **거리가 아니라 문구·진동 세기로만 구분**합니다 — **위험**: "경고! 경고! 멈추세요! 위험 요소가 식별되었습니다." + 강한 진동. **주의**: "주의하세요. 발밑을 확인하세요." + 짧은 진동 1회.
**2 m** 안은 음성 경고 없이 **근접만 기록**(평가 집계용). 오늘 TBM 작업 대상 물체는 경고에서 제외합니다 (아래 한계 참고).
같은 대상은 12초, 경고끼리는 5.5초 간격. Isaac 창으로 순찰하면 소리가 바로 나고, 시연 영상에는 소리 트랙으로 들어갑니다.

**공구 이름**: YOLO 가 공구를 종류별(8종)로 알아보고, 위험/안전은 **놓인 자리**로 에이전트가 판단합니다.
공구 박스 아래쪽을 작업대 윗면 높이로 투영해 작업대 위면 "정리된 공구", 아니면 "통로에 방치된 공구". 조치 지시서에는 "통로에 방치된 공구 (망치, 삽)" 처럼 이름까지 나옵니다.

**손동작 명령** (`factory_safety/assistant.py`, `hand_count.py`, `i18n.py`, `llm_agent.py`): 작업자가 멈춰 서서 오른손을 아래에서 바디캠 앞으로 들어 올려 손바닥을 보이고 손가락을 폅니다.

| 손가락 | 명령 | 에이전트가 하는 일 |
|:-:|---|---|
| 1 (검지) | 장비 설명 | 바로 전 바디캠 화면 가운데의 장비 (운반 카트, 공구, 소화기 등) 가 무엇인지, 안전하게 쓰는 법, 지금 상태 (바닥에 있으면 작업대로). **매뉴얼 RAG 근거가 없으면 추측하지 않고 거부**, 관리자 호출로 넘김 |
| 2 (+중지) | 공장 위험 스캔 | 위험물 대장(바디캠으로 확정한 것)에서 공장 전체의 위험을 LLM 이 고른 순서로 (같은 종류·구역은 한 번), 위험 영역 수 |
| 3 (+약지) | 오늘의 TBM | 아침 작업 전 안전 회의 (`data/tbm_today.json`: 오늘 작업, 주의할 위험, 지킬 것, 지난 순찰 조치) |
| 4 (+새끼) | 관리자 호출 | 작업자 위치·언어, 가까운 위험, LLM 요약 메시지를 조치 지시서 알림으로 |
| 5 (손바닥) | SOS 신고 | 경보음 + 위치·바디캠 화면을 관리자·안전팀에 전달 |

- **인식**: MediaPipe Hands 로 손 관절 21점 → 손가락마다 마디가 곧은지 (각도) 와 손목에서 먼지, 엄지는 약지 뿌리까지 거리로 수를 셈. 손이 기울어도 되게 화면 방향은 안 씀. 멈춘 손에서 같은 수가 3번 연속이면 명령, 손을 내려야 다시 받음
- **판단 (LLM 에이전트, `factory_safety/llm_agent.py`)**: 명령이 오면 지금 상황 (방금 바디캠에 보인 물체, 위험물 대장, 위험 영역, 오늘 TBM, 작업자 위치) 을 넘기고,
  LangGraph 그래프에서 로컬 **Qwen2.5-7B** (Ollama, 인터넷·API 키 없음) 가 도구를 골라 부른 뒤 `finish` 로 결정합니다. 도구 없이 글로만 답하면 `finish` 를 부르라고 한 번 더 요청하고, 그래도 안 되거나 Ollama 가 꺼져 있으면 규칙으로 정합니다

  ```mermaid
  flowchart LR
    S([손동작 명령]) --> A[agent<br/>Qwen2.5-7B]
    A -- 도구 호출 --> T[tools<br/>look_around · equipment_info · retrieve_manual<br/>hazard_log · todays_tbm · worker_status]
    T --> A
    A -. 글로만 답함 .-> M[remind] -.-> A
    A -- finish --> R[결정<br/>말할 항목 · 근거 · 관리자 메시지 · 거부 여부]
    R --> V[검수한 문장 틀로 작업자 언어 음성]
  ```

  | 도구 | 하는 일 |
  |---|---|
  | `look_around` | 최근 몇 초 바디캠에 보인 물체 (번호, 이름, 거리, 방향, 화면 가운데에서 얼마나 먼지) |
  | `equipment_info(id)` | 그 물체가 무엇이고 어떻게 안전하게 쓰는지 |
  | `retrieve_manual(query)` | 검증된 안전 매뉴얼 RAG (`factory_safety/manuals.py`, `data/manuals/*.md`) 검색. 근거 문서가 없으면 "no relevant manual passage" |
  | `hazard_log(limit)` | 공장 전체 위험물 대장 (우선순위 순, 구역, 작업자와 거리) |
  | `todays_tbm` / `worker_status` | 오늘 TBM / 작업자 위치·구역 |
  | `finish(say_ids, reason_ko, manager_ko, refuse)` | 말할 항목 (도구가 준 번호만 받음), 한국어 근거, 관리자 메시지, 매뉴얼 근거 없으면 `refuse=true` |

  LLM 은 안전 문장을 직접 쓰지 않습니다 (말할 것만 고름). 손가락 1(장비 설명)은 `retrieve_manual` 로 찾은 매뉴얼 근거가 없으면 추측 없이 거부하고 관리자 호출로 넘깁니다.
  판단 근거는 에이전트 기록에 `LLM 판단` 으로 남고, 관리자 메시지는 확인된 사실 (위치, 가까운 위험) 뒤에 `AI 요약` 으로 붙습니다
- **재판단 (별도 LLM 에이전트, `factory_safety/recheck_agent.py`)**: 손동작과는 다른 트리거 — "주의" 물체를 바디캠이 같은 자리에서 다시 포착했을 때만 호출됩니다.
  `evidence` 도구로 이전 관측·새 관측·경과 시간·구역을 보고, 필요하면 `retrieve_manual` 로 근거를 찾은 뒤 `finish(confident, verdict_class, reason_ko)` 로 위험/안전을 확정하거나 "아직 못 가름"을 답합니다. `run_patrol.py --recheck-llm off` 또는 `--agent rule` 이면 LLM 대신 규칙(고위험 후보 재검토)만 씁니다
- **언어**: 안내 문장은 사람이 검수한 4개 언어 문장 틀 + 현장 용어집으로 만듭니다. 번역 모델 (NLLB-200) 을 시험했더니 "안전화 → seat belt", "지게차 → parking lot" 처럼 현장 용어를 틀려서 안전 안내에는 쓰지 않습니다. 관리자에게는 한국어로 같이 남김
- **음성**: edge-tts (Microsoft 온라인 신경망 음성, 인터넷 필요). 안 되면 Windows 음성 (한국어·영어·일본어). 만든 음성은 문장별로 저장해 다시 씀
- **시뮬레이션**: 작업자 뼈대에 손가락 1~5 자세를 직접 만듭니다 (`walk_anim.py`). 손을 아래에서 위로 들어 올리며 손바닥이 카메라를 보게 돌리고, 손가락은 위로 조금 벌려 세우고, 엄지는 손바닥 쪽으로 접습니다 (팔은 2관절 IK).
  손가락을 붙이면 비스듬한 화면에서 약지가 가운뎃손가락 뒤에 가려 36/40 이었고, 벌려서 37~38/40 (RTX 렌더가 매번 조금 달라 시험마다 한 번쯤 다름).
  시연 이야기는 `story.py` 가, `--gestures demo` 는 `DemoScript` 가 작업자 역할로 손동작을 합니다
- **프로세스**: Isaac Sim 파이썬에는 MediaPipe·LangGraph 를 같이 깔 수 없어서 (Isaac 이 고정한 패키지 버전이 바뀜) 손 인식·음성·LLM 은 `.venv-assistant` 에서 따로 띄운 프로세스 (`scripts/assistant_worker.py`) 가 맡습니다

**시연 이야기** (`factory_safety/story.py`, `run_patrol.py --story`, 작업자 언어 영어, 음성 경고는 한 번만): 작업자 역할만 정해 두고 에이전트는 바디캠 화면과 손동작으로만 압니다.

1. 시작하자마자 손가락 3 → 오늘의 TBM ("카트로 상자 4개를 북쪽 보관 구역에서 남쪽 작업 구역으로", 주의할 위험, 지킬 것)
2. 순찰하며 걷기 (에이전트가 위험/안전 판정, 위험 영역, 닿기 직전 음성 경고 한 번)
3. 북쪽 보관 구역에서 운반 카트를 3초 보고 손가락 1 → LLM 이 `look_around` → `equipment_info` 로 카트를 골라, 쓰는 법과 주의점
4. 옆 팔레트의 상자 4개를 카트에 싣고 왼손으로 카트를 끌며 걷기 (오른손은 손동작)
5. 동쪽 통로 중간에서 손가락 2 → 공장 전체 위험 스캔 (LLM 이 `hazard_log` 에서 알릴 위험과 순서를 고름)
6. 한 바퀴를 다 돌면 손가락 4 → 관리자 호출 (위치·가까운 위험 + LLM 요약), 시뮬레이션 끝 (남은 점검 지점은 "현장 확인 필요"로 표시)

**조치 지시서** (`outputs/agent/dashboard_seed<시드>.html`, 시연 이야기는 `dashboard_story_seed5.html`): 평면도(번호 = 우선순위, 위험 영역), 조치 목록 (긴급/높음/보통, 위치, 조치 방법, 근거), 위험 영역, 음성 경고 기록(위험/주의 톤 구분), 점검표, 에이전트 기록. 우선순위 점수 = 위험 종류별 심각도 + 접근 경고 횟수 × 2.
조치 문구는 고정 테이블(`ACTIONS`)이 기본이고, `--report-llm` (기본 켬) 이면 `factory_safety/report_agent.py` 가 매뉴얼(`data/manuals/*.md`)에 더 구체적인 근거가 있는지 찾아 보강하고 출처를 표시 (근거 없으면 고정 문구 그대로).

**가시성 (시연영상에서 바로 보임)**: 처리 경로(LangGraph LLM 재판단 / 규칙 폴백)와 판정 상태 전환("주의"→"위험" 등)은 콘솔(`[hh:mm] <LLM 판단> ...`)과 `--record` 녹화 화면 아래쪽 배지(`overlay.draw_status_badge`)에 함께 표시됩니다.

---

## 🚧 위험한 상태 vs 안전한 상태 (YOLO 클래스)

같은 종류의 물체를 **위험한 상태와 안전한 상태로 함께** 놓고, YOLO 가 상태까지 구분합니다.

| 종류 | 위험 (🔴) | 안전 (🟢) | 실사 에셋 |
|---|---|---|---|
| 바닥 유출 | `spill` 기름/물 웅덩이 + 쓰러진 통, 아무 조치 없음 | `spill_marked` 같은 유출 + **미끄럼 주의 표지판, 라바콘** | 웅덩이는 직접 만든 광택 재질, 통·표지판·라바콘은 NVIDIA |
| 공구 | 통로 바닥에 방치 (`tool_floor`) | **작업대 위**에 정리 (`tool_stored`) | YOLO 는 종류 8개 (`hammer` `screwdriver` `saw` `power_saw` `pickaxe` `shovel` `wrench` `drill`) 로 알아보고 자리는 에이전트가 판단. Poly Haven 실물 스캔 (CC0), 전동톱은 직접 모델링, YCB 드릴 |
| 적재 | `stack_unstable` 맨 위 층이 밀려나 기울고 팔레트 밖으로 삐져나옴 (상자가 떨어져 있기도) | `stack_stable` 반듯하게 쌓인 상자 | NVIDIA 팔레트, 골판지 상자 |
| 소화기 | `ext_fallen` 바닥에 쓰러짐 · `ext_blocked` 앞을 상자가 가로막음 | `ext_ok` 랙 끝 제자리에 보이게 비치 | NVIDIA 소화기 |
| 작업자 | `worker` (바디캠 위치 추적, 거리 계산용) | | NVIDIA 건설 작업자 + 직접 만든 걷기 동작 |
| 위험 영역 표시 | `cone` 라바콘, `danger_sign` DANGER 표지 (A자형, "위험 구역 출입 금지") | | NVIDIA 라바콘, 표지는 직접 모델링, 안쪽에 뚜껑 열린 바닥 구멍이 있기도 함 |
| 장비 | | `cart` 운반 카트 (판정 대상 아님, 손동작 1 로 쓰는 법 안내) | Poly Haven 핸드트럭 (CC0, 세움·끄는 자세·상자 0~4개), 창고에 원래 있던 평판 카트 |

**배치** (시나리오마다 무작위): 위험 영역 2곳 (라바콘 링 4~6개 60% / DANGER 표지만 40%, 경로에서 경계까지 0.4~0.9 m), 유출 3곳 (45% 조치됨),
통로 공구 2~3곳 (1~3개씩, 종류 무작위), 작업대 2개, 팔레트 4곳 (절반 불안정), 소화기 6곳 (정상 60%, 쓰러짐 20%, 가로막힘 20%).
경로 바로 옆 (0.4~0.5 m) 에 위험물 하나는 꼭 둡니다 (닿기 직전 음성 경고 시험). 공구 모델은 `python scripts/get_assets.py` 로 받습니다.

---

## 🧑‍🏭 판정과 채점

**판정** (실시간, 정답 안 씀, `factory_safety/inspection.py` + `agent.py`): 바디캠 YOLO 추적 (ByteTrack) 으로 같은 물체에 번호를 붙이고, 3프레임 이상 같은 판정이면 위험/안전을 한 번 알립니다. 후보에 고위험 클래스가 있으면 즉시 위험으로 확정하고, 저위험 후보만 있으면 "주의"로 분류합니다. 가까이서 (또는 재관측으로) 판정이 바뀌면 다시 알립니다.

**채점** (끝나고, 정답표와 비교)

- 정답표: 시나리오를 만들 때 물체마다 `id, 클래스, 위험 여부, 위치, 구역` 을 `outputs/eval/answer_key_seed<시드>.json` 에 미리 저장
- **before (바디캠 프레임 원시판정)**: `BodycamInspector` 가 매 프레임 Replicator 정답 박스 (어느 물체인지 경로까지) 와 YOLO 박스를 겹침으로 맞추고, 물체마다 모인 프레임 단위 투표를 정답과 비교한 것 — 에이전트의 2단계 분류·재판단·위험 영역 반영 **이전** 단계
- **after (에이전트 최종 대장)**: 2단계 분류 + 재관측 재판단 + 점검표까지 거친 최종 위험물 대장을, **창고 전체 물체**(경로에서 안 보이는 것 포함)와 위치·종류로 맞춰 비교한 것
- 두 열은 집계 단위가 다릅니다 (프레임/박스 단위 vs 물체 단위) — "없는 위험 판정" 같은 항목은 `false_hazard_boxes`(before, 박스 수) / `false_reports`(after, 보고 건수) 로 필드 이름부터 구분해서 나란히 보여주고 하나로 합치지 않습니다
- `eval_patrol.py --agent both` 는 **프레임 원시판정 / 규칙 에이전트(`--agent rule`) / LangGraph 에이전트(`--agent langgraph`)** 3열로 비교하고, 주의 물체가 재관측으로 재판단된 건수(`caution_rejudged`, 그중 확정 전환 `caution_rejudged_confirmed`)도 함께 보여줘서 규칙과 LLM 의 차이가 몇 건에서 나왔는지 드러냅니다

---

## 📊 결과

### YOLO (YOLO26s, 960 px, 19 클래스, 합성 데이터 7700장, 18 클래스 모델에서 이어 15 epoch)

데이터: 일반 촬영 5000장 (학습 4286 + 검증 714, 바디캠 70%·자유 시점 30%) + 공구 가까이 1500장 (학습 1286 + 검증 214)
+ 운반 카트 1200장 (학습 1029 + 검증 171, 카트 라벨 2058개).

| 클래스 | mAP50 | | 클래스 | mAP50 | | 클래스 | mAP50 |
|---|:-:|---|---|:-:|---|---|:-:|
| 🔴 `spill` | 0.920 | | 🟢 `spill_marked` | 0.937 | | `worker` | 0.950 |
| 🔴 `stack_unstable` | 0.957 | | 🟢 `stack_stable` | 0.960 | | `cone` | 0.976 |
| 🔴 `ext_fallen` | 0.907 | | 🟢 `ext_ok` | 0.938 | | `danger_sign` | 0.950 |
| 🔴 `ext_blocked` | 0.909 | | `hammer` | 0.862 | | `screwdriver` | 0.828 |
| `saw` | 0.806 | | `power_saw` | 0.901 | | `pickaxe` | 0.845 |
| `shovel` | 0.819 | | `wrench` | 0.887 | | `drill` | 0.915 |
| `cart` (운반 카트) | 0.794 | | | | | | |

**전체 mAP50 0.898, mAP50-95 0.689** (정밀도 0.920, 재현율 0.827). 공구 8종과 운반 카트가 0.79~0.92 로 가장 약하고, 공구의 위험/안전은 YOLO 가 아니라 에이전트가 놓인 자리로 정합니다. (이 YOLO 모델 자체는 CCTV 제거와 무관하게 그대로 씁니다 — 학습 데이터의 촬영 시점 비중만 바디캠 위주로 재조정했습니다.)

<p align="center"><img src="docs/yolo_confusion.png" width="440" alt="정규화 혼동 행렬"> <img src="docs/yolo_training.png" width="440" alt="학습 곡선"></p>

학습된 가중치는 저장소에 들어있지 않고 Hugging Face ([IBDPLab/factory-safety-isaac-yolo](https://huggingface.co/IBDPLab/factory-safety-isaac-yolo))에 올려 두었습니다.
`python scripts/get_model.py` 로 `outputs/yolo/warehouse_v3/weights/best.pt` 에 받으면 데이터 생성과 학습 없이 바로 순찰을 돌릴 수 있습니다.

**실제 사진 시험** (`detect_image.py`): 책상 위 공구를 위에서 가까이 찍은 휴대폰 사진 3장 (커터칼, 일자 드라이버, 가위) 에서 학습한 클래스인 드라이버를 놓치고 손잡이를 작업자로 봤습니다.
학습 데이터의 드라이버는 화면의 0.05~4% 크기 (바디캠으로 1~5 m 앞 바닥) 인데 이 사진은 화면의 13% 를 채워서, 실제 현장 적용 전에 현장 사진으로 미세조정이 필요합니다.

### 순찰 채점

> ⚠️ **바디캠 단독 + 2단계 분류 + 재관측 재판단으로 구조를 바꾸면서, 옛 CCTV 기반 수치(접근 경고·재확인 횟수 등)는 더 이상 이 시스템에 해당하지 않아 표를 내렸습니다.**
> 최신 수치는 `python scripts/eval_patrol.py --seeds 0 1 2 3 4 --agent both` (Isaac Sim 필요) 로 다시 만들어 `outputs/eval/patrol_results_compare.md` 에 저장하세요 — 프레임 원시판정 / 규칙 에이전트 / LangGraph 에이전트 3열 비교, 주의 분류·재판단 건수가 함께 나옵니다.

### 손동작 명령 (`test_gestures.py`, 경로 8곳 x 손가락 1~5, 곳마다 조명 무작위)

순찰 때와 똑같이 손을 올리고 → 2초쯤 들고 → 내리는 동안의 바디캠 화면 (0.1초마다) 을 명령 확정 필터에 넣어 채점했습니다. (손 인식·손동작 로직은 CCTV 제거와 무관해 이 결과는 그대로 유효합니다.)

| 항목 | 결과 |
|---|:-:|
| 보인 손가락 수대로 명령이 한 번 나옴 | **37/40** |
| 다른 명령이 나옴 | 3 (손가락 2 → 1 두 번, 3 → 2 한 번: 어둡거나 역광인 손에서 MediaPipe 가 검지 관절을 접힌 것으로 봄) |
| 명령이 안 나옴 | 0 |
| 손을 다 올린 화면 중 수를 맞힘 | 478/520 |
| 손을 올리고 내리는 중 잘못 실행된 명령 | **0** (멈춘 손만 셈) |

`score_gestures.py` 는 저장한 시험 화면 (`--raw`) 으로 Isaac 없이 다시 채점합니다 (손 인식 기준을 바꿀 때, JPEG 로 저장한 화면이라 실시간 시험과 한두 번 다를 수 있음).

`eval_patrol.py` 를 돌리면 시나리오별 표가 `outputs/eval/patrol_results.md` (`--agent both` 는 `patrol_results_compare.md` 도), 물체별 판정은 `outputs/eval/inspection_seed<시드>.json`, 조치 지시서는 `outputs/agent/dashboard_seed<시드>.html` 에 생깁니다 (옛 CCTV 시절 결과는 `outputs/eval/_legacy_pre_cctv_removal/` 참고).
시드마다 약 5분 (Isaac 안 YOLO 는 CPU). RTX 렌더링이 매번 조금씩 달라서 같은 시드라도 결과가 한두 개 달라질 수 있습니다.

---

## ⚠️ 한계

- **작업 맥락 필터는 종류 단위**: 오늘 TBM 작업 항목에 구역 정보가 있으면(`TBM_TASK_ZONE`) 그 구역의 물체만 거리 경고에서 빼지만, 구역 정보가 없는 작업 항목은 **종류 단위로 제외**됩니다(그 종류의 물체가 창고 어디에 있든 전부 경고 제외). 더 정교하게 하려면 TBM 항목마다 구역을 지정해야 합니다.
- **CCTV 없음**: 바디캠이 안 보는 곳(사각지대, 뒤돌아선 순간)은 끝까지 "주의"나 미확인으로 남을 수 있습니다 — 재관측은 순찰 경로를 다시 지날 때만 일어나는 수동적 트리거라, 능동적으로 다시 확인하러 가는 기능은 없습니다.
- **합성 데이터만**: 실제 현장 사진으로 미세조정 전입니다 (위 YOLO 실제 사진 시험 참고).
- 그 밖의 한계(음성·언어, 처리 속도, LLM 응답 시간 등)는 `scripts/make_docs.py` 가 만드는 개발완료보고서 5.2 절 참고.

---

## 📁 폴더 구조

```
factory-safety-isaac/
├── README.md
├── requirements.txt          일반 파이썬 패키지 (YOLO 학습, 평가, 테스트)
├── docs/                     구조도, 평면도, YOLO 혼동 행렬과 학습 곡선, 보고서 그림
├── factory_safety/           ── 핵심 패키지 ──
│   ├── config.py             클래스 (위험/안전, 공구 8종, 라바콘·표지), 에셋 주소, 경고 거리
│   ├── warehouse.py          ★ 창고 배치: 순찰 경로, 물체 자리, 작업대, 구역 이름
│   ├── scenario.py           ★ 위험/안전 물체 무작위 배치 + 정답표
│   ├── scene.py              USD 장면: 창고 참조, 물체 묶음 + 의미 라벨, 작업자, 바디캠
│   ├── walk_anim.py          작업자 걷기 동작과 손동작 자세 (뼈대에 직접 만듦)
│   ├── walker.py             정해진 경로 걷기 + 가슴 바디캠 흔들림
│   ├── agent.py              ★ 안전 에이전트: 점검표 계획, 위험물 대장, 2단계 분류, 재관측 재판단 연결, 위험 영역·음성 경고, 조치 지시서, 채점
│   ├── zones.py              위험 영역 (라바콘 묶음, DANGER 표지, 에이전트 판단 — 회피형 위험만)
│   ├── voice.py              음성 경고 (경보음 + Windows 한국어 음성, 위험/주의 두 문구)
│   ├── overlay.py            화면에 박스와 한국어 이름, 손 관절, 처리 경로/상태 전환 배지 그리기
│   ├── assistant.py          손동작 명령 1~5 실행 (장비 설명, 공장 스캔, TBM, 호출, SOS)
│   ├── llm_agent.py          [.venv-assistant] 손동작 LLM 에이전트 (LangGraph + 로컬 Qwen2.5-7B, 도구 6개)
│   ├── manuals.py            [.venv-assistant] 매뉴얼 RAG (data/manuals/*.md 청크 → 로컬 임베딩·벡터스토어 → 검색, 근거 없으면 거부)
│   ├── recheck_agent.py      [.venv-assistant] 주의 물체 재관측 재판단 LLM 에이전트 (LangGraph + 로컬 Qwen2.5-7B, 바디캠 재포착이 트리거)
│   ├── report_agent.py       [.venv-assistant] 조치 지시서 문구를 매뉴얼 근거로 보강하는 LLM 에이전트 (LangGraph + 로컬 Qwen2.5-7B)
│   ├── story.py              시연 이야기 (TBM → 카트 설명 → 상자 싣고 끌기 → 공장 스캔 → 관리자 호출)
│   ├── assistant_client.py   손 인식·음성 도우미 프로세스 부르기
│   ├── hand_count.py         손 관절 21점 → 손가락 수, 연속 확인
│   ├── i18n.py               4개 언어 문장 틀, 현장 용어집, TBM 항목
│   ├── dashboard.py          조치 지시서 HTML (평면도, 위험 영역, 조치 목록, 음성 경고, 점검표, 기록)
│   ├── inspection.py         바닥 투영, 바디캠 프레임 채점 (BodycamInspector)
│   ├── detector.py           YOLO 래퍼 (ByteTrack 추적)
│   ├── dataset.py            학습 데이터 촬영 시점(바디캠 70% / 자유 30%), 후처리, YOLO 형식
│   ├── machines.py           끼임 위험 기계 작동 상태 레지스트리 (MachineRegistry, get_machine_state)
│   ├── pinch_detect.py       바디캠 depth + 손 관절로 손끝·끼임점 3D 위치 역투영 (끼임 경보용)
│   ├── geometry.py           카메라 투영, 회전 (Projector.unproject 포함)
│   ├── isaac_utils.py        Isaac Sim 버전 차이 흡수, Replicator 도우미
│   └── report.py             터미널 로그
├── scripts/
│   ├── get_assets.py         [파이썬] Poly Haven 공구 모델 받기
│   ├── get_model.py          [파이썬] 학습된 YOLO 가중치 받기 (Hugging Face)
│   ├── build_scene.py        [Isaac] 장면 + 정답표
│   ├── generate_dataset.py   [Isaac] YOLO 합성 데이터
│   ├── generate_pinch_dataset.py [Isaac] 끼임점(pinch_point) 합성 데이터
│   ├── run_patrol.py         [Isaac] 순찰 + 에이전트, 조치 지시서, 채점 (--story 로 시연 이야기)
│   ├── test_gestures.py      [Isaac] 손동작 인식 시험 (경로 여러 곳 x 손가락 1~5)
│   ├── score_gestures.py     [.venv-assistant] 저장한 시험 화면으로 다시 채점
│   ├── assistant_worker.py   [.venv-assistant] MediaPipe 손 인식 + 다국어 음성 + LLM 에이전트(손동작/재판단/조치문구)
│   ├── train_yolo.py         [파이썬] YOLO 학습
│   ├── val_yolo.py           [파이썬] YOLO 검증 점수 (metrics.json)
│   ├── compare_yolo.py       [파이썬] YOLO 가중치 비교
│   ├── eval_patrol.py        [파이썬] 여러 시나리오 평가, 3열 비교 (내부에서 Isaac)
│   ├── demo_all.py           [파이썬] 전체 시연 (내부에서 Isaac)
│   ├── make_video.py         [파이썬] 시연 영상 (run_patrol --record 결과로)
│   ├── make_figures.py       [파이썬] 구조도, 흐름도
│   ├── pick_figures.py       [파이썬] 녹화에서 보고서 그림 고르기
│   ├── make_docs.py          [파이썬] 개발완료보고서, 기술설명서 (docx)
│   ├── make_slides.py        [파이썬] 발표자료 (pptx)
│   ├── to_pdf.ps1            [PowerShell] docx, pptx → PDF (Word, PowerPoint 필요)
│   └── plot_layout.py        [파이썬] 평면도
├── data/tbm_today.json       오늘 TBM (작업, 위험, 지킬 것, 지난 순찰 조치)
├── data/manuals/*.md         안전 매뉴얼 RAG 원문 (공구·카트·적재·유출·소화기·위험구역·컨베이어, 공개 산업안전 자료 기반 큐레이션)
├── tests/test_core.py        Isaac 없이 도는 테스트
└── outputs/                  결과물 (저장소에는 eval/ 채점 결과, agent/ 조치 지시서만. yolo/warehouse_v3·pinch_v1/weights/best.pt 는 Hugging Face, get_model.py 로 받음)
```

`★` 두 파일을 고치면 경로, 물체 자리, 배치 확률이 장면, 학습 데이터, 순찰, 채점에 한꺼번에 반영됩니다.

**`_legacy_pre_cctv_removal/` 폴더 관례**: 구조를 크게 바꿀 때(예: CCTV 제거) 지금 코드와 안 맞게 된 결과물은 지우지 않고
`outputs/eval/_legacy_pre_cctv_removal/`, `submission/_legacy_pre_cctv_removal/` 처럼 같은 이름의 하위 폴더로 옮겨 둡니다.
평가·제출 문서 스크립트는 이 폴더를 보지 않으므로 최신 결과와 섞이지 않고, 과거 수치가 왜 다른지 추적할 때만 참고합니다.

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

# 3) 손동작·다국어 음성·LLM 도우미 (손동작 명령을 쓸 때만, 손 모델은 처음 실행 때 받음)
#    Isaac 환경 (.venv-isaac) 에는 깔지 마세요. Isaac 이 고정한 패키지 버전이 바뀝니다
py -3.12 -m venv .venv-assistant
.venv-assistant\Scripts\python -m pip install mediapipe edge-tts imageio-ffmpeg langgraph langchain-ollama langchain-chroma chromadb

# 4) 로컬 LLM (손동작 요청·재관측 재판단, 약 4.7 GB, 인터넷·API 키 불필요). 없으면 규칙으로 동작 (--llm off, --recheck-llm off 와 같음)
winget install Ollama.Ollama
ollama pull qwen2.5:7b
# 매뉴얼 RAG 임베딩 (손동작 1 "장비 설명" 과 재판단·조치문구가 검증된 안전 매뉴얼에 근거해 답하도록, 약 270 MB)
ollama pull nomic-embed-text
```

> Isaac Sim 첫 실행 때 NVIDIA Omniverse 라이선스(EULA) 동의를 물어봐요. 터미널에서 `Yes` 를 입력하거나 환경 변수 `OMNI_KIT_ACCEPT_EULA=YES`.
> `setx` 후에는 **VSCode를 완전히 껐다 켜야** 환경 변수가 적용돼요.

**확인** (Isaac 없이 1초): `.venv\Scripts\python -m pytest tests -q` → `35 passed`

---

## 🚀 빠른 시작

```powershell
# 0) 공구 모델 (Poly Haven) + 학습된 YOLO 가중치 (Hugging Face) 받기, 처음 한 번
.venv\Scripts\python scripts/get_assets.py
.venv\Scripts\python scripts/get_model.py

# 1) 장면 확인 (위에서 본 창고, 정답표 저장)
& $env:ISAACSIM_PYTHON scripts/build_scene.py

# 2) 학습 데이터 5000장 + 공구 가까이 1500장 + 운반 카트 1200장 → YOLO 학습 (약 2시간). get_model.py 로 받았으면 건너뛰어도 됨
& $env:ISAACSIM_PYTHON scripts/generate_dataset.py --num 5000 --scenario-every 40 --seed 7
& $env:ISAACSIM_PYTHON scripts/generate_dataset.py --num 1500 --scenario-every 30 --seed 11 --focus tools --prefix wt --out outputs/dataset_tools
& $env:ISAACSIM_PYTHON scripts/generate_dataset.py --num 1200 --scenario-every 30 --seed 21 --carts 2 --focus cart --prefix wc --out outputs/dataset_cart
.venv\Scripts\python scripts/train_yolo.py --data outputs/data_v3.yaml --epochs 80
.venv\Scripts\python scripts/val_yolo.py

# 3) 순찰 (바디캠 화면 / 관제 화면)
& $env:ISAACSIM_PYTHON scripts/run_patrol.py
& $env:ISAACSIM_PYTHON scripts/run_patrol.py --view top
& $env:ISAACSIM_PYTHON scripts/run_patrol.py --story --seed 5                 # 시연 이야기 (손동작, 카트, LLM, 영어 안내)
& $env:ISAACSIM_PYTHON scripts/run_patrol.py --pinch-demo                     # 끼임 위험 경보 시연 (컨베이어 on → 접근 → 끼임 경보 → off)

# 4) 여러 시나리오 채점, 전체 시연
.venv\Scripts\python scripts/eval_patrol.py --seeds 0 1 2 3 4
.venv\Scripts\python scripts/eval_patrol.py --seeds 0 1 2 --agent both   # 규칙 vs LangGraph 재판단 비교 (LLM 호출로 느려서 시드 적게)
.venv\Scripts\python scripts/demo_all.py

# 5) 시연 영상: 창 없이 녹화 (약 5분) → 영상 합치기
& $env:ISAACSIM_PYTHON scripts/run_patrol.py --headless --sim-dt 0.0333333 --seed 5 --yolo-every 3 --story --sound off --record outputs/record/seed5 --result outputs/record/seed5_result.json
.venv\Scripts\python scripts/make_video.py                                     # → outputs/video/demo.mp4

# 6) 제출 문서 (수치는 채점·녹화 결과에서). 발표자료는 저장소 옆 팀 양식 ..\조현준_IBDPppt양식.pptx 가 있으면 그 양식으로
.venv\Scripts\python scripts/pick_figures.py
.venv\Scripts\python scripts/make_docs.py
.venv\Scripts\python scripts/make_slides.py
powershell -ExecutionPolicy Bypass -File scripts/to_pdf.ps1                     # → submission/*.pdf, 쪽수 확인
```

VSCode 에서는 `Ctrl+Shift+P` → **Tasks: Run Task** 에 위 작업 (장면, 순찰 화면별, 시연 이야기, 손동작 시험, 데이터·학습, 평가, 시연 녹화·영상, 제출 문서·PDF, 평면도, 테스트) 이 들어 있어요.

---

## 📖 스크립트별 사용법

<details>
<summary><b>run_patrol.py</b> · 순찰과 채점</summary>

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--weights` | `outputs/yolo/warehouse_v3/weights/best.pt` | YOLO 가중치 |
| `--seed` | 무작위 | 위험 요소 배치 시드 (같으면 같은 배치) |
| `--laps` | `1` | 몇 바퀴 (한 바퀴 59 m, 걸음 1.25 m/s 로 약 47초) |
| `--view` | `bodycam` | `bodycam` / `top` (위에서) |
| `--sim-dt` | `0` | 0보다 크면 프레임마다 고정 시간 (평가 재현용, 예 `0.0333`) |
| `--yolo-every` | `6` | 몇 프레임마다 YOLO |
| `--record` | | 영상용: 바디캠 화면과 에이전트 상태를 저장할 폴더 (`make_video.py` 입력) |
| `--story` | | 시연 이야기 (TBM → 카트 설명 → 상자 싣고 끌기 → 공장 스캔 → 관리자 호출, 손 인식 켬, 음성 경고 1번) |
| `--llm` | `qwen2.5:7b` | 손동작 요청을 판단할 Ollama 모델, `off` 면 규칙만 |
| `--recheck-llm` | `qwen2.5:7b` | '주의' 물체 재관측 재판단을 맡길 Ollama 모델, `off` 면 규칙(고위험 후보 재적용)만 |
| `--agent` | `langgraph` | `langgraph` (위 LLM 사용) \| `rule` (`--recheck-llm` 무시, 규칙만). `eval_patrol.py --agent both` 비교용 |
| `--report-llm` | `qwen2.5:7b` | 조치 지시서 문구를 매뉴얼 근거로 보강할 Ollama 모델, `off` 면 고정 문구(`ACTIONS`)만 |
| `--voice-max` | 없음 (`--story` 는 `1`) | 음성 경고 최대 횟수 |
| `--gestures` | `off` | `demo`: 손가락 3 → 1 → 2 → 4 → 5 를 차례로 보임, `watch`: 손 인식만 (손동작은 안 함) |
| `--lang` | `zh,en,ja` (`--story` 는 `en`) | 작업자 언어 (`ko` `en` `zh` `ja`). 여러 개면 명령마다 돌아가며 |
| `--sound` | `auto` | 음성 경고·안내 소리 (`auto` 는 창이 있을 때만) |
| `--result` | `outputs/eval/inspection_seed<시드>.json` | 채점 결과 |
| `--pinch-weights` | `outputs/yolo/pinch_v1/weights/best.pt` | 끼임점(`pinch_point`) 1클래스 경량 모델 가중치. 없으면 등록 위치 폴백만 사용 |
| `--pinch-demo` | | 시연: 컨베이어(M1)를 켜고 순찰 중 가장 가까이 지날 때 끼임 경보를 보인 뒤 끔 |

끝나면 `outputs/agent/dashboard_seed<시드>.html` (조치 지시서), `outputs/agent/report_seed<시드>.json` (대장과 기록) 이 생깁니다.
`--story` 는 평가 결과를 덮지 않게 `dashboard_story_seed<시드>.html`, `report_story_seed<시드>.json` 으로 저장합니다.
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
| `--focus tools` / `cart` | | 공구만 가까이서 / 운반 카트를 절반은 가까이서 (`--carts 2` 와 같이) |
| `--carts` | `0` | 배치마다 운반 카트 수 |
| `--prefix` `--out` | `wh` `outputs/dataset` | 파일 이름 앞부분, 저장 폴더 (데이터셋을 합칠 때) |

**촬영 시점**: 바디캠 (경로 위 가슴 높이 1.2~1.55 m, 70%, 절반은 물체 쪽을 봄), 자유 시점 (물체 주변, 30%, 공구는 더 가까이). 배치는 바디캠 단독 운용에 맞춰 바디캠 비중을 높였습니다.
**출력**: `outputs/dataset/{images,labels}/{train,val}`, `data.yaml`, `README.txt` (클래스별 라벨 수)
</details>

<details>
<summary><b>train_yolo.py</b> · YOLO 학습</summary>

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--data` | `outputs/data_v3.yaml` | 데이터 (일반 촬영 + 공구 가까이 + 운반 카트, 19클래스) |
| `--model` | `yolo26s.pt` | 시작 가중치 (YOLO26: NMS 없는 출력, 작은 물체용 라벨 할당). 지금 가중치 (v3) 는 v2 (18클래스) 에서 이어 15 epoch, 저장소에는 v3 만 |
| `--imgsz` | `960` | 데이터 해상도 그대로 |
| `--epochs` | `100` | 처음부터 80, 이어 학습이면 40 |
| `--batch` | `16` | YOLO26s, 960 에서 약 11 GB |
| `--name` | `warehouse_v3` | 결과 `outputs/yolo/<name>/weights/best.pt` |
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
| 몇 개씩, 위험 확률 | `scenario.py` 의 `sample_scenario()` |
| 물체 모양 (기울기, 상자 수, 공구 종류) | `scene.py` 의 `_build_spill`, `_build_tool`, `_build_stack`, `_build_ext` |
| 근접 기록 거리 (경고 없음, 평가용) | `config.py` 의 `PROXIMITY_WARN_M` |
| 음성 경고 거리, 문구 | `config.py` 의 `TOUCH_WARN_M`, `VOICE_TEXT_HAZARD`, `VOICE_TEXT_CAUTION` |
| 거리 경고·위험 영역 대상 (회피형 위험) | `agent.py` 의 `AVOIDANCE_HAZARDS` |
| 애매한 판정의 즉시 확정/주의 분류 기준 | `agent.py` 의 `HAZARD_SEVERITY` |
| 오늘 TBM 작업 대상 제외 매핑 | `agent.py` 의 `TBM_TASK_KIND`, `TBM_TASK_ZONE` |
| 오늘 TBM 내용 | `data/tbm_today.json` (항목 키는 `i18n.py` 의 `TBM_ITEMS`, `ACTIONS`) |
| 손동작 안내 문장, 언어 추가 | `i18n.py` (문장 틀마다 언어별 문장), 음성은 `assistant_worker.py` 의 `VOICES` |
| 손가락 세는 기준 | `hand_count.py` 의 `BEND_MAX`, `REACH_MIN`, `THUMB_OUT` |
| 위험 영역 크기 (라바콘 연결 거리, 표지 반경, 에이전트 판단 반경) | `zones.py` 의 `CONE_LINK_M`, `SIGN_RADIUS`, `AGENT_RADIUS` |
| 위험 영역이 놓일 자리 | `warehouse.py` 의 `ZONE_SLOTS` |
| 걷는 속도, 바디캠 높이 | `walker.py` 의 `PathWalker` |
| 재관측 재판단 모델·신뢰 기준 | `recheck_agent.py` 의 `MODEL`, `MAX_STEPS`, `run_patrol.py` 의 `--recheck-llm`/`--agent` |
| 조치 방법, 심각도 | `agent.py` 의 `ACTIONS`, `SAFE_NOTES` |
| 끼임 위험구역 반경, 끼임 경보 거리 | `config.py` 의 `MACHINE_ZONE_M`, `PINCH_ALERT_M` |
| 끼임 위험 기계 위치·추가 | `warehouse.py` 의 `MACHINE_CONVEYOR`, `MACHINES` |

**새 위험 요소 추가**: `config.py` 의 `CLASSES`, `HAZARD`, `KIND` 에 위험/안전 두 클래스 추가 → `warehouse.py` 에 자리 → `scenario.py` 에서 배치 → `scene.py` 에 `_build_<종류>` → `agent.py` 의 `HAZARD_SEVERITY`/`AVOIDANCE_HAZARDS` 에 편입 여부 결정 → `pytest`.

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
| 에이전트 기록에 `LLM 판단 ... LLM 을 못 써서 규칙으로 정함` | Ollama 가 꺼져 있거나 모델이 없음. `ollama pull qwen2.5:7b` 후 다시 (규칙으로도 손동작 명령·재판단은 동작) |
| LangGraph 를 `.venv-isaac` 에 깔면 pip 가 버전 충돌을 경고 | Isaac 이 고정한 `idna`, `typing_extensions`, `websockets` 버전이 올라갑니다. LangGraph 는 `.venv-assistant` 에만 깔고, 이미 깔았으면 지우고 세 패키지를 원래 버전으로 |

---

## ✅ 검증 상태

| 부분 | 상태 |
|---|---|
| 배치, 정답표, 경로, 걷기, 판정·채점 로직, 바닥 투영, USD 장면, 에이전트 (2단계 분류·재관측 재판단·위치 보정·병합·조치 지시서), 위험 영역, 음성 경고, 공구 자리 판단, 손가락 세기, 다국어 문장, 손동작 자세, 시연 이야기, 카트 추적, LLM 결정 반영, 끼임 위험 기계(작동 on/off·끼임 경보·폴백) | 테스트 35개 통과 (`tests/test_core.py`) |
| 실제 Isaac Sim 6.0.1 (RTX 4060 Ti 16 GB, Windows 11) 장면, 라벨, 작업자 걷기 | ✅ |
| 학습 데이터 7700장 (19클래스), YOLO 학습 | ✅ (위 결과 표) |
| 손동작 명령 (손가락 1~5, 경로 8곳 x 조명) | ✅ 37/40 |
| LLM 에이전트 (Qwen2.5-7B 도구 호출) + 시연 이야기 | ✅ 손동작 4/4, LLM 결정 4/4 |
| 바디캠 단독 + 2단계 분류 + 재관측 재판단 구조의 순찰 채점 | ⏳ 재실행 필요 (`eval_patrol.py --agent both`, 위 결과 섹션 참고) |
| 끼임 위험 경보 (컨베이어 롤러 1대) — 아래 별도 절 | ✅ 핵심 규칙은 실제 Isaac Sim 순찰로 검증 |
| 실제 사진 | ❌ 아직 안 함 (합성 데이터만으로 학습) |

---

## 🗺 다음 단계

- **실사 테스트**: 휴대폰으로 찍은 실제 창고·실습실 사진에서 합성 데이터만으로 학습한 YOLO 시험
- **새 채점 재실행**: 바디캠 단독 + 2단계 분류 + 재관측 재판단 구조로 `eval_patrol.py --agent both` 를 돌려 결과 섹션 수치를 채움
- **동적 사고**: Isaac Sim 6.1 사고 이벤트 확장(넘어짐, 유출, 화재) 으로 "적재물이 쓰러지는 순간" 데이터
- **VR 체험**: 바디캠 시점을 XR 로 연결해 작업자 시점 안전 교육

### 🧷 끼임 위험 경보 (컨베이어 롤러)

작동 중인 기계(끼임점이 있는 설비)의 위험을 기존 정적 위험물(HAZARD) 체계와 별도로 다룬다. 기계 작동 상태는
실제 공장의 IoT/PLC 신호 조회를 흉내 낸 `get_machine_state(machine_id)` 로 노출하고, LLM 없이 T0 규칙만으로
판단한다: 작동 중인 기계의 끼임점 반경 1.5 m 를 위험구역으로 즉시 설정/해제하고, 손가락 끝(검지 끝)과
끼임점의 3D 거리가 10 cm 이내면 최고 등급 경보("손 빼세요" + 로그·HUD)를 낸다.

**구현 범위**
- 장면: 컨베이어 롤러 1대 (`factory_safety/scene.py:add_machine`), 진입 롤러에 `pinch_point` 라벨 (안전색 노랑으로 칠해 시각적으로도 구분)
- 기계 상태: `factory_safety/machines.py` (`MachineRegistry`, 기본값 off)
- 판단 규칙: `factory_safety/agent.py` (`_on_machine_change`, `check_pinch`, `machine_report`) — 위험구역은 기존 `_check_warnings` 경고 체계에 자연스럽게 편입됨
- 3D 위치 추정: 바디캠에 `distance_to_image_plane` 어노테이터를 붙여 MediaPipe 손 관절(검지 끝)과 `pinch_point` YOLO 박스를 역투영 (`geometry.Projector.unproject`, `factory_safety/pinch_detect.py`)
- 폴백: 끼임점 탐지 실패 시 장면 메타데이터의 등록 위치(`warehouse.MACHINE_CONVEYOR`)로 계산하고 로그("[폴백: ...]")·집계(`pinch_fallback`)에 남김
- `report()`/대시보드: 기계 ID·상태·접근(위험구역)/끼임 경보 횟수를 "끼임 위험 기계" 카드로 표시, 손동작 1(장비 설명)로 가리키면 안전수칙 안내 (`data/manuals/conveyor.md`, `i18n.INFO["machine_conveyor"]`)
- 테스트 6개 추가 (`tests/test_core.py`): 기계 on/off 위험구역 생성·해제, 위험구역 진입 경고, 10 cm+작동 중 경보, 꺼지면 무경보, 탐지 실패 폴백+로그

**합성 데이터 생성 방식** (`scripts/generate_pinch_dataset.py`, 기존 `generate_dataset.py` 패턴 재사용)
끼임점 주변 0.5~3.2 m, 눈높이~허리높이(0.5~1.9 m)에서 전 방위로 바디캠 시점처럼 렌더링. 조명 무작위화,
흔들림 블러·센서 노이즈 후처리(기존 `dataset.post_process` 재사용), 30% 확률로 작은 상자를 끼임점 앞에 놓아
가림 조건도 섞음. Replicator `bounding_box_2d_tight` 가 `pinch_point` 라벨이 붙은 진입 롤러만 자동으로
정답 박스화.

**학습 결과** (YOLO26n, 960 px, 1800장 — 학습 1543 / 검증 257, 라벨 1353장)

| 지표 | 값 |
|---|---|
| Precision | 0.983 |
| Recall | 0.910 |
| mAP50 | 0.966 |
| mAP50-95 | 0.760 |

6 epoch 만에 수렴(끼임점을 현장 안전색 노랑으로 칠해 시각적으로 뚜렷하게 만든 효과가 큼), 60 epoch 중
조기 종료. `outputs/yolo/pinch_v1/weights/best.pt`, `outputs/yolo/pinch_v1/metrics.json`.

**실제 Isaac Sim 종단 검증** (`run_patrol.py --pinch-demo --record`, headless, 실제 GPU 렌더링)
기계 on → 순찰 중 위험구역(1.5 m) 진입 시 실제 음성 경고 발생 → 끼임 경보("손 빼세요", 폴백 로그 포함) →
기계 off → 위험구역 즉시 해제·이후 무경보, 전 과정 로그·`state.jsonl`로 기록됨. 전체 `pytest tests -q` 35개 통과.

**알려진 한계 (의도적으로 범위 밖에 둔 것)**
- 시연 중 "손이 끼임점에 닿는" 순간은 손끝 3D 위치를 스크립트로 직접 주입해 규칙을 실행시킴 (`run_patrol.py` 의
  `--pinch-demo`). 작업자 캐릭터가 손을 실제로 끼임점(허리 높이)까지 뻗는 새 IK 애니메이션은 만들지 않았다 —
  기존 손동작(1~5) IK는 가슴 높이로 캘리브레이션돼 있어 재사용이 안 맞고, 새 reach 애니메이션은 범위를 넘어선다
  판단. 탐지·3D 역투영·T0 규칙 자체는 전부 실동작(위 종단 검증 참고).
- `.venv-assistant`(MediaPipe) 가 없으면 실시간 손 인식 자체가 꺼져서, 실제 깊이 기반 손끝 추정 경로는 그
  환경에서만 동작 확인 가능 (코드는 작성·단위 테스트 완료, 이번 세션엔 그 venv 를 새로 만들지 않음).
- 원래 `feature/pinch` 브랜치는 CCTV 제거 작업 3단계 커밋(`3d2abb3`) 위에서 개발됐고 (순수 추가분 624줄,
  파일 15개 + 대시보드 평가 요약이 새 `evaluate()` 필드와 안 맞던 무관한 버그 수정 1건), CCTV 제거 마무리
  작업(4~7단계, `d5345ac`) 이후 `main` 에 머지됐다.

---

## 📦 사용한 것

- NVIDIA Isaac Sim 6.0, OpenUSD, Omniverse Replicator
- NVIDIA Isaac Sim 에셋 (창고 `Simple_Warehouse`, 소품, 라바콘, 사람 모델) · YCB 드릴. 에셋은 NVIDIA 에셋 서버에서 참조로 불러오고 저장소에 포함하지 않음
- [Poly Haven](https://polyhaven.com/models) 공구 실물 스캔 (CC0): 망치 3종, 드라이버 2종, 톱 2종, 곡괭이, 삽, 렌치 3종. `get_assets.py` 로 받고 저장소에 포함하지 않음
- Windows 음성 합성 (SAPI, 한국어 Heami)
- [MediaPipe Hands](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) (Apache-2.0) 손 관절 인식
- [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) (Apache-2.0) 를 [Ollama](https://ollama.com) 로 로컬 실행, [LangGraph](https://github.com/langchain-ai/langgraph) (MIT) 로 도구 호출 그래프
- [edge-tts](https://github.com/rany2/edge-tts) (Microsoft 온라인 신경망 음성, 인터넷 필요)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (AGPL-3.0, 상업적으로 쓸 계획이면 라이선스 확인 필요)
- numpy, Pillow, matplotlib

<sub>제4회 경남AI·SW경진대회 (③ 제조·피지컬 AI Agent) 출품작 · IBDP 팀</sub>
