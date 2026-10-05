# outputs/eval/_legacy_pre_cctv_removal/

CCTV 제거(바디캠 단독화) 이전 구조로 돌린 `inspection_seed*.json` 채점 결과입니다. 지금 코드의 `evaluate()` 는
필드 구성이 다릅니다(`before`가 CCTV 포함 대장 기준 → 바디캠 프레임 원시판정 기준으로 바뀜, `recheck`→`rejudge`
등). `make_docs.py`/`make_slides.py`/`eval_patrol.py` 는 이 폴더를 보지 않고 `outputs/eval/inspection_seed*.json`
만 읽습니다 — 섞이지 않게 참고용으로만 보관합니다.

최신 채점은 `python scripts/eval_patrol.py --seeds 0 1 2 3 4 --agent both` 로 다시 만드세요.
