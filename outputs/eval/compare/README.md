# outputs/eval/compare/

`scripts/eval_patrol.py --agent both` 로 같은 시드를 규칙/LangGraph 두 모드로 각각 돌린 결과입니다
(`inspection_seed<시드>_rule.json` / `_langgraph.json`, `patrol_results_rule.md` / `patrol_results_langgraph.md`).
요약 비교표는 `outputs/eval/patrol_results_compare.md` 에 있습니다.

`scripts/make_docs.py`/`make_slides.py` 는 `outputs/eval/inspection_seed*.json` (이 폴더 밖, 접미사 없는 파일만)
을 모아 집계하므로, 이 폴더의 파일은 글롭에 안 걸리게 따로 둡니다 — 같은 시드를 두 번 세는 걸 방지합니다.
