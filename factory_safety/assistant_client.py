"""손 인식·음성 도우미 (scripts/assistant_worker.py, .venv-assistant) 를 띄우고 부른다. 못 띄우면 ok = False."""
import json
import os
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKER = os.path.join(ROOT, "scripts", "assistant_worker.py")


class AssistantClient:
    def __init__(self, python=None, work_dir=None):
        py = python or os.environ.get("ASSISTANT_PYTHON") or os.path.join(ROOT, ".venv-assistant", "Scripts", "python.exe")
        self.dir = work_dir or os.path.join(ROOT, "outputs", "assistant")
        os.makedirs(self.dir, exist_ok=True)
        self.proc, self.ok, self.n, self.error = None, False, 0, None
        if not os.path.exists(py):
            self.error = f"도우미 파이썬이 없어요: {py}"
            return
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        self._log = open(os.path.join(self.dir, "worker_log.txt"), "w", encoding="utf-8")
        self.proc = subprocess.Popen([py, "-u", WORKER], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log,
                                     text=True, encoding="utf-8", cwd=ROOT, env=env)
        res = self.call(op="ping")
        self.ok = bool(res.get("ok"))
        self.error = res.get("error")

    def call(self, **req):
        if self.proc is None or self.proc.poll() is not None:
            return {"error": "도우미가 꺼져 있음"}
        self.n += 1
        req["_id"] = self.n
        self.proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()
        while True:
            line = self.proc.stdout.readline()
            if not line:
                return {"error": "도우미 연결 끊김"}
            try:
                res = json.loads(line)
            except ValueError:
                continue
            if res.get("_id") == self.n:
                return res

    def hand(self, img):
        """화면 (H, W, 3) uint8 -> (손가락 수, 손 관절 21점 화면 좌표 또는 None). 손이 없거나 정해진 모양이 아니면 수는 0."""
        if not self.ok:
            return 0, None
        from PIL import Image
        path = os.path.join(self.dir, "hand_frame.jpg")
        Image.fromarray(img).save(path, quality=90)
        res = self.call(op="hand", path=path)
        return int(res.get("count", 0) or 0), res.get("pts")

    def tts(self, text, lang):
        """-> (wav 경로 또는 None, 길이 초)."""
        if not self.ok:
            return None, 0.0
        res = self.call(op="tts", text=text, lang=lang)
        return res.get("wav"), float(res.get("dur", 0.0) or 0.0)

    def agent(self, snapshot, model=None):
        """LLM 에이전트 결정 (LangGraph + 로컬 Qwen). 실패하면 {"llm": False, "error"}."""
        if not self.ok:
            return {"llm": False, "error": "도우미 없음"}
        return self.call(op="agent", snapshot=snapshot, model=model)

    def recheck(self, snapshot, model=None):
        """'주의' 물체 재관측 재판단 (factory_safety/recheck_agent.py). 실패하면 {"llm": False, "error"}."""
        if not self.ok:
            return {"llm": False, "error": "도우미 없음"}
        return self.call(op="recheck", snapshot=snapshot, model=model)

    def report_action(self, snapshot, model=None):
        """조치 문구 매뉴얼 보강 (factory_safety/report_agent.py). 실패하면 {"llm": False, "error"}."""
        if not self.ok:
            return {"llm": False, "error": "도우미 없음"}
        return self.call(op="report_action", snapshot=snapshot, model=model)

    def close(self):
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.proc.stdin.write(json.dumps({"op": "quit"}) + "\n")
                self.proc.stdin.flush()
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()
