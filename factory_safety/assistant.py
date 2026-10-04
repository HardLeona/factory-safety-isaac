"""작업자 손동작 명령: 바디캠 앞에 손을 가로로 내밀어 손가락 1~5개를 보이면 작업자 언어로 안내한다.

    1 장비 설명      바로 전 바디캠 화면 가운데에 있던 장비 (운반 카트, 공구, 소화기 등) 가 무엇인지, 어떻게 써야 안전한지
    2 공장 위험 스캔  CCTV 3대와 위험물 대장으로 공장 전체의 위험 요소 (우선순위 순, 구역), CCTV 확대로 차례로 비춤
    3 오늘의 TBM     아침 작업 전 안전 회의 내용 (오늘 작업, 주의할 위험, 지킬 것, 지난 순찰 조치)
    4 관리자 호출    작업자 위치와 바디캠 화면을 관리자에게 보냄
    5 SOS 신고       위치를 관리자·안전팀에 보내고, 작업자를 볼 수 있는 CCTV 가 작업자 쪽을 확대 촬영

무엇을 말할지는 LLM 에이전트 (llm_agent.py: LangGraph + 로컬 Qwen2.5-7B) 가 도구로 상황을 보고 정하고 (planner),
LLM 이 없거나 실패하면 규칙으로 정한다. 작업자에게 들려주는 문장은 i18n 의 검수한 틀로 만든다 (작업자 언어 + 관리자용 한국어).
손가락 수는 MediaPipe 손 관절로 센다 (hand_count).
"""
import json
import os
from collections import deque

import numpy as np

from . import i18n
from . import warehouse as W
from .config import EQUIPMENT, HAZARD, KIND
from .hand_count import GestureFilter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TBM_PATH = os.path.join(ROOT, "data", "tbm_today.json")
KINDS = {1: "equip", 2: "scan", 3: "tbm", 4: "manager", 5: "sos"}
EQUIP_RANK = {"equipment": -0.2, "tool": 0.0, "ext": 0.0, "stack": 0.3, "marker": 0.4, "spill": 0.5}     # 장비 설명에서 먼저 고를 종류


def load_tbm(path=TBM_PATH):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class SiteAssistant:
    VIEW_S = 5.0          # 명령 직전 이만큼의 바디캠 화면에서 장비·위험을 찾음 (손을 올리기 전 화면 포함)
    EQUIP_MAX_M = 8.0
    HAZ_MAX_M = 12.0
    ZONE_MAX_M = 8.0
    SOS_VIEW_S = 8.0      # SOS 뒤 CCTV 가 작업자를 확대해 보여 주는 시간
    SOS_WINDOW_M = 3.2    # 확대 화면 세로 폭
    SCAN_SAY = 3          # 공장 스캔에서 말해 주는 위험 수 (나머지는 조치 지시서)
    SCAN_DWELL_S = 1.6    # 스캔 때 CCTV 확대로 위험 하나를 비추는 시간
    SCAN_WINDOW_M = 3.0

    def __init__(self, agent, langs=("zh",), tbm=None, w=None, h=None):
        self.agent = agent
        self.langs = list(langs)
        self.n_cmd = 0
        self.frames = deque()
        self.filter = GestureFilter()
        self.tbm = tbm if tbm is not None else load_tbm()
        self.events = []
        self.sos = None
        self.scan = None
        self.planner = None       # 스냅샷 -> LLM 결정 (assistant_client.AssistantClient.agent). None 이면 규칙
        self.w, self.h = w or agent.w, h or agent.h

    # ------------------------------------------------------------ 입력
    def observe(self, t, cam, dets):
        """바디캠 YOLO 결과를 모아 둔다 (명령 1, 2 에서 '방금 보던 화면' 으로 씀)."""
        self.frames.append((float(t), cam, list(dets)))
        while self.frames and self.frames[0][0] < t - self.VIEW_S:
            self.frames.popleft()

    def on_hand(self, t, count, cam, worker_xy, worker_yaw, pts=None):
        """손가락 수 (0 = 손 없음/정해진 모양 아님) 와 손 관절. 멈춘 손에서 같은 수가 이어져 확정되면 명령을 실행하고 기록을 돌려준다."""
        c = self.filter.update(t, count, pts)
        return self.run(t, c, cam, worker_xy, worker_yaw) if c else None

    # ------------------------------------------------------------ 실행
    def run(self, t, count, cam, worker_xy, worker_yaw, lang=None):
        lang = lang or self.langs[self.n_cmd % len(self.langs)]
        self.n_cmd += 1
        wxy = np.asarray(worker_xy, float)
        ctx = {"t": t, "cam": cam, "xy": wxy, "yaw": float(worker_yaw), "zone": W.zone_name(*wxy)}
        ctx["decision"] = self._plan(ctx, count, lang)
        fn = {1: self._equip, 2: self._scan, 3: self._tbm, 4: self._manager, 5: self._sos}[count]
        texts, extra = {}, {}
        for lg in dict.fromkeys((lang, "ko")):
            texts[lg], extra = fn(ctx, lg)
        ack = {lg: i18n.fmt("ack", lg, n=count, cmd=i18n.COMMANDS[count][lg]) for lg in texts}
        ev = {"t": round(float(t), 2), "count": count, "kind": KINDS[count], "cmd": i18n.COMMANDS[count]["ko"],
              "cmd_lang": i18n.COMMANDS[count][lang], "lang": lang, "lang_name": i18n.LANGS[lang],
              "text": ack[lang] + " " + texts[lang], "text_ko": ack["ko"] + " " + texts["ko"],
              "zone": ctx["zone"], "worker": [round(float(wxy[0]), 2), round(float(wxy[1]), 2)], **extra}
        dec = ctx["decision"]
        if dec:
            ev["llm"] = {k: dec.get(k) for k in ("llm", "model", "say_ids", "reason_ko", "manager_ko", "refuse", "trace", "sec", "error")}
            if dec.get("llm"):
                self.agent.say(t, "LLM 판단", f"{dec['model']}: {' → '.join(dec.get('trace', []))} | {dec.get('reason_ko', '')}")
            else:
                self.agent.say(t, "LLM 판단", f"LLM 을 못 써서 규칙으로 정함 ({dec.get('error', '')[:60]})")
        self.events.append(ev)
        kind = {4: "관리자 호출", 5: "SOS"}.get(count, "손동작")
        self.agent.say(t, kind, f"손가락 {count}개 → {ev['cmd']} ({ev['lang_name']}): {texts['ko']}")
        return ev

    # ------------------------------------------------------------ LLM 에이전트에 넘길 상황
    def _plan(self, ctx, count, lang):
        """지금 상황 스냅샷을 만들어 planner (LLM) 에게 결정을 받는다. planner 가 없으면 None (규칙)."""
        seen = sorted(self._seen(ctx), key=lambda o: (EQUIP_RANK.get(KIND.get(o["cls"]), 0.5) + o["center"] + 0.06 * o["dist"]))
        ctx["view_ids"] = {f"V{k + 1}": o for k, o in enumerate(seen)}
        rep = self.agent.report()
        haz = [r for r in rep["findings"] if r["state"] == "위험"]
        ctx["haz_by_id"] = {r["id"]: r for r in haz}
        if self.planner is None:
            return None
        view = []
        for vid, o in ctx["view_ids"].items():
            key = o["tool"] or o["cls"]
            k = KIND.get(o["cls"], o["cls"])
            info = i18n.INFO.get(key, {}).get("en") or i18n.INFO.get(k, {}).get("en", "")
            view.append({"id": vid, "name": i18n.name(key, "en"), "name_ko": i18n.name(key, "ko"), "kind": k,
                         "dist": round(o["dist"], 1), "dir": o["dir"], "center": round(o["center"], 2), "info": info})
        hazards = []
        for r in haz:
            key = r["class"]
            what = i18n.name(key, "en") + (f" ({', '.join(r.get('tool_types') or [])})" if r.get("tool_types") else "")
            d = float(np.hypot(r["x"] - ctx["xy"][0], r["y"] - ctx["xy"][1]))
            hazards.append({"id": r["id"], "what": what, "what_ko": r["label"], "area": i18n.zone(r["zone"], "en"), "area_ko": r["zone"],
                            "dist": round(d, 1), "priority": {"긴급": "urgent", "높음": "high", "보통": "medium"}.get(r["priority"], "low")})
        tb = self.tbm
        snap = {"count": count, "t": round(float(ctx["t"]), 1), "lang": lang, "lang_name": i18n.LANGS[lang], "view": view,
                "hazards": hazards, "zones": [{"id": z["id"], "kind": z["source"], "area_ko": z["zone"]} for z in rep["zones"]],
                "tbm": {key: [i18n.TBM_ITEMS[k]["en"] for k in tb.get(key, [])] for key in ("work", "risks", "rules")},
                "worker": {"area": i18n.zone(ctx["zone"], "en"), "area_ko": ctx["zone"], "xy": [round(float(v), 1) for v in ctx["xy"]],
                           "cctv": bool(self.agent.cctvs)}}
        try:
            return self.planner(snap)
        except Exception as e:      # 도우미 연결 문제
            return {"llm": False, "error": f"{type(e).__name__}: {e}"}

    @staticmethod
    def _llm_ids(ctx):
        dec = ctx.get("decision") or {}
        return (dec.get("say_ids") or []) if dec.get("llm") else []

    # ------------------------------------------------------------ 바로 전 화면에서 본 물체
    def _seen(self, ctx):
        """최근 바디캠 화면의 물체: [{cls, tool, xy, dist, dir, center, conf}] (같은 추적 번호·자리는 하나로)."""
        objs = {}
        for t, cam, dets in self.frames:
            obs, marks = self.agent._normalize(cam, dets)
            items = [(n, c, b, tid, tool) for n, c, b, tid, tool in obs] + [(n, c, b, None, None) for n, c, b in marks]
            items += [(d[0], d[1], d[2], d[3] if len(d) > 3 else None, None) for d in dets if d[0] in EQUIPMENT]
            for name, conf, xyxy, tid, tool in items:
                if name in ("cone", "danger_sign") or name in EQUIPMENT:
                    xy = self._floor(cam, xyxy)
                else:
                    xy, _ = self.agent._locate(cam, xyxy, name)
                if xy is None:
                    continue
                dist = float(np.linalg.norm(xy - ctx["xy"]))
                key = ("t", tid) if tid is not None else (name, round(float(xy[0]) / 1.5), round(float(xy[1]) / 1.5))
                cx = ((xyxy[0] + xyxy[2]) / 2 - self.w / 2) / (self.w / 2)      # 바닥 물체는 원래 화면 아래쪽이라 가로만 봄
                o = objs.setdefault(key, {"cls": name, "tool": tool, "conf": 0.0, "n": 0, "center": 9.0, "votes": {}})
                o.update(xy=xy, dist=dist, t=t)
                o["votes"][name] = o["votes"].get(name, 0.0) + float(conf)     # 같은 추적 번호도 판정이 바뀔 수 있어서 투표
                o["cls"] = max(o["votes"], key=o["votes"].get)
                o["conf"] = max(o["conf"], float(conf))
                o["n"] += 1
                o["center"] = min(o["center"], abs(cx))
                if tool:
                    o["tool"] = tool
        out = []
        for o in objs.values():
            o["dir"] = i18n.direction(ctx["yaw"], ctx["xy"], o["xy"])
            out.append(o)
        # 끼임 위험 기계: YOLO 로 탐지하지 않으니(19클래스 재학습 안 함) 등록된 위치로 직접 후보에 넣는다
        for mid, m in self.agent.machines.machines.items():
            xy = np.array([m["x"], m["y"]], float)
            dist = float(np.linalg.norm(xy - ctx["xy"]))
            if dist > self.HAZ_MAX_M:
                continue
            out.append({"cls": "machine_conveyor", "tool": None, "conf": 1.0, "n": 1, "xy": xy, "dist": dist,
                       "center": 0.3, "dir": i18n.direction(ctx["yaw"], ctx["xy"], xy)})
        return out

    def _floor(self, cam, xyxy):
        from .inspection import floor_point
        return floor_point(cam, xyxy, "cone", self.w, self.h)

    # ------------------------------------------------------------ 1. 장비 설명
    def _equip(self, ctx, lang):
        dec = ctx.get("decision") or {}
        if dec.get("llm") and dec.get("refuse"):
            # 매뉴얼 RAG 에 근거가 없으면 LLM 이 추측하지 않고 거부 (finish(refuse=true)) -> 관리자 호출로 넘김
            return i18n.fmt("equip_refuse", lang), {"items": [], "notify": "관리자", "manager_ko": self._manager_note(ctx)}
        cand = [o for o in self._seen(ctx) if o["dist"] <= self.EQUIP_MAX_M]
        if not cand:
            return i18n.fmt("equip_none", lang), {"items": []}
        # LLM 이 고른 물체, 없으면 규칙: 장비(카트, 공구, 소화기) 를 먼저, 화면 가로 가운데에 가깝고 가까운 것
        picked = [ctx["view_ids"][i] for i in self._llm_ids(ctx) if i in ctx.get("view_ids", {})]
        o = picked[0] if picked else min(cand, key=lambda o: EQUIP_RANK.get(KIND.get(o["cls"]), 0.5) + o["center"] + 0.06 * o["dist"])
        if o["tool"]:
            nm = i18n.name(o["tool"], lang)
            info = i18n.INFO[o["tool"]][lang] + " " + i18n.fmt("tool_floor_note" if o["cls"] == "tool_floor" else "tool_stored_note", lang)
        else:
            nm = i18n.name(o["cls"], lang)
            k = KIND.get(o["cls"], o["cls"])
            info = " ".join(v for v in (i18n.INFO.get(k, {}).get(lang) if k in ("ext", "stack") else None,
                                         i18n.INFO.get(o["cls"], {}).get(lang)) if v)
        text = i18n.fmt("equip", lang, dir=i18n.DIRS[o["dir"]][lang], d=i18n.dist_text(o["dist"]), name=nm, info=info)
        return text, {"items": [self._item(o)]}

    # ------------------------------------------------------------ 2. 공장 위험 스캔
    def _scan(self, ctx, lang):
        """공장 전체: 위험물 대장 (바디캠·CCTV 확대로 확정한 것) 의 위험을 우선순위 순으로. 같은 종류·구역은 한 번만."""
        rep = self.agent.report()
        haz = [r for r in rep["findings"] if r["state"] == "위험"]
        # LLM 이 고른 위험을 먼저 (고른 순서대로), 나머지는 우선순위 순
        order = {i: k for k, i in enumerate(self._llm_ids(ctx))}
        haz.sort(key=lambda r: order.get(r["id"], len(order)))
        said, seen = [], set()
        for r in haz:
            key = (r["class"], r["zone"])
            if key in seen:
                continue
            seen.add(key)
            said.append(r)
        n_zone = len(rep["zones"])
        log_only = "" if self.agent.cctvs else "_log"           # CCTV 가 없으면 위험물 대장으로만
        if not said:
            return i18n.fmt("scan_none" + log_only, lang), {"items": [], "zones": n_zone}
        sep = " " if lang in ("ko", "en") else ""
        parts = [i18n.fmt("scan_head" + log_only, lang, n=len(said))]
        for r in said[:self.SCAN_SAY]:
            if r.get("tool_types"):
                nm = i18n.fmt("floor_tool", lang, tool=i18n.join([i18n.name(t, lang) for t in r["tool_types"][:2]], lang))
            else:
                nm = i18n.name(r["class"], lang)
            parts.append(i18n.fmt("scan_item", lang, name=nm, zone=i18n.zone(r["zone"], lang)))
        if len(said) > self.SCAN_SAY:
            parts.append(i18n.fmt("scan_more", lang, n=len(said) - self.SCAN_SAY))
        if n_zone:
            parts.append(i18n.fmt("scan_zones", lang, n=n_zone))
        if self.scan is None or self.scan.get("t0") != ctx["t"]:
            shots = []
            for r in said[:self.SCAN_SAY]:
                target = np.array([r["x"], r["y"], 0.4])
                cams = self.agent.camera_options(target) if self.agent.cctvs else []
                if cams:
                    shots.append((cams[0][0], target, r["id"]))
            self.scan = {"t0": float(ctx["t"]), "shots": shots}
        items = [{"id": r["id"], "cls": r["class"], "zone": r["zone"], "xy": [r["x"], r["y"]]} for r in said[:self.SCAN_SAY]]
        return sep.join(parts), {"items": items, "zones": n_zone, "n_hazards": len(said)}

    @staticmethod
    def _item(o):
        return {"cls": o["cls"], "tool": o["tool"], "xy": [round(float(o["xy"][0]), 2), round(float(o["xy"][1]), 2)],
                "dist": round(o["dist"], 1), "dir": o["dir"]}

    # ------------------------------------------------------------ 3. TBM
    def _tbm(self, ctx, lang):
        tb = self.tbm
        sep = " " if lang in ("ko", "en") else ""
        parts = []          # 날짜는 대시보드에만 (안내를 짧게)
        for key, tmpl in (("work", "tbm_work"), ("risks", "tbm_risk"), ("rules", "tbm_rule")):
            if tb.get(key):
                parts.append(i18n.fmt(tmpl, lang, items=i18n.join([i18n.TBM_ITEMS[k][lang] for k in tb[key]], lang)))
        todo = tb.get("todo") or []
        if todo:
            items = [i18n.fmt("todo_item", lang, zone=i18n.zone(a["zone"], lang), action=i18n.ACTIONS[a["cls"]][lang]) for a in todo]
            parts.append(i18n.fmt("tbm_todo", lang, n=len(todo), items=i18n.join(items, lang)))
        return sep.join(parts), {"date": tb["date"]}

    # ------------------------------------------------------------ 4. 관리자 호출
    def _manager(self, ctx, lang):
        return i18n.fmt("manager", lang, zone=i18n.zone(ctx["zone"], lang)), {"notify": "관리자", "manager_ko": self._manager_note(ctx)}

    def _manager_note(self, ctx):
        """관리자에게 보내는 한국어 메시지: 확인된 사실 (위치, 가까운 위험) + LLM 요약."""
        near = sorted(ctx.get("haz_by_id", {}).values(), key=lambda r: np.hypot(r["x"] - ctx["xy"][0], r["y"] - ctx["xy"][1]))[:2]
        facts = f"작업자 위치 {ctx['zone']} ({ctx['xy'][0]:+.1f}, {ctx['xy'][1]:+.1f})"
        if near:
            facts += ", 가까운 위험: " + ", ".join(f"{r['id']} {r['label']} ({r['zone']})" for r in near)
        dec = ctx.get("decision") or {}
        if dec.get("llm") and dec.get("manager_ko"):
            facts += f" | AI 요약: {dec['manager_ko']}"
        return facts

    # ------------------------------------------------------------ 5. SOS
    def _sos(self, ctx, lang):
        target = np.array([ctx["xy"][0], ctx["xy"][1], 1.0])
        cams = self.agent.camera_options(target) if self.agent.cctvs else []
        cam = cams[0][0] if cams else None
        if self.sos is None or self.sos.get("t0") != ctx["t"]:
            self.sos = {"t0": float(ctx["t"]), "cam": cam, "target": target, "until": float(ctx["t"]) + self.SOS_VIEW_S}
        zone = i18n.zone(ctx["zone"], lang)
        note = self._manager_note(ctx)
        if cam:
            return i18n.fmt("sos", lang, zone=zone, cam=i18n.CAMS.get(cam, {}).get(lang, cam)), \
                {"notify": "관리자, 안전팀", "cctv": cam, "manager_ko": note}
        return i18n.fmt("sos_nocam", lang, zone=zone), {"notify": "관리자, 안전팀", "cctv": None, "manager_ko": note}

    def sos_view(self, t):
        """SOS 뒤 작업자를 확대해 보는 CCTV (이름, 카메라 자세) 또는 None."""
        s = self.sos
        if not s or s["cam"] is None or not (s["t0"] <= t <= s["until"]):
            return None
        return s["cam"], self.agent.aim(s["cam"], s["target"], self.SOS_WINDOW_M)

    def ptz_view(self, t):
        """지금 비서가 쓰는 CCTV 확대: (이름, 카메라 자세, 무엇, 목표점) 또는 None. SOS 가 먼저, 다음은 공장 스캔."""
        v = self.sos_view(t)
        if v:
            return v[0], v[1], "SOS", self.sos["target"]
        sc = self.scan
        if sc and sc["shots"]:
            k = int((t - sc["t0"]) // self.SCAN_DWELL_S)
            if 0 <= k < len(sc["shots"]):
                cam, target, fid = sc["shots"][k]
                return cam, self.agent.aim(cam, target, self.SCAN_WINDOW_M), f"스캔 {fid}", target
        return None

    def report(self):
        return list(self.events)


class DemoScript:
    """시연용 손동작 순서 (작업자 역할). 앞 안내가 끝난 뒤에만 다음 손동작을 한다.

    3 TBM: 순찰을 시작하자마자 | 1 장비 설명: 공구나 소화기가 화면 가운데 1~4 m 에 보일 때 |
    2 위험 안내: 위험 요소가 2~7 m 에 보일 때 | 4 관리자 호출: 순찰 70% | 5 SOS: 순찰 85%
    (조건이 안 맞아도 순찰이 꽤 지나면 그 자리에서 함)."""
    HOLD = 2.0
    GAP_S = 1.0
    TRIES = 2           # 인식이 안 되면 한 번 더 보임

    def __init__(self):
        self.shown = []
        self.heard = set()
        self.busy_until = 0.0

    def _view(self, agent, cam, dets, wxy):
        equip = haz = False
        obs, _ = agent._normalize(cam, dets)
        for name, conf, xyxy, tid, tool in obs:
            xy, _ = agent._locate(cam, xyxy, name)
            if xy is None or conf < 0.5:
                continue
            d = float(np.linalg.norm(xy - wxy))
            cx = abs((xyxy[0] + xyxy[2]) / 2 / agent.w - 0.5)
            if (tool or KIND[name] == "ext") and 1.0 <= d <= 4.0 and cx < 0.22:
                equip = True
            if HAZARD.get(name) and 2.0 <= d <= 7.0:
                haz = True
        return equip, haz

    def next(self, t, frac, agent, cam, dets, wxy, walking):
        """지금 보일 손가락 수 또는 None."""
        if not walking or t < self.busy_until:
            return None
        tries = {}
        for _, c in self.shown:
            tries[c] = tries.get(c, 0) + 1
        done = self.heard | {c for c, n in tries.items() if n >= self.TRIES}
        equip, haz = self._view(agent, cam, dets, np.asarray(wxy, float))
        want = None
        if 3 not in done:
            want = 3 if t >= 1.0 else None
        elif 1 not in done:
            want = 1 if equip or frac > 0.5 else None
        elif 2 not in done:
            want = 2 if haz or frac > 0.62 else None
        elif 4 not in done:
            want = 4 if frac >= 0.7 else None
        elif 5 not in done:
            want = 5 if frac >= 0.85 else None
        if want:
            self.shown.append((float(t), want))
            self.busy_until = t + 1e9          # 안내가 나오면 (said) 그 길이만큼으로 바꿈
        return want

    def said(self, t, dur, count=None):
        self.busy_until = t + dur + self.GAP_S
        if count:
            self.heard.add(count)

    def missed(self, t):
        """손동작을 했는데 명령이 안 나옴: 다음으로."""
        self.busy_until = t + self.GAP_S


def gesture_eval(shown, events):
    """보인 손동작 (t, 수) 과 인식된 명령 (t, 수) 비교: 보인 것마다 4초 안에 같은 수가 나왔는지."""
    used, ok, wrong = set(), 0, 0
    for t, c in shown:
        hit = next((k for k, e in enumerate(events) if k not in used and 0 <= e["t"] - t <= 4.0), None)
        if hit is None:
            continue
        used.add(hit)
        ok += events[hit]["count"] == c
        wrong += events[hit]["count"] != c
    return {"shown": len(shown), "recognized": ok, "wrong": wrong, "missed": len(shown) - ok - wrong,
            "extra": len(events) - len(used)}
