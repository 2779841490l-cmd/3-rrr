"""
engagement_logger.py - Per-Engagement / Per-Kill Blackbox & Optional CS2 GSI Receiver
====================================================================================
Captures high-resolution engagement diagnostics around every shot/kill burst:
  1. Pre-shot & in-burst 144 Hz approach trajectory (`live_engagements.jsonl`)
  2. Physical 3200 DPI mouse-sensor vs commanded stage-delta coupling ratio
     (reveals immediately whether palm weight stalls open-loop micro-steps)
  3. Optional Valve CS2 Game State Integration (127.0.0.1:30001) real-time kill,
     headshot, and active weapon tagging
  4. Readable CSV summary (`live_kill_report.csv`) + terminal kill diagnostics
"""

from collections import deque
import csv
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import threading
import time
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np


class CS2GSIListener:
    """Lightweight non-blocking HTTP POST listener for Valve CS2 gamestate_integration."""

    def __init__(self, host: str = "127.0.0.1", port: int = 30001) -> None:
        self.host = str(host)
        self.port = int(port)
        self._lock = threading.Lock()
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self.connected = False
        self.packets = 0
        self.active_weapon = ""
        self.map_name = ""
        self.total_kills = 0
        self.total_hs = 0
        self._last_round_kills: Optional[int] = None
        self._last_match_kills: Optional[int] = None
        self._last_round_killhs: Optional[int] = None
        self._pending_kills = 0
        self._pending_hs = 0

    def start(self) -> bool:
        parent = self

        class _Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    raw = self.rfile.read(length) if length > 0 else b"{}"
                    payload = json.loads(raw.decode("utf-8", errors="ignore"))
                    parent._ingest(payload)
                except Exception:
                    pass
                self.send_response(200)
                self.end_headers()

            def log_message(self, fmt: str, *args: Any) -> None:
                return

        try:
            self._server = ThreadingHTTPServer((self.host, self.port), _Handler)
            self._server.daemon_threads = True
            self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
            self._thread.start()
            return True
        except Exception:
            self._server = None
            return False

    def _ingest(self, payload: Dict[str, Any]) -> None:
        with self._lock:
            self.connected = True
            self.packets += 1
            mp = payload.get("map", {})
            if isinstance(mp, dict) and mp.get("name"):
                self.map_name = str(mp.get("name"))
            player = payload.get("player", {})
            if not isinstance(player, dict):
                return
            state = player.get("state", {})
            dk_round = 0
            dk_match = 0
            if isinstance(state, dict):
                rk = state.get("round_kills")
                rhs = state.get("round_killhs")
                if rk is not None:
                    rk_i = int(rk)
                    if self._last_round_kills is not None and rk_i > self._last_round_kills:
                        dk_round = rk_i - self._last_round_kills
                    self._last_round_kills = rk_i
                if rhs is not None:
                    rhs_i = int(rhs)
                    if self._last_round_killhs is not None and rhs_i > self._last_round_killhs:
                        dhs = rhs_i - self._last_round_killhs
                        self.total_hs += dhs
                        self._pending_hs += dhs
                    self._last_round_killhs = rhs_i
            mstats = player.get("match_stats", {})
            if isinstance(mstats, dict) and mstats.get("kills") is not None:
                mk_i = int(mstats["kills"])
                if self._last_match_kills is not None and mk_i > self._last_match_kills:
                    dk_match = mk_i - self._last_match_kills
                self._last_match_kills = mk_i
            dk = max(dk_round, dk_match)
            if dk > 0:
                self.total_kills += dk
                self._pending_kills += dk
            weapons = player.get("weapons", {})
            if isinstance(weapons, dict):
                for w in weapons.values():
                    if isinstance(w, dict) and w.get("state") == "active" and w.get("name"):
                        self.active_weapon = str(w["name"]).replace("weapon_", "")
                        break

    def pop_delta(self) -> Tuple[int, int, str]:
        with self._lock:
            dk = self._pending_kills
            dhs = self._pending_hs
            self._pending_kills = 0
            self._pending_hs = 0
            return dk, dhs, self.active_weapon

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "gsi_connected": bool(self.connected),
                "gsi_packets": int(self.packets),
                "map": self.map_name,
                "active_weapon": self.active_weapon,
                "total_kills": int(self.total_kills),
                "total_hs": int(self.total_hs),
            }

    def close(self) -> None:
        if self._server is not None:
            try:
                self._server.shutdown()
                self._server.server_close()
            except Exception:
                pass
            self._server = None


def ensure_cs2_gsi_config(port: int = 30001) -> Optional[str]:
    """Locate Steam CS2 game/csgo/cfg directory and install gamestate_integration_3rrr.cfg if missing."""
    candidates = [
        r"C:\Program Files (x86)\Steam\steamapps\common\Counter-Strike Global Offensive\game\csgo\cfg",
        r"D:\SteamLibrary\steamapps\common\Counter-Strike Global Offensive\game\csgo\cfg",
        r"D:\Steam\steamapps\common\Counter-Strike Global Offensive\game\csgo\cfg",
        r"E:\SteamLibrary\steamapps\common\Counter-Strike Global Offensive\game\csgo\cfg",
        r"F:\SteamLibrary\steamapps\common\Counter-Strike Global Offensive\game\csgo\cfg",
    ]
    cfg_dir = next((p for p in candidates if os.path.isdir(p)), None)
    if cfg_dir is None:
        return None
    out_path = os.path.join(cfg_dir, "gamestate_integration_3rrr.cfg")
    content = (
        '"3-RRR HIL Kill Telemetry v.1"\n'
        '{\n'
        f'  "uri" "http://127.0.0.1:{int(port)}"\n'
        '  "timeout" "5.0"\n'
        '  "buffer" "0.0"\n'
        '  "throttle" "0.05"\n'
        '  "heartbeat" "5.0"\n'
        '  "data"\n'
        '  {\n'
        '    "provider" "1"\n'
        '    "map" "1"\n'
        '    "round" "1"\n'
        '    "player_id" "1"\n'
        '    "player_state" "1"\n'
        '    "player_weapons" "1"\n'
        '    "player_match_stats" "1"\n'
        '  }\n'
        '}\n'
    )
    if not os.path.isfile(out_path):
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(content)
    return out_path


class EngagementLogger:
    """Tracks per-burst/per-kill trajectories, physical mouse coupling, and diagnostics."""

    def __init__(self, logs_dir: str) -> None:
        self.logs_dir = str(logs_dir)
        os.makedirs(self.logs_dir, exist_ok=True)
        self.jsonl_path = os.path.join(self.logs_dir, "live_engagements.jsonl")
        self.csv_path = os.path.join(self.logs_dir, "live_kill_report.csv")
        self._pre_ring: Deque[Dict[str, Any]] = deque(maxlen=40)
        self._coupling_queue: Deque[Tuple[float, np.ndarray, np.ndarray]] = deque(maxlen=32)
        self._coupling_samples: List[Tuple[float, float]] = []
        self._recent_coupling: Deque[Tuple[float, float, float]] = deque(maxlen=32)
        self._active: Optional[Dict[str, Any]] = None
        self.engagements: List[Dict[str, Any]] = []
        self._prev_lmb = False
        self._lmb_release_t = 0.0

    def reset(self) -> None:
        self._pre_ring.clear()
        self._coupling_queue.clear()
        self._coupling_samples.clear()
        self._recent_coupling.clear()
        self._active = None
        self.engagements.clear()
        self._prev_lmb = False
        self._lmb_release_t = 0.0

    def record_frame(
        self,
        t_s: float,
        e_px: Optional[np.ndarray],
        v_hand_cnt_s: np.ndarray,
        p_aim_cnt: np.ndarray,
        p_rec_cnt: np.ndarray,
        gate_reason: str,
        dp_cmd_cnt: Optional[np.ndarray] = None,
    ) -> None:
        entry = {
            "t": round(float(t_s), 4),
            "e_px": [round(float(e_px[0]), 2), round(float(e_px[1]), 2)] if e_px is not None else None,
            "vh": round(float(np.hypot(v_hand_cnt_s[0], v_hand_cnt_s[1])), 1),
            "aim": [round(float(p_aim_cnt[0]), 1), round(float(p_aim_cnt[1]), 1)],
            "rec": [round(float(p_rec_cnt[0]), 1), round(float(p_rec_cnt[1]), 1)],
            "gate": str(gate_reason),
        }
        if dp_cmd_cnt is not None:
            entry["dp"] = [round(float(dp_cmd_cnt[0]), 1), round(float(dp_cmd_cnt[1]), 1)]
        self._pre_ring.append(entry)
        if self._active is not None and len(self._active["trace"]) < 240:
            self._active["trace"].append(entry)

    def clear_pending_probes(self) -> None:
        """Discard pending coupling samples/recent history (e.g. after F8 calibration sweep)."""
        self._coupling_queue.clear()
        self._recent_coupling.clear()

    def get_effective_coupling(self, default: float = 1.0, min_samples: int = 3) -> float:
        """Return measured physical stage->sensor transfer ratio when sufficient live samples exist."""
        now_t = time.perf_counter()
        recent = [(c, m) for t_c, c, m in self._recent_coupling if (now_t - t_c) <= 8.0]
        if len(recent) >= min_samples:
            rc_cmd = sum(x[0] for x in recent)
            if rc_cmd >= 25.0:
                return float(np.clip(sum(x[1] for x in recent) / rc_cmd, 0.05, 1.0))
        if len(self._coupling_samples) >= min_samples:
            all_cmd = sum(x[0] for x in self._coupling_samples)
            if all_cmd >= 25.0:
                return float(np.clip(sum(x[1] for x in self._coupling_samples) / all_cmd, 0.05, 1.0))
        return float(default)

    def on_stage_dispatch(self, t_now: float, dp_cmd_cnt: np.ndarray, sensor_xy: np.ndarray) -> None:
        if float(np.hypot(dp_cmd_cnt[0], dp_cmd_cnt[1])) >= 5.0:
            self._coupling_queue.append(
                (float(t_now) + 0.095, np.asarray(dp_cmd_cnt, dtype=float).copy(), np.asarray(sensor_xy, dtype=float).copy())
            )

    def poll_coupling(self, t_now: float, sensor_xy: np.ndarray, v_hand_norm: float) -> None:
        while self._coupling_queue and t_now >= self._coupling_queue[0][0]:
            _, dp_cmd, s0 = self._coupling_queue.popleft()
            cmd_norm = float(np.hypot(dp_cmd[0], dp_cmd[1]))
            if cmd_norm < 5.0 or v_hand_norm > 110.0:
                continue
            ds = np.asarray(sensor_xy, dtype=float) - s0
            proj = float(np.dot(ds, dp_cmd)) / max(1e-6, cmd_norm)
            sample = (cmd_norm, proj)
            self._coupling_samples.append(sample)
            self._recent_coupling.append((float(t_now), cmd_norm, proj))
            if self._active is not None:
                self._active["coupling"].append(sample)

    def _open_engagement(self, t_s: float, p_stage_now: np.ndarray, weapon: str) -> None:
        sp_norm = float(np.hypot(p_stage_now[0], p_stage_now[1]))
        now_perf = time.perf_counter()
        pre_c = [(c, m) for t_c, c, m in self._recent_coupling if (now_perf - t_c) <= 0.35]
        self._active = {
            "eng_id": len(self.engagements) + 1,
            "t_start_s": round(float(t_s), 3),
            "t_end_s": round(float(t_s), 3),
            "weapon": str(weapon or "unknown"),
            "sp_pre_cnt": [round(float(p_stage_now[0]), 2), round(float(p_stage_now[1]), 2)],
            "sp_pre_norm": round(sp_norm, 2),
            "sp_max_norm": round(sp_norm, 2),
            "shots": [],
            "gsi_kills": 0,
            "gsi_hs": 0,
            "kill_events": [],
            "coupling": pre_c,
            "pre_trace": list(self._pre_ring)[-18:],
            "trace": [],
        }

    def on_shot(
        self,
        t_s: float,
        p_stage_now: np.ndarray,
        verdict_dict: Dict[str, Any],
        weapon: str = "",
    ) -> None:
        if self._active is None:
            self._open_engagement(t_s, p_stage_now, weapon)
        assert self._active is not None
        if weapon and self._active["weapon"] == "unknown":
            self._active["weapon"] = str(weapon)
        self._active["t_end_s"] = round(float(t_s), 3)
        self._active["sp_max_norm"] = round(
            max(self._active["sp_max_norm"], float(np.hypot(p_stage_now[0], p_stage_now[1]))), 2
        )
        self._active["shots"].append(verdict_dict)

    def step_lifecycle(
        self,
        t_s: float,
        lmb_down: bool,
        p_stage_now: np.ndarray,
        gsi: Optional[CS2GSIListener] = None,
    ) -> Optional[Dict[str, Any]]:
        lmb_pressed = lmb_down and (not self._prev_lmb)
        lmb_released = self._prev_lmb and (not lmb_down)
        self._prev_lmb = lmb_down

        dk, dhs, weapon = gsi.pop_delta() if gsi is not None else (0, 0, "")
        if lmb_pressed and self._active is None:
            self._open_engagement(t_s, p_stage_now, weapon)
        if lmb_down and self._active is not None:
            self._active["sp_max_norm"] = round(
                max(self._active["sp_max_norm"], float(np.hypot(p_stage_now[0], p_stage_now[1]))), 2
            )
            self._active["t_end_s"] = round(float(t_s), 3)
        if lmb_released:
            self._lmb_release_t = t_s

        if dk > 0:
            if self._active is not None:
                self._active["gsi_kills"] += dk
                self._active["gsi_hs"] += dhs
                self._active["kill_events"].append(
                    {
                        "t_s": round(float(t_s), 3),
                        "shots_so_far": len(self._active["shots"]),
                        "dk": int(dk),
                        "dhs": int(dhs),
                    }
                )
            elif self.engagements and (float(t_s) - float(self.engagements[-1]["t_end_s"])) <= 0.65:
                last_e = self.engagements[-1]
                last_e["gsi_kills"] += dk
                last_e["gsi_hs"] += dhs
                kes = last_e.setdefault("kill_events", [])
                n_s = int(last_e.get("n_shots", len(last_e.get("shots", []))))
                kes.append(
                    {
                        "t_s": round(float(t_s), 3),
                        "shots_so_far": n_s,
                        "dk": int(dk),
                        "dhs": int(dhs),
                    }
                )
                if last_e.get("shots_to_first_kill") is None and kes:
                    ke0 = kes[0]
                    last_e["shots_to_first_kill"] = max(1, min(max(1, n_s), int(ke0["shots_so_far"]))) if n_s > 0 else None
                    last_e["t_to_first_kill_s"] = round(max(0.0, float(ke0["t_s"]) - float(last_e["t_start_s"])), 3)

        if self._active is not None and (not lmb_down) and (t_s - self._lmb_release_t) >= 0.22:
            return self._close_active()
        return None

    def _close_active(self) -> Optional[Dict[str, Any]]:
        eng = self._active
        self._active = None
        if eng is None or not eng["shots"]:
            return None
        s0 = eng["shots"][0]
        ea_vals = [float(s["e_actual_cnt"]) for s in eng["shots"]]
        head_vals = [float(s["head_dist_cnt"]) for s in eng["shots"]]
        rank = {"HEAD_HIT": 2, "BODY_HIT": 1, "WHIFF": 0}
        best_hb = max((s["hitbox"] for s in eng["shots"]), key=lambda h: rank.get(h, 0))

        c_samples = eng["coupling"]
        cmd_sum = sum(c[0] for c in c_samples)
        meas_sum = sum(c[1] for c in c_samples)
        ratio = (meas_sum / cmd_sum) if cmd_sum >= 5.0 else None

        flags: List[str] = []
        if eng["sp_pre_norm"] >= 100.0:
            flags.append("PRE_SAT")
        if ratio is not None and ratio < 0.55:
            flags.append("LOW_COUPLING")
        if float(s0["delta_e_cnt"]) < -2.0:
            flags.append("CONFLICT")
        if 10.5 < float(s0["head_dist_cnt"]) <= 22.0:
            flags.append("NEAR_HEAD_SHORT")
        if not flags and best_hb == "HEAD_HIT":
            flags.append("OK")

        n_s = len(eng["shots"])
        kes = eng.get("kill_events", [])
        if kes:
            ke0 = kes[0]
            stk = max(1, min(n_s, int(ke0["shots_so_far"])))
            t_fk = round(max(0.0, float(ke0["t_s"]) - float(eng["t_start_s"])), 3)
        else:
            stk = None
            t_fk = None

        eng["n_shots"] = n_s
        eng["shots_to_first_kill"] = stk
        eng["t_to_first_kill_s"] = t_fk
        eng["best_hitbox"] = best_hb
        eng["shot1_eh_cnt"] = round(float(s0["e_hand_cnt"]), 2)
        eng["shot1_ea_cnt"] = round(float(s0["e_actual_cnt"]), 2)
        eng["shot1_head_cnt"] = round(float(s0["head_dist_cnt"]), 2)
        eng["shot1_dE_cnt"] = round(float(s0["delta_e_cnt"]), 2)
        eng["min_ea_cnt"] = round(min(ea_vals), 2)
        eng["min_head_cnt"] = round(min(head_vals), 2)
        eng["stage_cmd_cnt"] = round(cmd_sum, 1)
        eng["sensor_meas_cnt"] = round(meas_sum, 1)
        eng["coupling_ratio"] = round(ratio, 3) if ratio is not None else None
        eng["diag_flags"] = flags
        self.engagements.append(eng)
        return eng

    def finalize_and_save(self) -> Dict[str, Any]:
        if self._active is not None:
            self._close_active()

        with open(self.jsonl_path, "w", encoding="utf-8") as f:
            for e in self.engagements:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

        fields = [
            "eng_id", "t_start_s", "t_end_s", "n_shots", "weapon", "gsi_kills", "gsi_hs",
            "shots_to_first_kill", "t_to_first_kill_s",
            "best_hitbox", "shot1_eh_cnt", "shot1_ea_cnt", "shot1_head_cnt", "shot1_dE_cnt",
            "min_ea_cnt", "min_head_cnt", "sp_pre_norm", "sp_max_norm",
            "stage_cmd_cnt", "sensor_meas_cnt", "coupling_ratio", "diag_flags",
        ]
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for e in self.engagements:
                row = {k: e.get(k) for k in fields}
                row["diag_flags"] = "|".join(e.get("diag_flags", []))
                writer.writerow(row)

        all_cmd = sum(c[0] for c in self._coupling_samples)
        all_meas = sum(c[1] for c in self._coupling_samples)
        return {
            "total_engagements": len(self.engagements),
            "total_gsi_kills": sum(int(e["gsi_kills"]) for e in self.engagements),
            "total_gsi_hs": sum(int(e["gsi_hs"]) for e in self.engagements),
            "mean_coupling_ratio": round(all_meas / all_cmd, 3) if all_cmd >= 10.0 else None,
            "coupling_samples": len(self._coupling_samples),
            "jsonl_path": self.jsonl_path,
            "csv_path": self.csv_path,
        }
