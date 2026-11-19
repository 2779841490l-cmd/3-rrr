"""
blackbox_sysid_trainer.py - Data-Driven System Identification & Loaded-Hand Sweep Collector
============================================================================================
Replaces blind trial-and-error parameter guessing with two fast empirical workflows:
  1. Offline System ID (`--analyze`, default): mines `live_engagements.jsonl` +
     `live_session_report.json` to extract true closed-loop `tau_50_ms` delay, step response
     fractions, directional palm coupling ratios, and ASSIST vs SHADOW TTK decomposition.
  2. 15s Loaded-Hand Wall Sweep (`--collect-sweep`): user simply keeps palm weight naturally on
     the mouse (aimed at any static CS2 wall, NO Bot head required) while 16 multi-amplitude
     step probes (`12..60 cnt`) measure real palm deadband, stiction threshold, and lag.
  3. `--apply`: safely backs up `config_hil.json` into `backups/` and writes identified params.
"""

import argparse
import json
import os
import shutil
import sys
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

HIL_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HIL_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

DEFAULT_CONFIG_PATH = os.path.join(HIL_DIR, "config_hil.json")
DEFAULT_JSONL_PATH = os.path.join(HIL_DIR, "logs", "live_engagements.jsonl")
DEFAULT_REPORT_PATH = os.path.join(HIL_DIR, "logs", "live_session_report.json")
DEFAULT_SWEEP_OUT = os.path.join(HIL_DIR, "logs", "palm_sweep_sysid.json")
BACKUPS_DIR = os.path.join(REPO_ROOT, "backups")


def _load_engagements(jsonl_path: str) -> List[Dict[str, Any]]:
    if not os.path.isfile(jsonl_path):
        return []
    out: List[Dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                out.append(json.loads(s))
    return out


def _estimate_empirical_step_response(engs: List[Dict[str, Any]], px_per_cnt: float = 0.254) -> Dict[str, Any]:
    """Identify true visual step response latency & gain fraction from logged LOCK_TRIM/SNAP steps."""
    buckets = [(15.0, 35.0), (40.0, 65.0), (70.0, 95.0), (100.0, 125.0), (130.0, 155.0)]
    samples: Dict[Tuple[float, float], List[float]] = {b: [] for b in buckets}
    n_steps = 0

    for e in engs:
        tr = list(e.get("pre_trace", [])) + list(e.get("trace", []))
        for i, fr in enumerate(tr):
            dp = fr.get("dp")
            e0 = fr.get("e_px")
            if dp is None or e0 is None or fr.get("gate") not in ("LOCK_TRIM", "LOCK_SNAP"):
                continue
            if abs(float(dp[0])) < 22.0:
                continue
            n_steps += 1
            t0 = float(fr["t"])
            expected_de_x = -float(px_per_cnt) * float(dp[0])
            if abs(expected_de_x) < 1.0:
                continue
            for j in range(i + 1, min(len(tr), i + 36)):
                if float(tr[j].get("vh", 0.0)) > 400.0:
                    break
                dp_j = tr[j].get("dp")
                if dp_j is not None and (abs(float(dp_j[0])) >= 8.0 or abs(float(dp_j[1])) >= 12.0):
                    # Isolate single-step dynamics cleanly without confounding subsequent commands
                    break
                ej = tr[j].get("e_px")
                if ej is None:
                    continue
                dt_ms = (float(tr[j]["t"]) - t0) * 1000.0
                frac = (float(ej[0]) - float(e0[0])) / expected_de_x
                for b in buckets:
                    if b[0] <= dt_ms <= b[1]:
                        samples[b].append(frac)

    curve = []
    tau_50_ms = 92.0
    for (lo, hi) in buckets:
        vals = samples[(lo, hi)]
        med = float(np.median(vals)) if vals else 0.0
        curve.append({"window_ms": [lo, hi], "n": len(vals), "med_frac": round(med, 3)})
    for c_prev, c_cur in zip(curve, curve[1:]):
        if c_prev["med_frac"] < 0.5 <= c_cur["med_frac"] and (c_cur["med_frac"] - c_prev["med_frac"]) > 1e-4:
            m0 = 0.5 * sum(c_prev["window_ms"])
            m1 = 0.5 * sum(c_cur["window_ms"])
            alpha = (0.5 - c_prev["med_frac"]) / (c_cur["med_frac"] - c_prev["med_frac"])
            tau_50_ms = float(np.clip(m0 + alpha * (m1 - m0), 65.0, 145.0))
            break

    return {"n_isolated_steps": n_steps, "curve": curve, "tau_50_ms": round(tau_50_ms, 1)}


def _eng_mode(e: Dict[str, Any]) -> str:
    sh = e.get("shots", [])
    if sh and all(s.get("mode") == "SHADOW_REFEREE_ONLY" for s in sh):
        return "SHADOW_REFEREE_ONLY"
    if any(s.get("mode") == "ASSIST_ON" for s in sh):
        return "ASSIST_ON"
    return "UNKNOWN"


def _extract_mode_blocks(
    engs: List[Dict[str, Any]],
    min_n: int = 5,
    pause_split_s: float = 4.0,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, float]]]:
    """Group consecutive same-mode fights into alternating blocks and compute exact matched-numerator/denominator active KPM."""
    raw_blocks: List[Tuple[str, List[Dict[str, Any]]]] = []
    cur: List[Dict[str, Any]] = []
    for e in engs:
        m = _eng_mode(e)
        if m == "UNKNOWN":
            continue
        if cur and _eng_mode(cur[-1]) != m:
            raw_blocks.append((_eng_mode(cur[0]), cur))
            cur = [e]
        else:
            cur.append(e)
    if cur:
        raw_blocks.append((_eng_mode(cur[0]), cur))

    matched: Dict[str, Dict[str, float]] = {
        "ASSIST_ON": {"dur_s": 0.0, "kills": 0.0, "hs": 0.0},
        "SHADOW_REFEREE_ONLY": {"dur_s": 0.0, "kills": 0.0, "hs": 0.0},
    }
    blocks_out: List[Dict[str, Any]] = []

    for b_idx, (mode_name, sub) in enumerate(raw_blocks, start=1):
        runs: List[List[Dict[str, Any]]] = []
        r_cur: List[Dict[str, Any]] = [sub[0]]
        for e_prev, e_now in zip(sub, sub[1:]):
            gap = float(e_now["t_start_s"]) - float(e_prev["t_end_s"])
            if 0.0 <= gap <= pause_split_s:
                r_cur.append(e_now)
            else:
                runs.append(r_cur)
                r_cur = [e_now]
        runs.append(r_cur)

        b_dur = 0.0
        b_kills = 0
        b_hs = 0
        for run in runs:
            if len(run) >= 2:
                d_r = max(0.0, float(run[-1]["t_end_s"]) - float(run[0]["t_end_s"]))
                # Skip run[0]'s kills so numerator and denominator span the exact same (run[0].t_end -> run[-1].t_end] window
                k_r = sum(int(e.get("gsi_kills", 0)) for e in run[1:])
                hs_r = sum(int(e.get("gsi_hs", 0)) for e in run[1:])
                b_dur += d_r
                b_kills += k_r
                b_hs += hs_r
        if mode_name in matched:
            matched[mode_name]["dur_s"] += b_dur
            matched[mode_name]["kills"] += b_kills
            matched[mode_name]["hs"] += b_hs

        if len(sub) >= min_n:
            s1_ea = [float(e["shot1_ea_cnt"]) for e in sub if e.get("shot1_ea_cnt") is not None]
            cps = [float(e["coupling_ratio"]) for e in sub if e.get("coupling_ratio") is not None]
            tot_k = sum(int(e.get("gsi_kills", 0)) for e in sub)
            tot_hs = sum(int(e.get("gsi_hs", 0)) for e in sub)
            blocks_out.append({
                "block_id": b_idx,
                "mode": mode_name,
                "n": len(sub),
                "matched_dur_s": round(b_dur, 1),
                "matched_kills": int(b_kills),
                "total_kills": int(tot_k),
                "total_hs": int(tot_hs),
                "kpm": round(60.0 * b_kills / b_dur, 1) if b_dur >= 1.0 and b_kills > 0 else None,
                "med_shots": round(float(np.median([int(e["n_shots"]) for e in sub])), 1),
                "shot1_ea_med_cnt": round(float(np.median(s1_ea)), 1) if s1_ea else None,
                "coupling_med": round(float(np.median(cps)), 3) if cps else None,
            })

    return blocks_out, matched


def _stratify_by_shot1_eh(engs: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Causal Aiming Stratification: separate Do-No-Harm (<=30 cnt) vs Whiff-Rescue (30..80 cnt) vs Wide (>80 cnt)."""
    tiers = [
        ("on_target_<=30cnt", lambda v: v <= 30.0),
        ("rescue_30_80cnt", lambda v: 30.0 < v <= 80.0),
        ("wide_>80cnt", lambda v: v > 80.0),
    ]
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for t_name, pred in tiers:
        out[t_name] = {}
        for mode_name in ("ASSIST_ON", "SHADOW_REFEREE_ONLY"):
            sub = [
                e for e in engs
                if _eng_mode(e) == mode_name
                and e.get("shot1_eh_cnt") is not None
                and pred(float(e["shot1_eh_cnt"]))
            ]
            if not sub:
                out[t_name][mode_name] = {"n": 0}
                continue
            shots = [int(e["n_shots"]) for e in sub]
            tot_k = sum(int(e.get("gsi_kills", 0)) for e in sub)
            has_gsi = any(int(e.get("gsi_kills", 0)) > 0 for e in engs if _eng_mode(e) == mode_name)
            s1 = [float(e["shots"][0]["e_actual_cnt"]) for e in sub if e.get("shots") and e["shots"][0].get("e_actual_cnt") is not None]
            s2 = [float(e["shots"][1]["e_actual_cnt"]) for e in sub if len(e.get("shots", [])) > 1 and e["shots"][1].get("e_actual_cnt") is not None]
            confl = sum(1 for e in sub if "CONFLICT" in e.get("diag_flags", []))
            zero_k = sum(1 for e in sub if int(e.get("gsi_kills", 0)) == 0)
            out[t_name][mode_name] = {
                "n": len(sub),
                "med_shots": round(float(np.median(shots)), 1),
                "shots_per_kill": round(sum(shots) / tot_k, 2) if tot_k > 0 else None,
                "zero_kill_pct": round(100.0 * zero_k / len(sub), 1) if has_gsi else None,
                "s1_to_s2_ea": [
                    round(float(np.median(s1)), 1) if s1 else None,
                    round(float(np.median(s2)), 1) if s2 else None,
                ],
                "conflict_pct": round(100.0 * confl / len(sub), 1),
            }
    return out


def _summarize_mode_split(
    engs: List[Dict[str, Any]],
    matched_stats: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, Dict[str, Any]]:
    def _agg(mode_name: str) -> Dict[str, Any]:
        sub = [e for e in engs if _eng_mode(e) == mode_name]
        if not sub:
            return {"n": 0}
        ttks = [float(e["t_end_s"]) - float(e["t_start_s"]) for e in sub]
        shots = [int(e["n_shots"]) for e in sub]
        s1_eh = [float(e["shot1_eh_cnt"]) for e in sub if e.get("shot1_eh_cnt") is not None]
        s1_ea = [float(e["shot1_ea_cnt"]) for e in sub if e.get("shot1_ea_cnt") is not None]
        min_ea = [float(e["min_ea_cnt"]) for e in sub if e.get("min_ea_cnt") is not None]
        hs = sum(1 for e in sub if e.get("best_hitbox") == "HEAD_HIT")
        bh = sum(1 for e in sub if e.get("best_hitbox") == "BODY_HIT")
        pre_sat = sum(1 for e in sub if "PRE_SAT" in e.get("diag_flags", []))
        confl = sum(1 for e in sub if "CONFLICT" in e.get("diag_flags", []))

        gaps: List[float] = []
        cycles: List[float] = []
        for e0, e1 in zip(engs, engs[1:]):
            if _eng_mode(e0) == mode_name and _eng_mode(e1) == mode_name:
                g = float(e1["t_start_s"]) - float(e0["t_end_s"])
                if 0.05 < g <= 4.0:
                    gaps.append(g)
                    cycles.append(float(e1["t_end_s"]) - float(e0["t_end_s"]))

        s_1_2 = sum(1 for n_s in shots if n_s <= 2)
        s_3_4 = sum(1 for n_s in shots if 3 <= n_s <= 4)
        s_5p = sum(1 for n_s in shots if n_s >= 5)

        per_shot_ea: Dict[str, Optional[float]] = {}
        for k in range(4):
            vals = [float(e["shots"][k]["e_actual_cnt"]) for e in sub if len(e.get("shots", [])) > k and e["shots"][k].get("e_actual_cnt") is not None]
            per_shot_ea[f"shot{k+1}"] = round(float(np.median(vals)), 1) if vals else None

        gsi_kills = sum(int(e.get("gsi_kills", 0)) for e in sub)
        gsi_hs = sum(int(e.get("gsi_hs", 0)) for e in sub)
        has_gsi = gsi_kills > 0
        m_entry = (matched_stats or {}).get(mode_name, {})
        m_dur = float(m_entry.get("dur_s", 0.0))
        m_kills = float(m_entry.get("kills", 0.0))
        med_cycle = float(np.median(cycles)) if cycles else None

        # True Single-Kill efficiency separated cleanly from 0-kill whiffs and multi-kill sprays
        sk_sub = [e for e in sub if int(e.get("gsi_kills", 0)) == 1]
        if sk_sub:
            sk_stks = [
                int(e["shots_to_first_kill"]) if e.get("shots_to_first_kill") is not None else int(e["n_shots"])
                for e in sk_sub
            ]
            sk_ttks = [
                float(e["t_to_first_kill_s"]) if e.get("t_to_first_kill_s") is not None else (float(e["t_end_s"]) - float(e["t_start_s"]))
                for e in sk_sub
            ]
            sk_12 = sum(1 for x in sk_stks if x <= 2)
            sk_34 = sum(1 for x in sk_stks if 3 <= x <= 4)
            sk_5p = sum(1 for x in sk_stks if x >= 5)
            sk_dist = {
                "1_2_shots": round(100.0 * sk_12 / len(sk_sub), 1),
                "3_4_shots": round(100.0 * sk_34 / len(sk_sub), 1),
                "5plus_shots": round(100.0 * sk_5p / len(sk_sub), 1),
            }
            sk_med_stk = round(float(np.median(sk_stks)), 1)
            sk_med_ttk = round(float(np.median(sk_ttks)), 3)
        else:
            sk_dist = None
            sk_med_stk = None
            sk_med_ttk = None

        zero_k = sum(1 for e in sub if int(e.get("gsi_kills", 0)) == 0)
        multi_k = sum(1 for e in sub if int(e.get("gsi_kills", 0)) >= 2)
        tot_shots = sum(shots)

        return {
            "n": len(sub),
            "med_ttk_s": round(float(np.median(ttks)), 3),
            "mean_ttk_s": round(float(np.mean(ttks)), 3),
            "med_gap_s": round(float(np.median(gaps)), 3) if gaps else None,
            "med_cycle_s": round(med_cycle, 3) if med_cycle else None,
            "est_bursts_per_min": round(60.0 / med_cycle, 1) if med_cycle and med_cycle > 0.05 else None,
            "med_shots": round(float(np.median(shots)), 1),
            "shot_dist_pct": {
                "1_2_shots": round(100.0 * s_1_2 / len(sub), 1),
                "3_4_shots": round(100.0 * s_3_4 / len(sub), 1),
                "5plus_shots": round(100.0 * s_5p / len(sub), 1),
            },
            "per_shot_ea_med_cnt": per_shot_ea,
            "gsi_kills": gsi_kills,
            "gsi_hs": gsi_hs,
            "matched_active_s": round(m_dur, 1),
            "matched_gsi_kills": int(m_kills),
            "true_gsi_kpm": round(60.0 * m_kills / m_dur, 1) if m_dur >= 1.0 and m_kills > 0 else None,
            "sec_per_kill": round(m_dur / m_kills, 3) if m_dur >= 1.0 and m_kills > 0 else None,
            "zero_kill_burst_pct": round(100.0 * zero_k / len(sub), 1) if has_gsi else None,
            "multi_kill_burst_pct": round(100.0 * multi_k / len(sub), 1) if has_gsi else None,
            "kills_per_burst": round(gsi_kills / len(sub), 3) if has_gsi else None,
            "shots_per_kill": round(tot_shots / gsi_kills, 2) if has_gsi else None,
            "single_kill_n": len(sk_sub),
            "single_kill_med_stk": sk_med_stk,
            "single_kill_med_ttk_s": sk_med_ttk,
            "single_kill_dist_pct": sk_dist,
            "hs_eng_rate": round(hs / len(sub), 4),
            "hit_eng_rate": round((hs + bh) / len(sub), 4),
            "shot1_eh_med_cnt": round(float(np.median(s1_eh)), 1) if s1_eh else None,
            "shot1_ea_med_cnt": round(float(np.median(s1_ea)), 1) if s1_ea else None,
            "min_ea_med_cnt": round(float(np.median(min_ea)), 1) if min_ea else None,
            "pre_sat_pct": round(100.0 * pre_sat / len(sub), 1),
            "conflict_pct": round(100.0 * confl / len(sub), 1),
        }

    return {"ASSIST_ON": _agg("ASSIST_ON"), "SHADOW_REFEREE_ONLY": _agg("SHADOW_REFEREE_ONLY")}


def analyze_blackbox(jsonl_path: str = DEFAULT_JSONL_PATH, config_path: str = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Run full empirical system identification and bottleneck breakdown from blackbox logs."""
    engs = _load_engagements(jsonl_path)
    if not engs:
        raise FileNotFoundError(f"No engagements found at {jsonl_path}")

    px_per_cnt = 0.254
    if os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        m_est = cfg.get("stage_kinematics", {}).get("J_screen_to_stage")
        if isinstance(m_est, list) and len(m_est) == 2:
            try:
                inv_m = np.linalg.inv(np.asarray(m_est, dtype=float))
                px_per_cnt = float(max(0.15, min(0.45, abs(float(inv_m[0, 0])))))
            except np.linalg.LinAlgError:
                pass

    step_id = _estimate_empirical_step_response(engs, px_per_cnt=px_per_cnt)
    alt_blocks, matched_stats = _extract_mode_blocks(engs)
    mode_split = _summarize_mode_split(engs, matched_stats=matched_stats)
    stratified = _stratify_by_shot1_eh(engs)
    cps = [float(e["coupling_ratio"]) for e in engs if e.get("coupling_ratio") is not None]
    pre_g: Counter = Counter()
    lmb_g: Counter = Counter()
    for e in engs:
        for fr in e.get("pre_trace", []):
            pre_g[fr.get("gate")] += 1
        for fr in e.get("trace", []):
            lmb_g[fr.get("gate")] += 1

    rec_lag_ms = float(np.clip(step_id["tau_50_ms"] + 12.0, 78.0, 125.0))
    return {
        "n_engagements": len(engs),
        "mode_split": mode_split,
        "alternating_blocks": alt_blocks,
        "stratified_by_shot1_eh": stratified,
        "coupling_summary": {
            "n": len(cps),
            "p10": round(float(np.percentile(cps, 10)), 3) if cps else None,
            "median": round(float(np.median(cps)), 3) if cps else None,
            "mean": round(float(np.mean(cps)), 3) if cps else None,
            "p90": round(float(np.percentile(cps, 90)), 3) if cps else None,
        },
        "step_response": step_id,
        "gate_counts": {"PRE": dict(pre_g), "LMB": dict(lmb_g)},
        "recommended_live_control": {"visual_lag_ms": round(rec_lag_ms, 1)},
    }


def apply_sysid_to_config(sysid: Dict[str, Any], config_path: str = DEFAULT_CONFIG_PATH) -> str:
    """Safely back up `config_hil.json` to `backups/` before writing identified parameters."""
    os.makedirs(BACKUPS_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    bak_path = os.path.join(BACKUPS_DIR, f"config_hil_{ts}.json")
    if os.path.isfile(config_path):
        shutil.copy2(config_path, bak_path)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    lc = dict(cfg.get("live_control", {}))
    rec = sysid.get("recommended_live_control", {})
    if "visual_lag_ms" in rec:
        lc["visual_lag_ms"] = float(rec["visual_lag_ms"])
    cfg["live_control"] = lc
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    return bak_path


def run_loaded_wall_sweep(config_path: str = DEFAULT_CONFIG_PATH, out_path: str = DEFAULT_SWEEP_OUT) -> Dict[str, Any]:
    """
    15-second hands-on CS2 loaded-palm calibration sweep:
    Keep palm resting naturally on mouse facing any blank wall (no Bot target needed).
    Injects 16 multi-amplitude pulses [14, 28, 44, 62] cnt along +-X and +-Y, recording
    1 kHz USB sensor displacement & real transfer efficiency.
    """
    from shared_control_hil.fast_self_calib import FastSelfCalibrator

    cal = FastSelfCalibrator(config_path=config_path)
    trainer = cal.ensure_trainer()
    amps = [14.0, 28.0, 44.0, 62.0]
    axes = [(1, 0, "+X"), (-1, 0, "-X"), (0, 1, "+Y"), (0, -1, "-Y")]
    results = []
    u0x, u0y = float(cal.u0_neutral[0]), float(cal.u0_neutral[1])
    last_u_lin = cal.u0_neutral.copy()

    print("[Loaded Wall Sweep] Rest palm normally on mouse facing any CS2 wall (starts in 1.5s)...")
    time.sleep(1.5)
    try:
        for sx, sy, tag in axes:
            for amp in amps:
                trainer.driver.move_smooth_raw(u0x, u0y, duration_s=0.14, rate_hz=50.0)
                last_u_lin = cal.u0_neutral.copy()
                time.sleep(0.11)
                p_tgt = np.array([sx * amp, sy * amp], dtype=float)
                s0 = np.array(trainer.sensor.get_position(), dtype=float)
                e1, e2, u_lin, _ = cal.stage_pos_to_pulses(trainer.manifold, float(p_tgt[0]), float(p_tgt[1]), dx=float(p_tgt[0]), dy=float(p_tgt[1]))
                bst = cal.compute_live_ballistic_boost(u_lin - last_u_lin, p_tgt, e1, e2, boost_scale=1.22, boost_dur_ms=30.0)
                if bst is not None and getattr(trainer.driver, "b_cmd_supported", False):
                    trainer.driver.send_ballistic_pulse(bst[0], bst[1], e1, e2, bst[2])
                else:
                    trainer.driver.send_raw_pulses(e1, e2)
                last_u_lin = u_lin.copy()
                time.sleep(0.12)
                s1 = np.array(trainer.sensor.get_position(), dtype=float)
                ds = s1 - s0
                par = float(ds[0] * sx + ds[1] * sy)
                results.append({"dir": tag, "cmd_cnt": amp, "meas_cnt": round(par, 2), "ratio": round(par / amp, 3)})
        trainer.driver.move_smooth_raw(u0x, u0y, duration_s=0.18, rate_hz=50.0)
    finally:
        cal.close()

    by_dir = {}
    for _, _, tag in axes:
        rs = [r["ratio"] for r in results if r["dir"] == tag]
        by_dir[tag] = round(float(np.median(rs)), 3) if rs else 0.0
    payload = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "by_dir_median_ratio": by_dir, "probes": results}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[Loaded Wall Sweep] Saved -> {out_path} | by_dir={by_dir}")
    return payload


def print_kill_efficiency_summary(rep: Dict[str, Any]) -> None:
    """Print 4-layer Chinese Aim-Botz Kill-Efficiency & Causal Aiming Evaluation."""
    ms = rep.get("mode_split", {})
    cp = rep.get("coupling_summary", {})
    blocks = rep.get("alternating_blocks", [])
    strat = rep.get("stratified_by_shot1_eh", {})

    print("\n================ 1. 交替波次真实 KPM 对照 ================")
    if blocks:
        for b in blocks:
            tag_cn = "开启辅助" if b["mode"] == "ASSIST_ON" else "纯手打  "
            kpm_s = f"{b['kpm']} KPM" if b.get("kpm") is not None else f"n={b['n']} (无GSI)"
            print(
                f"  Block {b['block_id']:02d} [{tag_cn} n={b['n']:3d}] {kpm_s} ({b['matched_kills']}杀/{b['matched_dur_s']}s) | "
                f"中位={b['med_shots']}发 | shot1_ea={b['shot1_ea_med_cnt']} | cp={b['coupling_med']}"
            )
    else:
        print("  (暂无 >=5 轮的连续波次)")

    print("\n================ 2. 真实击杀效率拆解 (KPM + STK + 空枪率) ================")
    for tag, label in (("ASSIST_ON", "开启机械辅助"), ("SHADOW_REFEREE_ONLY", "纯手打对照  ")):
        d = ms.get(tag, {})
        if not d.get("n"):
            continue
        sd = d.get("single_kill_dist_pct") or d.get("shot_dist_pct", {})
        ps = d.get("per_shot_ea_med_cnt", {})
        if d.get("true_gsi_kpm") is not None:
            kpm_str = (
                f"真KPM={d['true_gsi_kpm']} ({d['matched_gsi_kills']}杀/{d['matched_active_s']}s, "
                f"平均每杀={d.get('sec_per_kill')}s, 每杀={d.get('shots_per_kill')}发)"
            )
        else:
            kpm_str = f"等效交火={d.get('est_bursts_per_min')}/分 | 单只周期={d.get('med_cycle_s')}s"
        stk_str = f"单杀分布(n={d.get('single_kill_n',0)}, 中位={d.get('single_kill_med_stk')}发/{d.get('single_kill_med_ttk_s')}s)" if d.get("single_kill_n") else "全部发数分布"
        print(
            f"[{label} n={d['n']}] {kpm_str}\n"
            f"    切人={d.get('med_gap_s')}s + 开火={d.get('med_ttk_s')}s | "
            f"{stk_str} 1-2发:{sd.get('1_2_shots')}% 3-4发:{sd.get('3_4_shots')}% 5+发:{sd.get('5plus_shots')}% | "
            f"0杀空枪率={d.get('zero_kill_burst_pct')}% | 多杀={d.get('multi_kill_burst_pct')}%\n"
            f"    Shot1~4 实际误差(cnt): {ps.get('shot1')} -> {ps.get('shot2')} -> {ps.get('shot3')} -> {ps.get('shot4')}"
        )

    print("\n================ 3. 首枪手误分层因果检验 (到底有没有提高瞄准) ================")
    tier_labels = (
        ("on_target_<=30cnt", "层1: 已瞄准 <=30 cnt (检验: 绝不添乱)"),
        ("rescue_30_80cnt", "层2: 轻微擦边 30~80 cnt (检验: 弹间救场)"),
        ("wide_>80cnt", "层3: 大偏/转移 >80 cnt (检验: 控枪兜底)"),
    )
    for t_key, t_title in tier_labels:
        td = strat.get(t_key, {})
        a_d = td.get("ASSIST_ON", {})
        s_d = td.get("SHADOW_REFEREE_ONLY", {})
        if not (a_d.get("n") or s_d.get("n")):
            continue
        print(f"  {t_title}")
        if a_d.get("n"):
            print(
                f"    ASSIST (n={a_d['n']:3d}): med={a_d['med_shots']}发 | 每杀={a_d['shots_per_kill']}发 | "
                f"0杀={a_d['zero_kill_pct']}% | S1->S2={a_d['s1_to_s2_ea']} | 冲突={a_d['conflict_pct']}%"
            )
        if s_d.get("n"):
            print(
                f"    SHADOW (n={s_d['n']:3d}): med={s_d['med_shots']}发 | 每杀={s_d['shots_per_kill']}发 | "
                f"0杀={s_d['zero_kill_pct']}% | S1->S2={s_d['s1_to_s2_ea']}"
            )

    a_ms = ms.get("ASSIST_ON", {})
    print(
        f"\n[4. 硬件健康度] 耦合率 median={cp.get('median')} (p10={cp.get('p10')}, p90={cp.get('p90')}) | "
        f"冲突率={a_ms.get('conflict_pct')}% | PRE_SAT={a_ms.get('pre_sat_pct')}% | 推荐 visual_lag_ms={rep.get('recommended_live_control', {}).get('visual_lag_ms')}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Blackbox System-ID & Loaded-Palm Sweep Trainer")
    parser.add_argument("--jsonl", type=str, default=DEFAULT_JSONL_PATH)
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--apply", action="store_true", help="Apply identified visual_lag_ms cleanly with backup")
    parser.add_argument("--collect-sweep", action="store_true", help="Run 15s live COM4 loaded-hand wall sweep")
    parser.add_argument("--install-gsi", action="store_true", help="Install CS2 gamestate_integration_3rrr.cfg into Steam csgo/cfg")
    args = parser.parse_args()

    if args.install_gsi:
        from shared_control_hil.engagement_logger import ensure_cs2_gsi_config
        p = ensure_cs2_gsi_config(port=30001)
        print(f"[CS2 GSI] Installed -> {p}" if p else "[CS2 GSI] CS2 csgo/cfg folder not found.")
        return 0 if p else 1

    if args.collect_sweep:
        run_loaded_wall_sweep(config_path=args.config)
        return 0

    rep = analyze_blackbox(jsonl_path=args.jsonl, config_path=args.config)
    print("=== Blackbox System-ID Report ===")
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    print_kill_efficiency_summary(rep)
    if args.apply:
        bak = apply_sysid_to_config(rep, config_path=args.config)
        print(f"[Applied] Backed up old config -> {bak}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
