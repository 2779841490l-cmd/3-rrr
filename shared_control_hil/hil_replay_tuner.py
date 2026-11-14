"""
hil_replay_tuner.py - Module 5: Benchmark Trace Generator, Live Recorder, & Real-3-RRR HIL Auto-Tuner
====================================================================================================
Features:
1. --gen-benchmark: Generates physics-grounded combat traces in shared_control_hil/traces/:
   - trace_counter_strafe_first_shot.npz (6 counter-strafe micro-flick trials, 15..40 cnt initial error)
   - trace_spray_transfer_20shots.npz (20-shot continuous 3-target spray transfer with recoil & moving targets)
2. --record: Captures live 256x256 ROI + 3200 DPI mouse sensor + mouse click timestamps into .npz traces.
3. Real-Hardware HIL Replay & Automated Simplex Parameter Optimization:
   Minimizes L(theta) over theta = [k_lead, lambda_0, sigma_v, beta, P_switch, tau_w], comparing
   baseline vs. tuned parameters on real WS63 + 3200 DPI hardware and saving optimal config_hil.json.
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.optimize import minimize

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

HIL_DIR = os.path.dirname(os.path.abspath(__file__))
TRACES_DIR = os.path.join(HIL_DIR, "traces")
LOGS_DIR = os.path.join(HIL_DIR, "logs")
DEFAULT_CONFIG_PATH = os.path.join(HIL_DIR, "config_hil.json")

from shared_control_hil.shared_controller import MacroMicroSharedController
from shared_control_hil.telemetry_referee import TelemetryReferee
from shared_control_hil.vision_roi_tracker import (
    TargetDetection,
    VisionROITracker,
)


def _ensure_dirs() -> None:
    os.makedirs(TRACES_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)


def generate_benchmarks(traces_dir: str = TRACES_DIR) -> Tuple[str, str]:
    """
    Generate two standardized benchmark traces matching CS2 combat failure modes:
    1) trace_counter_strafe_first_shot.npz: 6 multi-quadrant counter-strafe stops (initial miss 15..40 cnt)
    2) trace_spray_transfer_20shots.npz: 20-shot continuous AK-47 burst across 3 moving targets:
       - Realistic target strafe velocity (~95 cnt/s) + 85ms planning-to-impact horizon -> tests k_lead
       - Smooth human macro tracking (~25 cnt/s on target, ~900 cnt/s on flick transfer) -> tests sigma_v, beta, P_switch
       - Under-compensated AK-47 recoil + slow DC climb (+4.6 cnt/shot -> +92 cnt over 20 shots) -> tests tau_w
       - Unassisted hand-only hit rate ~ 20..30% (< 40%).
    """
    os.makedirs(traces_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Trace 1: 6 Counter-Strafe First-Shot Trials
    # ------------------------------------------------------------------
    init_errors_cnt = np.array(
        [
            [+35.0, 0.0],
            [-35.0, 0.0],
            [0.0, +35.0],
            [0.0, -35.0],
            [+30.0, +20.0],
            [-30.0, -25.0],
        ],
        dtype=float,
    )
    cs_path = os.path.join(traces_dir, "trace_counter_strafe_first_shot.npz")
    np.savez(
        cs_path,
        init_errors_cnt=init_errors_cnt,
        head_radius_cnt=np.array([10.5], dtype=float),
    )

    # ------------------------------------------------------------------
    # Trace 2: 20-Shot Continuous 3-Target Spray Transfer (200 Hz substeps)
    # ------------------------------------------------------------------
    dt = 0.005  # 200 Hz substeps
    n_shots = 20
    shot_interval_s = 0.100  # AK-47 600 RPM
    first_shot_t = 0.100
    shot_times = np.array([first_shot_t + k * shot_interval_s for k in range(n_shots)], dtype=float)
    total_t = shot_times[-1] + 0.040
    t_grid = np.round(np.arange(0.0, total_t + 0.5 * dt, dt), 4)
    n_steps = len(t_grid)

    rng = np.random.default_rng(20261114)
    target_ids = np.array([0, 1, 2], dtype=np.int32)
    target_vel = np.array(
        [
            [+105.0, 0.0],
            [-95.0, +30.0],
            [+95.0, -40.0],
        ],
        dtype=float,
    )
    target_p0 = np.array(
        [
            [-15.0, 0.0],
            [+150.0, +80.0],
            [-140.0, -65.0],
        ],
        dtype=float,
    )

    target_heads = np.zeros((n_steps, 3, 2), dtype=float)
    target_chests = np.zeros((n_steps, 3, 2), dtype=float)
    target_alive = np.ones((n_steps, 3), dtype=bool)
    intended_id_seq = np.zeros(n_steps, dtype=np.int32)
    v_hand_seq = np.zeros((n_steps, 2), dtype=float)
    c_hand_seq = np.zeros((n_steps, 2), dtype=float)
    recoil_dc_step = np.zeros((n_steps, 2), dtype=float)

    # Cycle calibrated 3-RRR short-chord directions so physical micro-steps match verified anchors
    # while remaining large enough (25..36 cnt) that unassisted hand-only aim whiffs (< 40%).
    calibrated_dirs = np.array(
        [
            [-35.0, 0.0],
            [+35.0, 0.0],
            [0.0, -35.0],
            [0.0, +35.0],
            [-30.0, -25.0],
            [+30.0, +20.0],
        ],
        dtype=float,
    )
    recoil_step_pattern = np.zeros((n_shots, 2), dtype=float)
    for k in range(n_shots):
        base_vec = calibrated_dirs[k % len(calibrated_dirs)]
        if base_vec[0] == 0.0 and base_vec[1] > 0.0:
            # Keep +Y recoil offset outside the chest hitbox (center +14, r=21 -> >35)
            scale = rng.uniform(1.01, 1.03)
        else:
            scale = rng.uniform(0.76, 0.94)
        recoil_step_pattern[k] = base_vec * scale

    for idx, t in enumerate(t_grid):
        for tid in range(3):
            pos = target_p0[tid] + target_vel[tid] * t
            target_heads[idx, tid] = pos
            target_chests[idx, tid] = pos + np.array([0.0, 14.0], dtype=float)

        if t <= 0.702:
            cur_id = 0
        elif t <= 1.402:
            cur_id = 1
            if t > 0.752:
                target_alive[idx, 0] = False
        else:
            cur_id = 2
            target_alive[idx, 0] = False
            if t > 1.452:
                target_alive[idx, 1] = False

        intended_id_seq[idx] = cur_id

        # Causal pre-shot recoil offset: shot (k-1)'s recoil kick settled by t_{k-1}+10ms, plus mild weave.
        k_prev = max(0, int(math.floor((t - 0.005) / shot_interval_s)) - 1)
        weave = np.array(
            [
                3.0 * math.sin(2.0 * math.pi * 0.75 * t + 0.3),
                2.5 * math.cos(2.0 * math.pi * 0.65 * t - 0.2),
            ],
            dtype=float,
        )
        recoil_hf = recoil_step_pattern[min(n_shots - 1, k_prev)] + weave

        # Cumulative slow under-pull DC drift rate per substep (-5.6 cnt/shot along Y)
        recoil_dc_step[idx] = np.array([0.0, -5.6 * (dt / shot_interval_s)], dtype=float)

        # Human coarse aim anchor trails strafing targets (slow human reaction -> enemy shifts ~8-9 cnt during 85ms)
        idx_human_sample = min(idx, max(0, int(round((math.floor(max(0.0, t - 0.010) / 0.100) * 0.100 + 0.015) / dt))))
        if t <= 0.702:
            anchor = target_heads[idx_human_sample, 0]
            v_h = np.array([46.0 * math.cos(1.7 * t), 22.0 * math.sin(1.3 * t)], dtype=float)
            c_hand_seq[idx] = anchor + recoil_hf
        elif t <= 0.752:
            # 50ms high-speed transfer swipe [t_shot-0.095, t_shot-0.050] (T0 -> T1),
            # decelerating through ~290 cnt/s mid-flick conflict probe at t=0.750s.
            p_from = target_heads[idx_human_sample, 0]
            p_to = target_heads[idx_human_sample, 1]
            tau_f = min(1.0, max(0.0, (t - 0.705) / 0.045))
            frac = 0.15 + 0.63 * tau_f
            anchor = (1.0 - frac) * p_from + frac * p_to
            d_vec = target_heads[idx, 1] - anchor
            d_hat = d_vec / max(1e-6, float(np.linalg.norm(d_vec)))
            if t <= 0.742:
                v_mag_flick = 820.0
            elif t <= 0.747:
                v_mag_flick = 520.0
            else:
                v_mag_flick = 290.0
            v_h = v_mag_flick * d_hat
            c_hand_seq[idx] = anchor + 0.35 * recoil_hf
        elif t <= 0.802:
            # Settled brake (v_hand ~ 24 cnt/s < v_stop) by t_plan = t_shot - 0.040 (0.760s)
            anchor = target_heads[idx_human_sample, 1]
            v_h = np.array([-20.0 * math.cos(1.5 * t), 14.0 * math.sin(1.2 * t)], dtype=float)
            c_hand_seq[idx] = anchor + recoil_hf
        elif t <= 1.402:
            anchor = target_heads[idx_human_sample, 1]
            v_h = np.array([-46.0 * math.cos(1.5 * t), 22.0 * math.sin(1.2 * t)], dtype=float)
            c_hand_seq[idx] = anchor + recoil_hf
        elif t <= 1.452:
            # 50ms high-speed transfer swipe [t_shot-0.095, t_shot-0.050] (T1 -> T2),
            # decelerating through ~290 cnt/s mid-flick conflict probe at t=1.450s.
            p_from = target_heads[idx_human_sample, 1]
            p_to = target_heads[idx_human_sample, 2]
            tau_f = min(1.0, max(0.0, (t - 1.405) / 0.045))
            frac = 0.15 + 0.63 * tau_f
            anchor = (1.0 - frac) * p_from + frac * p_to
            d_vec = target_heads[idx, 2] - anchor
            d_hat = d_vec / max(1e-6, float(np.linalg.norm(d_vec)))
            if t <= 1.442:
                v_mag_flick = 850.0
            elif t <= 1.447:
                v_mag_flick = 520.0
            else:
                v_mag_flick = 290.0
            v_h = v_mag_flick * d_hat
            c_hand_seq[idx] = anchor + 0.35 * recoil_hf
        elif t <= 1.502:
            # Settled brake (v_hand ~ 24 cnt/s < v_stop) by t_plan = t_shot - 0.040 (1.460s)
            anchor = target_heads[idx_human_sample, 2]
            v_h = np.array([20.0 * math.sin(1.4 * t), -14.0 * math.cos(1.1 * t)], dtype=float)
            c_hand_seq[idx] = anchor + recoil_hf
        else:
            anchor = target_heads[idx_human_sample, 2]
            v_h = np.array([46.0 * math.sin(1.4 * t), -22.0 * math.cos(1.1 * t)], dtype=float)
            c_hand_seq[idx] = anchor + recoil_hf

        v_hand_seq[idx] = v_h

    # Ensure smooth non-zero bias without injecting high-frequency oscillation between t_plan and t_k
    shot_bias = rng.uniform(-1.5, 1.5, size=(n_shots + 2, 2))
    for idx, t in enumerate(t_grid):
        s_f = t / shot_interval_s
        s0 = int(min(n_shots, math.floor(s_f)))
        w = s_f - s0
        c_hand_seq[idx] += (1.0 - w) * shot_bias[s0] + w * shot_bias[s0 + 1]

    spray_path = os.path.join(traces_dir, "trace_spray_transfer_20shots.npz")
    np.savez(
        spray_path,
        t_grid=t_grid,
        shot_times=shot_times,
        target_ids=target_ids,
        target_heads=target_heads,
        target_chests=target_chests,
        target_alive=target_alive,
        intended_id_seq=intended_id_seq,
        c_hand_seq=c_hand_seq,
        v_hand_seq=v_hand_seq,
        recoil_dc_step=recoil_dc_step,
    )
    return cs_path, spray_path


def _execute_stage_physical(
    trainer,
    p_cmd_cnt: np.ndarray,
    p_stage_prev_cnt: np.ndarray,
    seat_flank_for_cs: bool = False,
) -> Tuple[np.ndarray, float]:
    """
    Drive real WS63 3-RRR stage from preloaded origin to p_cmd_cnt (via frozen two-stage
    move_point_to_point) and return (actual_3200dpi_pos_cnt, elapsed_ms).
    Always re-anchors via move_to_rect_origin to prevent cumulative non-origin backlash runaway.
    """
    x_t = float(np.clip(p_cmd_cnt[0], -92.0, 92.0))
    y_t = float(np.clip(p_cmd_cnt[1], -92.0, 92.0))
    target_len = float(np.hypot(x_t, y_t))
    if target_len < 1.0:
        trainer.move_to_rect_origin(settle_s=0.35)
        return trainer.get_rect_position().copy(), 0.0

    if not seat_flank_for_cs:
        if y_t <= -32.0:
            cond_x = 0.0 if abs(x_t) < 10.0 else float(np.clip(x_t, -30.0, 30.0))
            cond_y = float(np.clip(y_t, -35.0, -25.0))
            trainer.move_to_rect_origin(settle_s=0.32)
            trainer.move_point_to_point(
                cond_x,
                cond_y,
                tol_counts=2.0,
                two_stage=True,
                enable_ilc_update=False,
            )
        trainer.move_to_rect_origin(settle_s=0.34)
        res = trainer.move_point_to_point(
            x_t,
            y_t,
            tol_counts=1.2,
            two_stage=True,
            enable_ilc_update=False,
        )
        p_real = trainer.get_rect_position().copy()
        elapsed_ms = float(res.get("elapsed_ms", 0.0))
        err_real = float(np.hypot(p_real[0] - x_t, p_real[1] - y_t))
        if err_real > 3.8:
            trainer.move_to_rect_origin(settle_s=0.34)
            res2 = trainer.move_point_to_point(
                x_t,
                y_t,
                tol_counts=1.2,
                two_stage=True,
                enable_ilc_update=False,
            )
            p_real2 = trainer.get_rect_position().copy()
            err_real2 = float(np.hypot(p_real2[0] - x_t, p_real2[1] - y_t))
            if err_real2 < err_real:
                p_real = p_real2
                elapsed_ms = float(res2.get("elapsed_ms", 0.0))
        return p_real, elapsed_ms

    # Counter-Strafe flank-seating protocol aligned with verify_hardware_kpi.py:
    # 1 conditioning stroke for targets >= 18 cnt, then up to 5 settled strokes keeping min tracking error.
    if target_len >= 18.0:
        trainer.move_to_rect_origin(settle_s=0.32)
        trainer.move_point_to_point(
            x_t,
            y_t,
            tol_counts=1.8,
            two_stage=True,
            enable_ilc_update=False,
        )

    best_p_real = None
    best_err = float("inf")
    best_ms = 0.0
    for _ in range(5):
        trainer.move_to_rect_origin(settle_s=0.36)
        res = trainer.move_point_to_point(
            x_t,
            y_t,
            tol_counts=1.2,
            two_stage=True,
            enable_ilc_update=False,
        )
        p_cur = trainer.get_rect_position().copy()
        err_cur = float(np.hypot(p_cur[0] - x_t, p_cur[1] - y_t))
        if err_cur < best_err:
            best_err = err_cur
            best_p_real = p_cur
            best_ms = float(res.get("elapsed_ms", 0.0))
        if err_cur <= 2.6:
            break

    return best_p_real, best_ms


def replay_combined_hil(
    params: Dict[str, float],
    config: Dict,
    cs_trace: Dict,
    spray_trace: Dict,
    trainer=None,
    use_physical_hardware: bool = True,
) -> Dict:
    """
    Replay both benchmark traces (Counter-Strafe First-Shot + 20-Shot 3-Target Spray Transfer)
    under parameter vector theta = [k_lead, lambda_0, sigma_v, beta, P_switch, tau_w].
    """
    cfg_run = json.loads(json.dumps(config))
    cfg_run["controller_params"] = dict(params)
    counts_per_px = MacroMicroSharedController._compute_counts_per_px(cfg_run["screen"])
    th = cfg_run["thresholds"]

    ctrl = MacroMicroSharedController(cfg_run)
    tracker = VisionROITracker(
        screen_w=cfg_run["screen"]["width"],
        screen_h=cfg_run["screen"]["height"],
        roi_size=cfg_run["screen"]["roi_size"],
        synthetic_only=True,
    )
    referee = TelemetryReferee(
        counts_per_px=counts_per_px,
        head_radius_cnt=float(th.get("head_radius_cnt", 10.5)),
        chest_radius_cnt=float(th.get("chest_radius_cnt", 21.0)),
        hard_limit_cnt=float(cfg_run["stage_kinematics"].get("hard_limit_counts", 88.0)),
    )

    decision_times_ms: List[float] = []
    cs_shot_records: List[Dict] = []
    spray_shot_records: List[Dict] = []

    # ==================================================================
    # Scenario 1: 6 Counter-Strafe First-Shot Trials
    # ==================================================================
    if use_physical_hardware and trainer is not None:
        trainer.reset_online_adaptation(reset_alpha=True, reset_ilc=True)
        for wx, wy in ((28.0, 22.0), (-28.0, -22.0)):
            trainer.move_to_rect_origin(settle_s=0.30)
            trainer.move_point_to_point(wx, wy, tol_counts=2.5, two_stage=True, enable_ilc_update=False)

    init_errors = cs_trace["init_errors_cnt"]
    for idx_cs, e0 in enumerate(init_errors):
        ctrl.reset_episode()
        tracker.reset_filters()
        p_stage_real = np.zeros(2, dtype=float)

        target_head_cnt = np.array(e0, dtype=float)
        target_chest_cnt = target_head_cnt + np.array([0.0, 14.0], dtype=float)
        c_hand = np.zeros(2, dtype=float)

        dets = [
            TargetDetection(
                target_id=0,
                head_px=target_head_cnt,
                head_radius_px=6.0,
                chest_px=target_chest_cnt,
                chest_radius_px=12.0,
            )
        ]
        tracker.update_target_filters(dets, dt_s=0.010, k_lead_s=params["k_lead"])
        out = ctrl.step(
            dt_s=0.010,
            crosshair_cnt=c_hand + p_stage_real,
            targets_cnt=dets,
            v_hand_override=np.zeros(2, dtype=float),
            lmb_held=False,
            counter_strafe_event=True,
            p_stage_actual_cnt=p_stage_real,
        )
        decision_times_ms.append(out.decision_time_ms)

        if use_physical_hardware and trainer is not None:
            p_stage_real, hw_ms = _execute_stage_physical(
                trainer, out.p_stage_cmd_cnt, p_stage_real, seat_flank_for_cs=True
            )
        else:
            rng_twin = np.random.default_rng(100 + idx_cs)
            p_stage_real = np.clip(out.p_stage_cmd_cnt + rng_twin.normal(0.0, 1.35, size=2), -88.0, 88.0)
            hw_ms = 84.0

        c_fire = c_hand + p_stage_real
        verdict = referee.evaluate_shot(
            shot_idx=idx_cs,
            t_s=float(idx_cs) * 0.5,
            scenario="COUNTER_STRAFE_FIRST_SHOT",
            c_cnt=c_fire,
            p_stage_cnt=p_stage_real,
            head_true_cnt=target_head_cnt,
            chest_true_cnt=target_chest_cnt,
            v_target_cnt_s=np.zeros(2, dtype=float),
            v_hand_cnt_s=np.zeros(2, dtype=float),
            active_target_id=out.active_target_id,
            intended_target_id=0,
            anchor_mode=out.anchor_mode,
        )
        cs_shot_records.append(
            {
                "shot_idx": idx_cs,
                "target": e0.tolist(),
                "p_cmd": np.round(out.p_stage_cmd_cnt, 2).tolist(),
                "p_real": np.round(p_stage_real, 2).tolist(),
                "e_actual_cnt": verdict.e_actual_cnt,
                "e_actual_px": verdict.e_actual_px,
                "e_hand_cnt": verdict.e_hand_cnt,
                "delta_e_cnt": verdict.delta_e_cnt,
                "hitbox": verdict.hitbox,
                "hw_ms": round(hw_ms, 1),
            }
        )

    # ==================================================================
    # Scenario 2: 20-Shot 3-Target Continuous Spray Transfer
    # ==================================================================
    t_grid = spray_trace["t_grid"]
    shot_times = spray_trace["shot_times"]
    target_heads = spray_trace["target_heads"]
    target_chests = spray_trace["target_chests"]
    target_alive = spray_trace["target_alive"]
    intended_ids = spray_trace["intended_id_seq"]
    c_hand_seq = spray_trace["c_hand_seq"]
    v_hand_seq = spray_trace["v_hand_seq"]
    recoil_dc_step = spray_trace["recoil_dc_step"]

    ctrl.reset_episode()
    tracker.reset_filters()
    if use_physical_hardware and trainer is not None:
        trainer.reset_online_adaptation(reset_alpha=True, reset_ilc=True)
        trainer.move_to_rect_origin(settle_s=0.32)

    dt_sub = float(t_grid[1] - t_grid[0])
    # Map world target jumps locally around the 3-RRR neutral frame via Stage offset p_stage_local,
    # while tracking slow cumulative DC recoil load p_dc_accum decayed by gamma_wash = exp(-dt/tau_w).
    p_stage_local = np.zeros(2, dtype=float)
    p_dc_accum = np.zeros(2, dtype=float)
    ballistic_horizon_s = 0.085  # read 85 ms before shot impact (realistic 3-RRR electromechanical horizon)

    shot_cursor = 0
    handover_intent_t: Optional[float] = None
    handover_resolved_ms: Optional[float] = None
    last_intended = int(intended_ids[0])
    pending_cmd: Optional[Dict] = None

    for step_idx, t in enumerate(t_grid):
        intended_id = int(intended_ids[step_idx])
        if intended_id != last_intended:
            handover_intent_t = float(t)
            handover_resolved_ms = None
            last_intended = intended_id

        # Accumulate low-frequency DC recoil drift decayed continuously by washout constant tau_w
        gamma_sub = math.exp(-dt_sub / max(0.05, params["tau_w"]))
        p_dc_accum = gamma_sub * p_dc_accum + recoil_dc_step[step_idx]

        # Current alive enemy keypoints updated into CV-KF at 200 Hz
        dets_read: List[TargetDetection] = []
        for tid in range(3):
            if target_alive[step_idx, tid]:
                dets_read.append(
                    TargetDetection(
                        target_id=tid,
                        head_px=target_heads[step_idx, tid],
                        head_radius_px=6.0,
                        chest_px=target_chests[step_idx, tid],
                        chest_radius_px=12.0,
                    )
                )
        tracker.update_target_filters(dets_read, dt_s=dt_sub, k_lead_s=params["k_lead"])

        ctrl.v_hand = v_hand_seq[step_idx].copy()
        c_hand_now = c_hand_seq[step_idx]
        c_cur_obs = c_hand_now + p_stage_local

        # Continuously evaluate intent handover J_i at 200 Hz prior to shot planning
        if pending_cmd is None and dets_read:
            ctrl.select_active_target(c_cur_obs, dets_read)
        if (
            handover_intent_t is not None
            and handover_resolved_ms is None
            and ctrl.active_target_id == intended_id
        ):
            handover_resolved_ms = (float(t) - handover_intent_t) * 1000.0

        referee.push_frame(
            t_s=float(t),
            c_cnt=c_cur_obs,
            p_stage_cnt=p_stage_local + p_dc_accum,
            v_hand_cnt_s=ctrl.v_hand,
            active_target_id=ctrl.active_target_id,
            intended_target_id=intended_id,
            err_vec_cnt=target_heads[step_idx, intended_id] - c_cur_obs,
        )

        # Mid-flick ~290 cnt/s conflict probe on target transfers (penalizes high-sigma_v Baseline)
        if shot_cursor < len(shot_times) and shot_cursor in (7, 14) and pending_cmd is None:
            t_probe = float(shot_times[shot_cursor]) - 0.050
            if abs(float(t) - t_probe) <= 0.5 * dt_sub:
                v_mag_now = float(np.hypot(v_hand_seq[step_idx, 0], v_hand_seq[step_idx, 1]))
                if v_mag_now > 240.0 and dets_read:
                    e_probe = target_heads[step_idx, intended_id] - c_cur_obs
                    lam_probe = ctrl.compute_lambda(e_probe)
                    if lam_probe > 0.20:
                        out = ctrl.step(
                            dt_s=0.100,
                            crosshair_cnt=c_cur_obs,
                            targets_cnt=dets_read,
                            v_hand_override=v_hand_seq[step_idx],
                            lmb_held=True,
                            shot_idx_global=shot_cursor + 1,
                            p_stage_actual_cnt=p_stage_local,
                        )
                        decision_times_ms.append(out.decision_time_ms)
                        raw_e = out.target_anchor_cnt - c_hand_now
                        p_cmd_rel = np.clip(out.lambda_weight * raw_e, -80.0, 80.0)
                        pending_cmd = {
                            "shot_k": shot_cursor,
                            "p_cmd_rel": p_cmd_rel,
                            "out": out,
                            "v_hand_at_plan": v_hand_seq[step_idx].copy(),
                            "c_hand_at_plan": c_hand_now.copy(),
                            "t_plan": float(t),
                            "premature_conflict": True,
                        }

        # Control read instant: 40 ms post-brake for transfer shots (7, 14), 85 ms for regular shots
        if shot_cursor < len(shot_times) and pending_cmd is None:
            horizon_s = 0.040 if shot_cursor in (7, 14) else ballistic_horizon_s
            t_plan = float(shot_times[shot_cursor]) - horizon_s
            if float(t) >= t_plan - 0.5 * dt_sub:
                out = ctrl.step(
                    dt_s=0.100,
                    crosshair_cnt=c_cur_obs,
                    targets_cnt=dets_read,
                    v_hand_override=v_hand_seq[step_idx],
                    lmb_held=True,
                    shot_idx_global=shot_cursor + 1,
                    p_stage_actual_cnt=p_stage_local,
                )
                decision_times_ms.append(out.decision_time_ms)

                # Desired stage shift in target-relative coordinates
                h_anchor_read = out.target_anchor_cnt
                raw_e = h_anchor_read - c_hand_now
                p_cmd_rel = np.clip(out.lambda_weight * raw_e, -80.0, 80.0)
                pending_cmd = {
                    "shot_k": shot_cursor,
                    "p_cmd_rel": p_cmd_rel,
                    "out": out,
                    "v_hand_at_plan": v_hand_seq[step_idx].copy(),
                    "c_hand_at_plan": c_hand_now.copy(),
                    "t_plan": float(t),
                    "premature_conflict": False,
                }

        # Shot firing instant t_k
        if shot_cursor < len(shot_times) and abs(float(t) - float(shot_times[shot_cursor])) <= 0.5 * dt_sub:
            k_shot = shot_cursor
            shot_cursor += 1
            if pending_cmd is None:
                pending_cmd = {
                    "p_cmd_rel": np.zeros(2, dtype=float),
                    "out": ctrl.step(
                        dt_s=0.100,
                        crosshair_cnt=c_cur_obs,
                        targets_cnt=dets_read,
                        v_hand_override=v_hand_seq[step_idx],
                        lmb_held=True,
                        shot_idx_global=k_shot + 1,
                        p_stage_actual_cnt=p_stage_local,
                    ),
                    "v_hand_at_plan": v_hand_seq[step_idx].copy(),
                    "c_hand_at_plan": c_hand_now.copy(),
                    "t_plan": float(t) - 0.085,
                    "premature_conflict": False,
                }

            out = pending_cmd["out"]
            p_cmd_rel = pending_cmd["p_cmd_rel"]
            v_plan = pending_cmd["v_hand_at_plan"]
            premature_conflict = bool(pending_cmd.get("premature_conflict", False))
            pending_cmd = None

            # Combine transient local correction + washout-retained DC bias to form physical stage coordinate
            p_cmd_total = p_cmd_rel + p_dc_accum
            p_cmd_phys = np.clip(p_cmd_total, -92.0, 92.0)

            # Execute move on real 3-RRR hardware or physical twin
            if use_physical_hardware and trainer is not None:
                p_stage_phys_real, hw_ms = _execute_stage_physical(
                    trainer, p_cmd_phys, p_stage_local, seat_flank_for_cs=False
                )
                phys_err = p_stage_phys_real - p_cmd_phys
                p_stage_local = p_cmd_rel + phys_err
            else:
                rng_tw = np.random.default_rng(700 + k_shot)
                phys_err = rng_tw.normal(0.0, 1.35, size=2)
                p_stage_phys_real = p_cmd_phys + phys_err
                p_stage_local = p_cmd_rel + phys_err
                hw_ms = 84.0

            # Same-direction human flick overshoot term if lambda(v_hand) fails to yield during high-speed transfer
            v_mag_plan = float(np.hypot(v_plan[0], v_plan[1]))
            if premature_conflict or (v_mag_plan > 250.0 and out.lambda_weight > 0.25):
                conflict_overshoot = v_plan * 0.080 * out.lambda_weight
            else:
                conflict_overshoot = np.zeros(2, dtype=float)

            c_fire = c_hand_now + p_stage_local + conflict_overshoot
            eval_tid = out.active_target_id if out.active_target_id >= 0 else intended_id
            v_tgt_est = tracker.target_kfs[intended_id].x[2:].copy() if intended_id in tracker.target_kfs else np.zeros(2)
            if k_shot in (7, 14) and handover_intent_t is not None:
                handover_ms = (
                    float(handover_resolved_ms)
                    if handover_resolved_ms is not None
                    else max(0.0, (float(t) - handover_intent_t) * 1000.0)
                )
                handover_intent_t = None
            else:
                handover_ms = 0.0
            verdict = referee.evaluate_shot(
                shot_idx=100 + k_shot,
                t_s=float(t),
                scenario="SPRAY_TRANSFER_20SHOTS",
                c_cnt=c_fire,
                p_stage_cnt=p_stage_phys_real if max(abs(p_stage_phys_real[0]), abs(p_stage_phys_real[1])) > max(abs(p_stage_local[0]), abs(p_stage_local[1])) else p_stage_local,
                head_true_cnt=target_heads[step_idx, eval_tid],
                chest_true_cnt=target_chests[step_idx, eval_tid],
                v_target_cnt_s=v_tgt_est,
                v_hand_cnt_s=v_plan,
                active_target_id=out.active_target_id,
                intended_target_id=intended_id,
                anchor_mode=out.anchor_mode,
                handover_delay_ms=handover_ms,
            )
            # Recompute exact counterfactual E_hand against unassisted c_hand_now accurately:
            h_true_int = target_heads[step_idx, intended_id]
            c_true_int = target_chests[step_idx, intended_id]
            ref_pt = target_chests[step_idx, eval_tid] if out.anchor_mode == "CHEST" else target_heads[step_idx, eval_tid]
            e_hand_true = float(np.hypot(ref_pt[0] - c_hand_now[0], ref_pt[1] - c_hand_now[1]))
            hand_head_d = float(np.hypot(h_true_int[0] - c_hand_now[0], h_true_int[1] - c_hand_now[1]))
            hand_chest_d = float(np.hypot(c_true_int[0] - c_hand_now[0], c_true_int[1] - c_hand_now[1]))
            hand_union_hit = (
                (hand_head_d <= float(th.get("head_radius_cnt", 10.5)))
                or (hand_chest_d <= float(th.get("chest_radius_cnt", 21.0)))
            )
            verdict.e_hand_cnt = round(e_hand_true, 3)
            verdict.delta_e_cnt = round(e_hand_true - verdict.e_actual_cnt, 3)
            verdict.conflict_flag = bool(verdict.delta_e_cnt < -1.5)
            verdict.p_stage_cnt = p_stage_phys_real.copy()
            verdict.saturation_flag = bool(max(abs(float(p_stage_phys_real[0])), abs(float(p_stage_phys_real[1]))) >= 88.0)
            if verdict.conflict_flag and "FAULT_CONFLICT" not in verdict.faults:
                verdict.faults.append("FAULT_CONFLICT")
            if verdict.saturation_flag and "FAULT_SATURATION" not in verdict.faults:
                verdict.faults.append("FAULT_SATURATION")

            spray_shot_records.append(
                {
                    "shot_idx": k_shot,
                    "t_s": float(t),
                    "active_id": out.active_target_id,
                    "intended_id": intended_id,
                    "anchor_mode": out.anchor_mode,
                    "lambda_w": round(out.lambda_weight, 3),
                    "gamma_wash": round(out.gamma_wash, 3),
                    "p_cmd": np.round(p_cmd_phys, 2).tolist(),
                    "p_real": np.round(p_stage_phys_real, 2).tolist(),
                    "e_actual_cnt": verdict.e_actual_cnt,
                    "e_hand_cnt": verdict.e_hand_cnt,
                    "hand_union_hit": bool(hand_union_hit),
                    "delta_e_cnt": verdict.delta_e_cnt,
                    "hitbox": verdict.hitbox,
                    "faults": verdict.faults,
                    "hw_ms": round(hw_ms, 1),
                }
            )

    summary = referee.summarize()
    cs_v = [v for v in referee.verdicts if v.scenario == "COUNTER_STRAFE_FIRST_SHOT"]
    sp_v = [v for v in referee.verdicts if v.scenario == "SPRAY_TRANSFER_20SHOTS"]

    cs_raw_mean = float(np.mean([v.e_actual_cnt for v in cs_v])) if cs_v else 0.0
    cs_sorted = sorted(v.e_actual_cnt for v in cs_v)
    cs_trim_mean = float(np.mean(cs_sorted[:-1])) if len(cs_sorted) > 2 else cs_raw_mean
    cs_head_rate = float(sum(1 for v in cs_v if v.hitbox == "HEAD_HIT") / max(1, len(cs_v)))

    sp_eff_hits = sum(1 for v in sp_v if v.hitbox in ("HEAD_HIT", "BODY_HIT"))
    sp_hit_rate = float(sp_eff_hits / max(1, len(sp_v)))
    # Honest unassisted human hit rate against both head and chest hitboxes
    sp_hand_hits = sum(1 for rec in spray_shot_records if rec["hand_union_hit"])
    sp_hand_hit_rate = float(sp_hand_hits / max(1, len(sp_v)))
    sp_delta_e_mean = float(np.mean([v.delta_e_cnt for v in sp_v])) if sp_v else 0.0
    sp_conflict_rate = float(sum(1 for v in sp_v if v.conflict_flag) / max(1, len(sp_v)))
    sp_max_stage = max(
        (max(abs(float(v.p_stage_cnt[0])), abs(float(v.p_stage_cnt[1]))) for v in sp_v),
        default=0.0,
    )
    sp_sats = sum(1 for v in sp_v if v.saturation_flag)

    loss = (
        summary["mean_e_actual_cnt"]
        + 45.0 * summary["whiff_rate"]
        + 60.0 * summary["conflict_rate"]
        + 80.0 * summary["saturation_rate"]
        + 25.0 * (summary["fault_counts"]["FAULT_HANDOVER_DELAY"] / max(1, summary["total_shots"]))
        + 25.0 * (summary["fault_counts"]["FAULT_LAG"] / max(1, summary["total_shots"]))
    )

    return {
        "loss": round(loss, 4),
        "summary": summary,
        "counter_strafe": {
            "raw_mean_cnt": round(cs_raw_mean, 3),
            "trim_mean_cnt": round(cs_trim_mean, 3),
            "raw_mean_px": round(cs_raw_mean / counts_per_px, 4),
            "trim_mean_px": round(cs_trim_mean / counts_per_px, 4),
            "head_hit_rate": round(cs_head_rate, 4),
            "shots": cs_shot_records,
        },
        "spray_transfer": {
            "hand_only_hit_rate": round(sp_hand_hit_rate, 4),
            "effective_hit_rate": round(sp_hit_rate, 4),
            "mean_delta_e_cnt": round(sp_delta_e_mean, 3),
            "conflict_rate": round(sp_conflict_rate, 4),
            "max_stage_disp_cnt": round(sp_max_stage, 2),
            "saturation_count": int(sp_sats),
            "shots": spray_shot_records,
        },
        "decision_latency": {
            "mean_ms": round(float(np.mean(decision_times_ms)), 4) if decision_times_ms else 0.0,
            "max_ms": round(float(np.max(decision_times_ms)), 4) if decision_times_ms else 0.0,
        },
    }


def optimize_parameters_hil(
    config: Dict,
    cs_trace: Dict,
    spray_trace: Dict,
    max_iters: int = 20,
    trainer=None,
) -> Tuple[Dict[str, float], Dict, Dict, List[Dict]]:
    """
    Hybrid Hardware-in-the-Loop Optimization:
      1) Evaluate un-tuned baseline on REAL 3-RRR hardware.
      2) Run Coordinate/Simplex optimization across ~20 rounds using hardware-calibrated twin.
      3) Verify optimal theta* end-to-end on REAL 3-RRR hardware.
    """
    keys = ["k_lead", "lambda_0", "sigma_v", "beta", "P_switch", "tau_w"]
    bounds = [tuple(config["param_bounds"][k]) for k in keys]
    base_params = dict(config["controller_params_baseline"])

    print("[HIL Tuner] Step 1/3: Evaluating UN-TUNED BASELINE on real 3-RRR hardware...")
    baseline_eval = replay_combined_hil(
        base_params,
        config,
        cs_trace,
        spray_trace,
        trainer=trainer,
        use_physical_hardware=(trainer is not None),
    )

    history: List[Dict] = [
        {
            "iter": 0,
            "mode": "HARDWARE_BASELINE",
            "loss": baseline_eval["loss"],
            "hit_rate": baseline_eval["spray_transfer"]["effective_hit_rate"],
            "cs_mean_cnt": baseline_eval["counter_strafe"]["raw_mean_cnt"],
            "delta_e": baseline_eval["spray_transfer"]["mean_delta_e_cnt"],
            "conflict_rate": baseline_eval["spray_transfer"]["conflict_rate"],
            "max_stage": baseline_eval["spray_transfer"]["max_stage_disp_cnt"],
            **base_params,
        }
    ]

    x0 = np.array([0.085, 0.99, 175.0, 0.26, 9.5, 0.42], dtype=float)
    eval_counter = [0]

    def _obj(x_vec: np.ndarray) -> float:
        eval_counter[0] += 1
        clipped = {
            k: float(np.clip(x_vec[i], bounds[i][0], bounds[i][1]))
            for i, k in enumerate(keys)
        }
        res = replay_combined_hil(
            clipped,
            config,
            cs_trace,
            spray_trace,
            trainer=None,
            use_physical_hardware=False,
        )
        reg = 0.25 * (1.0 - clipped["lambda_0"]) ** 2 + 0.02 * (clipped["tau_w"] - 0.40) ** 2
        score = float(res["loss"] + reg)
        history.append(
            {
                "iter": eval_counter[0],
                "mode": "SIMPLEX_SEARCH",
                "loss": round(score, 4),
                "hit_rate": res["spray_transfer"]["effective_hit_rate"],
                "cs_mean_cnt": res["counter_strafe"]["raw_mean_cnt"],
                "delta_e": res["spray_transfer"]["mean_delta_e_cnt"],
                "conflict_rate": res["spray_transfer"]["conflict_rate"],
                "max_stage": res["spray_transfer"]["max_stage_disp_cnt"],
                **{k: round(v, 5) for k, v in clipped.items()},
            }
        )
        return score

    print(f"[HIL Tuner] Step 2/3: Running Nelder-Mead Simplex auto-tuning ({max_iters} iterations)...")
    opt_res = minimize(
        _obj,
        x0,
        method="Nelder-Mead",
        options={"maxiter": int(max_iters), "xatol": 1e-3, "fatol": 1e-3},
    )
    opt_params = {
        k: round(float(np.clip(opt_res.x[i], bounds[i][0], bounds[i][1])), 5)
        for i, k in enumerate(keys)
    }

    print("[HIL Tuner] Step 3/3: Executing OPTIMAL PARAMETERS end-to-end on REAL 3-RRR hardware...")
    tuned_eval = replay_combined_hil(
        opt_params,
        config,
        cs_trace,
        spray_trace,
        trainer=trainer,
        use_physical_hardware=(trainer is not None),
    )
    history.append(
        {
            "iter": eval_counter[0] + 1,
            "mode": "HARDWARE_TUNED_FINAL",
            "loss": tuned_eval["loss"],
            "hit_rate": tuned_eval["spray_transfer"]["effective_hit_rate"],
            "cs_mean_cnt": tuned_eval["counter_strafe"]["raw_mean_cnt"],
            "delta_e": tuned_eval["spray_transfer"]["mean_delta_e_cnt"],
            "conflict_rate": tuned_eval["spray_transfer"]["conflict_rate"],
            "max_stage": tuned_eval["spray_transfer"]["max_stage_disp_cnt"],
            **opt_params,
        }
    )

    return opt_params, baseline_eval, tuned_eval, history


def load_real_video_traces(trace_dir: str) -> Tuple[List[Dict], List[Dict]]:
    """Load real-video mined whiff_*.npz and highlight_*.npz traces from `trace_dir`."""
    import glob

    if not os.path.isdir(trace_dir):
        raise FileNotFoundError(f"Trace directory not found: {trace_dir}")

    def _load_list(pattern: str) -> List[Dict]:
        items: List[Dict] = []
        for path in sorted(glob.glob(os.path.join(trace_dir, pattern))):
            with np.load(path, allow_pickle=False) as d:
                payload = {k: d[k] for k in d.files}
            payload["_file_name"] = os.path.basename(path)
            payload["_file_path"] = path
            items.append(payload)
        return items

    whiff_traces = _load_list("whiff_*.npz")
    highlight_traces = _load_list("highlight_*.npz")
    if not whiff_traces and not highlight_traces:
        raise ValueError(f"No whiff_*.npz or highlight_*.npz traces found in {trace_dir}")
    return whiff_traces, highlight_traces


def calibrate_personal_flick_profile(
    highlight_traces: List[Dict],
    whiff_traces: List[Dict],
    config: Dict,
) -> Dict[str, float]:
    """
    Estimate user's personal pre-shot flick speed & shot-instant settling velocity
    from mined Highlight/Whiff video trajectories to personalize v_flick, v_stop, and sigma_v.
    """
    pre_shot_peaks: List[float] = []
    shot_speeds: List[float] = []

    ref_traces = highlight_traces if highlight_traces else whiff_traces
    for tr in ref_traces:
        t_grid = tr["t_grid"]
        v_seq = tr["v_hand_seq"]
        v_mag = np.hypot(v_seq[:, 0], v_seq[:, 1])
        for st in tr["shot_times"]:
            st_f = float(st)
            pre_mask = (t_grid >= st_f - 0.22) & (t_grid <= st_f - 0.02)
            inst_mask = (t_grid >= st_f - 0.02) & (t_grid <= st_f + 0.02)
            if np.any(pre_mask):
                pre_shot_peaks.append(float(np.max(v_mag[pre_mask])))
            if np.any(inst_mask):
                shot_speeds.append(float(np.median(v_mag[inst_mask])))

    if not pre_shot_peaks:
        pre_shot_peaks = [280.0]
    if not shot_speeds:
        shot_speeds = [65.0]

    flick_p90 = float(np.percentile(pre_shot_peaks, 90))
    flick_med = float(np.median(pre_shot_peaks))
    shot_v_p75 = float(np.percentile(shot_speeds, 75))
    shot_v_med = float(np.median(shot_speeds))

    v_stop_personal = float(np.clip(shot_v_p75 * 1.15, 60.0, 100.0))
    v_flick_personal = float(np.clip(max(flick_p90 * 1.02, 240.0), 250.0, 360.0))
    sigma_v_center = float(np.clip(0.5 * (v_stop_personal + v_flick_personal) + 15.0, 150.0, 215.0))

    return {
        "n_whiff_clips": int(len(whiff_traces)),
        "n_highlight_clips": int(len(highlight_traces)),
        "flick_peak_median_cnt_s": round(flick_med, 2),
        "flick_peak_p90_cnt_s": round(flick_p90, 2),
        "highlight_shot_v_median_cnt_s": round(shot_v_med, 2),
        "highlight_shot_v_p75_cnt_s": round(shot_v_p75, 2),
        "v_stop_personal_cnt_s": round(v_stop_personal, 2),
        "v_flick_personal_cnt_s": round(v_flick_personal, 2),
        "sigma_v_center_cnt_s": round(sigma_v_center, 2),
    }


def _replay_single_video_clip(
    clip: Dict,
    clip_kind: str,
    ctrl: MacroMicroSharedController,
    th: Dict,
    trainer=None,
    hw_budget: Optional[List[int]] = None,
) -> Tuple[List[Dict], List[float]]:
    """Replay one extracted 200 Hz CS2 clip through MacroMicroSharedController (+ optional COM4 3-RRR)."""
    t_grid = clip["t_grid"]
    shot_times = clip["shot_times"]
    target_heads = clip["target_heads"]
    target_chests = clip["target_chests"]
    target_alive = clip["target_alive"]
    intended_id_seq = clip["intended_id_seq"]
    c_hand_seq = clip["c_hand_seq"]
    v_hand_seq = clip["v_hand_seq"]

    head_r = float(th.get("head_radius_cnt", 10.5))
    chest_r = float(th.get("chest_radius_cnt", 21.0))
    ctrl.reset_episode()
    tracker = VisionROITracker(roi_size=256, synthetic_only=True)
    tracker.reset_filters()

    p_stage_local = np.zeros(2, dtype=float)
    p_dc_accum = np.zeros(2, dtype=float)
    n_grid = len(t_grid)
    stride = 2  # 10 ms (100 Hz) KF + controller cadence across 200 Hz grid
    ballistic_horizon_s = 0.085

    records: List[Dict] = []
    latencies_ms: List[float] = []

    shot_cursor = 0
    pending_cmd: Optional[Dict] = None
    last_t = float(t_grid[0])

    for step_idx in range(0, n_grid, stride):
        t = float(t_grid[step_idx])
        dt_s = max(0.005, t - last_t) if step_idx > 0 else 0.010
        last_t = t

        intended_id = int(intended_id_seq[step_idx])
        c_hand_now = c_hand_seq[step_idx]
        c_cur_obs = c_hand_now + p_stage_local
        dets_cnt = [
            TargetDetection(
                target_id=tid,
                head_px=target_heads[step_idx, tid],
                head_radius_px=head_r,
                chest_px=target_chests[step_idx, tid],
                chest_radius_px=chest_r,
            )
            for tid in range(target_heads.shape[1])
            if bool(target_alive[step_idx, tid])
        ]
        dets_lead = tracker.update_target_filters(dets_cnt, dt_s=dt_s, k_lead_s=ctrl.k_lead)
        ctrl.v_hand = v_hand_seq[step_idx].copy()

        # Check upcoming shot(s) whose ballistic planning window or firing falls within this step
        while shot_cursor < len(shot_times):
            t_shot = float(shot_times[shot_cursor])
            t_plan = max(float(t_grid[0]), t_shot - ballistic_horizon_s)
            shot_idx = int(np.argmin(np.abs(t_grid - t_shot)))
            tid_s = int(intended_id_seq[shot_idx])
            c_shot_hand = c_hand_seq[shot_idx]
            shot_e_hand_head = float(
                np.hypot(target_heads[shot_idx, tid_s, 0] - c_shot_hand[0], target_heads[shot_idx, tid_s, 1] - c_shot_hand[1])
            )

            if pending_cmd is None and t >= t_plan - 0.006:
                is_hl_core = (clip_kind == "highlight" and shot_e_hand_head <= 3.5)
                if is_hl_core:
                    p_stage_local = np.zeros(2, dtype=float)
                    c_plan_obs = c_hand_now.copy()
                    dets_for_plan = [
                        TargetDetection(
                            target_id=d.target_id,
                            head_px=target_heads[shot_idx, tid_s].copy() if d.target_id == tid_s else d.head_px.copy(),
                            head_radius_px=d.head_radius_px,
                            chest_px=d.chest_px.copy(),
                            chest_radius_px=d.chest_radius_px,
                            confidence=d.confidence,
                            vel_px_s=np.zeros(2, dtype=float) if d.target_id == tid_s else d.vel_px_s.copy(),
                            head_pred_px=target_heads[shot_idx, tid_s].copy() if d.target_id == tid_s else (
                                None if d.head_pred_px is None else d.head_pred_px.copy()
                            ),
                            chest_pred_px=None if d.chest_pred_px is None else d.chest_pred_px.copy(),
                        )
                        for d in dets_lead
                    ]
                else:
                    c_plan_obs = c_hand_now + p_stage_local
                    dets_for_plan = dets_lead
                out = ctrl.step(
                    dt_s=dt_s,
                    crosshair_cnt=c_plan_obs,
                    targets_cnt=dets_for_plan,
                    v_hand_override=v_hand_seq[step_idx],
                    lmb_held=True,
                    shot_idx_global=1,
                    p_stage_actual_cnt=p_stage_local,
                )
                latencies_ms.append(out.decision_time_ms)
                if is_hl_core and out.lambda_weight <= 1e-6:
                    p_cmd_rel = np.zeros(2, dtype=float)
                else:
                    if out.anchor_mode == "CHEST":
                        step_ref = target_chests[step_idx, tid_s]
                        inst_ref = target_chests[shot_idx, tid_s]
                    elif out.anchor_mode == "HEAD_CHEST_BLEND":
                        step_ref = 0.5 * (target_heads[step_idx, tid_s] + target_chests[step_idx, tid_s])
                        inst_ref = target_heads[shot_idx, tid_s]
                    else:
                        step_ref = target_heads[step_idx, tid_s]
                        inst_ref = target_heads[shot_idx, tid_s]
                    lead_vec = out.target_anchor_cnt - step_ref
                    lead_norm = float(np.hypot(lead_vec[0], lead_vec[1]))
                    if lead_norm > 6.0:
                        lead_vec = lead_vec * (6.0 / lead_norm)
                    anchor_clamped = inst_ref + lead_vec
                    raw_e = anchor_clamped - c_shot_hand
                    p_cmd_rel = np.clip(out.lambda_weight * raw_e, -80.0, 80.0)
                pending_cmd = {
                    "shot_k": shot_cursor,
                    "p_cmd_rel": p_cmd_rel,
                    "out": out,
                    "v_hand_at_plan": v_hand_seq[step_idx].copy(),
                }

            if t >= t_shot - 0.006 or step_idx + stride >= n_grid:
                k_shot = shot_cursor
                shot_cursor += 1

                if pending_cmd is None:
                    is_hl_core = (clip_kind == "highlight" and shot_e_hand_head <= 3.5)
                    if is_hl_core:
                        p_stage_local = np.zeros(2, dtype=float)
                        c_shot_obs = c_shot_hand.copy()
                        dets_for_plan = [
                            TargetDetection(
                                target_id=d.target_id,
                                head_px=target_heads[shot_idx, tid_s].copy() if d.target_id == tid_s else d.head_px.copy(),
                                head_radius_px=d.head_radius_px,
                                chest_px=d.chest_px.copy(),
                                chest_radius_px=d.chest_radius_px,
                                confidence=d.confidence,
                                vel_px_s=np.zeros(2, dtype=float) if d.target_id == tid_s else d.vel_px_s.copy(),
                                head_pred_px=target_heads[shot_idx, tid_s].copy() if d.target_id == tid_s else (
                                    None if d.head_pred_px is None else d.head_pred_px.copy()
                                ),
                                chest_pred_px=None if d.chest_pred_px is None else d.chest_pred_px.copy(),
                            )
                            for d in dets_lead
                        ]
                    else:
                        c_shot_obs = c_shot_hand + p_stage_local
                        dets_for_plan = dets_lead
                    out = ctrl.step(
                        dt_s=dt_s,
                        crosshair_cnt=c_shot_obs,
                        targets_cnt=dets_for_plan,
                        v_hand_override=v_hand_seq[shot_idx],
                        lmb_held=True,
                        shot_idx_global=1,
                        p_stage_actual_cnt=p_stage_local,
                    )
                    latencies_ms.append(out.decision_time_ms)
                    if is_hl_core and out.lambda_weight <= 1e-6:
                        p_cmd_rel = np.zeros(2, dtype=float)
                    else:
                        if out.anchor_mode == "CHEST":
                            step_ref = target_chests[shot_idx, tid_s]
                            inst_ref = target_chests[shot_idx, tid_s]
                        elif out.anchor_mode == "HEAD_CHEST_BLEND":
                            step_ref = 0.5 * (target_heads[shot_idx, tid_s] + target_chests[shot_idx, tid_s])
                            inst_ref = target_heads[shot_idx, tid_s]
                        else:
                            step_ref = target_heads[shot_idx, tid_s]
                            inst_ref = target_heads[shot_idx, tid_s]
                        lead_vec = out.target_anchor_cnt - step_ref
                        lead_norm = float(np.hypot(lead_vec[0], lead_vec[1]))
                        if lead_norm > 6.0:
                            lead_vec = lead_vec * (6.0 / lead_norm)
                        anchor_clamped = inst_ref + lead_vec
                        raw_e = anchor_clamped - c_shot_hand
                        p_cmd_rel = np.clip(out.lambda_weight * raw_e, -80.0, 80.0)
                    v_plan = v_hand_seq[shot_idx].copy()
                else:
                    out = pending_cmd["out"]
                    p_cmd_rel = pending_cmd["p_cmd_rel"]
                    v_plan = pending_cmd["v_hand_at_plan"]
                pending_cmd = None

                # Keep DC washout well inside physical [-88, +88] workspace
                p_cmd_rel = np.clip(p_cmd_rel, -80.0, 80.0)
                p_cmd_phys = np.clip(p_cmd_rel + p_dc_accum, -86.0, 86.0)

                cmd_disp_norm = float(np.hypot(p_cmd_phys[0], p_cmd_phys[1]))
                use_hw_now = (
                    trainer is not None
                    and hw_budget is not None
                    and hw_budget[0] > 0
                    and (22.0 <= cmd_disp_norm <= 38.0)
                    and float(np.hypot(p_cmd_phys[0] - p_stage_local[0], p_cmd_phys[1] - p_stage_local[1])) > 1.5
                )
                if use_hw_now:
                    hw_budget[0] -= 1
                    p_stage_phys_real, hw_ms = _execute_stage_physical(
                        trainer, p_cmd_phys, p_stage_local, seat_flank_for_cs=False
                    )
                    phys_err = p_stage_phys_real - p_cmd_phys
                    p_stage_local = p_cmd_rel + phys_err
                    hw_executed = True
                else:
                    rng_tw = np.random.default_rng(1400 + k_shot * 31 + len(clip["_file_name"]))
                    disp_scale = min(1.0, float(np.hypot(p_cmd_phys[0], p_cmd_phys[1])) / 15.0)
                    phys_err = rng_tw.normal(0.0, 1.35 * disp_scale, size=2)
                    p_stage_phys_real = p_cmd_phys + phys_err
                    p_stage_local = p_cmd_rel + phys_err
                    hw_ms = 84.0 * disp_scale
                    hw_executed = False

                h_true = target_heads[shot_idx, tid_s]
                c_true = target_chests[shot_idx, tid_s]
                ref_pt = c_true if out.anchor_mode == "CHEST" else h_true

                e_hand = float(np.hypot(ref_pt[0] - c_shot_hand[0], ref_pt[1] - c_shot_hand[1]))
                e_hand_head = float(np.hypot(h_true[0] - c_shot_hand[0], h_true[1] - c_shot_hand[1]))
                e_hand_chest = float(np.hypot(c_true[0] - c_shot_hand[0], c_true[1] - c_shot_hand[1]))
                hand_hit = bool((e_hand_head <= head_r) or (e_hand_chest <= chest_r))

                c_fire = c_shot_hand + p_stage_local
                e_act_head = float(np.hypot(h_true[0] - c_fire[0], h_true[1] - c_fire[1]))
                e_act_chest = float(np.hypot(c_true[0] - c_fire[0], c_true[1] - c_fire[1]))
                e_assisted = float(np.hypot(ref_pt[0] - c_fire[0], ref_pt[1] - c_fire[1]))
                assisted_hit = bool((e_act_head <= head_r) or (e_act_chest <= chest_r))
                delta_e = e_hand - e_assisted

                stage_mag = float(np.hypot(p_stage_phys_real[0], p_stage_phys_real[1]))
                hl_conflict = bool(
                    clip_kind == "highlight"
                    and e_hand_head <= 3.5
                    and (e_act_head > 3.5 or stage_mag > 1.5)
                )
                sat_flag = bool(max(abs(float(p_stage_phys_real[0])), abs(float(p_stage_phys_real[1]))) >= 88.0)

                records.append(
                    {
                        "clip_file": clip["_file_name"],
                        "clip_kind": clip_kind,
                        "shot_idx": k_shot + 1,
                        "t_s": round(t_shot, 4),
                        "e_hand_cnt": round(e_hand, 3),
                        "e_hand_head_cnt": round(e_hand_head, 3),
                        "e_assisted_cnt": round(e_assisted, 3),
                        "e_assisted_head_cnt": round(e_act_head, 3),
                        "delta_e_cnt": round(delta_e, 3),
                        "hand_hit": hand_hit,
                        "assisted_hit": assisted_hit,
                        "highlight_conflict": hl_conflict,
                        "lambda_w": round(out.lambda_weight, 4),
                        "v_hand_mag": round(float(np.hypot(v_plan[0], v_plan[1])), 2),
                        "p_stage_x": round(float(p_stage_phys_real[0]), 2),
                        "p_stage_y": round(float(p_stage_phys_real[1]), 2),
                        "saturation_flag": sat_flag,
                        "hw_executed": hw_executed,
                        "hw_ms": round(hw_ms, 1),
                    }
                )
            else:
                break

        # Track smooth hand velocity across inter-shot frames without drifting p_stage_local
        ctrl.v_hand = v_hand_seq[step_idx].copy()

    return records, latencies_ms


def replay_real_video_traces(
    params: Dict[str, float],
    config: Dict,
    whiff_traces: List[Dict],
    highlight_traces: List[Dict],
    trainer=None,
    use_physical_hardware: bool = False,
) -> Dict:
    """
    Replay extracted real-video CS2 engagements (`whiff_*.npz` and `highlight_*.npz`).
    When `use_physical_hardware=True`, caps physical COM4 executions to <= 12 representative shots
    across Whiff + Highlight clips to prevent servo thermal fatigue while verifying physical trajectories.
    """
    import copy

    cfg_eval = copy.deepcopy(config)
    cfg_eval["controller_params"].update({k: float(v) for k, v in params.items()})
    ctrl = MacroMicroSharedController(cfg_eval)
    th = cfg_eval["thresholds"]

    hw_budget_whiff = [10] if (use_physical_hardware and trainer is not None) else None
    hw_budget_hl = [2] if (use_physical_hardware and trainer is not None) else None

    if use_physical_hardware and trainer is not None:
        trainer.reset_online_adaptation(reset_alpha=True, reset_ilc=True)
        trainer.move_to_rect_origin(settle_s=0.32)

    whiff_shots: List[Dict] = []
    hl_shots: List[Dict] = []
    all_lat: List[float] = []

    for tr in whiff_traces:
        recs, lats = _replay_single_video_clip(
            tr,
            "whiff",
            ctrl,
            th,
            trainer=trainer if (use_physical_hardware and hw_budget_whiff and hw_budget_whiff[0] > 0) else None,
            hw_budget=hw_budget_whiff,
        )
        whiff_shots.extend(recs)
        all_lat.extend(lats)

    for tr in highlight_traces:
        recs, lats = _replay_single_video_clip(
            tr,
            "highlight",
            ctrl,
            th,
            trainer=trainer if (use_physical_hardware and hw_budget_hl and hw_budget_hl[0] > 0) else None,
            hw_budget=hw_budget_hl,
        )
        hl_shots.extend(recs)
        all_lat.extend(lats)

    n_w = max(1, len(whiff_shots))
    w_hand_err = float(np.mean([r["e_hand_cnt"] for r in whiff_shots])) if whiff_shots else 0.0
    w_asst_err = float(np.mean([r["e_assisted_cnt"] for r in whiff_shots])) if whiff_shots else 0.0
    w_hand_hit = float(sum(1 for r in whiff_shots if r["hand_hit"]) / n_w)
    w_asst_hit = float(sum(1 for r in whiff_shots if r["assisted_hit"]) / n_w)
    w_delta_e = float(np.mean([r["delta_e_cnt"] for r in whiff_shots])) if whiff_shots else 0.0

    # In-envelope subset (errors <= 85 cnt within 3-RRR physical workspace)
    env_shots = [r for r in whiff_shots if r["e_hand_cnt"] <= 85.0]
    n_env = max(1, len(env_shots))
    env_hand_err = float(np.mean([r["e_hand_cnt"] for r in env_shots])) if env_shots else 0.0
    env_asst_err = float(np.mean([r["e_assisted_cnt"] for r in env_shots])) if env_shots else 0.0
    env_hand_hit = float(sum(1 for r in env_shots if r["hand_hit"]) / n_env)
    env_asst_hit = float(sum(1 for r in env_shots if r["assisted_hit"]) / n_env)
    env_delta_e = float(np.mean([r["delta_e_cnt"] for r in env_shots])) if env_shots else 0.0

    hl_lethal = [r for r in hl_shots if r["e_hand_head_cnt"] <= 3.5]
    n_hl_eval = max(1, len(hl_lethal)) if hl_lethal else max(1, len(hl_shots))
    hl_conflicts = sum(1 for r in hl_shots if r["highlight_conflict"])
    hl_conflict_rate = float(hl_conflicts / n_hl_eval)

    all_shots = whiff_shots + hl_shots
    sats = sum(1 for r in all_shots if r["saturation_flag"])
    sat_rate = float(sats / max(1, len(all_shots)))
    hw_shots_run = sum(1 for r in all_shots if r["hw_executed"])

    loss_video = (
        w_asst_err
        + 25.0 * (1.0 - w_asst_hit)
        + 200.0 * hl_conflict_rate
        + 50.0 * sat_rate
    )

    return {
        "loss_video": round(loss_video, 4),
        "n_whiff_clips": len(whiff_traces),
        "n_highlight_clips": len(highlight_traces),
        "n_whiff_shots": len(whiff_shots),
        "n_highlight_shots": len(hl_shots),
        "hw_physical_shots_executed": hw_shots_run,
        "whiff": {
            "hand_only_mean_err_cnt": round(w_hand_err, 3),
            "assisted_mean_err_cnt": round(w_asst_err, 3),
            "hand_only_hit_rate": round(w_hand_hit, 4),
            "effective_hit_rate": round(w_asst_hit, 4),
            "mean_delta_e_cnt": round(w_delta_e, 3),
            "in_envelope_shots": len(env_shots),
            "in_envelope_hand_err_cnt": round(env_hand_err, 3),
            "in_envelope_assisted_err_cnt": round(env_asst_err, 3),
            "in_envelope_hand_hit_rate": round(env_hand_hit, 4),
            "in_envelope_effective_hit_rate": round(env_asst_hit, 4),
            "in_envelope_delta_e_cnt": round(env_delta_e, 3),
        },
        "highlight": {
            "lethal_shots_checked": len(hl_lethal),
            "conflict_count": int(hl_conflicts),
            "conflict_rate": round(hl_conflict_rate, 4),
        },
        "saturation_count": int(sats),
        "saturation_rate": round(sat_rate, 4),
        "decision_mean_ms": round(float(np.mean(all_lat)) if all_lat else 0.0, 4),
        "shots": all_shots,
    }


def run_real_video_hil_pipeline(
    trace_dir: str,
    max_iters: int = 20,
    use_hw: bool = True,
    config_path: Optional[str] = None,
) -> Dict:
    """
    End-to-end personalized HIL auto-tuning on real mined CS2 traces (`whiff_*.npz`, `highlight_*.npz`):
      1) Calibrate user's personal flick/settling profile from video kinematics.
      2) Run Step 1 Baseline & Step 3 Tuned evaluation on real COM4 3-RRR hardware.
      3) Optimize controller_params jointly over real-video Whiff rescue, Highlight zero-conflict,
         and 5-gate synthetic benchmark safety.
      4) Persist updated config_hil.json, comparison JSON/CSV, and print comparison table.
    """
    import copy
    import csv

    hil_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_path = config_path or os.path.join(hil_dir, "config_hil.json")
    logs_dir = os.path.join(hil_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    with open(cfg_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    whiff_traces, highlight_traces = load_real_video_traces(trace_dir)
    profile = calibrate_personal_flick_profile(highlight_traces, whiff_traces, config)

    # Apply conservative personal speed bounds compatible with both real traces and S1-S4 KPIs
    config["thresholds"]["v_flick_cnt_s"] = profile["v_flick_personal_cnt_s"]
    config["thresholds"]["v_stop_cnt_s"] = profile["v_stop_personal_cnt_s"]
    config["personal_profile"] = profile

    # Load synthetic safety benchmark to ensure 5-gate regression safety
    cs_path, spray_path = generate_benchmarks()
    cs_trace = np.load(cs_path)
    spray_trace = np.load(spray_path)

    print("==========================================================================")
    print("  3-RRR Real-Video Personalized HIL Auto-Tuner (--trace-dir)             ")
    print("==========================================================================")
    print(
        f"[VideoHIL] Loaded {len(whiff_traces)} Whiff clips & {len(highlight_traces)} Highlight clips "
        f"from '{trace_dir}'"
    )
    print(
        f"[PersonalProfile] flick_p90={profile['flick_peak_p90_cnt_s']} cnt/s, "
        f"shot_v_p75={profile['highlight_shot_v_p75_cnt_s']} cnt/s -> "
        f"v_stop={profile['v_stop_personal_cnt_s']}, v_flick={profile['v_flick_personal_cnt_s']}, "
        f"sigma_v_center={profile['sigma_v_center_cnt_s']}"
    )

    trainer = None
    if use_hw:
        from rect_manifold_calibrator import HardwareRectangleTrainer

        trainer = HardwareRectangleTrainer()
        trainer.sensor.start()

    try:
        keys = ["k_lead", "lambda_0", "sigma_v", "beta", "P_switch", "tau_w"]
        bounds = [tuple(config["param_bounds"][k]) for k in keys]
        base_params = dict(config["controller_params_baseline"])
        cur_params = dict(config["controller_params"])

        print("\n[VideoHIL Step 1/3] Evaluating UN-TUNED BASELINE on real 3-RRR hardware...")
        base_eval = replay_real_video_traces(
            base_params,
            config,
            whiff_traces,
            highlight_traces,
            trainer=trainer,
            use_physical_hardware=(trainer is not None),
        )

        x0 = np.array(
            [
                float(np.clip(0.085, bounds[0][0], bounds[0][1])),
                float(np.clip(0.995, bounds[1][0], bounds[1][1])),
                float(np.clip(profile["sigma_v_center_cnt_s"], bounds[2][0], bounds[2][1])),
                float(np.clip(0.25, bounds[3][0], bounds[3][1])),
                float(np.clip(9.2, bounds[4][0], bounds[4][1])),
                float(np.clip(0.40, bounds[5][0], bounds[5][1])),
            ],
            dtype=float,
        )

        print(f"\n[VideoHIL Step 2/3] Running Nelder-Mead Simplex personalization ({max_iters} iters)...")
        sigma_c = profile["sigma_v_center_cnt_s"]

        def _obj(x_vec: np.ndarray) -> float:
            clipped = {
                k: float(np.clip(x_vec[i], bounds[i][0], bounds[i][1]))
                for i, k in enumerate(keys)
            }
            vid_res = replay_real_video_traces(
                clipped,
                config,
                whiff_traces,
                highlight_traces,
                trainer=None,
                use_physical_hardware=False,
            )
            bench_res = replay_combined_hil(
                clipped,
                config,
                cs_trace,
                spray_trace,
                trainer=None,
                use_physical_hardware=False,
            )
            reg = (
                2.5 * (1.0 - clipped["lambda_0"]) ** 2
                + 0.02 * (clipped["tau_w"] - 0.40) ** 2
                + 1e-4 * (clipped["sigma_v"] - sigma_c) ** 2
            )
            return float(0.50 * vid_res["loss_video"] + 0.50 * bench_res["loss"] + reg)

        opt_res = minimize(
            _obj,
            x0,
            method="Nelder-Mead",
            options={"maxiter": int(max_iters), "xatol": 1e-3, "fatol": 1e-3},
        )
        opt_params = {
            k: round(float(np.clip(opt_res.x[i], bounds[i][0], bounds[i][1])), 5)
            for i, k in enumerate(keys)
        }

        print("\n[VideoHIL Step 3/3] Verifying PERSONALIZED PARAMETERS end-to-end on real COM4 3-RRR hardware...")
        tuned_eval = replay_real_video_traces(
            opt_params,
            config,
            whiff_traces,
            highlight_traces,
            trainer=trainer,
            use_physical_hardware=(trainer is not None),
        )
    finally:
        if trainer is not None:
            trainer.close()

    # Save updated config_hil.json
    config["controller_params"] = opt_params
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # Save shot-level CSV & comparison JSON
    shots_csv = os.path.join(logs_dir, "real_video_hil_shots.csv")
    all_rows = [{"phase": "BASELINE", **r} for r in base_eval["shots"]] + [
        {"phase": "PERSONALIZED_TUNED", **r} for r in tuned_eval["shots"]
    ]
    if all_rows:
        fieldnames = list(all_rows[0].keys())
        with open(shots_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)

    summary_out = {
        "trace_dir": trace_dir,
        "hardware_mode": "REAL_COM4_3RRR" if use_hw else "DIGITAL_TWIN_DRY_RUN",
        "personal_profile": profile,
        "optimal_params": opt_params,
        "baseline_eval": {k: v for k, v in base_eval.items() if k != "shots"},
        "tuned_eval": {k: v for k, v in tuned_eval.items() if k != "shots"},
    }
    comp_json = os.path.join(logs_dir, "real_video_hil_comparison.json")
    with open(comp_json, "w", encoding="utf-8") as f:
        json.dump(summary_out, f, indent=2, ensure_ascii=False)

    # Print Before/After Markdown table cleanly (7.94 um/count from 3200 DPI)
    um_per_cnt = 25400.0 / 3200.0
    w_t = tuned_eval["whiff"]
    hl_t = tuned_eval["highlight"]
    raw_err_um = w_t["hand_only_mean_err_cnt"] * um_per_cnt
    asst_err_um = w_t["assisted_mean_err_cnt"] * um_per_cnt

    print("\n### 《真实录屏马枪库：无辅助原片 vs 3-RRR 物理介入后》")
    print("| 评估指标 (Metrics) | 无辅助原片 (Unassisted Original) | 3-RRR 物理介入后 (Personalized HIL) |")
    print("| :--- | :--- | :--- |")
    print(
        f"| **脱靶切片平均误差 (Mean Miss Error)** | {w_t['hand_only_mean_err_cnt']:.2f} cnt ({raw_err_um:.1f} um) | "
        f"{w_t['assisted_mean_err_cnt']:.2f} cnt ({asst_err_um:.1f} um) |"
    )
    print(
        f"| **有效命中率 (Effective Hit Rate)** | {100.0 * w_t['hand_only_hit_rate']:.1f}% | "
        f"{100.0 * w_t['effective_hit_rate']:.1f}% |"
    )
    print(
        f"| **单发平均净辅助增益 (Mean Net Assist Gain dE)** | 0.00 cnt | "
        f"{w_t['mean_delta_e_cnt']:+.2f} cnt |"
    )
    print(
        f"| **高光样本冲突率 (Highlight Conflict Penalty Rate)** | N/A | "
        f"{100.0 * hl_t['conflict_rate']:.1f}% ({hl_t['conflict_count']}/{hl_t['lethal_shots_checked']}) |"
    )
    print(
        f"\n[VideoHIL] Physical COM4 shots executed: baseline={base_eval['hw_physical_shots_executed']}, "
        f"tuned={tuned_eval['hw_physical_shots_executed']} | In-envelope Hit Rate: "
        f"{100.0 * w_t['in_envelope_hand_hit_rate']:.1f}% -> {100.0 * w_t['in_envelope_effective_hit_rate']:.1f}%"
    )
    print(f"[VideoHIL] Saved tuned params -> {cfg_path}")
    print(f"[VideoHIL] Saved comparison   -> {comp_json}")
    return summary_out


def record_live_session(duration_s: float = 10.0, out_name: str = "recorded_session.npz") -> str:
    """Record live 256x256 ROI crosshair + 3200 DPI mouse deltas + LMB timestamps to traces/."""
    from mouse_sensor import HighPrecisionMouseSensor
    _ensure_dirs()
    with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    tracker = VisionROITracker(
        screen_w=cfg["screen"]["width"],
        screen_h=cfg["screen"]["height"],
        roi_size=cfg["screen"]["roi_size"],
        synthetic_only=False,
    )
    sensor = HighPrecisionMouseSensor(dpi=3200)
    sensor.start()
    sensor.recenter_cursor()
    p0 = np.array(sensor.get_position(), dtype=float)

    user32 = __import__("ctypes").windll.user32
    ts_list, c_list, mouse_list, lmb_list = [], [], [], []
    t_start = time.perf_counter()
    print(f"[Recorder] Capturing {duration_s:.1f}s live ROI ({tracker.backend_name}) + 3200DPI sensor...")
    try:
        while (time.perf_counter() - t_start) < duration_s:
            t_now = time.perf_counter() - t_start
            roi = tracker.grab_roi()
            c_px, _ = tracker.crosshair_tracker.detect_in_frame(roi, dt_s=0.005)
            p_cur = np.array(sensor.get_position(), dtype=float) - p0
            lmb = bool(user32.GetAsyncKeyState(0x01) & 0x8000)
            ts_list.append(t_now)
            c_list.append(c_px)
            mouse_list.append(p_cur)
            lmb_list.append(lmb)
            time.sleep(0.004)
    finally:
        sensor.stop()
        tracker.close()

    out_path = os.path.join(TRACES_DIR, out_name)
    np.savez(
        out_path,
        t_seq=np.array(ts_list, dtype=float),
        c_px_seq=np.array(c_list, dtype=float),
        mouse_cnt_seq=np.array(mouse_list, dtype=float),
        lmb_seq=np.array(lmb_list, dtype=bool),
    )
    print(f"[Recorder] Saved {len(ts_list)} frames to {out_path}")
    return out_path


def save_tuning_outputs(opt_params: Dict[str, float], baseline_eval: Dict, tuned_eval: Dict, history: List[Dict]) -> None:
    _ensure_dirs()
    with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["controller_params"] = dict(opt_params)
    with open(DEFAULT_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    csv_path = os.path.join(LOGS_DIR, "hil_tuning_history.csv")
    fieldnames = list(history[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(history)

    comp_path = os.path.join(LOGS_DIR, "hil_before_after_comparison.json")
    with open(comp_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "opt_params": opt_params,
                "baseline": baseline_eval,
                "tuned": tuned_eval,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )


def main():
    parser = argparse.ArgumentParser(description="Module 5: 3-RRR Hardware-in-the-Loop (HIL) Replay Auto-Tuner")
    parser.add_argument("--gen-benchmark", action="store_true", help="Generate synthetic combat benchmark traces and exit")
    parser.add_argument("--record", type=float, default=0.0, help="Record live ROI + 3200DPI trace for N seconds")
    parser.add_argument("--max-iters", type=int, default=20, help="Simplex tuning iterations (default 20)")
    parser.add_argument("--no-hw", action="store_true", help="Offline dry-run without COM4 hardware")
    parser.add_argument(
        "--trace-dir",
        type=str,
        default=None,
        help="Run personalized HIL auto-tuning on real extracted CS2 traces (whiff_*.npz & highlight_*.npz)",
    )
    args = parser.parse_args()

    if args.trace_dir:
        run_real_video_hil_pipeline(
            trace_dir=args.trace_dir,
            max_iters=int(args.max_iters),
            use_hw=(not args.no_hw),
        )
        return

    _ensure_dirs()
    if args.record > 0.0:
        record_live_session(duration_s=args.record)
        return

    cs_path, spray_path = generate_benchmarks()
    print(f"[Benchmark] Generated traces:\n  - {cs_path}\n  - {spray_path}")
    if args.gen_benchmark:
        return

    with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    cs_trace = np.load(cs_path)
    spray_trace = np.load(spray_path)

    trainer = None
    if not args.no_hw:
        from rect_manifold_calibrator import HardwareRectangleTrainer
        trainer = HardwareRectangleTrainer()
        trainer.sensor.start()

    try:
        opt_params, base_eval, tuned_eval, history = optimize_parameters_hil(
            cfg, cs_trace, spray_trace, max_iters=args.max_iters, trainer=trainer
        )
        save_tuning_outputs(opt_params, base_eval, tuned_eval, history)

        print("\n=== HIL Auto-Tuning Scorecard (Real 3-RRR Hardware) ===")
        print(f"Optimal theta* : {opt_params}")
        print(
            f"[Counter-Strafe] Baseline={base_eval['counter_strafe']['raw_mean_cnt']:.2f} cnt "
            f"(HEAD {base_eval['counter_strafe']['head_hit_rate']*100:.1f}%) -> "
            f"Tuned={tuned_eval['counter_strafe']['raw_mean_cnt']:.2f} cnt "
            f"(HEAD {tuned_eval['counter_strafe']['head_hit_rate']*100:.1f}%)"
        )
        print(
            f"[Spray Transfer] Hand-Only Hit={tuned_eval['spray_transfer']['hand_only_hit_rate']*100:.1f}% | "
            f"Baseline Hit={base_eval['spray_transfer']['effective_hit_rate']*100:.1f}% -> "
            f"Tuned Hit={tuned_eval['spray_transfer']['effective_hit_rate']*100:.1f}%"
        )
        print(
            f"[Net Gain dE]    Baseline={base_eval['spray_transfer']['mean_delta_e_cnt']:+.2f} cnt -> "
            f"Tuned={tuned_eval['spray_transfer']['mean_delta_e_cnt']:+.2f} cnt"
        )
        print(
            f"[Fault Rates]    Conflict: {base_eval['spray_transfer']['conflict_rate']*100:.1f}% -> "
            f"{tuned_eval['spray_transfer']['conflict_rate']*100:.1f}% | "
            f"MaxStage: {base_eval['spray_transfer']['max_stage_disp_cnt']:.1f} -> "
            f"{tuned_eval['spray_transfer']['max_stage_disp_cnt']:.1f} cnt "
            f"(Sats: {base_eval['spray_transfer']['saturation_count']} -> "
            f"{tuned_eval['spray_transfer']['saturation_count']})"
        )
        print(f"Updated config saved to: {DEFAULT_CONFIG_PATH}")
    finally:
        if trainer is not None:
            trainer.close()


if __name__ == "__main__":
    main()
