"""In-Game Loaded Closed-Loop Auto-Trainer (F5 Stage 1 & Stage 2 + F8 YOLO Flow).

Stage 1 (`calibrate_yolo_head_jacobian`):
    Uses live YOLO `head` bounding-box detections in the 256x256 ROI while driving
    the 3-RRR stage by (+/- step_cnt) under real palm load (with atomic B_CMD boost)
    to measure exact screen-to-stage Jacobian `J_screen_to_stage` (`YOLO_HEAD_FLOW`).

Stage 2 (`run_in_game_damping_autotune`):
    Runs up to `n_trials` (default 16) automated offset-and-pullback trials against a
    stationary Aim Botz head target. Measures screen overshoot (px), sign flips,
    settling error (px), and settling time (ms), tuning
    `[kp_small_head, kp_large_error, in_box_deadband_ratio, cooldown_ms, boost_scale]`
    until 5 consecutive zero-overshoot (< 2.0 px) trials converge.

Can also be exercised offline with real visual latency + palm load + backlash
simulation via:
    python -m shared_control_hil.in_game_auto_trainer --sim-test
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, asdict
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


HIL_DIR = Path(__file__).resolve().parent
REPO_ROOT = HIL_DIR.parent
DEFAULT_CONFIG_PATH = HIL_DIR / "config_hil.json"
DEFAULT_REPORT_PATH = HIL_DIR / "logs" / "in_game_autotune_report.json"

SAFE_MIN_US = 1150
SAFE_MAX_US = 1850
MAX_X_CNT = 240.0
MAX_Y_CNT = 300.0


def _clamp_pulse(u: float) -> float:
    return round(float(max(SAFE_MIN_US, min(SAFE_MAX_US, float(u)))), 1)


def _extract_head_obs(
    det: Any,
    roi_center_px: Tuple[float, float],
    c_px: Optional[np.ndarray] = None,
) -> Optional[Dict[str, Any]]:
    """Normalize a TargetDetection object or dict into head center/size/error."""
    if det is None:
        return None

    ref_px = (
        np.asarray(c_px, dtype=np.float64)
        if c_px is not None
        else np.array([float(roi_center_px[0]), float(roi_center_px[1])], dtype=np.float64)
    )

    if isinstance(det, dict):
        bbox = det.get("head_bbox_px")
        head_px = det.get("head_px")
        raw_err = det.get("error_px")
        conf = float(det.get("confidence", det.get("conf", 1.0)))
        tid = int(det.get("target_id", det.get("track_id", 1)))
        head_r = float(det.get("head_radius_px", 5.0))
    else:
        bbox = getattr(det, "head_bbox_px", None)
        head_px = getattr(det, "head_px", None)
        raw_err = getattr(det, "error_px", None)
        conf = float(getattr(det, "confidence", getattr(det, "conf", 1.0)))
        tid = int(getattr(det, "target_id", getattr(det, "track_id", 1)))
        head_r = float(getattr(det, "head_radius_px", 5.0))

    if bbox is not None and len(bbox) == 4:
        x1, y1, x2, y2 = [float(v) for v in bbox]
        w = max(2.0, abs(x2 - x1))
        h = max(2.0, abs(y2 - y1))
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        err = np.array([cx - ref_px[0], cy - ref_px[1]], dtype=np.float64)
        return {
            "center_px": np.array([cx, cy], dtype=np.float64),
            "error_px": err,
            "size_px": np.array([w, h], dtype=np.float64),
            "conf": conf,
            "track_id": tid,
            "is_head_box": True,
        }

    if head_px is not None and len(head_px) >= 2:
        cx, cy = float(head_px[0]), float(head_px[1])
        err = np.array([cx - ref_px[0], cy - ref_px[1]], dtype=np.float64)
        sz = max(4.0, 2.0 * head_r)
        return {
            "center_px": np.array([cx, cy], dtype=np.float64),
            "error_px": err,
            "size_px": np.array([sz, sz], dtype=np.float64),
            "conf": conf,
            "track_id": tid,
            "is_head_box": False,
        }

    if raw_err is not None and len(raw_err) >= 2:
        ex, ey = float(raw_err[0]), float(raw_err[1])
        return {
            "center_px": np.array([ref_px[0] + ex, ref_px[1] + ey], dtype=np.float64),
            "error_px": np.array([ex, ey], dtype=np.float64),
            "size_px": np.array([10.0, 10.0], dtype=np.float64),
            "conf": conf,
            "track_id": tid,
            "is_head_box": False,
        }
    return None


@dataclass
class AutoTuneTrialMetrics:
    trial_idx: int
    offset_cnt: List[float]
    initial_err_px: float
    final_err_px: float
    overshoot_px: float
    sign_flips: int
    settling_ms: float
    steps_executed: int
    converged: bool
    params_snapshot: Dict[str, float]


class InGameAutoTrainer:
    """Loaded-hand closed-loop YOLO Jacobian & damping auto-tuner."""

    def __init__(
        self,
        config_path: Path = DEFAULT_CONFIG_PATH,
        report_path: Path = DEFAULT_REPORT_PATH,
        calibrator: Any = None,
        vision: Any = None,
        trainer: Any = None,
        dispatch_fn: Optional[Callable[..., bool]] = None,
        recenter_fn: Optional[Callable[[float], None]] = None,
        detect_fn: Optional[Callable[[np.ndarray], Tuple[np.ndarray, List[Any]]]] = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        clock_fn: Callable[[], float] = time.perf_counter,
        roi_size_px: Tuple[float, float] = (256.0, 256.0),
    ):
        self.config_path = Path(config_path)
        self.report_path = Path(report_path)
        self.calibrator = calibrator
        self.vision = vision
        self.trainer = trainer
        self._dispatch_fn = dispatch_fn
        self._recenter_fn = recenter_fn
        self._detect_fn = detect_fn
        self._sleep = sleep_fn
        self._clock = clock_fn
        self.roi_center_px = (0.5 * float(roi_size_px[0]), 0.5 * float(roi_size_px[1]))
        self._last_u_kin: Optional[np.ndarray] = None
        self._locked_play_g = np.zeros(2, dtype=np.float64)

    # ------------------------------------------------------------------
    # Config I/O
    # ------------------------------------------------------------------
    def load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            return {}
        return json.loads(self.config_path.read_text(encoding="utf-8"))

    def get_live_control_params(self) -> Dict[str, float]:
        cfg = self.load_config()
        lc = cfg.get("live_control", {})
        return {
            "kp_small_head": float(lc.get("kp_small_head", 0.42)),
            "kp_large_error": float(lc.get("kp_large_error", 0.68)),
            "small_error_px": float(lc.get("small_error_px", 14.0)),
            "in_box_deadband_ratio": float(lc.get("in_box_deadband_ratio", 0.38)),
            "min_deadband_px": float(lc.get("min_deadband_px", 2.2)),
            "cooldown_ms": float(lc.get("cooldown_ms", 85.0)),
            "recoil_cooldown_ms": float(lc.get("recoil_cooldown_ms", 20.0)),
            "boost_scale": float(lc.get("boost_scale", 1.35)),
            "boost_dur_ms": float(lc.get("boost_dur_ms", 22.0)),
            "recenter_idle_ms": float(lc.get("recenter_idle_ms", 150.0)),
            "aim_washout_tau_s": float(lc.get("aim_washout_tau_s", 0.20)),
            "no_reversal_window_ms": float(lc.get("no_reversal_window_ms", 250.0)),
            "auto_recenter_threshold_cnt": float(lc.get("auto_recenter_threshold_cnt", 180.0)),
        }

    def save_live_control_params(self, params: Dict[str, float]) -> None:
        cfg = self.load_config()
        lc = dict(cfg.get("live_control", {}))
        for k, v in params.items():
            lc[k] = round(float(v), 5)
        cfg["live_control"] = lc
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    # ------------------------------------------------------------------
    # Vision & Hardware Helpers
    # ------------------------------------------------------------------
    def _sample_head_obs(self, n_frames: int = 3, require_head_box: bool = True) -> Optional[Dict[str, Any]]:
        """Grab `n_frames` from vision tracker and median-average valid head observations."""
        if self.vision is None:
            return None

        errs = []
        centers = []
        sizes = []
        head_flags = []
        confs = []
        tids = []

        for _ in range(max(1, int(n_frames))):
            frame = self.vision.grab_roi()
            if frame is None:
                continue
            c_px: Optional[np.ndarray] = None
            det = None
            if self._detect_fn is not None:
                c_px, dets = self._detect_fn(frame)
                if dets:
                    # Pick detection closest to c_px (or highest seen_frames)
                    best_d = min(
                        dets,
                        key=lambda d: float(
                            np.hypot(float(d.head_px[0]) - float(c_px[0]), float(d.head_px[1]) - float(c_px[1]))
                        ),
                    )
                    det = best_d
            else:
                out = self.vision.process_frame(frame)
                det = out[0] if isinstance(out, tuple) else out
            obs = _extract_head_obs(det, self.roi_center_px, c_px=c_px)
            if obs is None:
                continue
            if require_head_box and not obs["is_head_box"]:
                continue
            errs.append(obs["error_px"])
            centers.append(obs["center_px"])
            sizes.append(obs["size_px"])
            head_flags.append(obs["is_head_box"])
            confs.append(obs["conf"])
            tids.append(obs["track_id"])

        if not errs:
            return None

        return {
            "center_px": np.median(np.vstack(centers), axis=0),
            "error_px": np.median(np.vstack(errs), axis=0),
            "size_px": np.median(np.vstack(sizes), axis=0),
            "conf": float(np.median(confs)),
            "track_id": tids[-1],
            "is_head_box": any(head_flags),
            "samples": len(errs),
        }

    def _recenter(self, duration_s: float = 0.12) -> None:
        if self._recenter_fn is not None:
            self._recenter_fn(duration_s)
            return
        self._dispatch(np.zeros(2, dtype=np.float64), cooldown_s=duration_s, freeze_play_g=False, use_boost=False)

    def _dispatch(
        self,
        p_target_cnt: np.ndarray,
        cooldown_s: float = 0.085,
        freeze_play_g: bool = False,
        use_boost: bool = True,
        boost_scale: float = 1.35,
        boost_dur_ms: float = 22.0,
    ) -> bool:
        p_clamped = np.array(
            [
                float(np.clip(p_target_cnt[0], -MAX_X_CNT, MAX_X_CNT)),
                float(np.clip(p_target_cnt[1], -MAX_Y_CNT, MAX_Y_CNT)),
            ],
            dtype=np.float64,
        )

        if self._dispatch_fn is not None:
            return bool(
                self._dispatch_fn(
                    p_clamped,
                    cooldown_s=cooldown_s,
                    freeze_play_g=freeze_play_g,
                    use_boost=use_boost,
                    boost_scale=boost_scale,
                    boost_dur_ms=boost_dur_ms,
                )
            )

        if self.trainer is None:
            raise RuntimeError("InGameAutoTrainer requires either dispatch_fn or trainer.")

        manifold = self.trainer.manifold
        driver = self.trainer.driver
        x_cmd, y_cmd = float(p_clamped[0]), float(p_clamped[1])

        if freeze_play_g and hasattr(manifold, "evaluate_kinematic_pulse"):
            u_kin = np.asarray(manifold.evaluate_kinematic_pulse(x_cmd, y_cmd), dtype=np.float64)
            u_cmd = u_kin + self._locked_play_g
        else:
            u_cmd = np.asarray(manifold.evaluate_pulse(x_cmd, y_cmd, dir_hint=(0.0, 0.0)), dtype=np.float64)
            if hasattr(manifold, "evaluate_kinematic_pulse"):
                u_kin = np.asarray(manifold.evaluate_kinematic_pulse(x_cmd, y_cmd), dtype=np.float64)
                self._locked_play_g = u_cmd - u_kin
            else:
                u_kin = u_cmd

        ux_t = _clamp_pulse(u_cmd[0])
        uy_t = _clamp_pulse(u_cmd[1])

        if use_boost and hasattr(driver, "send_ballistic_pulse") and self._last_u_kin is not None:
            du = u_kin - self._last_u_kin
            du_b = np.clip(du * 0.40 * float(boost_scale), -26.0, 26.0)
            b1 = _clamp_pulse(u_cmd[0] + du_b[0])
            b2 = _clamp_pulse(u_cmd[1] + du_b[1])
            bst_us = int(round(max(10.0, min(35.0, float(boost_dur_ms))) * 1000.0))
            driver.send_ballistic_pulse(b1, b2, ux_t, uy_t, bst_us)
        else:
            driver.send_raw_pulses(ux_t, uy_t, round(3000.0 - uy_t, 1))

        self._last_u_kin = u_kin.copy()
        self._sleep(max(0.005, float(cooldown_s)))
        return True

    # ------------------------------------------------------------------
    # Stage 1: Loaded YOLO Head Jacobian Calibration (F8 primary / F5 Stage 1)
    # ------------------------------------------------------------------
    def calibrate_yolo_head_jacobian(
        self,
        step_cnt: float = 80.0,
        settle_s: float = 0.095,
        require_head_box: bool = True,
    ) -> Dict[str, Any]:
        """Measure loaded screen-to-stage Jacobian from live YOLO `head` detections."""
        if self.vision is None:
            return {"ok": False, "reason": "no_vision_tracker"}

        lc = self.get_live_control_params()
        boost_scale = lc["boost_scale"]
        boost_dur_ms = lc["boost_dur_ms"]

        self._recenter(0.12)
        self._sleep(max(0.06, settle_s))
        if hasattr(self.vision, "reset_filters"):
            self.vision.reset_filters()

        base = self._sample_head_obs(n_frames=3, require_head_box=require_head_box)
        if base is None:
            return {"ok": False, "reason": "no_yolo_head_in_roi"}

        def _measure_axis(axis_idx: int) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
            deltas = []
            for sgn in (+1.0, -1.0):
                p = np.zeros(2, dtype=np.float64)
                p[axis_idx] = sgn * float(step_cnt)
                self._dispatch(
                    p,
                    cooldown_s=settle_s,
                    freeze_play_g=False,
                    use_boost=True,
                    boost_scale=boost_scale,
                    boost_dur_ms=boost_dur_ms,
                )
                # Allow 80-95ms screen/capture latency to flush completely before sampling
                self._sleep(max(0.06, settle_s))
                obs = self._sample_head_obs(n_frames=3, require_head_box=require_head_box)
                self._recenter(0.11)
                self._sleep(max(0.09, settle_s))
                if obs is None:
                    return None, {"ok": False, "reason": f"lost_head_axis_{axis_idx}"}
                # Delta in error_px caused by +sgn * step_cnt stage move
                shift = (obs["error_px"] - base["error_px"]) * sgn
                deltas.append(shift / float(step_cnt))
            avg_col = 0.5 * (deltas[0] + deltas[1])
            return avg_col, {
                "pos_px_per_cnt": deltas[0].round(4).tolist(),
                "neg_px_per_cnt": deltas[1].round(4).tolist(),
            }

        col_x, dbg_x = _measure_axis(0)
        if col_x is None:
            self._recenter(0.12)
            return dbg_x

        col_y, dbg_y = _measure_axis(1)
        if col_y is None:
            self._recenter(0.12)
            return dbg_y

        # M maps dp_cnt -> d(error_px); to cancel e_px we want M @ dp_cnt = -e_px => J_s2st = -inv(M)
        M = np.column_stack([col_x, col_y])
        det_m = float(np.linalg.det(M))
        if abs(det_m) < 1e-4:
            return {
                "ok": False,
                "reason": f"singular_yolo_jacobian(det={det_m:.6f})",
                "M": M.tolist(),
            }

        cond_m = float(np.linalg.cond(M))
        if cond_m > 12.0:
            return {
                "ok": False,
                "reason": f"ill_conditioned_yolo_jacobian(cond={cond_m:.2f})",
                "M": M.tolist(),
            }

        J_screen_to_stage = -np.linalg.inv(M)
        # Estimate mounting angle from -M (dp > 0 moves error < 0 when aligned)
        A_pos = -M
        angle_rad = math.atan2(0.5 * (A_pos[1, 0] - A_pos[0, 1]), 0.5 * (A_pos[0, 0] + A_pos[1, 1]))
        angle_deg = math.degrees(angle_rad)

        if self.calibrator is not None:
            self.calibrator.J_screen_to_stage = J_screen_to_stage.copy()
            self.calibrator.visual_calib_source = "YOLO_HEAD_FLOW"
            self.calibrator.mounting_angle_deg = float(angle_deg)
            if hasattr(self.calibrator, "save_config"):
                self.calibrator.save_config()

        self._recenter(0.12)
        return {
            "ok": True,
            "source": "YOLO_HEAD_FLOW",
            "J_screen_to_stage": J_screen_to_stage.tolist(),
            "M_stage_to_err": M.tolist(),
            "det": det_m,
            "cond": cond_m,
            "mounting_angle_deg": float(angle_deg),
            "axis_x": dbg_x,
            "axis_y": dbg_y,
        }

    # ------------------------------------------------------------------
    # Stage 2: Zero-Overshoot Small-Head Damping Auto-Tuner (F5)
    # ------------------------------------------------------------------
    def _error_to_stage_delta(self, e_px: np.ndarray) -> np.ndarray:
        if self.calibrator is not None and hasattr(self.calibrator, "aim_error_to_stage_delta"):
            return np.asarray(self.calibrator.aim_error_to_stage_delta(float(e_px[0]), float(e_px[1])), dtype=np.float64)
        if self.calibrator is not None and hasattr(self.calibrator, "J_screen_to_stage"):
            return np.asarray(self.calibrator.J_screen_to_stage, dtype=np.float64) @ np.asarray(e_px, dtype=np.float64)
        return np.asarray(e_px, dtype=np.float64) * 1.1

    def run_in_game_damping_autotune(
        self,
        n_trials: int = 16,
        converge_streak: int = 5,
        err_tol_px: float = 2.0,
        offset_cnt: float = 42.0,
        max_steps_per_trial: int = 8,
        on_trial_done: Optional[Callable[[AutoTuneTrialMetrics], None]] = None,
    ) -> Dict[str, Any]:
        """Run offset-and-pullback trials against stationary Bot head & auto-tune damping."""
        if self.vision is None:
            return {"ok": False, "reason": "no_vision_tracker"}

        params = self.get_live_control_params()
        offsets = [
            np.array([+offset_cnt, 0.0]),
            np.array([-offset_cnt, 0.0]),
            np.array([0.0, +offset_cnt]),
            np.array([0.0, -offset_cnt]),
            np.array([+0.72 * offset_cnt, +0.72 * offset_cnt]),
            np.array([-0.72 * offset_cnt, -0.72 * offset_cnt]),
            np.array([+0.72 * offset_cnt, -0.72 * offset_cnt]),
            np.array([-0.72 * offset_cnt, +0.72 * offset_cnt]),
        ]

        trials: List[AutoTuneTrialMetrics] = []
        streak = 0
        t_all_start = self._clock()

        for trial_idx in range(1, int(n_trials) + 1):
            off = offsets[(trial_idx - 1) % len(offsets)]
            self._recenter(0.12)
            self._sleep(0.08)

            # Offset stage under loaded palm, wait past visual pipeline lag, read initial e0_px
            self._dispatch(
                off,
                cooldown_s=0.10,
                freeze_play_g=False,
                use_boost=True,
                boost_scale=params["boost_scale"],
                boost_dur_ms=params["boost_dur_ms"],
            )
            self._sleep(0.04)

            obs0 = self._sample_head_obs(n_frames=2, require_head_box=True)
            if obs0 is None:
                # Try once more with fallback detection if YOLO missed 1 frame
                obs0 = self._sample_head_obs(n_frames=2, require_head_box=False)
            if obs0 is None:
                self._recenter(0.12)
                return {
                    "ok": False,
                    "reason": f"lost_head_at_trial_{trial_idx}",
                    "trials": [asdict(t) for t in trials],
                    "params": params,
                }

            e0 = obs0["error_px"].copy()
            e0_norm = float(np.linalg.norm(e0))
            if e0_norm < 1e-3:
                e0_norm = 1e-3
            u0 = e0 / e0_norm

            # Lock initial polarity & play_g on first shot of this trial
            dp0_init = self._error_to_stage_delta(e0)
            s_init = np.sign(dp0_init)
            p_cmd = off.copy()

            t_trial_start = self._clock()
            e_hist: List[np.ndarray] = [e0]
            overshoot_px = 0.0
            sign_flips = 0
            prev_proj = float(np.dot(e0, u0))
            settling_ms = 0.0
            steps_exec = 0

            for step in range(max_steps_per_trial):
                obs = obs0 if step == 0 else self._sample_head_obs(n_frames=1, require_head_box=False)
                if obs is None:
                    break
                e_cur = obs["error_px"]
                e_hist.append(e_cur.copy())
                e_norm = float(np.linalg.norm(e_cur))

                proj = float(np.dot(e_cur, u0))
                if proj < 0.0:
                    overshoot_px = max(overshoot_px, -proj)
                    if proj < -0.75 and prev_proj > 0.75:
                        sign_flips += 1
                prev_proj = proj

                w_h, h_h = float(obs["size_px"][0]), float(obs["size_px"][1])
                r_lock = max(
                    params["min_deadband_px"],
                    params["in_box_deadband_ratio"] * min(w_h, h_h),
                )

                if e_norm <= r_lock or e_norm <= err_tol_px:
                    settling_ms = (self._clock() - t_trial_start) * 1000.0
                    break

                gain = (
                    params["kp_small_head"]
                    if e_norm <= params["small_error_px"]
                    else params["kp_large_error"]
                )
                dp_raw = self._error_to_stage_delta(e_cur) * gain

                # One-Way No-Reversal Clamp
                dp = dp_raw.copy()
                for ax in (0, 1):
                    if s_init[ax] != 0.0 and dp[ax] * s_init[ax] < 0.0:
                        dp[ax] = 0.0

                if float(np.linalg.norm(dp)) < 0.3:
                    settling_ms = (self._clock() - t_trial_start) * 1000.0
                    break

                p_cmd = p_cmd + dp
                cooldown_s = params["cooldown_ms"] / 1000.0
                # Freeze play_g after first entry or when inside small_error_px zone!
                freeze_pg = (e_norm <= params["small_error_px"]) or (step > 0)
                self._dispatch(
                    p_cmd,
                    cooldown_s=cooldown_s,
                    freeze_play_g=freeze_pg,
                    use_boost=True,
                    boost_scale=params["boost_scale"],
                    boost_dur_ms=params["boost_dur_ms"],
                )
                steps_exec += 1

            obs_final = self._sample_head_obs(n_frames=2, require_head_box=False)
            e_final = obs_final["error_px"] if obs_final is not None else e_hist[-1]
            final_err_px = float(np.linalg.norm(e_final))
            proj_final = float(np.dot(e_final, u0))
            if proj_final < 0.0:
                overshoot_px = max(overshoot_px, -proj_final)
            if settling_ms <= 0.0:
                settling_ms = (self._clock() - t_trial_start) * 1000.0

            trial_ok = (overshoot_px <= 1.2) and (sign_flips == 0) and (final_err_px <= err_tol_px)
            streak = (streak + 1) if trial_ok else 0

            tm = AutoTuneTrialMetrics(
                trial_idx=trial_idx,
                offset_cnt=[round(float(off[0]), 2), round(float(off[1]), 2)],
                initial_err_px=round(e0_norm, 3),
                final_err_px=round(final_err_px, 3),
                overshoot_px=round(overshoot_px, 3),
                sign_flips=int(sign_flips),
                settling_ms=round(settling_ms, 2),
                steps_executed=steps_exec,
                converged=trial_ok,
                params_snapshot={k: round(float(v), 4) for k, v in params.items()},
            )
            trials.append(tm)
            if on_trial_done is not None:
                on_trial_done(tm)

            # Adaptive damping/gain update rule
            if not trial_ok:
                if overshoot_px > 1.2 or sign_flips > 0:
                    params["kp_small_head"] = max(0.22, params["kp_small_head"] * 0.86)
                    params["kp_large_error"] = max(0.38, params["kp_large_error"] * 0.88)
                    params["in_box_deadband_ratio"] = min(0.50, params["in_box_deadband_ratio"] + 0.03)
                    params["cooldown_ms"] = min(145.0, params["cooldown_ms"] + 15.0)
                elif final_err_px > err_tol_px:
                    # Undershoot / friction stall under palm load
                    params["boost_scale"] = min(1.90, params["boost_scale"] + 0.12)
                    params["kp_small_head"] = min(0.75, params["kp_small_head"] * 1.10)

            if streak >= converge_streak:
                break

        self._recenter(0.12)
        self.save_live_control_params(params)

        converged_all = streak >= converge_streak
        report = {
            "ok": True,
            "converged": converged_all,
            "converge_streak": int(streak),
            "target_streak": int(converge_streak),
            "elapsed_s": round(self._clock() - t_all_start, 2),
            "final_params": params,
            "trials": [asdict(t) for t in trials],
        }
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        return report


# ----------------------------------------------------------------------
# Synthetic Closed-Loop Simulator (Visual Latency + Palm Load + Backlash)
# ----------------------------------------------------------------------
class _SimClock:
    def __init__(self) -> None:
        self.t = 0.0

    def now(self) -> float:
        return self.t

    def sleep(self, dt: float) -> None:
        self.t += max(0.0, float(dt))


class _SimVisionAndHardware:
    """Simulates 80ms screen latency, 300-500g palm friction attenuation, & backlash step."""

    def __init__(self, clock: _SimClock, angle_deg: float = 11.5, px_per_cnt: float = 0.72) -> None:
        self.clock = clock
        th = math.radians(angle_deg)
        R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]], dtype=np.float64)
        # Moving stage +dp moves target on screen in opposite direction
        self.M_true = -px_per_cnt * R
        self.roi_center = np.array([128.0, 128.0], dtype=np.float64)
        self.head_size = np.array([8.0, 8.0], dtype=np.float64)

        self.p_phys = np.zeros(2, dtype=np.float64)
        self.last_cmd = np.zeros(2, dtype=np.float64)
        self.last_dir = np.zeros(2, dtype=np.float64)
        self.backlash_state = np.zeros(2, dtype=np.float64)
        # Delay buffer of (timestamp, err_px) for 80 ms frame lag
        self.frame_pipe: deque = deque([(0.0, np.zeros(2, dtype=np.float64))], maxlen=64)

    def dispatch(
        self,
        p_target_cnt: np.ndarray,
        cooldown_s: float = 0.085,
        freeze_play_g: bool = False,
        use_boost: bool = True,
        boost_scale: float = 1.35,
        boost_dur_ms: float = 22.0,
    ) -> bool:
        dp_cmd = np.asarray(p_target_cnt, dtype=np.float64) - self.last_cmd
        # Palm load causes ~0.68x attenuation unless compensated by B_CMD boost
        eff_gain = min(1.06, 0.66 * (1.0 + 0.34 * (boost_scale - 1.0) if use_boost else 1.0))

        # Simulate pi_nbm_net step jump (~9 cnt equivalent) if play_g is NOT frozen on reversal
        jump = np.zeros(2, dtype=np.float64)
        if not freeze_play_g:
            for ax in (0, 1):
                sgn = float(np.sign(dp_cmd[ax]))
                if sgn != 0.0 and self.last_dir[ax] != 0.0 and sgn != self.last_dir[ax]:
                    jump[ax] = sgn * 8.5
                if sgn != 0.0:
                    self.last_dir[ax] = sgn

        self.p_phys += dp_cmd * eff_gain + jump
        self.last_cmd = np.asarray(p_target_cnt, dtype=np.float64).copy()

        # Servo settles in ~40ms, while cooldown_s controls dispatch interval
        dt = max(0.015, float(cooldown_s))
        settle_dt = min(dt, 0.040)
        self.clock.sleep(settle_dt)
        err_true = self.M_true @ self.p_phys
        self.frame_pipe.append((self.clock.now(), err_true))
        if dt > settle_dt:
            self.clock.sleep(dt - settle_dt)
        return True

    def recenter(self, duration_s: float = 0.12) -> None:
        self.p_phys = np.zeros(2, dtype=np.float64)
        self.last_cmd = np.zeros(2, dtype=np.float64)
        dt = max(0.02, float(duration_s))
        settle_dt = min(dt, 0.045)
        self.clock.sleep(settle_dt)
        self.frame_pipe.append((self.clock.now(), np.zeros(2, dtype=np.float64)))
        if dt > settle_dt:
            self.clock.sleep(dt - settle_dt)

    def grab_roi(self) -> np.ndarray:
        return np.zeros((256, 256, 3), dtype=np.uint8)

    def reset_filters(self) -> None:
        pass

    def process_frame(self, _frame: np.ndarray) -> Tuple[Dict[str, Any], None]:
        t_query = self.clock.now() - 0.080  # 80 ms camera + capture latency
        err_seen = self.frame_pipe[-1][1]
        for ts, val in self.frame_pipe:
            if ts <= t_query:
                err_seen = val
        # Small pixel quantization noise
        noise = np.array([0.05, -0.05], dtype=np.float64)
        e = err_seen + noise
        c = self.roi_center + e
        hw, hh = 0.5 * self.head_size[0], 0.5 * self.head_size[1]
        det = {
            "head_bbox_px": (c[0] - hw, c[1] - hh, c[0] + hw, c[1] + hh),
            "error_px": (float(e[0]), float(e[1])),
            "conf": 0.91,
            "track_id": 7,
        }
        return det, None


class _SimCalibrator:
    def __init__(self) -> None:
        # Deliberately wrong initial angle & scale
        self.J_screen_to_stage = np.array([[-1.0, 0.0], [0.0, -1.0]], dtype=np.float64)
        self.visual_calib_source = "PINHOLE_FALLBACK"
        self.mounting_angle_deg = 0.0

    def aim_error_to_stage_delta(self, ex: float, ey: float) -> np.ndarray:
        return self.J_screen_to_stage @ np.array([ex, ey], dtype=np.float64)

    def save_config(self) -> None:
        pass


def run_synthetic_simulation_test() -> int:
    """Prove Stage 1 YOLO Jacobian recovery + Stage 2 5-streak convergence from bad initial gains."""
    import tempfile

    clock = _SimClock()
    sim = _SimVisionAndHardware(clock, angle_deg=11.5, px_per_cnt=0.72)
    calib = _SimCalibrator()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        cfg_path = tmp / "sim_config.json"
        rep_path = tmp / "sim_report.json"
        # Intentionally aggressive, under-damped & under-boosted initial params
        cfg_path.write_text(
            json.dumps(
                {
                    "live_control": {
                        "kp_small_head": 0.92,
                        "kp_large_error": 1.15,
                        "small_error_px": 14.0,
                        "in_box_deadband_ratio": 0.16,
                        "min_deadband_px": 1.5,
                        "cooldown_ms": 50.0,
                        "boost_scale": 1.00,
                        "boost_dur_ms": 22.0,
                    }
                }
            ),
            encoding="utf-8",
        )

        trainer = InGameAutoTrainer(
            config_path=cfg_path,
            report_path=rep_path,
            calibrator=calib,
            vision=sim,
            dispatch_fn=sim.dispatch,
            recenter_fn=sim.recenter,
            sleep_fn=clock.sleep,
            clock_fn=clock.now,
        )

        print("=== [Sim Test] Stage 1: YOLO Head Flow Jacobian Recovery ===")
        res1 = trainer.calibrate_yolo_head_jacobian(step_cnt=70.0, settle_s=0.095)
        print(
            f"Stage 1 ok={res1['ok']} source={res1.get('source')} "
            f"angle={res1.get('mounting_angle_deg', 0.0):.2f} deg (true=11.50 deg) "
            f"cond={res1.get('cond', 0.0):.2f}"
        )
        if not res1.get("ok") or abs(res1["mounting_angle_deg"] - 11.5) > 1.5:
            print("[FAIL] Stage 1 Jacobian recovery accuracy check failed.")
            return 1

        print("\n=== [Sim Test] Stage 2: Loaded-Hand Zero-Overshoot Damping Auto-Tuner ===")
        res2 = trainer.run_in_game_damping_autotune(
            n_trials=16,
            converge_streak=5,
            err_tol_px=2.0,
            offset_cnt=42.0,
            on_trial_done=lambda tm: print(
                f"  [Trial {tm.trial_idx:02d}] init={tm.initial_err_px:5.2f}px -> "
                f"final={tm.final_err_px:4.2f}px | ovs={tm.overshoot_px:4.2f}px | "
                f"flips={tm.sign_flips} | ok={tm.converged} | "
                f"kp_s={tm.params_snapshot['kp_small_head']:.3f} boost={tm.params_snapshot['boost_scale']:.2f}"
            ),
        )
        print(
            f"Stage 2 converged={res2['converged']} streak={res2['converge_streak']}/{res2['target_streak']} "
            f"final_params={res2['final_params']}"
        )
        if not res2.get("converged"):
            print("[FAIL] Stage 2 failed to converge to 5 consecutive zero-overshoot trials.")
            return 2

    print("\n[PASS] InGameAutoTrainer Stage 1 & Stage 2 synthetic verification passed!")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="CS2 3-RRR In-Game Loaded Closed-Loop Auto-Trainer")
    parser.add_argument("--sim-test", action="store_true", help="Run synthetic latency+load convergence test")
    args = parser.parse_args()
    if args.sim_test:
        return run_synthetic_simulation_test()
    print("Use --sim-test or press F5/F8 inside shared_control_hil/run_live_game.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
