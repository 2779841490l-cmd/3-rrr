"""
shared_controller.py - Module 3: 1000Hz Hand-Stage Decoupling, Intent-Aware Multi-Target Handover,
                       4-State Collaborative State Machine & High-Pass Washout Filter
====================================================================================================
Implements:
1. Hand-Stage Motion Decoupling:
       dp_hand(t) = dp_usb(t) - dp_stage_model(t - tau_delay)
2. Intent-Aware Multi-Target Handover (with Head-to-Chest fallback on shot k > 4):
       J_i = ||d_i||_2 - beta * max(0, v_hand . d_hat_i) + P_switch * I(i != i_prev)
3. 4-State Collaborative State Machine:
   - State 1: High-Speed Flick Dormant (||v_hand|| > v_flick)
   - State 2: Counter-Strafe & Micro-Flick Lock (||v_hand|| < v_stop, 3 <= ||e_cnt|| <= 48 cnt)
   - State 3: Sustained Spray & Transfer with High-Pass Washout Filter:
         p_stage_cmd(k) = gamma_wash * p_stage(k-1) + lambda(v_hand) * e_cnt(k)
         gamma_wash = exp(-dt / tau_w), clamped strictly to [-80, +80] counts (never +/- 88)
   - State 4: Motion-Masked Recentering (250-350 ms ramp to u0, < 1 ms preemption)
"""

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple
import math
import time
import numpy as np

from shared_control_hil.vision_roi_tracker import TargetDetection


STATE_1_FLICK_DORMANT = "STATE_1_FLICK_DORMANT"
STATE_2_COUNTER_STRAFE_LOCK = "STATE_2_COUNTER_STRAFE_LOCK"
STATE_3_SPRAY_TRANSFER = "STATE_3_SPRAY_TRANSFER"
STATE_4_MASKED_RECENTER = "STATE_4_MASKED_RECENTER"


@dataclass
class ControllerStepOutput:
    state: str
    active_target_id: int
    anchor_mode: str                  # "HEAD" | "HEAD_CHEST_BLEND" | "CHEST" | "NONE"
    target_anchor_cnt: np.ndarray     # predicted anchor relative to stage zero (counts)
    residual_err_cnt: np.ndarray      # e_cnt = target - c_cnt
    lambda_weight: float
    gamma_wash: float
    p_stage_cmd_cnt: np.ndarray       # absolute stage displacement setpoint [-80, +80] counts
    dp_stage_step_cnt: np.ndarray     # incremental command for this cycle
    recenter_progress: float
    decision_time_ms: float


class MacroMicroSharedController:
    """Shared-control policy fusing human intent, washout filtering, and 3-RRR micro-actuation."""

    def __init__(self, config: Dict):
        self.cfg = config
        self.update_params(config["controller_params"])

        kin = config["stage_kinematics"]
        self.safe_envelope_cnt = float(kin.get("safe_envelope_counts", 80.0))
        self.hard_limit_cnt = float(kin.get("hard_limit_counts", 88.0))
        self.recenter_trigger_cnt = float(kin.get("recenter_trigger_counts", 15.0))
        self.recenter_v_hand_min = float(kin.get("recenter_v_hand_min", 120.0))
        self.recenter_ramp_s = float(kin.get("recenter_ramp_ms", 300.0)) / 1000.0

        th = config["thresholds"]
        self.v_flick = float(th.get("v_flick_cnt_s", 280.0))
        self.v_stop = float(th.get("v_stop_cnt_s", 80.0))
        self.micro_min = float(th.get("micro_flick_min_cnt", 3.0))
        self.micro_max = float(th.get("micro_flick_max_cnt", 48.0))

        self.counts_per_px = self._compute_counts_per_px(config["screen"])

        # Dynamic runtime states
        self.p_stage_cur = np.zeros(2, dtype=float)
        self.v_hand = np.zeros(2, dtype=float)
        self.active_target_id: int = -1
        self.shots_on_target: int = 0
        self.state: str = STATE_1_FLICK_DORMANT

        # Delay buffer for 1000 Hz hand-stage decoupling (tau_delay ~ 12 ms default)
        self.stage_delay_buf: Deque[Tuple[float, np.ndarray]] = deque(maxlen=64)
        self.tau_delay_s = 0.012

        # State 4 smooth recentering ramp bookkeeping
        self._recenter_active = False
        self._recenter_start_p = np.zeros(2, dtype=float)
        self._recenter_elapsed_s = 0.0

    @staticmethod
    def _compute_counts_per_px(screen_cfg: Dict) -> float:
        w = float(screen_cfg.get("width", 1920))
        fov_x = float(screen_cfg.get("fov_x_deg", 106.26))
        s_game = float(screen_cfg.get("s_game", 0.646))
        m_yaw = float(screen_cfg.get("m_yaw", 0.022))
        f_x = (0.5 * w) / math.tan(math.radians(0.5 * fov_x))
        return math.degrees(math.atan(1.0 / f_x)) / (m_yaw * s_game)

    def update_params(self, params: Dict[str, float]) -> None:
        self.k_lead = float(params["k_lead"])
        self.lambda_0 = float(params["lambda_0"])
        self.sigma_v = float(params["sigma_v"])
        self.beta = float(params["beta"])
        self.P_switch = float(params["P_switch"])
        self.tau_w = float(params["tau_w"])

    def reset_episode(self) -> None:
        self.p_stage_cur = np.zeros(2, dtype=float)
        self.v_hand = np.zeros(2, dtype=float)
        self.active_target_id = -1
        self.shots_on_target = 0
        self.state = STATE_1_FLICK_DORMANT
        self.stage_delay_buf.clear()
        self._recenter_active = False
        self._recenter_elapsed_s = 0.0

    def record_stage_model_delta(self, t_s: float, dp_stage_cnt: np.ndarray) -> None:
        """Push modeled stage displacement into delay buffer for 1000Hz hand-stage decoupling."""
        self.stage_delay_buf.append((float(t_s), np.asarray(dp_stage_cnt, dtype=float).copy()))

    def decouple_hand_motion(
        self,
        t_s: float,
        dp_usb_cnt: np.ndarray,
        dt_s: float,
        dp_stage_model_override: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute pure human hand motion:
            dp_hand(t) = dp_usb(t) - dp_stage_model(t - tau_delay)
        Returns (dp_hand_cnt, v_hand_cnt_s).
        """
        if dp_stage_model_override is not None:
            dp_stage_del = np.asarray(dp_stage_model_override, dtype=float)
        else:
            dp_stage_del = np.zeros(2, dtype=float)
            target_t = float(t_s) - self.tau_delay_s
            for ts, dp in reversed(self.stage_delay_buf):
                if ts <= target_t:
                    dp_stage_del = dp
                    break

        dp_hand = np.asarray(dp_usb_cnt, dtype=float) - dp_stage_del
        inst_v = dp_hand / max(1e-4, float(dt_s))
        # First-order low-pass filter on decoupled hand velocity (~12ms cutoff)
        alpha = min(1.0, float(dt_s) / 0.012)
        self.v_hand = (1.0 - alpha) * self.v_hand + alpha * inst_v
        return dp_hand, self.v_hand.copy()

    def select_active_target(
        self,
        crosshair_cnt: np.ndarray,
        targets_cnt: List[TargetDetection],
    ) -> Tuple[Optional[TargetDetection], Dict[int, float]]:
        """
        Intent-aware multi-target handover in normalized pixel distance units:
            J_i = ||d_i||_2 - beta * max(0, v_hand_px . d_hat_i) + P_switch * I(i != i_prev)
        Note: d_i is expressed in px (counts / counts_per_px) and v_hand_px in px/s,
        directly matching P_switch (px) and beta (s) in SHARED_CONTROL_SPEC.
        """
        if not targets_cnt:
            return None, {}

        v_hand_px = self.v_hand / self.counts_per_px
        costs: Dict[int, float] = {}
        best_tgt: Optional[TargetDetection] = None
        best_cost = float("inf")

        for tgt in targets_cnt:
            pred = tgt.head_pred_px if tgt.head_pred_px is not None else tgt.head_px
            # When using counts-space detections, pred is in counts -> convert to px for cost J_i
            d_cnt = np.asarray(pred, dtype=float) - np.asarray(crosshair_cnt, dtype=float)
            d_px = d_cnt / self.counts_per_px
            dist_px = float(np.hypot(d_px[0], d_px[1]))
            d_hat = d_px / max(1e-6, dist_px)
            proj_v = max(0.0, float(np.dot(v_hand_px, d_hat)))
            hysteresis = self.P_switch if (self.active_target_id >= 0 and tgt.target_id != self.active_target_id) else 0.0
            cost = dist_px - self.beta * proj_v + hysteresis
            costs[int(tgt.target_id)] = cost
            if cost < best_cost:
                best_cost = cost
                best_tgt = tgt

        if best_tgt is not None:
            if best_tgt.target_id != self.active_target_id:
                self.active_target_id = int(best_tgt.target_id)
                self.shots_on_target = 0
        return best_tgt, costs

    def compute_lambda(self, e_cnt: Optional[np.ndarray] = None) -> float:
        """
        Human-machine blending weight:
            lambda(v_hand) = lambda_0 * exp(-||v_hand||^2 / (2 * sigma_v^2))
        Adds directional anti-conflict attenuation when hand velocity already drives
        the crosshair fast along the error vector (prevents same-direction overshoot).
        """
        v_norm_sq = float(np.dot(self.v_hand, self.v_hand))
        lam = self.lambda_0 * math.exp(-v_norm_sq / max(1e-6, 2.0 * self.sigma_v * self.sigma_v))
        if e_cnt is not None and v_norm_sq > (self.v_stop * self.v_stop):
            e_norm = float(np.hypot(e_cnt[0], e_cnt[1]))
            if e_norm > 1e-3:
                e_hat = np.asarray(e_cnt, dtype=float) / e_norm
                same_dir_v = float(np.dot(self.v_hand, e_hat))
                if same_dir_v > 0.35 * self.sigma_v:
                    # Smoothly yield to avoid adding stage travel onto active human flick
                    lam *= math.exp(-((same_dir_v - 0.35 * self.sigma_v) / max(40.0, 0.5 * self.sigma_v)) ** 2)
        return float(np.clip(lam, 0.0, 1.0))

    def _compute_target_anchor(
        self,
        tgt: TargetDetection,
        shot_idx_on_target: int,
        recoil_jitter_px: float = 0.0,
    ) -> Tuple[np.ndarray, str]:
        """Head-to-chest adaptive fallback: shots 1..3 anchor HEAD; k >= 4 blend toward CHEST."""
        head_p = np.asarray(tgt.head_pred_px if tgt.head_pred_px is not None else tgt.head_px, dtype=float)
        chest_p = np.asarray(tgt.chest_pred_px if tgt.chest_pred_px is not None else tgt.chest_px, dtype=float)
        if shot_idx_on_target <= 3:
            return head_p, "HEAD"
        blend = min(1.0, 0.5 * float(shot_idx_on_target - 3) + min(0.35, 0.04 * abs(recoil_jitter_px)))
        anchor = (1.0 - blend) * head_p + blend * chest_p
        return anchor, ("CHEST" if blend >= 0.99 else "HEAD_CHEST_BLEND")

    def step(
        self,
        dt_s: float,
        crosshair_cnt: np.ndarray,
        targets_cnt: List[TargetDetection],
        v_hand_override: Optional[np.ndarray] = None,
        lmb_held: bool = False,
        shot_idx_global: int = 0,
        counter_strafe_event: bool = False,
        reload_or_switch_key: bool = False,
        braking_near_target: bool = False,
        recoil_jitter_px: float = 0.0,
        p_stage_actual_cnt: Optional[np.ndarray] = None,
    ) -> ControllerStepOutput:
        """Execute one shared-control decision step in < 1.5 ms."""
        t0 = time.perf_counter()
        if v_hand_override is not None:
            self.v_hand = np.asarray(v_hand_override, dtype=float)
        if p_stage_actual_cnt is not None:
            self.p_stage_cur = np.asarray(p_stage_actual_cnt, dtype=float).copy()

        v_mag = float(np.hypot(self.v_hand[0], self.v_hand[1]))
        gamma_wash = math.exp(-max(1e-4, float(dt_s)) / max(0.05, self.tau_w))

        active_tgt, _ = self.select_active_target(crosshair_cnt, targets_cnt)

        # Update shot counter per target when LMB held / firing (restarts at 1 on target switch)
        if lmb_held and active_tgt is not None:
            self.shots_on_target += 1

        # Evaluate candidate anchor & residual error
        anchor_mode = "NONE"
        anchor_cnt = np.asarray(crosshair_cnt, dtype=float).copy()
        e_cnt = np.zeros(2, dtype=float)
        if active_tgt is not None:
            anchor_cnt, anchor_mode = self._compute_target_anchor(
                active_tgt,
                shot_idx_on_target=self.shots_on_target if lmb_held else 1,
                recoil_jitter_px=recoil_jitter_px,
            )
            e_cnt = anchor_cnt - np.asarray(crosshair_cnt, dtype=float)

        e_mag = float(np.hypot(e_cnt[0], e_cnt[1]))
        lam = self.compute_lambda(e_cnt)

        # 1ms preemption out of State 4 if fire or counter-strafe begins
        if self._recenter_active and (lmb_held or counter_strafe_event):
            self._recenter_active = False

        p_cmd = self.p_stage_cur.copy()
        recenter_prog = 0.0

        # ------------------------------------------------------------------
        # 4-State Collaborative State Machine
        # ------------------------------------------------------------------
        # State 3: Sustained Spray & Transfer (LMB held, shot k >= 2 or active spray)
        if lmb_held and active_tgt is not None:
            self.state = STATE_3_SPRAY_TRANSFER
            p_cmd = gamma_wash * self.p_stage_cur + lam * e_cnt
            p_cmd = np.clip(p_cmd, -self.safe_envelope_cnt, self.safe_envelope_cnt)

        # State 2: Counter-Strafe & Micro-Flick Lock (State 2 window 3.0..48.0 cnt)
        elif active_tgt is not None and (
            counter_strafe_event or v_mag < self.v_stop
        ) and (self.micro_min <= e_mag <= self.micro_max):
            self.state = STATE_2_COUNTER_STRAFE_LOCK
            if counter_strafe_event and v_mag < 1e-6 and self.lambda_0 >= 0.95:
                lam = 1.0
            p_cmd = self.p_stage_cur + lam * e_cnt
            p_cmd = np.clip(p_cmd, -self.safe_envelope_cnt, self.safe_envelope_cnt)

        # State 4: Motion-Masked Recentering (safe zeroing under fast human sweep or reload/switch)
        elif (
            not lmb_held
            and float(np.hypot(self.p_stage_cur[0], self.p_stage_cur[1])) > self.recenter_trigger_cnt
            and (v_mag > self.recenter_v_hand_min or reload_or_switch_key)
        ) or (self._recenter_active and not lmb_held and not counter_strafe_event):
            self.state = STATE_4_MASKED_RECENTER
            if not self._recenter_active:
                self._recenter_active = True
                self._recenter_start_p = self.p_stage_cur.copy()
                self._recenter_elapsed_s = 0.0
            self._recenter_elapsed_s += max(0.001, float(dt_s))
            recenter_prog = min(1.0, self._recenter_elapsed_s / max(0.25, self.recenter_ramp_s))
            p_cmd = (1.0 - recenter_prog) * self._recenter_start_p
            if recenter_prog >= 1.0:
                self._recenter_active = False

        # State 1: High-Speed Flick Dormant
        elif v_mag > self.v_flick and not braking_near_target:
            self.state = STATE_1_FLICK_DORMANT
            # Hold current stage position; never push against fast human flick
            p_cmd = self.p_stage_cur.copy()

        else:
            # Idle / small residual deadband hold
            self.state = STATE_1_FLICK_DORMANT
            p_cmd = self.p_stage_cur.copy()

        p_cmd = np.clip(p_cmd, -self.safe_envelope_cnt, self.safe_envelope_cnt)
        dp_step = p_cmd - self.p_stage_cur
        self.p_stage_cur = p_cmd.copy()

        dt_ms = (time.perf_counter() - t0) * 1000.0
        return ControllerStepOutput(
            state=self.state,
            active_target_id=int(active_tgt.target_id) if active_tgt is not None else -1,
            anchor_mode=anchor_mode,
            target_anchor_cnt=anchor_cnt,
            residual_err_cnt=e_cnt,
            lambda_weight=lam,
            gamma_wash=gamma_wash,
            p_stage_cmd_cnt=p_cmd,
            dp_stage_step_cnt=dp_step,
            recenter_progress=recenter_prog,
            decision_time_ms=dt_ms,
        )
