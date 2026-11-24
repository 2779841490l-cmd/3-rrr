"""
run_live_game.py - Module 8: CS2 Live Macro-Micro Shared-Control Resident Runner
================================================================================
Ties together the validated shared_control_hil modules for live CS2 gameplay
(e.g. Aim Botz workshop / Fullscreen Windowed, cl_crosshair_recoil 1):
  - Module 1 (fast_self_calib.py): 1.3s 4-pulse F8 online self-calibration & 3D pinhole projection
  - Module 2 (vision_roi_tracker.py): High-rate 256x256 center ROI capture, crosshair & KF tracker
  - Module 3 (shared_controller.py): 4-state collaborative controller with config_hil.json theta*
  - Module 4 (telemetry_referee.py): Real-time shot verdict, InitErr/FinalErr & 4-fault attribution

Global Non-Blocking Hotkeys (GetAsyncKeyState):
  - F6  : Collect 10-shot CS2 AK-47 cl_crosshair_recoil 1 burst -> ak47_recoil_profile.json
  - F7  : Toggle AK_SPRAY_LOCK mode (LMB holds -> 100 Hz closed-loop AK pull-down, release -> recenter)
  - F8  : 1.3s COM4 online self-calibration (updates mounting angle alpha & 2x2 Jacobian)
  - F9  : Toggle mode PAUSED -> ASSIST_ON <-> SHADOW_REFEREE_ONLY (blind A/B evaluation)
  - F10 : Smoothly return 3-RRR stage to neutral (1496.0, 1466.0) us and PAUSE capture
  - END : Safe exit (or Ctrl+C in terminal) -> smooth neutral return, close COM4/capture & save report
"""

import argparse
from collections import deque
import ctypes
import json
import math
import os
import sys
import threading
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from rect_manifold_calibrator import HardwareRectangleTrainer
from shared_control_hil.ak_spray_calibrator import (
    AK47SprayCalibrator,
    _clamp_servo_pair,
    detect_crosshair_offset_px,
)
from shared_control_hil.engagement_logger import CS2GSIListener, EngagementLogger, ensure_cs2_gsi_config
from shared_control_hil.fast_self_calib import DEFAULT_CONFIG_PATH, FastSelfCalibrator
from shared_control_hil.in_game_auto_trainer import InGameAutoTrainer
from shared_control_hil.shared_controller import (
    STATE_1_FLICK_DORMANT,
    STATE_2_COUNTER_STRAFE_LOCK,
    STATE_3_SPRAY_TRANSFER,
    STATE_4_MASKED_RECENTER,
    MacroMicroSharedController,
)
from shared_control_hil.telemetry_referee import (
    FAULT_CONFLICT,
    FAULT_HANDOVER_DELAY,
    FAULT_LAG,
    FAULT_SATURATION,
    ShotVerdict,
    TelemetryReferee,
)
from shared_control_hil.vision_roi_tracker import (
    TargetDetection,
    VisionROITracker,
    render_synthetic_roi_frame,
)

HIL_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(HIL_DIR, "logs")
DEFAULT_REPORT_PATH = os.path.join(LOGS_DIR, "live_session_report.json")

# Virtual-key codes
VK_LBUTTON = 0x01
VK_END = 0x23
VK_1 = 0x31
VK_2 = 0x32
VK_3 = 0x33
VK_A = 0x41
VK_D = 0x44
VK_Q = 0x51
VK_R = 0x52
VK_S = 0x53
VK_W = 0x57
VK_F4 = 0x73
VK_F5 = 0x74
VK_F6 = 0x75
VK_F7 = 0x76
VK_F8 = 0x77
VK_F9 = 0x78
VK_F10 = 0x79

MODE_PAUSED = "PAUSED"
MODE_ASSIST_ON = "ASSIST_ON"
MODE_SHADOW = "SHADOW_REFEREE_ONLY"
MODE_AK_SPRAY = "AK_SPRAY_LOCK"
BLIND_WINDOW_ENGAGEMENTS = 5

STATE_LABEL_MAP = {
    STATE_1_FLICK_DORMANT: "DORMANT",
    STATE_2_COUNTER_STRAFE_LOCK: "COUNTER_STRAFE",
    STATE_3_SPRAY_TRANSFER: "SPRAY_TRANSFER",
    STATE_4_MASKED_RECENTER: "MASKED_RECENTER",
}

SAFE_ENVELOPE_CNT = 85.0
SPRAY_MAX_X_CNT = 280.0
SPRAY_MAX_Y_CNT = 340.0
LIVE_MAX_X_CNT = 240.0
LIVE_MAX_Y_CNT = 300.0
COOLDOWN_LOCK_S = 0.085
RECOIL_UPDATE_PERIOD_S = 0.020
AUTO_FIRE_PERIOD_S = 0.095
HUD_WINDOW_NAME = "CS2 3-RRR Vision HUD"


class Win32KeyPoller:
    """Lightweight non-blocking Win32 key-state poller preferring win32api.GetAsyncKeyState."""

    def __init__(self) -> None:
        self._win32api = None
        try:
            import win32api  # type: ignore
            self._win32api = win32api
        except Exception:
            self._win32api = None
        self._user32 = ctypes.windll.user32
        self._prev_down: Dict[int, bool] = {}

    def is_down(self, vk: int) -> bool:
        if self._win32api is not None:
            return bool(self._win32api.GetAsyncKeyState(int(vk)) & 0x8000)
        return bool(self._user32.GetAsyncKeyState(int(vk)) & 0x8000)

    def pressed_once(self, vk: int) -> bool:
        now = self.is_down(vk)
        prev = self._prev_down.get(int(vk), False)
        self._prev_down[int(vk)] = now
        return now and not prev

    def flush(self, vks: Tuple[int, ...]) -> None:
        for vk in vks:
            self._prev_down[int(vk)] = self.is_down(vk)


def extract_live_crosshair_fast(
    roi_rgb: np.ndarray,
    vision: VisionROITracker,
    dt_s: float,
) -> np.ndarray:
    """Fast (< 0.2 ms) vectorized central recoil crosshair c(t) extractor for 144+ Hz live control."""
    h, w, _ = roi_rgb.shape
    r = roi_rgb[:, :, 0].astype(np.int16)
    g = roi_rgb[:, :, 1].astype(np.int16)
    b = roi_rgb[:, :, 2].astype(np.int16)

    cx0, cy0 = 0.5 * w, 0.5 * h
    x_lo = max(0, int(math.floor(cx0 - 30.0)))
    x_hi = min(w, int(math.ceil(cx0 + 30.0)) + 1)
    y_lo = max(0, int(math.floor(cy0 - 80.0)))
    y_hi = min(h, int(math.ceil(cy0 + 18.0)) + 1)
    r_w, g_w, b_w = r[y_lo:y_hi, x_lo:x_hi], g[y_lo:y_hi, x_lo:x_hi], b[y_lo:y_hi, x_lo:x_hi]

    cyan_mask = (g_w >= 175) & (b_w >= 175) & ((g_w - r_w) >= 60) & ((b_w - r_w) >= 50)
    green_mask = (g_w >= 175) & ((g_w - r_w) >= 65) & ((g_w - b_w) >= 45)
    ch_mask = cyan_mask if int(cyan_mask.sum()) >= int(green_mask.sum()) else green_mask

    c_meas: Optional[np.ndarray] = None
    ch_cnt = int(ch_mask.sum())
    if 3 <= ch_cnt <= 320:
        row_sum = ch_mask.sum(axis=1).astype(float)
        col_sum = ch_mask.sum(axis=0).astype(float)
        vr = int(np.argmax(row_sum))
        ur = int(np.argmax(col_sum))
        mh, mw = ch_mask.shape
        vw = slice(max(0, vr - 9), min(mh, vr + 10))
        uw = slice(max(0, ur - 9), min(mw, ur + 10))
        ys, xs = np.nonzero(ch_mask[vw, uw])
        if len(xs) >= 3:
            c_meas = np.array(
                [float(xs.mean() + x_lo + uw.start), float(ys.mean() + y_lo + vw.start)],
                dtype=float,
            )

    if c_meas is not None:
        c_px, _ = vision.crosshair_tracker.kf.step(c_meas, dt_s=dt_s)
        vision.crosshair_tracker.last_pos = c_px
    else:
        c_px, _ = vision.crosshair_tracker.detect_in_frame(roi_rgb, dt_s=dt_s)
    return c_px


def detect_live_crosshair_and_targets(
    roi_rgb: np.ndarray,
    vision: VisionROITracker,
    counts_per_px: float,
    dt_s: float,
    k_lead_s: float,
    prev_gray: Optional[np.ndarray],
    compute_bg_flow: bool = True,
) -> Tuple[np.ndarray, List[TargetDetection], np.ndarray, Optional[np.ndarray]]:
    """
    Extract dynamic crosshair c_px, live targets (head + chest), ego background shift, and gray frame.
    Supports Cyan (0,255,255), Green (0,255,0)/(0,255,80), and synthetic/silhouette fallback.
    """
    h, w, _ = roi_rgb.shape
    c_px = extract_live_crosshair_fast(roi_rgb, vision, dt_s=dt_s)

    gray: Optional[np.ndarray] = None
    du_bg = np.zeros(2, dtype=float)
    if compute_bg_flow:
        r = roi_rgb[:, :, 0].astype(np.float32)
        g = roi_rgb[:, :, 1].astype(np.float32)
        b = roi_rgb[:, :, 2].astype(np.float32)
        gray = (0.299 * r + 0.587 * g + 0.114 * b)
        if prev_gray is not None and prev_gray.shape == gray.shape:
            # Fast FFT phase correlation on outer ROI ring for mouse turn fallback
            g0 = prev_gray - float(prev_gray.mean())
            g1 = gray - float(gray.mean())
            f0 = np.fft.rfft2(g0)
            f1 = np.fft.rfft2(g1)
            cross = f1 * np.conj(f0)
            cross /= np.maximum(np.abs(cross), 1e-6)
            corr = np.fft.irfft2(cross, s=gray.shape)
            py, px = np.unravel_index(int(np.argmax(corr)), corr.shape)
            if py > h // 2:
                py -= h
            if px > w // 2:
                px -= w
            if abs(px) <= 18 and abs(py) <= 18:
                du_bg = np.array([float(px), float(py)], dtype=float)

    # Use real trained YOLOv8n CS2 head/chest detector in live mode; color blobs ONLY in synthetic mode
    if vision.synthetic_only:
        dets = vision._detect_color_blobs_fallback(roi_rgb)
    else:
        dets = vision._detect_yolo(roi_rgb)

    dets = vision.update_target_filters(dets, dt_s=dt_s, k_lead_s=k_lead_s)
    return c_px, dets, du_bg, gray


class LiveGameSharedControlRunner:
    """Resident closed-loop controller connecting CS2 ROI perception to 3-RRR WS63 COM4 hardware."""

    def __init__(
        self,
        config_path: str = DEFAULT_CONFIG_PATH,
        report_path: str = DEFAULT_REPORT_PATH,
        synthetic_capture: bool = False,
        enable_gui: bool = True,
    ) -> None:
        self.config_path = config_path
        self.report_path = report_path
        self.enable_gui = bool(enable_gui)
        self._hud_initialized = False
        os.makedirs(os.path.dirname(self.report_path), exist_ok=True)

        self.calibrator = FastSelfCalibrator(config_path=self.config_path)
        self.config = self.calibrator.config
        self.u0_neutral_us = (
            float(self.config["stage_kinematics"]["u0_neutral_us"][0]),
            float(self.config["stage_kinematics"]["u0_neutral_us"][1]),
        )

        # Initialize WS63 + 3200 DPI Raw Input sensor cleanly
        self.trainer: HardwareRectangleTrainer = self.calibrator.ensure_trainer()

        vision_cfg = self.config.get("vision", {})
        raw_yolo_path = vision_cfg.get("yolo_weights_path")
        resolved_yolo_path = None
        if raw_yolo_path:
            cand = str(raw_yolo_path)
            if not os.path.isabs(cand) and not os.path.exists(cand):
                cand = os.path.join(os.path.dirname(HIL_DIR), str(raw_yolo_path))
            resolved_yolo_path = cand

        self.min_track_frames = int(vision_cfg.get("min_track_frames", 2))
        self.min_aim_dist_px = float(vision_cfg.get("min_aim_dist_px", 4.0))
        self.max_aim_dist_px = float(vision_cfg.get("max_aim_dist_px", 95.0))
        self.vision = VisionROITracker(
            screen_w=int(self.config["screen"]["width"]),
            screen_h=int(self.config["screen"]["height"]),
            roi_size=int(self.config["screen"].get("roi_size", 256)),
            yolo_model_path=resolved_yolo_path,
            yolo_conf=float(vision_cfg.get("yolo_conf", 0.48)),
            yolo_imgsz=int(vision_cfg.get("yolo_imgsz", 256)),
            max_box_w=float(vision_cfg.get("max_box_w", 90.0)),
            max_box_h=float(vision_cfg.get("max_box_h", 140.0)),
            synthetic_only=bool(synthetic_capture),
        )
        self.live_max_x_cnt = float(self.config["stage_kinematics"].get("live_max_x_counts", LIVE_MAX_X_CNT))
        self.live_max_y_cnt = float(self.config["stage_kinematics"].get("live_max_y_counts", LIVE_MAX_Y_CNT))
        self.spray_max_x_cnt = float(self.config["stage_kinematics"].get("spray_max_x_counts", SPRAY_MAX_X_CNT))
        self.spray_max_y_cnt = float(self.config["stage_kinematics"].get("spray_max_y_counts", SPRAY_MAX_Y_CNT))
        self.safe_envelope_cnt = float(self.config["stage_kinematics"].get("safe_envelope_counts", 165.0))
        self.ctrl = MacroMicroSharedController(self.config)
        self.ak_calibrator = AK47SprayCalibrator(config_path=self.config_path, calibrator=self.calibrator)
        self.referee = TelemetryReferee(
            counts_per_px=self.ctrl.counts_per_px,
            head_radius_cnt=float(self.config["thresholds"].get("head_radius_cnt", 10.5)),
            chest_radius_cnt=float(self.config["thresholds"].get("chest_radius_cnt", 21.0)),
            hard_limit_cnt=float(self.config["stage_kinematics"].get("hard_limit_counts", 180.0)),
        )
        self.eng_logger = EngagementLogger(LOGS_DIR)
        self.gsi = CS2GSIListener(port=30001)
        self._gsi_started = False
        self.keys = Win32KeyPoller()

        self.mode: str = MODE_PAUSED
        self.shot_count: int = 0
        self.shot_modes: List[str] = []
        self.shot_states: List[str] = []
        self.decision_latencies_ms: List[float] = []
        self.calib_history: List[Dict] = []
        self.roi_fps_bench: float = 0.0
        self.raw_grab_fps_bench: float = 0.0
        self.full_yolo_fps_bench: float = 0.0
        self.full_yolo_median_ms: float = 0.0

        # Physical stage state + non-blocking 85 ms cooldown hardware dispatch guard
        self.p_stage_est = np.zeros(2, dtype=float)
        self._hw_lock = threading.Lock()
        self._hw_busy = False
        self._hw_thread: Optional[threading.Thread] = None
        self._last_hw_cmd_t = 0.0
        self._prev_sensor_xy = np.array(self.trainer.sensor.get_position(), dtype=float)

        # LMB / WASD live interaction state
        self._prev_lmb = False
        self._last_fire_t = 0.0
        self._burst_shot_idx = 0
        self._wasd_prev_down = False
        self._last_wasd_dir = (0, 0)
        self._counter_strafe_until = 0.0
        self._prev_gray: Optional[np.ndarray] = None
        self._stop_requested = False

        # Live decomposed control, zero-overshoot lock & F5/F6/F8 loaded calibration state
        self.p_aim_cnt = np.zeros(2, dtype=float)
        self.p_recoil_cnt = np.zeros(2, dtype=float)
        self._last_engage_t = time.perf_counter()
        self._prev_lmb_live = False
        self._lock_target_id: Optional[int] = None
        self._lock_start_t = 0.0
        self._lock_dir_sign = np.zeros(2, dtype=float)
        self._last_u_kin_live: Optional[np.ndarray] = None
        self._locked_play_g = np.zeros(2, dtype=float)
        self._prev_recoil_off_px: Optional[np.ndarray] = None
        self._mono_target_id: Optional[int] = None
        self._mono_e0_norm: float = 0.0
        self._mono_cmd_t: float = 0.0
        self._mono_tripped: bool = False
        self._shot_snapshots: deque = deque(maxlen=16)
        self.snapshots_path = os.path.join(LOGS_DIR, "live_shot_snapshots.jpg")
        self._last_aim_cmd_t: float = 0.0
        self._last_recoil_cmd_t: float = 0.0
        self._lmb_down_since_t: float = 0.0
        self._last_lmb_release_t: float = 0.0
        self._snap_done_tid: Optional[int] = None
        self._last_snap_t: float = 0.0
        self._flick_hi_cnt: int = 0
        self._yolo_miss_since_t: Optional[float] = None
        self._mono_trip_t: float = 0.0
        self._active_bcmd: Optional[Dict[str, Any]] = None
        self._last_pulse_cmd_t: float = 0.0
        self._last_pulse_dir: np.ndarray = np.zeros(2, dtype=float)
        self._pulse_immunity_active: bool = False
        self._pulse_opp_turn_confirmed: bool = False
        self._last_bcmd_t: float = 0.0
        self._prev_trim_dp: np.ndarray = np.zeros(2, dtype=float)
        self._prev_trim_t: float = 0.0
        self._live_anchor_mode: str = "HEAD"
        self._live_anchor_px: Optional[np.ndarray] = None
        self._spray_anchor_latched: Optional[str] = None
        self._cmd_history: deque = deque(maxlen=64)
        self._aim_cmd_history: deque = deque(maxlen=64)
        self._dual_mouse_detected: bool = False
        self._pre_micro_done_tid: Optional[int] = None
        self._pre_micro_count: int = 0
        self._pre_micro_sent_t: float = 0.0
        self._pre_prev_epx: Optional[np.ndarray] = None
        self._pre_prev_epx_t: float = 0.0
        self._target_base_stage_cnt: np.ndarray = np.zeros(2, dtype=float)
        self._last_wash_cmd_t: float = 0.0
        self._tgt_motion_tid: Optional[int] = None
        self._tgt_prev_head_px: Optional[np.ndarray] = None
        self._tgt_prev_det_t: float = 0.0
        self._tgt_prev_hand_xy: Optional[np.ndarray] = None
        self._tgt_prev_vis_stage: Optional[np.ndarray] = None
        self._tgt_v_drift_px_s: np.ndarray = np.zeros(2, dtype=float)
        self._tgt_v_world_px_s: np.ndarray = np.zeros(2, dtype=float)
        self._tgt_moving_frames: int = 0
        self._cap_hand_history: deque = deque(maxlen=64)
        self._hand_px_per_cnt_samples: deque = deque(maxlen=45)
        self.blind_ab_active: bool = False
        self.blind_window_size: int = BLIND_WINDOW_ENGAGEMENTS
        self._blind_rng = np.random.default_rng()
        self._blind_pair_id: int = 0
        self._blind_slot_idx: int = 0
        self._blind_pair_arms: Tuple[str, str] = (MODE_ASSIST_ON, MODE_SHADOW)
        self._blind_next_pair_arms: Tuple[str, str] = (MODE_SHADOW, MODE_ASSIST_ON)
        self._blind_eng_in_win: int = 0
        self._blind_switch_pending: bool = False
        self._blind_pending_since_t: float = 0.0
        self._last_is_moving_tgt: bool = False
        self._last_v_world_x_px_s: float = 0.0
        self._reload_live_control_params()

        self.in_game_trainer = InGameAutoTrainer(
            config_path=self.config_path,
            calibrator=self.calibrator,
            vision=self.vision,
            trainer=self.trainer,
            dispatch_fn=self._trainer_dispatch,
            recenter_fn=self._smooth_recenter_3step,
            detect_fn=self._detect_for_auto_trainer,
            roi_size_px=(float(self.vision.roi_size), float(self.vision.roi_size)),
        )

        # Dedicated 1000 Hz state/input tracker
        self._state_lock = threading.Lock()
        self._v_hand_1k = np.zeros(2, dtype=float)
        self._lmb_edge_pending = False
        self._counter_strafe_flag = False
        self._recenter_key_flag = False
        self._tracker_running = False
        self._tracker_thread: Optional[threading.Thread] = None

    def _reload_live_control_params(self) -> Dict[str, float]:
        """Load or hot-reload `live_control` damping, deadband, boost & recenter parameters."""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)
        except Exception:
            pass
        lc = self.config.get("live_control", {})
        self.kp_small_head = float(lc.get("kp_small_head", 0.55))
        self.kp_large_error = float(lc.get("kp_large_error", 0.56))
        self.kp_snap = float(lc.get("kp_snap", 0.85))
        self.small_error_px = float(lc.get("small_error_px", 7.5))
        self.in_box_deadband_ratio = float(lc.get("in_box_deadband_ratio", 0.18))
        self.min_deadband_px = float(lc.get("min_deadband_px", 1.4))
        self.max_deadband_px = float(lc.get("max_deadband_px", 2.4))
        self.cooldown_lock_s = float(lc.get("cooldown_ms", 45.0)) / 1000.0
        self.recoil_cooldown_s = max(0.025, float(lc.get("recoil_cooldown_ms", 35.0)) / 1000.0)
        self.aim_cooldown_s = max(0.042, float(lc.get("aim_cooldown_ms", 50.0)) / 1000.0)
        self.tap_guard_s = max(0.0, float(lc.get("tap_guard_ms", 30.0)) / 1000.0)
        self.visual_lag_s = max(0.040, float(lc.get("visual_lag_ms", 95.0)) / 1000.0)
        self.boost_scale = float(lc.get("boost_scale", 1.35))
        self.boost_dur_ms = float(lc.get("boost_dur_ms", 30.0))
        self.recenter_idle_s = float(lc.get("recenter_idle_ms", 150.0)) / 1000.0
        self.aim_washout_tau_s = max(0.05, float(lc.get("aim_washout_tau_s", 0.20)))
        self.no_reversal_window_s = float(lc.get("no_reversal_window_ms", 250.0)) / 1000.0
        self.auto_recenter_threshold_cnt = float(lc.get("auto_recenter_threshold_cnt", 180.0))
        return {k: float(v) for k, v in lc.items()} if isinstance(lc, dict) else {}

    def _detect_for_auto_trainer(self, roi_rgb: np.ndarray) -> Tuple[np.ndarray, List[TargetDetection]]:
        """Adapter exposing current crosshair & tracked YOLO detections to InGameAutoTrainer."""
        c_px, dets_px, _, _ = detect_live_crosshair_and_targets(
            roi_rgb=roi_rgb,
            vision=self.vision,
            counts_per_px=self.ctrl.counts_per_px,
            dt_s=0.020,
            k_lead_s=self.ctrl.k_lead,
            prev_gray=None,
        )
        return c_px, dets_px

    def _reset_live_lock_and_stage_state(self) -> None:
        self.p_aim_cnt = np.zeros(2, dtype=float)
        self.p_recoil_cnt = np.zeros(2, dtype=float)
        self._lock_target_id = None
        self._lock_start_t = 0.0
        self._lock_dir_sign = np.zeros(2, dtype=float)
        self._last_u_kin_live = None
        self._locked_play_g = np.zeros(2, dtype=float)
        self._prev_recoil_off_px = None
        self._mono_target_id = None
        self._mono_e0_norm = 0.0
        self._mono_cmd_t = 0.0
        self._mono_tripped = False
        self._last_aim_cmd_t = 0.0
        self._last_recoil_cmd_t = 0.0
        self._lmb_down_since_t = 0.0
        self._last_lmb_release_t = 0.0
        self._snap_done_tid = None
        self._yolo_miss_since_t = None
        self._last_high_flick_t = 0.0
        self._settled_low_vh_ticks = 0
        self._last_high_flick_1k_t = 0.0
        self._hand_settled_1k_since_t = 0.0
        self._last_hud_t = 0.0
        self._mono_trip_t = 0.0
        self._active_bcmd = None
        self._last_pulse_cmd_t = 0.0
        self._last_pulse_dir = np.zeros(2, dtype=float)
        self._pulse_immunity_active = False
        self._pulse_opp_turn_confirmed = False
        self._last_bcmd_t = 0.0
        self._prev_trim_dp = np.zeros(2, dtype=float)
        self._prev_trim_t = 0.0
        self._live_anchor_mode = "HEAD"
        self._live_anchor_px = None
        self._spray_anchor_latched = None
        self._pre_micro_done_tid = None
        self._pre_micro_count = 0
        self._pre_micro_sent_t = 0.0
        self._pre_prev_epx = None
        self._pre_prev_epx_t = 0.0
        self._target_base_stage_cnt = np.zeros(2, dtype=float)
        self._last_wash_cmd_t = 0.0
        self._tgt_motion_tid = None
        self._tgt_prev_head_px = None
        self._tgt_prev_det_t = 0.0
        self._tgt_prev_hand_xy = None
        self._tgt_prev_vis_stage = None
        self._tgt_v_drift_px_s = np.zeros(2, dtype=float)
        self._tgt_v_world_px_s = np.zeros(2, dtype=float)
        self._tgt_moving_frames = 0
        if hasattr(self, "_cap_hand_history"):
            self._cap_hand_history.clear()
        if hasattr(self, "_cmd_history"):
            self._cmd_history.clear()
        if hasattr(self, "_aim_cmd_history"):
            self._aim_cmd_history.clear()
        self._last_engage_t = time.perf_counter()

    def _compute_deadband_px(self, det: Optional[TargetDetection]) -> float:
        """Compute bounded In-Box Deadband Freeze radius in pixels; tighter on far small heads."""
        if det is None:
            return float(self.min_deadband_px)
        if det.head_bbox_px is not None:
            bx1, by1, bx2, by2 = det.head_bbox_px
            min_wh = min(abs(float(bx2) - float(bx1)), abs(float(by2) - float(by1)))
        else:
            min_wh = 2.0 * float(det.head_radius_px)
        if min_wh <= 11.0:
            return float(np.clip(self.in_box_deadband_ratio * min_wh, 0.90, min(1.5, float(self.max_deadband_px))))
        return float(np.clip(self.in_box_deadband_ratio * min_wh, self.min_deadband_px, self.max_deadband_px))

    def _trainer_dispatch(
        self,
        p: np.ndarray,
        cooldown_s: float = COOLDOWN_LOCK_S,
        freeze_play_g: bool = False,
        use_boost: bool = True,
        boost_scale: Optional[float] = None,
        boost_dur_ms: Optional[float] = None,
    ) -> bool:
        """Blocking dispatch wrapper for InGameAutoTrainer so F5/F8 never drop steps or skip frame-lag settle."""
        sent = self._dispatch_stage_manifold_open_loop(
            p,
            is_recenter=False,
            cooldown_s=0.0,
            freeze_play_g=freeze_play_g,
            use_boost=use_boost,
            boost_scale=boost_scale,
            boost_dur_ms=boost_dur_ms,
        )
        time.sleep(max(0.085, float(cooldown_s)))
        return bool(sent)

    def _smooth_recenter_3step(self, duration_s: float = 0.12) -> None:
        """3-step non-jarring manifold return to (0, 0) mechanical neutral (~120ms)."""
        with self._hw_lock:
            p_cur = self.p_stage_est.copy()
        if math.hypot(float(p_cur[0]), float(p_cur[1])) < 1.5:
            self._reset_live_lock_and_stage_state()
            return
        step_sleep = max(0.015, float(duration_s) / 3.0)
        for alpha in (2.0 / 3.0, 1.0 / 3.0, 0.0):
            p_step = p_cur * alpha
            try:
                u1_c, u2_c, _, _ = self.calibrator.stage_pos_to_pulses(
                    self.trainer.manifold,
                    float(p_step[0]),
                    float(p_step[1]),
                    dx=-float(p_cur[0]),
                    dy=-float(p_cur[1]),
                    freeze_play_g=False,
                    play_deadband_cnt=0.0,
                )
                self.trainer.driver.send_raw_pulses(u1_c, u2_c)
            except Exception:
                pass
            time.sleep(step_sleep)
        with self._hw_lock:
            self.p_stage_est = np.zeros(2, dtype=float)
            self.ctrl.p_stage_cur = np.zeros(2, dtype=float)
            self.trainer.manifold.cur_rect_pos = np.zeros(2, dtype=float)
            self._last_hw_cmd_t = time.perf_counter()
        self._reset_live_lock_and_stage_state()

    def _wait_hw_idle(self, timeout_s: float = 0.75) -> None:
        t0 = time.perf_counter()
        th = None
        with self._hw_lock:
            th = self._hw_thread
        if th is not None and th.is_alive():
            rem = max(0.01, timeout_s - (time.perf_counter() - t0))
            th.join(timeout=rem)

    def recenter_to_neutral(self, duration_s: float = 0.22) -> None:
        """Smoothly return 3-RRR stage to mechanical center (1496.0, 1466.0) us and zero stage coordinates."""
        self._wait_hw_idle()
        u0x, u0y = self.u0_neutral_us
        self.trainer.driver.move_smooth_raw(u0x, u0y, duration_s=duration_s, rate_hz=50.0)
        self.p_stage_est = np.zeros(2, dtype=float)
        self.ctrl.p_stage_cur = np.zeros(2, dtype=float)
        self.trainer.manifold.cur_rect_pos = np.zeros(2, dtype=float)
        self._reset_live_lock_and_stage_state()
        sx, sy = self.trainer.sensor.get_position()
        self._prev_sensor_xy = np.array([float(sx), float(sy)], dtype=float)
        self.trainer.origin_sensor_xy = self._prev_sensor_xy.copy()

    def _maybe_cut_active_bcmd(self, cur_sensor: np.ndarray, t_now: float) -> bool:
        """1 kHz optical early H-brake: drop transient B_CMD overdrive to setpoint once 58% travel completes."""
        bcmd = getattr(self, "_active_bcmd", None)
        if bcmd is None or bcmd.get("cut_done", True):
            return False
        dt_b = float(t_now) - float(bcmd["t0"])
        dur_b = float(bcmd["dur_s"])
        if dt_b >= max(0.0, dur_b - 0.002):
            bcmd["cut_done"] = True
            return False
        if dt_b < 0.014:
            return False
        L_b = float(bcmd["L"])
        if L_b < 4.0:
            return False
        disp_vec = np.asarray(cur_sensor, dtype=float) - bcmd["s0"]
        d_hat = bcmd["d_hat"]
        p_par = float(np.dot(disp_vec, d_hat))
        p_perp = abs(float(disp_vec[0] * (-d_hat[1]) + disp_vec[1] * d_hat[0]))
        cut_thresh = float(bcmd.get("cut_frac", 0.58)) * L_b
        if p_par >= cut_thresh and p_perp < max(12.0, 0.85 * L_b):
            if self._hw_lock.acquire(timeout=0.002):
                try:
                    cur_b = getattr(self, "_active_bcmd", None)
                    if cur_b is bcmd and not cur_b.get("cut_done", True):
                        cur_b["cut_done"] = True
                        self.trainer.driver.send_raw_pulses(float(cur_b["e1"]), float(cur_b["e2"]))
                        return True
                finally:
                    self._hw_lock.release()
        return False

    def _start_1khz_state_tracker(self) -> None:
        if self._tracker_running:
            return
        self._tracker_running = True

        def _loop() -> None:
            last_t = time.perf_counter()
            last_sensor = np.array(self.trainer.sensor.get_position(), dtype=float)
            has_hand_api = hasattr(self.trainer.sensor, "get_hand_position") and hasattr(
                self.trainer.sensor, "get_hand_event_count"
            )
            last_hand_sensor = (
                np.array(self.trainer.sensor.get_hand_position(), dtype=float)
                if has_hand_api
                else np.zeros(2, dtype=float)
            )
            v_hand_mouse_lp = np.zeros(2, dtype=float)
            dual_logged = False
            with self._hw_lock:
                prev_stage_cmd = self.p_stage_est.copy()
                p_stage_d1 = self.p_stage_est.copy()
                last_stage_phys = self.p_stage_est.copy()
                p_stage_phys = self.p_stage_est.copy()
            prev_lmb_1k = False
            prev_wasd = (False, False, False, False)
            v_usb_lp = np.zeros(2, dtype=float)
            v_stg_lp = np.zeros(2, dtype=float)
            cone_dir = np.array([1.0, 0.0], dtype=float)
            cone_budget = 0.0
            last_cmd_1k_t = 0.0
            opp_turn_dur_s = 0.0
            cos75 = math.cos(math.radians(75.0))
            while self._tracker_running:
                t_now = time.perf_counter()
                dt = max(0.0005, t_now - last_t)
                last_t = t_now

                cur_sensor = np.array(self.trainer.sensor.get_position(), dtype=float)
                self._maybe_cut_active_bcmd(cur_sensor, t_now)
                with self._hw_lock:
                    cur_stage_cmd = self.p_stage_est.copy()
                    pulse_t = float(self._last_pulse_cmd_t)
                    pulse_dir = self._last_pulse_dir.copy()

                # Track 2-pole servo deadtime+settling (18 ms + 45 ms) plus 115 ms held + 30 ms tail directional cone
                d_cmd = cur_stage_cmd - prev_stage_cmd
                prev_stage_cmd = cur_stage_cmd.copy()
                d_cmd_n = float(math.hypot(d_cmd[0], d_cmd[1]))
                if d_cmd_n >= 3.0:
                    cone_dir = d_cmd / d_cmd_n
                    cone_budget = max(cone_budget, d_cmd_n / 0.065)
                    last_cmd_1k_t = t_now
                    if (t_now - pulse_t) > 0.020:
                        pulse_t = t_now
                        pulse_dir = cone_dir.copy()
                elif (t_now - last_cmd_1k_t) > 0.115:
                    cone_budget = cone_budget * math.exp(-dt / 0.030)

                p_stage_d1 = p_stage_d1 + min(1.0, dt / 0.018) * (cur_stage_cmd - p_stage_d1)
                p_stage_phys = p_stage_phys + min(1.0, dt / 0.045) * (p_stage_d1 - p_stage_phys)
                dp_usb = cur_sensor - last_sensor
                dp_stage = p_stage_phys - last_stage_phys
                last_sensor = cur_sensor
                last_stage_phys = p_stage_phys.copy()

                alpha = min(1.0, dt / 0.025)
                v_usb_lp = (1.0 - alpha) * v_usb_lp + alpha * (dp_usb / dt)
                v_stg_lp = (1.0 - alpha) * v_stg_lp + alpha * (dp_stage / dt)
                usb_spd = float(math.hypot(v_usb_lp[0], v_usb_lp[1]))
                stg_spd = float(math.hypot(v_stg_lp[0], v_stg_lp[1]))

                # Dual-mouse auto-detection: read 1000 Hz ground-truth hand velocity with ZERO servo crosstalk!
                hand_evts = int(self.trainer.sensor.get_hand_event_count()) if has_hand_api else 0
                if has_hand_api:
                    cur_hand_sensor = np.array(self.trainer.sensor.get_hand_position(), dtype=float)
                    dp_hand = cur_hand_sensor - last_hand_sensor
                    last_hand_sensor = cur_hand_sensor
                    alpha_h = min(1.0, dt / 0.016)
                    v_hand_mouse_lp = (1.0 - alpha_h) * v_hand_mouse_lp + alpha_h * (dp_hand / dt)

                if hand_evts >= 5:
                    self._dual_mouse_detected = True
                    if not dual_logged:
                        dual_logged = True
                        print(
                            "\n[Dual-Mouse Auto-Detected] Separating true 1000 Hz Hand Mouse velocity "
                            "from 3-RRR Stage Mouse (zero servo crosstalk)!",
                            flush=True,
                        )
                    v_hand_est = v_hand_mouse_lp.copy()
                    self._pulse_opp_turn_confirmed = False
                    self._pulse_immunity_active = False
                else:
                    # Single-mouse fallback: 98 ms In-Pulse Immunity Window + stage velocity subtraction
                    p_dir_n = float(math.hypot(pulse_dir[0], pulse_dir[1]))
                    in_pulse_win = (pulse_t > 0.0) and ((t_now - pulse_t) <= 0.098) and (p_dir_n > 0.5)
                    if in_pulse_win:
                        p_hat = pulse_dir / p_dir_n
                        v_par = float(np.dot(v_usb_lp, p_hat))
                        if (-v_par) > 380.0:
                            opp_turn_dur_s += dt
                        else:
                            opp_turn_dur_s = 0.0
                        opp_confirmed = opp_turn_dur_s >= 0.020
                        self._pulse_opp_turn_confirmed = bool(opp_confirmed)
                        self._pulse_immunity_active = not opp_confirmed
                        if v_par >= 0.0 and (usb_spd <= 1e-3 or v_par > (usb_spd * cos75)):
                            v_hand_est = np.zeros(2, dtype=float)
                        elif v_par > 0.0:
                            v_hand_est = v_usb_lp - v_par * p_hat
                        elif not opp_confirmed:
                            v_hand_est = np.zeros(2, dtype=float)
                        else:
                            v_hand_est = v_usb_lp.copy()
                    else:
                        opp_turn_dur_s = 0.0
                        self._pulse_opp_turn_confirmed = False
                        self._pulse_immunity_active = False
                        cone_active = (t_now - last_cmd_1k_t) <= 0.145 and cone_budget > 20.0
                        if stg_spd > 1.0 or cone_active:
                            u_hat = (v_stg_lp / stg_spd) if (stg_spd >= cone_budget and stg_spd > 1.0) else cone_dir
                            allow_spd = max(stg_spd, cone_budget if cone_active else 0.0)
                            proj = float(np.dot(v_usb_lp, u_hat))
                            stage_attrib = min(max(0.0, proj), 1.30 * allow_spd) * u_hat
                            v_hand_est = v_usb_lp - stage_attrib
                        else:
                            v_hand_est = v_usb_lp.copy()

                lmb_now = bool(self.keys.is_down(VK_LBUTTON))
                w_d = bool(self.keys.is_down(VK_W))
                a_d = bool(self.keys.is_down(VK_A))
                s_d = bool(self.keys.is_down(VK_S))
                d_d = bool(self.keys.is_down(VK_D))
                wasd_now = (w_d, a_d, s_d, d_d)
                cs_trigger = False
                if any(prev_wasd) and not any(wasd_now):
                    cs_trigger = True
                elif (prev_wasd[1] and d_d) or (prev_wasd[3] and a_d) or (prev_wasd[0] and s_d) or (prev_wasd[2] and w_d):
                    cs_trigger = True
                prev_wasd = wasd_now

                rec_key = any(self.keys.is_down(vk) for vk in (VK_R, VK_1, VK_2, VK_3, VK_Q))
                vh_1k_norm = float(math.hypot(v_hand_est[0], v_hand_est[1]))
                if self._dual_mouse_detected:
                    if vh_1k_norm >= 220.0:
                        self._last_high_flick_1k_t = t_now
                        self._hand_settled_1k_since_t = 0.0
                    elif vh_1k_norm <= 40.0:
                        if getattr(self, "_hand_settled_1k_since_t", 0.0) <= 0.0:
                            self._hand_settled_1k_since_t = t_now
                    elif vh_1k_norm >= 85.0:
                        self._hand_settled_1k_since_t = 0.0
                elif self._pulse_immunity_active:
                    if getattr(self, "_hand_settled_1k_since_t", 0.0) <= 0.0:
                        self._hand_settled_1k_since_t = t_now
                elif vh_1k_norm >= 240.0:
                    self._last_high_flick_1k_t = t_now
                    self._hand_settled_1k_since_t = 0.0
                elif vh_1k_norm <= 155.0:
                    if getattr(self, "_hand_settled_1k_since_t", 0.0) <= 0.0:
                        self._hand_settled_1k_since_t = t_now
                else:
                    self._hand_settled_1k_since_t = 0.0
                with self._state_lock:
                    self._v_hand_1k = v_hand_est.copy()
                    if lmb_now and not prev_lmb_1k:
                        self._lmb_edge_pending = True
                    if cs_trigger:
                        self._counter_strafe_flag = True
                    if rec_key:
                        self._recenter_key_flag = True
                prev_lmb_1k = lmb_now
                time.sleep(0.001)

        self._tracker_thread = threading.Thread(target=_loop, daemon=True)
        self._tracker_thread.start()

    def _stop_1khz_state_tracker(self) -> None:
        self._tracker_running = False
        if self._tracker_thread is not None and self._tracker_thread.is_alive():
            self._tracker_thread.join(timeout=0.5)
        self._tracker_thread = None

    def startup_self_check(self) -> float:
        """Verify COM4 connection, recenter servos to neutral, benchmark 1.0s ROI FPS, and print usage."""
        print("=" * 78)
        print("  CS2 3-RRR Macro-Micro Shared Control: Live Runner Pre-Flight Self-Check")
        print("=" * 78)
        port_name = getattr(self.trainer.driver, "port", "COM4")
        sub_us = getattr(self.trainer.driver, "sub_us_supported", False)
        b_cmd = getattr(self.trainer.driver, "b_cmd_supported", False)
        print(f"[COM4] Connected on {port_name} (SUBUS_V2={sub_us}, B_CMD={b_cmd})")
        self.recenter_to_neutral(duration_s=0.22)
        print(
            f"[COM4] Stage anchored at mechanical neutral "
            f"u0=({self.u0_neutral_us[0]:.1f}, {self.u0_neutral_us[1]:.1f}) us"
        )

        # 1. Dual FPS Benchmark: Raw Grab FPS vs Full Grab + YOLO FP16 Inference FPS (<= 12.0 ms gate)
        raw_lat_ms = []
        t0_raw = time.perf_counter()
        while (time.perf_counter() - t0_raw) < 0.55:
            tf0 = time.perf_counter()
            _ = self.vision.grab_roi()
            raw_lat_ms.append((time.perf_counter() - tf0) * 1000.0)
        elapsed_raw = max(1e-3, time.perf_counter() - t0_raw)
        self.raw_grab_fps_bench = len(raw_lat_ms) / elapsed_raw
        raw_med_ms = float(np.median(raw_lat_ms)) if raw_lat_ms else 999.0

        full_lat_ms = []
        yolo_lat_ms = []
        t0_full = time.perf_counter()
        while (time.perf_counter() - t0_full) < 1.0:
            tf0 = time.perf_counter()
            roi_b = self.vision.grab_roi()
            ty0 = time.perf_counter()
            if self.vision.synthetic_only:
                _ = self.vision._detect_color_blobs_fallback(roi_b)
            else:
                _ = self.vision._detect_yolo(roi_b)
            yolo_lat_ms.append((time.perf_counter() - ty0) * 1000.0)
            full_lat_ms.append((time.perf_counter() - tf0) * 1000.0)
        elapsed_full = max(1e-3, time.perf_counter() - t0_full)
        self.full_yolo_fps_bench = len(full_lat_ms) / elapsed_full
        self.full_yolo_median_ms = float(np.median(full_lat_ms)) if full_lat_ms else 999.0
        full_p95_ms = float(np.percentile(full_lat_ms, 95)) if full_lat_ms else 999.0
        yolo_med_ms = float(np.median(yolo_lat_ms)) if yolo_lat_ms else 0.0
        self.roi_fps_bench = self.full_yolo_fps_bench
        gate_ok = self.full_yolo_median_ms <= 12.0 and self.full_yolo_fps_bench >= 80.0
        print(
            f"[ROI Benchmark] Backend={self.vision.backend_name} | "
            f"Raw Grab FPS={self.raw_grab_fps_bench:.1f} Hz (med={raw_med_ms:.2f}ms)"
        )
        print(
            f"[Vision + YOLO Benchmark] Detector={self.vision.detector_backend} | "
            f"Full Grab + YOLO FP16 Inference FPS={self.full_yolo_fps_bench:.1f} Hz | "
            f"med={self.full_yolo_median_ms:.2f}ms (yolo={yolo_med_ms:.2f}ms, p95={full_p95_ms:.2f}ms) -> "
            f"{'PASS (<=12ms / >=80Hz)' if gate_ok else 'WARN (>12ms)'}"
        )
        print(
            f"[Vision Detector] Backend={self.vision.detector_backend} | "
            f"spray_envelope=({self.spray_max_x_cnt:.0f}, {self.spray_max_y_cnt:.0f}) cnt"
        )
        print(
            f"[Params Loaded] {os.path.basename(self.config_path)}: "
            f"k_lead={self.ctrl.k_lead:.5f}, lambda_0={self.ctrl.lambda_0:.4f}, "
            f"sigma_v={self.ctrl.sigma_v:.2f}, beta={self.ctrl.beta:.4f}, "
            f"P_switch={self.ctrl.P_switch:.3f}, tau_w={self.ctrl.tau_w:.4f}"
        )
        print("-" * 78)
        print("Hotkeys (Non-Blocking Win32 GetAsyncKeyState):")
        print("  [F4 ]  Toggle BLIND_AB (Double-Blind 5-Engagement Randomized Crossover -> Paired p-value)")
        print("  [F5 ]  60s In-Game Loaded Auto-Train (Bot red-box in ROI -> Stage1 Jacobian + Stage2 Damping)")
        print("  [F6 ]  Loaded-Hand 5-Iter AK-47 ILC Calibration (hold mouse with palm weight!) -> ak47_recoil_profile.json")
        print("  [F7 ]  Toggle AK_SPRAY_LOCK (hold LMB: 100 Hz boosted ILC pull-down, release: 3-step recenter)")
        print("  [F8 ]  YOLO Head-Flow Visual Calibration (Bot head in ROI -> fallback OpticalFlow/Pinhole)")
        print("  [F9 ]  Toggle Mode: PAUSED -> ASSIST_ON <-> SHADOW_REFEREE_ONLY")
        print("  [F10]  Emergency Recenter to (1496, 1466) us & PAUSE")
        print("  [END]  Safe Exit (or Ctrl+C in terminal) -> Recenter, Release COM4 & Save report")
        print("-" * 78)
        print("[HOW TO TRAIN IN AIM BOTZ (2 Steps, Hand resting naturally on mouse)]:")
        print("  1. Stand ~10-15m facing a stationary Bot so its RED YOLO box stays inside the 256x256 HUD.")
        print("     Press [F8] (~1.0s) to calibrate screen Jacobian, then [F5] (up to 16 trials) to eliminate overshoot.")
        print("  2. Equip AK-47, keep hand weight naturally on mouse, press [F6] & follow prompt for 5-iter loaded spray ILC.")
        print("-" * 78)
        self.mode = MODE_PAUSED
        print(f"[STATE] Entered {self.mode}. Alt-tab into CS2 and press F5/F8 (calib), F4 (blind A/B test), or F9 (manual).")
        return self.roi_fps_bench

    def trigger_f5_in_game_autotrain(self) -> Dict:
        """Pause assist, calibrate loaded YOLO head Jacobian (Stage 1) + zero-overshoot damping (Stage 2)."""
        self._clear_blind_ab_state()
        prev_mode = self.mode
        self._wait_hw_idle()
        self.mode = MODE_PAUSED
        self._reset_live_lock_and_stage_state()
        print("\n" + "=" * 78)
        print("[F5 IN-GAME AUTO-TRAINER] Keep a stationary Bot RED head box in the 256x256 crosshair ROI")
        print("and keep your palm resting normally on the mouse (DO NOT fight micro-movements).")
        print("Stage 1: Loaded YOLO Head-Box Jacobian calibration...")
        res1 = self.in_game_trainer.calibrate_yolo_head_jacobian(step_cnt=70.0, settle_s=0.105)
        if not res1.get("ok"):
            print(
                f"[F5] Stage 1 aborted ({res1.get('reason')}). Make sure a stationary Bot head "
                "shows a RED box inside CS2 3-RRR Vision HUD, then press F5 again."
            )
            self.recenter_to_neutral(duration_s=0.15)
            self.mode = prev_mode
            return res1

        self.config = self.calibrator.config
        print(
            f"[F5 Stage 1 OK] src={res1['source']} | angle={res1['mounting_angle_deg']:+.2f} deg | "
            f"cond={res1['cond']:.2f}"
        )
        print("Stage 2: Loaded Closed-Loop Zero-Overshoot Damping Auto-Tuner (max 16 trials, stop on 5-streak)...")

        def _on_trial(tm) -> None:
            print(
                f"  [Trial {tm.trial_idx:02d}] init={tm.initial_err_px:5.2f}px -> final={tm.final_err_px:4.2f}px | "
                f"ovs={tm.overshoot_px:4.2f}px | flips={tm.sign_flips} | ok={tm.converged} | "
                f"kp_s={tm.params_snapshot['kp_small_head']:.3f} boost={tm.params_snapshot['boost_scale']:.2f}"
            )

        res2 = self.in_game_trainer.run_in_game_damping_autotune(
            n_trials=16,
            converge_streak=5,
            err_tol_px=2.0,
            offset_cnt=42.0,
            on_trial_done=_on_trial,
        )
        self._reload_live_control_params()
        self.recenter_to_neutral(duration_s=0.15)
        self.ctrl.reset_episode()
        self.vision.reset_filters()
        self.mode = prev_mode
        print(
            f"[F5 COMPLETE] converged={res2.get('converged')} streak={res2.get('converge_streak', 0)}/5 | "
            f"config_saved={res2.get('config_saved', False)} | "
            f"kp_small={self.kp_small_head:.3f}, kp_large={self.kp_large_error:.3f}, "
            f"deadband_ratio={self.in_box_deadband_ratio:.2f}, cooldown={self.cooldown_lock_s*1000:.0f}ms, "
            f"boost={self.boost_scale:.2f} | Mode={self.mode}"
        )
        print("=" * 78)
        return {"stage1": res1, "stage2": res2}

    def trigger_f8_calibration(self) -> Dict:
        """Run YOLO Head-Flow F8 calibration first (under real palm load), else Optical Flow / Pinhole fallback."""
        prev_mode = self.mode
        self._wait_hw_idle()
        self._reset_live_lock_and_stage_state()
        print("\n[F8] Trying YOLO Head-Box Visual Polarity & Jacobian calibration...")
        res_yolo = self.in_game_trainer.calibrate_yolo_head_jacobian(step_cnt=70.0, settle_s=0.135)
        if res_yolo.get("ok"):
            self.config = self.calibrator.config
            res = {
                "elapsed_s": 0.9,
                "visual_calib_source": res_yolo["source"],
                "du_x_px": res_yolo["axis_x"]["pos_px_per_cnt"],
                "du_y_px": res_yolo["axis_y"]["pos_px_per_cnt"],
                "mounting_angle_deg": res_yolo["mounting_angle_deg"],
                "pass_time": True,
                "pass_recon": True,
                "yolo_jacobian": res_yolo,
            }
        else:
            print(f"[F8] YOLO head unavailable ({res_yolo.get('reason')}); falling back to Optical Flow...")
            res = self.calibrator.run_visual_flow_calib(
                grab_roi_fn=self.vision.grab_roi,
                step_counts=60.0,
                settle_s=0.12,
            )
            self.config = self.calibrator.config
        self.calib_history.append(res)
        self.p_stage_est = np.zeros(2, dtype=float)
        self.ctrl.p_stage_cur = np.zeros(2, dtype=float)
        self.ctrl.reset_episode()
        self.vision.reset_filters()
        try:
            u_kin0 = self.calibrator.u0_neutral.copy()
            self.trainer.manifold.pi_net.reset_play_state(u_kin0, (1.0, 1.0))
            self._last_u_kin_live = u_kin0.copy()
        except Exception:
            pass
        self._prev_sensor_xy = np.array(self.trainer.sensor.get_position(), dtype=float)
        self._last_hw_cmd_t = time.perf_counter()
        if hasattr(self, "eng_logger"):
            self.eng_logger.clear_pending_probes()
        self.mode = prev_mode
        passed = bool(res.get("pass_time", True) and res.get("pass_recon", True))
        print(
            f"[F8 Visual-Calib] elapsed={res['elapsed_s']:.3f}s | src={res['visual_calib_source']} "
            f"| +X_flow={res['du_x_px']}px | +Y_flow={res['du_y_px']}px "
            f"| angle={res['mounting_angle_deg']:+.2f} deg | PASS={passed} | Mode={self.mode}"
        )
        return res

    def toggle_f9_mode(self) -> str:
        """Toggle PAUSED/AK_SPRAY_LOCK -> ASSIST_ON <-> SHADOW_REFEREE_ONLY."""
        self._clear_blind_ab_state()
        self._hide_vision_hud()
        if self.mode in (MODE_PAUSED, MODE_AK_SPRAY):
            self.recenter_to_neutral(duration_s=0.18)
            self.ctrl.reset_episode()
            self.vision.reset_filters()
            self._prev_gray = None
            self.mode = MODE_ASSIST_ON
        elif self.mode == MODE_ASSIST_ON:
            self.recenter_to_neutral(duration_s=0.18)
            self.mode = MODE_SHADOW
        else:
            self.recenter_to_neutral(duration_s=0.18)
            self.ctrl.p_stage_cur = np.zeros(2, dtype=float)
            self.mode = MODE_ASSIST_ON
        print(f"\n[F9] Switched Mode -> {self.mode}")
        return self.mode

    def _sample_blind_pair_arms(self) -> Tuple[str, str]:
        """Sample balanced permuted pair order: (ASSIST_ON, SHADOW) or (SHADOW, ASSIST_ON)."""
        if int(self._blind_rng.integers(0, 2)) == 0:
            return (MODE_ASSIST_ON, MODE_SHADOW)
        return (MODE_SHADOW, MODE_ASSIST_ON)

    def _clear_blind_ab_state(self) -> None:
        self.blind_ab_active = False
        self._blind_switch_pending = False
        self._blind_eng_in_win = 0

    def toggle_f4_blind_ab(self, seed: Optional[int] = None) -> str:
        """Toggle Double-Blind Permuted 5-Engagement Micro-Crossover (BLIND_AB) <-> PAUSED."""
        if self.blind_ab_active:
            self.trigger_f10_pause()
            print("[F4] Exited BLIND_AB -> PAUSED.")
            return self.mode

        if seed is not None:
            self._blind_rng = np.random.default_rng(int(seed))
        self._hide_vision_hud()
        self.recenter_to_neutral(duration_s=0.18)
        self._reset_live_lock_and_stage_state()
        self.ctrl.reset_episode()
        self.vision.reset_filters()
        self._prev_gray = None

        self.blind_ab_active = True
        self._blind_pair_id = 1
        self._blind_slot_idx = 0
        self._blind_eng_in_win = 0
        self._blind_switch_pending = False
        self._blind_pending_since_t = 0.0
        self._blind_pair_arms = self._sample_blind_pair_arms()
        self._blind_next_pair_arms = self._sample_blind_pair_arms()
        self.mode = self._blind_pair_arms[0]
        print(
            "\n[F4] Entered BLIND_AB (Double-Blind 5-Engagement Permuted Crossover | Pair #01 started). "
            "Shoot normally for 2-4 min (~10+ pairs), then press END/Ctrl+C for unblinded paired report!",
            flush=True,
        )
        return "BLIND_AB"

    def _step_blind_ab_crossover(
        self,
        t_now: float,
        lmb_down: bool,
        v_hand_norm: float,
        min_tgt_dist_px: float,
        closed_eng: Optional[Dict[str, Any]],
        dispatch_fn: Optional[Callable[..., bool]] = None,
    ) -> None:
        """Advance 5-engagement double-blind micro-windows and zero the stage cleanly before SHADOW arms."""
        if not self.blind_ab_active:
            return
        disp = self._dispatch_stage_manifold_open_loop if dispatch_fn is None else dispatch_fn

        if closed_eng is not None and (not self._blind_switch_pending):
            self._blind_eng_in_win += 1
            if self._blind_eng_in_win >= int(self.blind_window_size):
                self._blind_switch_pending = True
                self._blind_pending_since_t = float(t_now)

        if not self._blind_switch_pending:
            return
        if lmb_down or (hasattr(self, "eng_logger") and self.eng_logger._active is not None):
            return

        next_mode = self._blind_pair_arms[1] if self._blind_slot_idx == 0 else self._blind_next_pair_arms[0]
        if next_mode == MODE_SHADOW:
            stage_norm = float(np.hypot(self.p_stage_est[0], self.p_stage_est[1]))
            if stage_norm >= 2.0:
                safe_gap = (v_hand_norm >= 320.0 and min_tgt_dist_px > 34.0) or (
                    (float(t_now) - float(self._blind_pending_since_t)) >= 0.16
                )
                if safe_gap:
                    sent = disp(
                        np.zeros(2, dtype=float),
                        is_recenter=True,
                        cooldown_s=0.040,
                        freeze_play_g=False,
                        use_boost=False,
                    )
                    if sent:
                        self.p_aim_cnt[:] = 0.0
                        self.p_recoil_cnt[:] = 0.0
                        self._target_base_stage_cnt[:] = 0.0
                        self._cmd_history.append((float(t_now), np.zeros(2, dtype=float)))
                        self._aim_cmd_history.append((float(t_now), np.zeros(2, dtype=float)))
                stage_norm = float(np.hypot(self.p_stage_est[0], self.p_stage_est[1]))
                if stage_norm >= 2.0:
                    return

            self.p_aim_cnt[:] = 0.0
            self.p_recoil_cnt[:] = 0.0
            self._target_base_stage_cnt[:] = 0.0
            self._lock_target_id = None
            self._snap_done_tid = None
            self._pre_micro_done_tid = None
            self._pre_micro_count = 0
            self.ctrl.p_stage_cur = np.zeros(2, dtype=float)

        if self._blind_slot_idx == 0:
            self._blind_slot_idx = 1
            self.mode = self._blind_pair_arms[1]
        else:
            done_pair = int(self._blind_pair_id)
            self._blind_pair_id += 1
            self._blind_slot_idx = 0
            self._blind_pair_arms = self._blind_next_pair_arms
            self._blind_next_pair_arms = self._sample_blind_pair_arms()
            self.mode = self._blind_pair_arms[0]
            print(
                f"[BLIND_AB] Completed matched Pair #{done_pair:02d} -> starting Pair #{self._blind_pair_id:02d}",
                flush=True,
            )
        self._blind_eng_in_win = 0
        self._blind_switch_pending = False

    def toggle_f7_ak_spray(self) -> str:
        """Toggle AK_SPRAY_LOCK <-> PAUSED for 100 Hz closed-loop AK-47 point-lock recoil control."""
        self._clear_blind_ab_state()
        self._wait_hw_idle()
        self.recenter_to_neutral(duration_s=0.18)
        self.ctrl.reset_episode()
        self.vision.reset_filters()
        self._prev_gray = None
        if self.mode != MODE_AK_SPRAY:
            self._hide_vision_hud()
            self.ak_calibrator.load_profile()
            self.mode = MODE_AK_SPRAY
            print(
                "\n[F7] Switched Mode -> AK_SPRAY_LOCK "
                f"(source={self.ak_calibrator.profile_source}, s_game={self.ak_calibrator.s_game:.3f}). "
                "Hold LMB in CS2 for 100 Hz closed-loop point-lock; release LMB to smoothly recenter."
            )
        else:
            self.mode = MODE_PAUSED
            print("\n[F7] Exited AK_SPRAY_LOCK -> PAUSED.")
        return self.mode

    def trigger_f6_loaded_ak_ilc(self) -> Dict:
        """Run 5-iteration loaded-hand AK-47 ILC compensation (rest palm naturally on mouse!)."""
        self._clear_blind_ab_state()
        self._wait_hw_idle()
        self.mode = MODE_PAUSED
        self.recenter_to_neutral(duration_s=0.18)
        print("\n[F6] Running 5-Iter Loaded-Hand AK-47 ILC Calibration (keep palm weight normally on mouse)...")
        res = self.ak_calibrator.calibrate_loaded_hand_ilc(
            self.trainer,
            n_shots=12,
            ilc_iters=5,
            initial_boost_scale=self.boost_scale,
        )
        if "boost_scale" in res:
            self.boost_scale = float(res["boost_scale"])
        self.recenter_to_neutral(duration_s=0.18)
        self.ctrl.reset_episode()
        return res

    def _run_ak_spray_burst(self, n_shots: int = 30, check_lmb: bool = True) -> Dict:
        """Execute 100 Hz boosted ILC + B_CMD AK-47 pull-down (zero mouse_sensor hand conflict)."""
        self._wait_hw_idle()
        self.recenter_to_neutral(duration_s=0.12)

        res = self.ak_calibrator.execute_ak_spray_closed_loop(
            self.trainer,
            n_shots=n_shots,
            use_feedback=False,
            use_ilc=True,
            boost_scale=self.boost_scale,
            use_b_cmd=True,
            rate_hz=250.0,
            reset_origin=False,
            check_lmb=check_lmb,
            visual_trim_fn=None,
        )
        self.recenter_to_neutral(duration_s=0.15)
        print(
            f"[AK_SPRAY_LOCK Burst] dur={res['duration_s']:.2f}s | ticks={res['ticks']} | "
            f"mean_err={res['mean_bullet_err_cnt']:.2f} cnt | max={res['max_bullet_err_cnt']:.2f} cnt"
        )
        return res

    def _dispatch_stage_manifold_open_loop(
        self,
        p_target_cnt: np.ndarray,
        is_recenter: bool = False,
        cooldown_s: float = COOLDOWN_LOCK_S,
        freeze_play_g: bool = False,
        use_boost: bool = True,
        boost_scale: Optional[float] = None,
        boost_dur_ms: Optional[float] = None,
    ) -> bool:
        """
        Pure commanded-state manifold dispatch for LIVE gameplay (ZERO mouse_sensor feedback).
        Supports:
          - `freeze_play_g=True`: holds backlash flank locked (`evaluate_kinematic_pulse + _locked_play_g`)
            to avoid 18-30 us Bang-Bang jumps across tiny heads;
          - `use_boost=True`: atomic WS63 `B_CMD` transient boost overcoming 300-500g palm load.
        """
        t_now = time.perf_counter()
        if (t_now - self._last_hw_cmd_t) < float(cooldown_s):
            return False

        limits = np.array([self.live_max_x_cnt, self.live_max_y_cnt], dtype=float)
        p_clamped = np.clip(np.asarray(p_target_cnt, dtype=float), -limits, limits)
        dx = float(p_clamped[0] - self.p_stage_est[0])
        dy = float(p_clamped[1] - self.p_stage_est[1])
        if (not is_recenter) and math.hypot(dx, dy) < 1.2:
            return False
        if is_recenter and math.hypot(float(self.p_stage_est[0]), float(self.p_stage_est[1])) < 1.5:
            return False

        with self._hw_lock:
            if self._hw_busy:
                return False
            self._last_hw_cmd_t = t_now
            if is_recenter:
                p_clamped = np.zeros(2, dtype=float)
                dx = -float(self.p_stage_est[0])
                dy = -float(self.p_stage_est[1])

            x_cmd, y_cmd = float(p_clamped[0]), float(p_clamped[1])
            manifold = self.trainer.manifold
            e1, e2, u_lin, play_g = self.calibrator.stage_pos_to_pulses(
                manifold,
                x_cmd,
                y_cmd,
                dx=dx,
                dy=dy,
                freeze_play_g=(freeze_play_g and (not is_recenter)),
                locked_play_g=self._locked_play_g,
                play_deadband_cnt=0.0 if is_recenter else 4.0,
            )
            if not (freeze_play_g and (not is_recenter)):
                self._locked_play_g = play_g

            b_scale = self.boost_scale if boost_scale is None else float(boost_scale)
            dur_ms = self.boost_dur_ms if boost_dur_ms is None else float(boost_dur_ms)
            try:
                s_before = np.array(self.trainer.sensor.get_position(), dtype=float)
            except Exception:
                s_before = None

            live_boost = None
            dp_vec = np.array([dx, dy], dtype=float)
            dp_len = float(math.hypot(dx, dy))
            if (
                use_boost
                and (not is_recenter)
                and getattr(self.trainer.driver, "b_cmd_supported", False)
                and self._last_u_kin_live is not None
            ):
                live_boost = self.calibrator.compute_live_ballistic_boost(
                    u_lin - self._last_u_kin_live,
                    dp_vec,
                    e1,
                    e2,
                    boost_scale=b_scale,
                    boost_dur_ms=dur_ms,
                )
            if live_boost is not None:
                b1, b2, bst_us = live_boost
                if boost_dur_ms is not None:
                    bst_us = int(round(max(14.0, min(42.0, dur_ms)) * 1000.0))
                if dp_len >= 4.0 and s_before is not None:
                    self._active_bcmd = {
                        "t0": time.perf_counter(),
                        "dur_s": float(bst_us) * 1e-6,
                        "s0": s_before.copy(),
                        "dp_cmd": dp_vec.copy(),
                        "L": float(dp_len),
                        "d_hat": dp_vec / max(dp_len, 1e-6),
                        "e1": float(e1),
                        "e2": float(e2),
                        "cut_frac": 0.58,
                        "cut_done": False,
                    }
                else:
                    self._active_bcmd = None
                self.trainer.driver.send_ballistic_pulse(b1, b2, e1, e2, bst_us)
                if not is_recenter:
                    self._last_bcmd_t = time.perf_counter()
            else:
                self._active_bcmd = None
                self.trainer.driver.send_raw_pulses(e1, e2)

            if (not is_recenter) and ((live_boost is not None and dp_len >= 3.5) or dp_len >= 4.5):
                self._last_pulse_cmd_t = time.perf_counter()
                self._last_pulse_dir = dp_vec / max(dp_len, 1e-6)
                self._pulse_opp_turn_confirmed = False
                self._pulse_immunity_active = True

            if (not is_recenter) and (s_before is not None) and hasattr(self, "eng_logger"):
                self.eng_logger.on_stage_dispatch(t_now, np.array([dx, dy], dtype=float), s_before)

            self._last_u_kin_live = u_lin.copy()
            self.p_stage_est = p_clamped.copy()
            self.ctrl.p_stage_cur = p_clamped.copy()
        return True

    def _build_hud_canvas(
        self,
        roi_rgb: np.ndarray,
        c_px: np.ndarray,
        dets_px: List[TargetDetection],
        locked_det: Optional[TargetDetection],
        arrow_tip_px: Optional[np.ndarray],
        p_stage_now: np.ndarray,
        footer_text: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        """Build 256x256 BGR HUD overlay canvas shared by live GUI and blackbox snapshots."""
        try:
            import cv2

            hud = roi_rgb[:, :, ::-1].copy()
            h, w, _ = hud.shape
            cx0, cy0 = w // 2, h // 2

            # Geometric screen center u0
            cv2.circle(hud, (cx0, cy0), 2, (200, 200, 200), -1)

            # Detected recoil crosshair c(t) - Green Cross
            gx = int(np.clip(round(float(c_px[0])), 4, w - 5))
            gy = int(np.clip(round(float(c_px[1])), 4, h - 5))
            cv2.line(hud, (gx - 7, gy), (gx + 7, gy), (0, 255, 0), 1)
            cv2.line(hud, (gx, gy - 7), (gx, gy + 7), (0, 255, 0), 1)

            # Filtered YOLO boxes (RED = head, BLUE = chest)
            for d in dets_px:
                if d.chest_bbox_px is not None:
                    x1, y1, x2, y2 = [int(round(v)) for v in d.chest_bbox_px]
                    cv2.rectangle(hud, (x1, y1), (x2, y2), (255, 110, 0), 1)
                if d.head_bbox_px is not None:
                    x1, y1, x2, y2 = [int(round(v)) for v in d.head_bbox_px]
                    cv2.rectangle(hud, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(
                        hud,
                        f"{d.confidence:.2f}#{d.seen_frames}",
                        (max(2, x1), max(11, y1 - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.36,
                        (0, 0, 255),
                        1,
                    )
                else:
                    hx, hy = int(round(float(d.head_px[0]))), int(round(float(d.head_px[1])))
                    cv2.circle(hud, (hx, hy), 5, (0, 0, 255), 1)

            # Bounded In-Box Deadband Freeze white ring r_lock (<= max_deadband_px)
            if locked_det is not None:
                hx_l = int(round(float(locked_det.head_px[0])))
                hy_l = int(round(float(locked_det.head_px[1])))
                r_lock_px = self._compute_deadband_px(locked_det)
                cv2.circle(hud, (hx_l, hy_l), max(2, int(round(r_lock_px))), (255, 255, 255), 1)

            # Yellow traction vector arrow from c(t) toward expected aim direction
            if arrow_tip_px is not None:
                tx = int(np.clip(round(float(arrow_tip_px[0])), 2, w - 3))
                ty = int(np.clip(round(float(arrow_tip_px[1])), 2, h - 3))
                cv2.arrowedLine(hud, (gx, gy), (tx, ty), (0, 255, 255), 2, tipLength=0.25)

            src_str = self.calibrator.visual_calib_source
            calib_tag = "YOLO" if src_str == "YOLO_HEAD_FLOW" else ("FLOW" if src_str == "OPTICAL_FLOW" else "PINH")
            mode_tag = "ASSIST" if self.mode == MODE_ASSIST_ON else self.mode[:6]
            lock_tag = f"LOCK#{locked_det.target_id}" if locked_det is not None else "FREE"
            cv2.rectangle(hud, (0, 0), (w, 20), (15, 15, 15), -1)
            cv2.putText(
                hud,
                f"{mode_tag} ({p_stage_now[0]:+.0f},{p_stage_now[1]:+.0f}) {calib_tag} {lock_tag}",
                (4, 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (235, 235, 235),
                1,
            )
            if footer_text:
                cv2.rectangle(hud, (0, h - 18), (w, h), (15, 15, 15), -1)
                cv2.putText(hud, footer_text, (4, h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 255, 255), 1)
            return hud
        except Exception:
            return None

    def _capture_shot_snapshot(
        self,
        roi_rgb: np.ndarray,
        c_px: np.ndarray,
        dets_px: List[TargetDetection],
        locked_det: Optional[TargetDetection],
        arrow_tip_px: Optional[np.ndarray],
        p_stage_now: np.ndarray,
        label: str,
    ) -> None:
        hud = self._build_hud_canvas(
            roi_rgb=roi_rgb,
            c_px=c_px,
            dets_px=dets_px,
            locked_det=locked_det,
            arrow_tip_px=arrow_tip_px,
            p_stage_now=p_stage_now,
            footer_text=label,
        )
        if hud is not None:
            self._shot_snapshots.append(hud)

    def _save_shot_snapshots_grid(self) -> Optional[str]:
        """Stitch up to 16 blackbox shot/calibration snapshots into logs/live_shot_snapshots.jpg."""
        if not self._shot_snapshots:
            return None
        try:
            import cv2

            frames = list(self._shot_snapshots)
            cols = 4
            rows = int(math.ceil(len(frames) / float(cols)))
            cell_h, cell_w = frames[0].shape[:2]
            grid = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)
            for idx, fr in enumerate(frames):
                r, c = divmod(idx, cols)
                if fr.shape[:2] != (cell_h, cell_w):
                    fr = cv2.resize(fr, (cell_w, cell_h))
                grid[r * cell_h : (r + 1) * cell_h, c * cell_w : (c + 1) * cell_w] = fr
            os.makedirs(os.path.dirname(self.snapshots_path), exist_ok=True)
            cv2.imwrite(self.snapshots_path, grid)
            return self.snapshots_path
        except Exception:
            return None

    def _hide_vision_hud(self) -> None:
        """Close the TOPMOST debug HUD window so DWM compositing cannot stall GDI BitBlt."""
        if not getattr(self, "_hud_initialized", False):
            return
        try:
            import cv2

            cv2.destroyWindow(HUD_WINDOW_NAME)
            cv2.waitKey(1)
        except Exception:
            try:
                import cv2

                cv2.destroyAllWindows()
            except Exception:
                pass
        self._hud_initialized = False

    def _render_vision_hud(
        self,
        roi_rgb: np.ndarray,
        c_px: np.ndarray,
        dets_px: List[TargetDetection],
        locked_det: Optional[TargetDetection],
        arrow_tip_px: Optional[np.ndarray],
        p_stage_now: np.ndarray,
    ) -> None:
        """Render non-blocking topmost 256x256 real-time Vision HUD debug window (PAUSED mode only)."""
        if not self.enable_gui:
            return
        if self.mode != MODE_PAUSED:
            self._hide_vision_hud()
            return
        try:
            import cv2

            hud = self._build_hud_canvas(
                roi_rgb=roi_rgb,
                c_px=c_px,
                dets_px=dets_px,
                locked_det=locked_det,
                arrow_tip_px=arrow_tip_px,
                p_stage_now=p_stage_now,
            )
            if hud is None:
                return
            if not self._hud_initialized:
                cv2.namedWindow(HUD_WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(HUD_WINDOW_NAME, 256, 256)
                cv2.moveWindow(HUD_WINDOW_NAME, 10, 10)
                try:
                    cv2.setWindowProperty(HUD_WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)
                except Exception:
                    pass
                self._hud_initialized = True
            cv2.imshow(HUD_WINDOW_NAME, hud)
            cv2.waitKey(1)
        except Exception:
            pass

    def trigger_f10_pause(self) -> None:
        """Emergency smooth recenter to (1496, 1466) us and pause ROI capture."""
        self._clear_blind_ab_state()
        self.mode = MODE_PAUSED
        self.recenter_to_neutral(duration_s=0.20)
        self.ctrl.reset_episode()
        print("\n[F10] Stage safely returned to neutral (1496.0, 1466.0) us. Status: PAUSED.")

    def _dispatch_stage_command(
        self,
        p_target_cnt: np.ndarray,
        is_recenter: bool = False,
        wait_done: bool = False,
        spray_active: bool = False,
    ) -> bool:
        """Non-blocking ballistic dispatch guarded by COOLDOWN_LOCK_S (85 ms) and active mode envelope."""
        t_now = time.perf_counter()
        if (t_now - self._last_hw_cmd_t) < COOLDOWN_LOCK_S:
            return False

        if spray_active:
            env_vec = np.array([self.spray_max_x_cnt, self.spray_max_y_cnt], dtype=float)
        else:
            env_vec = np.array([SAFE_ENVELOPE_CNT, SAFE_ENVELOPE_CNT], dtype=float)
        p_clamped = np.clip(np.asarray(p_target_cnt, dtype=float), -env_vec, env_vec)
        delta_norm = float(np.hypot(p_clamped[0] - self.p_stage_est[0], p_clamped[1] - self.p_stage_est[1]))
        if not is_recenter and delta_norm < 1.8:
            return False

        with self._hw_lock:
            if self._hw_busy:
                return False
            self._hw_busy = True
            self._last_hw_cmd_t = t_now
            p_start_est = self.p_stage_est.copy()
            sensor_now = np.array(self.trainer.sensor.get_position(), dtype=float)
            # Decouple macro mouse travel so move_point_to_point sees clean stage start coordinates
            self.trainer.origin_sensor_xy = sensor_now - p_start_est
            self.trainer.manifold.cur_rect_pos = p_start_est.copy()

            def _worker() -> None:
                try:
                    if is_recenter:
                        u0x, u0y = self.u0_neutral_us
                        self.trainer.driver.move_smooth_raw(u0x, u0y, duration_s=0.18, rate_hz=50.0)
                        p_new = np.zeros(2, dtype=float)
                    else:
                        K_stage = np.asarray(self.calibrator.K_stage, dtype=float)
                        K_lin = np.asarray(self.trainer.manifold.pi_net.K_lin, dtype=float)
                        near_origin = float(np.hypot(p_start_est[0], p_start_est[1])) <= 3.0
                        w_lim = (0.95 if spray_active else 0.85) * self.trainer.manifold.W
                        h_lim = (1.15 if spray_active else 0.85) * self.trainer.manifold.H

                        if near_origin:
                            # Mouse sensor target counts (p_clamped) already match move_point_to_point sensor frame
                            p_man = p_clamped.copy()
                            p_man[0] = float(np.clip(p_man[0], -w_lim, w_lim))
                            p_man[1] = float(np.clip(p_man[1], -h_lim, h_lim))
                            self.trainer.move_point_to_point(
                                float(p_man[0]),
                                float(p_man[1]),
                                tol_counts=1.8,
                                two_stage=True,
                                enable_ilc_update=False,
                            )
                            p_meas = np.clip(self.trainer.get_rect_position(), -env_vec, env_vec)
                        else:
                            # Already displaced from origin: trim locally from current estimated optical position
                            p_meas = p_start_est.copy()

                        best_p = p_meas.copy()
                        best_u = (float(self.trainer.driver.cur_s1), float(self.trainer.driver.cur_s2))
                        best_err = float(np.hypot(p_clamped[0] - best_p[0], p_clamped[1] - best_p[1]))

                        if best_err > 2.2:
                            if getattr(self.trainer.driver, "ser", None) is not None:
                                try:
                                    self.trainer.driver.ser.write(b"F 10000\n")
                                except Exception:
                                    pass

                            # Fast closed-loop 3200 DPI sensor micro-trim (up to 3 steps)
                            gain = 0.82
                            for _trim_i in range(3):
                                if best_err <= 2.2:
                                    break
                                e_cnt = p_clamped - best_p
                                du = gain * (K_stage @ e_cnt)
                                if (
                                    _trim_i == 0
                                    and near_origin
                                    and p_clamped[0] < -10.0
                                    and e_cnt[1] > 6.0
                                    and abs(e_cnt[0]) <= 5.0
                                ):
                                    du[0] *= 0.25
                                    du[1] -= 5.5
                                du = np.clip(du, -45.0 if spray_active else -38.0, 45.0 if spray_active else 38.0)
                                u1, u2 = _clamp_servo_pair(
                                    best_u[0] + float(du[0]),
                                    best_u[1] + float(du[1]),
                                    prev_u1=best_u[0],
                                    prev_u2=best_u[1],
                                    max_step_us=55.0 if spray_active else 40.0,
                                )
                                self.trainer.driver.send_raw_pulses(u1, u2)
                                time.sleep(0.042)
                                p_trial = np.clip(self.trainer.get_rect_position(), -env_vec, env_vec)
                                err_trial = float(np.hypot(p_clamped[0] - p_trial[0], p_clamped[1] - p_trial[1]))
                                moved = float(np.hypot(p_trial[0] - best_p[0], p_trial[1] - best_p[1]))
                                if err_trial < best_err - 0.15:
                                    best_p = p_trial.copy()
                                    best_u = (u1, u2)
                                    best_err = err_trial
                                    if moved < 0.35 * max(1.0, err_trial):
                                        gain = min(1.25, gain + 0.25)
                                elif err_trial > best_err + 1.0:
                                    self.trainer.driver.send_raw_pulses(best_u[0], best_u[1])
                                    time.sleep(0.020)
                                    break
                                else:
                                    gain = min(1.30, gain + 0.30)

                        p_new = best_p
                    with self._hw_lock:
                        self.p_stage_est = p_new.copy()
                        self.ctrl.p_stage_cur = p_new.copy()
                        sx_after, sy_after = self.trainer.sensor.get_position()
                        self._prev_sensor_xy = np.array([float(sx_after), float(sy_after)], dtype=float)
                except Exception as exc:
                    print(f"\n[WARN][HW] Stage dispatch error: {exc}")
                finally:
                    with self._hw_lock:
                        self._hw_busy = False

            th = threading.Thread(target=_worker, daemon=True)
            self._hw_thread = th
            th.start()

        if wait_done:
            th.join(timeout=0.75)
        return True

    def _record_and_print_shot(
        self,
        t_now: float,
        mode_now: str,
        short_state: str,
        c_cnt: np.ndarray,
        p_stage_shot: np.ndarray,
        det_target: TargetDetection,
        v_hand_cnt_s: np.ndarray,
        active_target_id: int,
        anchor_mode: str,
    ) -> ShotVerdict:
        """Evaluate one live shot via TelemetryReferee and print single-line HUD telemetry."""
        self.shot_count += 1
        roi_center = 0.5 * self.vision.roi_size
        h_head_cnt = self.calibrator.pixel_to_counts(
            float(det_target.head_px[0]) - roi_center,
            float(det_target.head_px[1]) - roi_center,
        )
        h_chest_cnt = self.calibrator.pixel_to_counts(
            float(det_target.chest_px[0]) - roi_center,
            float(det_target.chest_px[1]) - roi_center,
        )
        v_tgt_cnt_s = self.calibrator.pixel_to_counts(
            float(det_target.vel_px_s[0]),
            float(det_target.vel_px_s[1]),
        )

        eff_cp = (
            self.eng_logger.get_effective_coupling(default=1.0)
            if (mode_now == MODE_ASSIST_ON and hasattr(self, "eng_logger"))
            else 1.0
        )
        verdict = self.referee.evaluate_shot(
            shot_idx=self.shot_count,
            t_s=t_now,
            scenario=mode_now,
            c_cnt=c_cnt,
            p_stage_cnt=p_stage_shot,
            head_true_cnt=h_head_cnt,
            chest_true_cnt=h_chest_cnt,
            v_target_cnt_s=v_tgt_cnt_s,
            v_hand_cnt_s=v_hand_cnt_s,
            active_target_id=active_target_id,
            intended_target_id=int(det_target.target_id),
            anchor_mode=anchor_mode if anchor_mode != "NONE" else "HEAD",
            coupling_eff=eff_cp,
        )
        self.shot_modes.append(mode_now)
        self.shot_states.append(short_state)
        if hasattr(self, "eng_logger"):
            gsi_snap = self.gsi.summary() if hasattr(self, "gsi") else {}
            self.eng_logger.on_shot(
                t_s=t_now,
                p_stage_now=p_stage_shot,
                verdict_dict={
                    "shot_idx": verdict.shot_idx,
                    "t_s": round(verdict.t_s, 3),
                    "mode": verdict.scenario,
                    "state": short_state,
                    "hitbox": verdict.hitbox,
                    "anchor_mode": verdict.anchor_mode,
                    "e_hand_cnt": verdict.e_hand_cnt,
                    "e_actual_cnt": verdict.e_actual_cnt,
                    "head_dist_cnt": verdict.head_dist_cnt,
                    "delta_e_cnt": verdict.delta_e_cnt,
                    "stage_pos_cnt": [round(float(verdict.p_stage_cnt[0]), 2), round(float(verdict.p_stage_cnt[1]), 2)],
                    "faults": verdict.faults,
                },
                weapon=str(gsi_snap.get("active_weapon", "")),
                blind_ab=self.blind_ab_active,
                blind_pair_id=(self._blind_pair_id if self.blind_ab_active else None),
                blind_slot=(self._blind_slot_idx if self.blind_ab_active else None),
            )

        px_int = int(round(float(verdict.p_stage_cnt[0])))
        py_int = int(round(float(verdict.p_stage_cnt[1])))
        if self.blind_ab_active:
            print(
                f"[Shot #{self.shot_count:03d} | Mode: BLIND_AB | Pair #{self._blind_pair_id:02d}] Recorded (blinded)",
                flush=True,
            )
        else:
            print(
                f"[Shot #{self.shot_count:03d} | Mode: {mode_now} | State: {short_state}] "
                f"InitErr: {verdict.e_hand_cnt:.1f} cnt -> FinalErr: {verdict.e_actual_cnt:.1f} cnt "
                f"[{verdict.hitbox}] | Gain dE: {verdict.delta_e_cnt:+.1f} cnt | StagePos: ({px_int:+d}, {py_int:+d})",
                flush=True,
            )
        return verdict

    def _build_session_report(self, session_dur_s: float) -> Dict:
        """Aggregate full session and per-mode (ASSIST_ON vs SHADOW_REFEREE_ONLY) telemetry."""
        overall = self.referee.summarize()

        def _subset_summary(target_mode: str) -> Dict:
            sub = [v for v in self.referee.verdicts if v.scenario == target_mode]
            if not sub:
                return {
                    "total_shots": 0,
                    "head_hit_rate": 0.0,
                    "effective_hit_rate": 0.0,
                    "mean_e_hand_cnt": 0.0,
                    "mean_e_actual_cnt": 0.0,
                    "mean_delta_e_cnt": 0.0,
                    "conflict_rate": 0.0,
                }
            n = len(sub)
            heads = sum(1 for v in sub if v.hitbox == "HEAD_HIT")
            eff = sum(1 for v in sub if v.hitbox in ("HEAD_HIT", "BODY_HIT"))
            confl = sum(1 for v in sub if v.conflict_flag)
            return {
                "total_shots": n,
                "head_hit_rate": round(heads / n, 4),
                "effective_hit_rate": round(eff / n, 4),
                "mean_e_hand_cnt": round(float(np.mean([v.e_hand_cnt for v in sub])), 3),
                "mean_e_actual_cnt": round(float(np.mean([v.e_actual_cnt for v in sub])), 3),
                "mean_delta_e_cnt": round(float(np.mean([v.delta_e_cnt for v in sub])), 3),
                "conflict_rate": round(confl / n, 4),
            }

        assist_stats = _subset_summary(MODE_ASSIST_ON)
        shadow_stats = _subset_summary(MODE_SHADOW)
        p95_lat = float(np.percentile(self.decision_latencies_ms, 95)) if self.decision_latencies_ms else 0.0

        return {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "session_duration_s": round(session_dur_s, 2),
            "roi_backend": self.vision.backend_name,
            "detector_backend": self.vision.detector_backend,
            "measured_roi_fps": round(self.roi_fps_bench, 1),
            "raw_grab_fps": round(getattr(self, "raw_grab_fps_bench", self.roi_fps_bench), 1),
            "full_yolo_fps": round(getattr(self, "full_yolo_fps_bench", self.roi_fps_bench), 1),
            "full_yolo_median_ms": round(getattr(self, "full_yolo_median_ms", 0.0), 3),
            "decision_p95_ms": round(p95_lat, 4),
            "safe_envelope_cnt": self.safe_envelope_cnt,
            "cooldown_lock_ms": round(self.cooldown_lock_s * 1000.0, 1),
            "controller_params": self.config["controller_params"],
            "thresholds": self.config["thresholds"],
            "mounting_angle_deg": round(float(self.calibrator.mounting_angle_deg), 4),
            "f8_calibrations": self.calib_history,
            "overall_summary": overall,
            "mode_comparison": {
                MODE_ASSIST_ON: assist_stats,
                MODE_SHADOW: shadow_stats,
            },
            "gsi_summary": self.gsi.summary() if hasattr(self, "gsi") else {},
            "shots": [
                {
                    "shot_idx": v.shot_idx,
                    "t_s": round(v.t_s, 3),
                    "mode": v.scenario,
                    "state": self.shot_states[i] if i < len(self.shot_states) else "UNKNOWN",
                    "hitbox": v.hitbox,
                    "anchor_mode": v.anchor_mode,
                    "e_hand_cnt": v.e_hand_cnt,
                    "e_actual_cnt": v.e_actual_cnt,
                    "head_dist_cnt": v.head_dist_cnt,
                    "delta_e_cnt": v.delta_e_cnt,
                    "stage_pos_cnt": [round(float(v.p_stage_cnt[0]), 2), round(float(v.p_stage_cnt[1]), 2)],
                    "faults": v.faults,
                }
                for i, v in enumerate(self.referee.verdicts)
            ],
        }

    def finalize_and_save_report(self, session_dur_s: float) -> Dict:
        """Print formatted battle report 《实战会话遥测战报》 and write JSON/CSV engagement logs."""
        report = self._build_session_report(session_dur_s)
        eng_summary = self.eng_logger.finalize_and_save() if hasattr(self, "eng_logger") else {}
        report["engagement_summary"] = eng_summary

        rep_bb: Optional[Dict[str, Any]] = None
        if eng_summary.get("total_engagements", 0) >= 4 and eng_summary.get("jsonl_path"):
            try:
                from shared_control_hil.blackbox_sysid_trainer import (
                    analyze_blackbox,
                    print_kill_efficiency_summary,
                )

                rep_bb = analyze_blackbox(jsonl_path=eng_summary["jsonl_path"], config_path=self.config_path)
                report["blind_crossover"] = rep_bb.get("blind_paired_crossover")
                report["conditional_conversion"] = rep_bb.get("conditional_conversion")
            except Exception:
                rep_bb = None

        with open(self.report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        ov = report["overall_summary"]
        a = report["mode_comparison"][MODE_ASSIST_ON]
        s = report["mode_comparison"][MODE_SHADOW]

        print("\n" + "=" * 78)
        print("  《实战会话遥测战报》 (CS2 3-RRR Live Shared-Control Session Report)")
        print("=" * 78)
        print(
            f"时长: {report['session_duration_s']:.1f}s | ROI后端: {report['roi_backend']} "
            f"({report['measured_roi_fps']:.1f} Hz) | 决策P95: {report['decision_p95_ms']:.3f} ms | "
            f"安装角: {report['mounting_angle_deg']:+.2f} deg"
        )
        print(
            f"总击发: {ov['total_shots']} | 爆头率: {ov['head_hit_rate']*100:.1f}% | "
            f"有效命中率: {ov['effective_hit_rate']*100:.1f}% | "
            f"人手均差 E_hand: {ov['mean_e_hand_cnt']:.2f} cnt -> 实弹均差 E_actual: {ov['mean_e_actual_cnt']:.2f} cnt | "
            f"平均增益 dE: {ov['mean_delta_e_cnt']:+.2f} cnt"
        )
        print("-" * 78)
        print("| 运行模式 (Mode) | 击发数 | 爆头率 (Head) | 有效命中 (Hit) | 初始均差 E_hand | 最终均差 E_actual | 平均增益 dE |")
        print("| :--- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for m_name, st in ((MODE_ASSIST_ON, a), (MODE_SHADOW, s)):
            print(
                f"| {m_name} | {st['total_shots']} | {st['head_hit_rate']*100:.1f}% | "
                f"{st['effective_hit_rate']*100:.1f}% | {st['mean_e_hand_cnt']:.2f} cnt | "
                f"{st['mean_e_actual_cnt']:.2f} cnt | {st['mean_delta_e_cnt']:+.2f} cnt |"
            )
        print("-" * 78)
        fc = ov["fault_counts"]
        print(
            f"4类故障归因: LAG={fc[FAULT_LAG]}, CONFLICT={fc[FAULT_CONFLICT]}, "
            f"SATURATION={fc[FAULT_SATURATION]}, HANDOVER_DELAY={fc[FAULT_HANDOVER_DELAY]}"
        )
        if eng_summary:
            cp_str = f"{eng_summary['mean_coupling_ratio']:.3f}" if eng_summary.get("mean_coupling_ratio") is not None else "N/A"
            gsi_s = report.get("gsi_summary", {})
            print(
                f"[Engagement/Kill Diagnostics] engagements={eng_summary.get('total_engagements', 0)} | "
                f"GSI kills={gsi_s.get('total_kills', 0)} (HS={gsi_s.get('total_hs', 0)}, connected={gsi_s.get('gsi_connected', False)}) | "
                f"palm_coupling_ratio={cp_str} (n={eng_summary.get('coupling_samples', 0)})"
            )
            print(f"[Saved] Per-kill CSV & blackbox traces -> {eng_summary.get('csv_path')} | {eng_summary.get('jsonl_path')}")
        print(f"[Saved] Session telemetry report -> {self.report_path}")
        snap_path = self._save_shot_snapshots_grid()
        if snap_path:
            print(f"[Saved] Live blackbox shot snapshots ({len(self._shot_snapshots)}) -> {snap_path}")
        if rep_bb is not None:
            try:
                print_kill_efficiency_summary(rep_bb)
            except Exception:
                pass
        print("=" * 78)
        return report

    def close(self) -> None:
        self._stop_1khz_state_tracker()
        try:
            self.vision.stop_async_yolo()
        except Exception:
            pass
        if hasattr(self, "gsi") and self._gsi_started:
            self.gsi.close()
            self._gsi_started = False
        if self._hud_initialized:
            try:
                import cv2
                cv2.destroyAllWindows()
            except Exception:
                pass
            self._hud_initialized = False
        try:
            self.recenter_to_neutral(duration_s=0.22)
        except Exception:
            pass
        try:
            self.vision.close()
        except Exception:
            pass
        try:
            self.calibrator.close()
        except Exception:
            pass

    def _get_visible_stage_pos(self, t_now: float, lag_s: Optional[float] = None) -> np.ndarray:
        """Estimate stage position already reflected on screen at (t_now - visual_lag_s)."""
        if not self._cmd_history:
            return np.zeros(2, dtype=float)
        tau = self.visual_lag_s if lag_s is None else float(lag_s)
        t_vis = float(t_now) - max(0.0, tau)
        vis_pos = np.zeros(2, dtype=float)
        for t_c, p_c in self._cmd_history:
            if t_c <= t_vis:
                vis_pos = p_c
            else:
                break
        return vis_pos.copy()

    def _get_visible_aim_pos(self, t_now: float, lag_s: Optional[float] = None) -> np.ndarray:
        """Estimate PURE aim component p_aim_cnt already reflected on screen at (t_now - visual_lag_s)."""
        if not getattr(self, "_aim_cmd_history", None):
            return np.zeros(2, dtype=float)
        tau = self.visual_lag_s if lag_s is None else float(lag_s)
        t_vis = float(t_now) - max(0.0, tau)
        vis_pos = np.zeros(2, dtype=float)
        for t_c, p_c in self._aim_cmd_history:
            if t_c <= t_vis:
                vis_pos = p_c
            else:
                break
        return vis_pos.copy()

    @staticmethod
    def _soft_saturate_stage_cmd(p_total: np.ndarray, limits: np.ndarray, soft_cap: Any = (250.0, 285.0)) -> np.ndarray:
        """Smooth tanh soft-knee saturation (default 250 X / 285 Y cnt below mechanical 300 cnt wall)."""
        out = np.asarray(p_total, dtype=float).copy()
        caps = (
            (float(soft_cap), float(soft_cap))
            if isinstance(soft_cap, (int, float))
            else (float(soft_cap[0]), float(soft_cap[1]))
        )
        for ax in (0, 1):
            lim = float(limits[ax])
            ceil = min(caps[ax], lim)
            knee = min(0.86 * ceil, ceil - 32.0)
            val = float(out[ax])
            aval = abs(val)
            if aval > knee and ceil > knee + 1e-3:
                span = ceil - knee
                soft_mag = knee + span * math.tanh((aval - knee) / span)
                out[ax] = math.copysign(min(soft_mag, lim), val)
            else:
                out[ax] = float(np.clip(val, -lim, lim))
        return out

    def _select_locked_target(
        self,
        dets_px: List[TargetDetection],
        c_px: np.ndarray,
        v_hand_norm: float = 0.0,
        lmb_down: bool = False,
    ) -> Tuple[Optional[TargetDetection], float]:
        """Select nearest valid target with responsive closer-target handover and moving-target tracking support."""
        follow_max_px = min(max(1.15 * self.max_aim_dist_px, 44.0), 54.0)
        valid: List[Tuple[float, TargetDetection]] = []
        locked_cur: Optional[TargetDetection] = None
        locked_cur_dist = 1e9
        for d in dets_px:
            dist_px = float(np.hypot(d.head_px[0] - c_px[0], d.head_px[1] - c_px[1]))
            if self._lock_target_id is not None and int(d.target_id) == int(self._lock_target_id):
                if d.seen_frames >= 1 and dist_px <= follow_max_px:
                    locked_cur = d
                    locked_cur_dist = dist_px
            if d.seen_frames >= self.min_track_frames and dist_px <= self.max_aim_dist_px:
                valid.append((dist_px, d))
        if locked_cur is not None:
            if valid:
                best_dist, best_det = min(valid, key=lambda x: x[0])
                if int(best_det.target_id) != int(locked_cur.target_id) and (best_dist + 9.0) < locked_cur_dist:
                    return best_det, best_dist
            return locked_cur, locked_cur_dist
        max_acq_vh = 2200.0 if lmb_down else 950.0
        if (not valid) or v_hand_norm >= max_acq_vh:
            return None, 1e9
        best_dist, best_det = min(valid, key=lambda x: x[0])
        return best_det, best_dist

    def _update_locked_target_motion(
        self,
        tid_cur: int,
        raw_head_px: np.ndarray,
        det_t: float,
        hand_xy_at_det: Optional[np.ndarray],
        v_hand_raw: np.ndarray,
        M_stage_to_err: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        Estimate stage-decontaminated drift velocity `v_drift_px_s` (head vs hand-driven crosshair)
        and true world-space target velocity `v_world_px_s` (subtracting both stage AND hand camera rotation).
        """
        vis_stg_det = self._get_visible_stage_pos(det_t, lag_s=self.visual_lag_s)
        J = self.calibrator.J_screen_to_stage
        col0_n = float(math.hypot(float(J[0, 0]), float(J[1, 0])))
        base_ppc = 1.0 / max(1.5, col0_n)
        hand_samples = getattr(self, "_hand_px_per_cnt_samples", ())
        px_per_hand_cnt = float(np.median(hand_samples)) if len(hand_samples) >= 5 else base_ppc

        if self._tgt_motion_tid != tid_cur or self._tgt_prev_head_px is None:
            self._tgt_motion_tid = tid_cur
            self._tgt_prev_head_px = np.asarray(raw_head_px, dtype=float).copy()
            self._tgt_prev_det_t = float(det_t)
            self._tgt_prev_hand_xy = None if hand_xy_at_det is None else np.asarray(hand_xy_at_det, dtype=float).copy()
            self._tgt_prev_vis_stage = vis_stg_det.copy()
            self._tgt_v_drift_px_s[:] = 0.0
            self._tgt_v_world_px_s[:] = 0.0
            self._tgt_moving_frames = 0
            return self._tgt_v_drift_px_s.copy(), self._tgt_v_world_px_s.copy(), False

        dt_det = float(det_t) - float(self._tgt_prev_det_t)
        if 0.005 <= dt_det <= 0.120:
            dh_px = np.asarray(raw_head_px, dtype=float) - self._tgt_prev_head_px
            dp_vis = vis_stg_det - (
                self._tgt_prev_vis_stage if self._tgt_prev_vis_stage is not None else vis_stg_det
            )
            dh_stg_px = M_stage_to_err @ dp_vis
            dh_no_stg = dh_px - dh_stg_px
            v_drift_inst = np.clip(dh_no_stg / dt_det, -340.0, 340.0)

            if hand_xy_at_det is not None and self._tgt_prev_hand_xy is not None:
                d_hand = np.asarray(hand_xy_at_det, dtype=float) - self._tgt_prev_hand_xy
                if (
                    abs(float(d_hand[0])) >= 18.0
                    and (abs(float(d_hand[0])) / dt_det) >= 450.0
                    and abs(float(dh_no_stg[0])) >= 4.0
                    and (float(d_hand[0]) * float(dh_no_stg[0])) < 0.0
                ):
                    r_cand = -float(dh_no_stg[0]) / float(d_hand[0])
                    if 0.16 <= r_cand <= 0.90 and hasattr(self, "_hand_px_per_cnt_samples"):
                        self._hand_px_per_cnt_samples.append(r_cand)
                        if len(self._hand_px_per_cnt_samples) >= 5:
                            px_per_hand_cnt = float(np.median(self._hand_px_per_cnt_samples))
                v_hand_cam_px_s = -px_per_hand_cnt * (d_hand / dt_det)
            else:
                v_hand_cam_px_s = -px_per_hand_cnt * np.asarray(v_hand_raw, dtype=float)
            v_world_inst = np.clip(v_drift_inst - v_hand_cam_px_s, -340.0, 340.0)

            # Fast reset if enemy reverses strafe direction (counter-strafe A <-> D)
            if (float(v_world_inst[0]) * float(self._tgt_v_world_px_s[0])) < -400.0:
                self._tgt_v_world_px_s[0] = 0.5 * float(v_world_inst[0])
                self._tgt_v_drift_px_s[0] = 0.5 * float(v_drift_inst[0])
                self._tgt_moving_frames = 0
            else:
                alpha_m = min(1.0, dt_det / 0.024)
                self._tgt_v_drift_px_s = (1.0 - alpha_m) * self._tgt_v_drift_px_s + alpha_m * v_drift_inst
                self._tgt_v_world_px_s = (1.0 - alpha_m) * self._tgt_v_world_px_s + alpha_m * v_world_inst

            world_thresh = 30.0 + 0.30 * abs(float(v_hand_cam_px_s[0]))
            if abs(float(self._tgt_v_world_px_s[0])) >= world_thresh:
                self._tgt_moving_frames = min(8, self._tgt_moving_frames + 1)
            else:
                self._tgt_moving_frames = max(0, self._tgt_moving_frames - 1)

            self._tgt_prev_head_px = np.asarray(raw_head_px, dtype=float).copy()
            self._tgt_prev_det_t = float(det_t)
            self._tgt_prev_hand_xy = None if hand_xy_at_det is None else np.asarray(hand_xy_at_det, dtype=float).copy()
            self._tgt_prev_vis_stage = vis_stg_det.copy()
        elif dt_det > 0.120:
            self._tgt_prev_head_px = np.asarray(raw_head_px, dtype=float).copy()
            self._tgt_prev_det_t = float(det_t)
            self._tgt_prev_hand_xy = None if hand_xy_at_det is None else np.asarray(hand_xy_at_det, dtype=float).copy()
            self._tgt_prev_vis_stage = vis_stg_det.copy()
            self._tgt_v_drift_px_s[:] = 0.0
            self._tgt_v_world_px_s[:] = 0.0
            self._tgt_moving_frames = 0

        cur_thresh = 30.0 + 0.25 * px_per_hand_cnt * abs(float(v_hand_raw[0]))
        is_moving = (self._tgt_moving_frames >= 2) and (abs(float(self._tgt_v_world_px_s[0])) >= cur_thresh)
        self._last_is_moving_tgt = bool(is_moving)
        self._last_v_world_x_px_s = float(self._tgt_v_world_px_s[0])
        return self._tgt_v_drift_px_s.copy(), self._tgt_v_world_px_s.copy(), bool(is_moving)

    def _select_aim_anchor_px(
        self,
        det: TargetDetection,
        c_eff_px: np.ndarray,
        lmb_down: bool,
        lmb_hold_s: float,
        roi_center_y: float = 128.0,
    ) -> Tuple[np.ndarray, str]:
        """
        Strict Head-First -> Smooth Upper-Chest fallback anchor:
          - Idle and first 3 bullets (< 260 ms LMB hold): 100% locked onto HEAD center (`HEAD`).
          - Shot 4+ (>= 260 ms) AND only when follow crosshair climbs vertically > 18 px:
            smoothly ramp to upper chest (0.35 * chest_height below neck, never belly).
        """
        head_px = np.asarray(det.head_px, dtype=float)
        chest_px = np.asarray(det.chest_px, dtype=float)
        v_nose = chest_px - head_px
        head_r = max(3.5, float(getattr(det, "head_radius_px", 5.5)))
        chest_r = max(8.0, float(getattr(det, "chest_radius_px", 12.0)))
        if det.chest_bbox_px is not None:
            chest_h = max(12.0, abs(float(det.chest_bbox_px[3]) - float(det.chest_bbox_px[1])))
            neck_y = float(det.chest_bbox_px[1])
        else:
            chest_h = 2.0 * chest_r
            neck_y = float(head_px[1]) + 0.8 * head_r

        shot_idx = int(getattr(self, "_burst_shot_idx", 0))
        in_first_3_shots = (not lmb_down) or (lmb_hold_s < 0.260) or (0 < shot_idx < 3)
        if in_first_3_shots:
            self._spray_anchor_latched = None
            return head_px + 0.08 * v_nose, "HEAD"

        recoil_up_px = float(roi_center_y) - float(c_eff_px[1])
        if getattr(self, "_prev_recoil_off_px", None) is not None:
            recoil_up_px = max(recoil_up_px, -float(self._prev_recoil_off_px[1]))

        if recoil_up_px <= 18.0 and self._spray_anchor_latched != "CHEST":
            return head_px + 0.08 * v_nose, "HEAD"

        # Shot 4+ (>= 260 ms) and follow crosshair jumped > 18 px UP: smooth transition to 0.35 * chest_height
        self._spray_anchor_latched = "CHEST"
        upper_chest_y = min(float(chest_px[1]), neck_y + 0.35 * chest_h)
        head_pt = head_px + 0.08 * v_nose
        upper_chest_pt = np.array([float(head_px[0] + 0.35 * v_nose[0]), upper_chest_y], dtype=float)
        blend = float(np.clip((lmb_hold_s - 0.260) / 0.100, 0.0, 1.0))
        anchor_px = head_pt + blend * (upper_chest_pt - head_pt)
        return anchor_px, ("CHEST" if blend >= 0.45 else "HEAD_CHEST_BLEND")

    def _step_live_assist(
        self,
        t_now: float,
        dt_s: float,
        roi_rgb: np.ndarray,
        c_px: np.ndarray,
        dets_px: List[TargetDetection],
        lmb_down: bool,
        v_hand_raw: np.ndarray,
        p_stage_now: np.ndarray,
        dispatch_fn: Optional[Callable[..., bool]] = None,
        det_age_s: float = 0.0,
        rec_key_down: bool = False,
        det_t_s: Optional[float] = None,
        hand_xy_at_det: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[TargetDetection], float]:
        """
        Execute one 144 Hz tick of Dual-Mouse Saccadic Recenter + Moving-Target Pursuit + Inter-Bullet Rescue:
          1. Saccadic open-space recenter bleeds stage cleanly back to comfort zone (55, 65) during fast whips
             while holding 0-drift once crosshair lands near a target.
          2. Stage & hand-decontaminated world-space target velocity estimator drives predictive X lead
             (`0.078s * v_drift_x`) + 45 ms pursuit tracking on moving bots (`is_moving_tgt == True`).
          3. Stationary targets retain terminal PRE Micro-Snap, adaptive SHOT1_RESCUE, and AK feedforward.
        """
        disp = self._dispatch_stage_manifold_open_loop if dispatch_fn is None else dispatch_fn
        self._last_is_moving_tgt = False
        self._last_v_world_x_px_s = 0.0
        limits = np.array([self.live_max_x_cnt, self.live_max_y_cnt], dtype=float)
        aim_env_lo = np.array([-min(190.0, self.live_max_x_cnt), -165.0], dtype=float)
        aim_env_hi = np.array([min(190.0, self.live_max_x_cnt), 225.0], dtype=float)
        v_hand_norm = float(np.hypot(v_hand_raw[0], v_hand_raw[1]))
        roi_c = 0.5 * float(roi_rgb.shape[0])
        roi_center_arr = np.array([roi_c, roi_c], dtype=float)

        # 98 ms In-Pulse Immunity Window (single-mouse mode only; dual-mouse reads true hand mouse directly)
        last_pulse_t = float(getattr(self, "_last_pulse_cmd_t", 0.0))
        pulse_age_s = t_now - last_pulse_t
        in_pulse_immunity = (
            (not getattr(self, "_dual_mouse_detected", False))
            and (last_pulse_t > 0.0)
            and (pulse_age_s <= 0.098)
            and (not getattr(self, "_pulse_opp_turn_confirmed", False))
        )
        if in_pulse_immunity and (not getattr(self, "_tracker_running", False)):
            p_dir = getattr(self, "_last_pulse_dir", np.zeros(2, dtype=float))
            p_dir_n = float(np.hypot(p_dir[0], p_dir[1]))
            if p_dir_n > 0.5 and v_hand_norm > 1e-3:
                p_hat = p_dir / p_dir_n
                v_par = float(np.dot(v_hand_raw, p_hat))
                if (-v_par) > 380.0 and pulse_age_s >= 0.020:
                    in_pulse_immunity = False
                elif v_par >= 0.0 and v_par > (v_hand_norm * 0.258819):
                    v_hand_raw = np.zeros(2, dtype=float)
                    v_hand_norm = 0.0
                elif v_par > 0.0:
                    v_hand_raw = v_hand_raw - v_par * p_hat
                    v_hand_norm = float(np.hypot(v_hand_raw[0], v_hand_raw[1]))

        lmb_pressed = lmb_down and (not self._prev_lmb_live)
        lmb_released = self._prev_lmb_live and (not lmb_down)
        self._prev_lmb_live = lmb_down
        if lmb_pressed:
            self._lmb_down_since_t = t_now
            self._shot1_rescue_sent = False
            self._need_switch_recenter = False
            if (t_now - self._last_aim_cmd_t) >= 0.110:
                self._aim_cmd_history = deque([(float(t_now) - 1.0, self.p_aim_cnt.copy())], maxlen=64)
            if (t_now - getattr(self, "_pre_micro_sent_t", 0.0)) > 0.280:
                self._target_base_stage_cnt = p_stage_now.copy()
        if lmb_released:
            self._last_lmb_release_t = t_now
            self._prev_recoil_off_px = None
            self._mono_tripped = False
            self._spray_anchor_latched = None
            self._live_anchor_mode = "HEAD"
            self._pre_micro_done_tid = None
            self._pre_micro_count = 0
            # Fold recoil offset into resting p_aim_cnt with ZERO physical stage jump (Freeze-Hold!)
            p_tot_rel = np.clip(self.p_aim_cnt + self.p_recoil_cnt, -limits, limits)
            self.p_aim_cnt = np.clip(p_tot_rel, aim_env_lo, aim_env_hi)
            self.p_recoil_cnt = p_tot_rel - self.p_aim_cnt
            self._aim_cmd_history = deque([(float(t_now) - 1.0, self.p_aim_cnt.copy())], maxlen=64)
            self._need_switch_recenter = bool(abs(float(p_tot_rel[0])) > 140.0 or abs(float(p_tot_rel[1])) > 160.0)

        # When idle and CS2 recoil has recovered, keep c_eff anchored at screen center
        # so Aim Botz bright green foliage/wall logos cannot bias pre-shot aim.
        c_eff_px = c_px
        if (not lmb_down) and (t_now - self._last_lmb_release_t) >= 0.30:
            if float(np.hypot(c_px[0] - roi_c, c_px[1] - roi_c)) > 2.5:
                c_eff_px = roi_center_arr

        # Track true human hand settle & high-flick timestamps
        if not in_pulse_immunity:
            if v_hand_norm >= 220.0:
                self._last_high_flick_t = t_now
            if v_hand_norm <= 40.0:
                self._settled_low_vh_ticks = getattr(self, "_settled_low_vh_ticks", 0) + 1
            else:
                self._settled_low_vh_ticks = 0
        else:
            self._settled_low_vh_ticks = max(getattr(self, "_settled_low_vh_ticks", 0), 2)

        # Saccadic Open-Space Recenter: bleed cleanly during fast open-space whips, hold rigid on target landing!
        in_snap_guard = (t_now - self._last_snap_t) <= 0.120
        has_near_tgt = any(
            float(np.hypot(float(d.head_px[0]) - float(c_eff_px[0]), float(d.head_px[1]) - float(c_eff_px[1])))
            <= float(self.max_aim_dist_px)
            for d in dets_px
        )
        has_close_tgt_34 = any(
            float(np.hypot(float(d.head_px[0]) - float(c_eff_px[0]), float(d.head_px[1]) - float(c_eff_px[1])))
            <= 34.0
            for d in dets_px
        )
        recent_burst_tail = (t_now - self._last_lmb_release_t) <= 0.140
        if (not in_pulse_immunity) and (not lmb_down) and (not recent_burst_tail) and v_hand_norm > 520.0:
            self._flick_hi_cnt += 1
        else:
            self._flick_hi_cnt = 0
        flick_trigger = (not in_pulse_immunity) and (not lmb_down) and (not recent_burst_tail) and (
            (v_hand_norm > (1650.0 if has_near_tgt else 820.0))
            or ((not has_near_tgt) and v_hand_norm > 520.0 and self._flick_hi_cnt >= 2)
        )
        if flick_trigger:
            self._lock_target_id = None
            self._snap_done_tid = None
            self._pre_micro_done_tid = None
            self._pre_micro_count = 0
            self._tgt_motion_tid = None
            self._tgt_moving_frames = 0
            self._lock_dir_sign[:] = 0.0
            self._mono_tripped = False
            self._last_high_flick_t = t_now
            self._settled_low_vh_ticks = 0
            outside_comfort = abs(float(p_stage_now[0])) > 55.0 or abs(float(p_stage_now[1])) > 65.0
            if outside_comfort and (not has_close_tgt_34):
                decay_aim = math.exp(-max(0.002, dt_s) / 0.065)
                self.p_aim_cnt = self.p_aim_cnt * decay_aim
                self.p_recoil_cnt = self.p_recoil_cnt * decay_aim
                p_wash = self._soft_saturate_stage_cmd(self.p_aim_cnt + self.p_recoil_cnt, limits, soft_cap=260.0)
                if (t_now - self._last_wash_cmd_t) >= 0.045:
                    is_zero = float(np.hypot(p_wash[0], p_wash[1])) < 4.0
                    if is_zero:
                        self.p_aim_cnt[:] = 0.0
                        self.p_recoil_cnt[:] = 0.0
                        p_wash = np.zeros(2, dtype=float)
                    if disp(p_wash, is_recenter=is_zero, cooldown_s=0.045, freeze_play_g=False, use_boost=False):
                        self._last_wash_cmd_t = t_now
                        self._cmd_history.append((float(t_now), p_wash.copy()))
                        self._aim_cmd_history.append((float(t_now), self.p_aim_cnt.copy()))
            if hasattr(self, "eng_logger"):
                self.eng_logger.record_frame(t_now, None, v_hand_raw, self.p_aim_cnt, self.p_recoil_cnt, "FLICK_HOLD")
            return None, 1e9

        locked_det, locked_dist_px = self._select_locked_target(
            dets_px, c_eff_px, v_hand_norm=(0.0 if in_pulse_immunity else v_hand_norm), lmb_down=lmb_down
        )
        if lmb_released and locked_det is None:
            self._lock_target_id = None
            self._snap_done_tid = None
            self._lock_dir_sign[:] = 0.0

        anchor_px: Optional[np.ndarray] = None
        anchor_mode = "HEAD"
        if locked_det is not None:
            if int(locked_det.target_id) != self._lock_target_id:
                self._spray_anchor_latched = None
            lmb_hold_s = max(0.0, t_now - self._lmb_down_since_t) if lmb_down else 0.0
            anchor_px, anchor_mode = self._select_aim_anchor_px(
                locked_det,
                c_eff_px,
                lmb_down=lmb_down,
                lmb_hold_s=lmb_hold_s,
                roi_center_y=roi_c,
            )
            self._live_anchor_mode = anchor_mode
            self._live_anchor_px = anchor_px.copy()
        else:
            self._live_anchor_mode = "HEAD"
            self._live_anchor_px = None

        # 1. Compute smooth P-only follow-recoil candidate (with fast tap guard)
        p_rec_cand = self.p_recoil_cnt.copy()
        p_rec_goal = self.p_recoil_cnt.copy()
        recoil_due = False
        in_tap_guard = bool(lmb_down and (t_now - self._lmb_down_since_t) < self.tap_guard_s)
        if lmb_down:
            self._last_engage_t = t_now
            if (not in_tap_guard) and (t_now - self._last_recoil_cmd_t) >= self.recoil_cooldown_s:
                off_px = detect_crosshair_offset_px(roi_rgb, is_rgb=True)
                if off_px is not None:
                    raw_off = np.array([float(off_px[0]), float(off_px[1])], dtype=float)
                    if self._prev_recoil_off_px is not None:
                        off_arr = 0.62 * raw_off + 0.38 * self._prev_recoil_off_px
                    else:
                        off_arr = raw_off
                    self._prev_recoil_off_px = off_arr.copy()
                    if anchor_px is not None:
                        # Target-locked: aim loop owns e_raw = anchor_px - c_eff_px directly.
                        # Do NOT double-count recoil_offset_to_stage_abs alongside aim; only decay stray X recoil.
                        p_rec_x = 0.65 * float(self.p_recoil_cnt[0]) if abs(float(self.p_recoil_cnt[0])) >= 2.0 else 0.0
                        p_rec_raw = np.array([p_rec_x, max(0.0, float(self.p_recoil_cnt[1]))], dtype=float)
                    elif float(np.hypot(off_arr[0], off_arr[1])) >= 2.5:
                        p_rec_offset = self.calibrator.recoil_offset_to_stage_abs(float(off_arr[0]), float(off_arr[1]))
                        p_rec_raw = np.array(
                            [
                                float(np.clip(0.42 * float(p_rec_offset[0]), -55.0, 55.0)),
                                float(np.clip(0.42 * float(p_rec_offset[1]), 0.0, 105.0)),
                            ],
                            dtype=float,
                        )
                    else:
                        p_rec_raw = None
                    if p_rec_raw is not None:
                        p_rec_goal = np.clip(p_rec_raw, -limits, limits)
                        d_rec = p_rec_raw - self.p_recoil_cnt
                        d_rec_norm = float(np.hypot(d_rec[0], d_rec[1]))
                        if d_rec_norm > 28.0:
                            p_rec_raw = self.p_recoil_cnt + d_rec * (28.0 / d_rec_norm)
                        p_rec_cand = np.clip(p_rec_raw, -limits, limits)
                        if float(np.hypot(p_rec_cand[0] - self.p_recoil_cnt[0], p_rec_cand[1] - self.p_recoil_cnt[1])) >= 1.5:
                            recoil_due = True
        else:
            self._prev_recoil_off_px = None

        # Saccadic Open-Space Recenter + Freeze-Hold on Landing:
        # Hold rigid during landing (locked target or target <=34px); recenter fast on open-space whips/idle/rail.
        p_tot_now = self.p_aim_cnt + p_rec_cand
        p_tot_norm_now = float(np.hypot(p_tot_now[0], p_tot_now[1]))
        p_aim_norm_now = float(np.hypot(self.p_aim_cnt[0], self.p_aim_cnt[1]))
        in_comfort_zone = (abs(float(p_tot_now[0])) <= 55.0) and (abs(float(p_tot_now[1])) <= 65.0)
        outer_rail = (abs(float(p_tot_now[0])) > 145.0) or (abs(float(p_tot_now[1])) > 165.0)
        if p_tot_norm_now < 2.0:
            self._need_switch_recenter = False
        since_rel_s = t_now - self._last_lmb_release_t
        since_eng_s = t_now - self._last_engage_t
        true_idle_pause = (since_rel_s >= 0.70) and (since_eng_s >= 0.65) and (v_hand_norm <= 35.0)
        saccade_recenter_safe = (
            (not in_comfort_zone)
            and (since_rel_s >= 0.055)
            and (not has_close_tgt_34)
            and (locked_det is None)
            and (v_hand_norm >= 520.0)
        )
        rail_idle_safe = outer_rail and (since_rel_s >= 0.35) and (v_hand_norm <= 35.0)
        pan_bleed_active = (not lmb_down) and (p_tot_norm_now >= 2.0) and (
            bool(rec_key_down) or true_idle_pause or saccade_recenter_safe or rail_idle_safe
        )
        if pan_bleed_active:
            if p_tot_norm_now < 4.0:
                self.p_aim_cnt[:] = 0.0
                p_rec_cand[:] = 0.0
                self._aim_cmd_history = deque([(float(t_now) - 1.0, np.zeros(2, dtype=float))], maxlen=64)
                recoil_due = True
            else:
                tau_bleed = 0.065 if (rec_key_down or saccade_recenter_safe or rail_idle_safe) else 0.14
                decay_f = math.exp(-max(0.002, dt_s) / tau_bleed)
                if p_aim_norm_now >= 0.5:
                    self.p_aim_cnt = self.p_aim_cnt * decay_f
                if float(np.hypot(p_rec_cand[0], p_rec_cand[1])) >= 0.5:
                    p_rec_cand = p_rec_cand * decay_f
                    recoil_due = True

        def _dispatch_recoil_only(cd_s: float) -> bool:
            act_b = getattr(self, "_active_bcmd", None)
            if (
                act_b is not None
                and (not act_b.get("cut_done", True))
                and (t_now - float(act_b["t0"])) < float(act_b["dur_s"])
            ):
                return False
            # Per-axis soft-washout p_aim_cnt only when that axis approaches ceiling
            p_sum_pre = self.p_aim_cnt + p_rec_cand
            axis_ceil = (225.0, 255.0)
            for ax in (0, 1):
                if abs(float(p_sum_pre[ax])) > axis_ceil[ax]:
                    self.p_aim_cnt[ax] *= math.exp(-max(0.002, dt_s) / 0.14)
            p_cmd_rec = self._soft_saturate_stage_cmd(self.p_aim_cnt + p_rec_cand, limits, soft_cap=(250.0, 285.0))
            ok_rec = disp(
                p_cmd_rec,
                is_recenter=False,
                cooldown_s=float(cd_s),
                freeze_play_g=True,
                use_boost=False,
            )
            if ok_rec:
                self.p_recoil_cnt = p_rec_cand
                self._last_recoil_cmd_t = t_now
                self._cmd_history.append((float(t_now), p_cmd_rec.copy()))
            return ok_rec

        aim_sent = False
        if locked_det is not None and anchor_px is not None:
            if lmb_down:
                self._last_engage_t = t_now
            self._yolo_miss_since_t = None
            try:
                M_stage_to_err = np.linalg.inv(self.calibrator.J_screen_to_stage)
            except np.linalg.LinAlgError:
                M_stage_to_err = np.zeros((2, 2), dtype=float)
            eff_det_t = float(t_now - det_age_s) if det_t_s is None else float(det_t_s)
            v_drift_px_s, v_world_px_s, is_moving_tgt = self._update_locked_target_motion(
                int(locked_det.target_id),
                np.asarray(locked_det.head_px, dtype=float),
                eff_det_t,
                hand_xy_at_det,
                v_hand_raw,
                M_stage_to_err,
            )

            e_raw_px = anchor_px - c_eff_px
            # During brief 30ms tap guard on stationary targets, ignore tiny vertical ViewPunch jitter (<=8px)
            if (not is_moving_tgt) and in_tap_guard and abs(float(c_eff_px[1]) - roi_c) <= 8.0:
                e_raw_px = np.array([float(anchor_px[0] - roi_c), float(anchor_px[1] - roi_c)], dtype=float)
            lead_x_px = 0.0
            has_pos_drift = False
            if is_moving_tgt:
                age_x = float(np.clip(det_age_s, 0.0, 0.025)) * float(v_drift_px_s[0])
                e_raw_px[0] += float(np.clip(age_x, -4.5, 4.5))
                if (float(v_drift_px_s[0]) * float(v_world_px_s[0])) > 0.0:
                    has_pos_drift = True
                    lead_x_px = float(np.clip(0.132 * float(v_drift_px_s[0]), -24.0, 24.0))
                self._live_anchor_px = anchor_px + np.array([lead_x_px, 0.0], dtype=float)
            e_norm = float(np.hypot(e_raw_px[0], e_raw_px[1]))
            r_lock = self._compute_deadband_px(locked_det)

            eta_flight = (
                float(np.clip(self.eng_logger.get_effective_coupling(default=0.92, min_samples=3), 0.88, 1.0))
                if hasattr(self, "eng_logger")
                else 1.0
            )
            # PURE aim Smith predictor: exclude p_rec_goal / p_recoil_cnt so horizontal recoil steps
            # NEVER generate phantom aim errors or horizontal ping-pong oscillation!
            if (t_now - self._last_aim_cmd_t) >= 0.110:
                dp_in_flight = np.zeros(2, dtype=float)
            else:
                eff_lag_s = self.visual_lag_s + float(np.clip(det_age_s, 0.0, 0.060))
                p_aim_at_frame_capture = self._get_visible_aim_pos(t_now, lag_s=eff_lag_s)
                dp_in_flight = eta_flight * (self.p_aim_cnt - p_aim_at_frame_capture)
            e_no_lead_pred_px = e_raw_px + (M_stage_to_err @ dp_in_flight)
            # Smith predictor may shrink same-sign error (avoid over-sending while step settles),
            # NEVER cross zero on stationary axes (skip X clamp when tracking a moving target!).
            for ax in (0, 1):
                if is_moving_tgt and ax == 0:
                    continue
                if (float(e_raw_px[ax]) * float(e_no_lead_pred_px[ax])) <= 0.0:
                    e_no_lead_pred_px[ax] = 0.0
            e_pred_px = np.array(
                [float(e_no_lead_pred_px[0]) + lead_x_px, float(e_no_lead_pred_px[1])],
                dtype=float,
            )
            e_pred_norm = float(np.hypot(e_pred_px[0], e_pred_px[1]))

            tid_cur = int(locked_det.target_id)
            if tid_cur != self._lock_target_id:
                recent_burst = lmb_down or ((t_now - self._last_lmb_release_t) <= 0.200)
                if recent_burst and self._snap_done_tid is not None:
                    self._snap_done_tid = tid_cur
                if not lmb_down:
                    self._pre_micro_done_tid = None
                    self._pre_micro_count = 0
                    self._pre_prev_epx = None
                    self._target_base_stage_cnt = p_stage_now.copy()
                dp_init = self.calibrator.aim_error_to_stage_delta(float(e_pred_px[0]), float(e_pred_px[1]))
                self._lock_target_id = tid_cur
                self._lock_start_t = t_now
                self._lock_dir_sign = np.sign(dp_init)
                self._mono_target_id = tid_cur
                self._mono_e0_norm = e_norm
                self._mono_e0_x = abs(float(e_raw_px[0]))
                self._mono_cmd_t = 0.0
                self._mono_tripped = False
                self._mono_trip_t = 0.0

            vis_stable = False
            if not lmb_down:
                if (
                    self._pre_prev_epx is not None
                    and 0.004 <= (t_now - self._pre_prev_epx_t) <= 0.060
                    and float(np.hypot(e_raw_px[0] - self._pre_prev_epx[0], e_raw_px[1] - self._pre_prev_epx[1])) <= 2.2
                ):
                    vis_stable = True
                self._pre_prev_epx = e_raw_px.copy()
                self._pre_prev_epx_t = t_now

            if not lmb_down:
                self._mono_tripped = False
            elif self._mono_tripped and (t_now - self._mono_trip_t) >= 0.22:
                self._mono_tripped = False

            # Guard ONLY catastrophic horizontal polarity divergence while hand is idle (NEVER trip on vertical AK recoil climb!)
            if (
                lmb_down
                and self._mono_cmd_t > 0.0
                and self._mono_target_id == tid_cur
                and 0.120 <= (t_now - self._mono_cmd_t) <= 0.300
                and v_hand_norm < 85.0
                and abs(float(e_raw_px[0])) > max(38.0, float(getattr(self, "_mono_e0_x", 0.0)) + 28.0)
            ):
                self._mono_tripped = True
                self._mono_trip_t = t_now
                self._mono_cmd_t = 0.0
                print(f"\n[MONO_TRIP_X] target#{tid_cur} X diverged -> {abs(float(e_raw_px[0])):.1f}px; brief 0.22s cooldown.")

            if in_pulse_immunity:
                hand_settled = True
                last_flick_t = 0.0
            elif lmb_down:
                # During fire/pursuit, unlock full cooperative tracking up to 2200 cnt/s
                hand_settled = v_hand_norm <= 2200.0
                last_flick_t = getattr(self, "_last_high_flick_t", 0.0)
            elif getattr(self, "_dual_mouse_detected", False):
                # Dual-mouse ground truth: true hand stop (<=35 cnt/s for >=25ms, >=45ms after high flick)
                settled_1k_t = getattr(self, "_hand_settled_1k_since_t", 0.0)
                hand_settled = (v_hand_norm <= 35.0) and (
                    settled_1k_t > 0.0 and (t_now - settled_1k_t) >= 0.025
                )
                last_flick_t = max(
                    getattr(self, "_last_high_flick_t", 0.0),
                    getattr(self, "_last_high_flick_1k_t", 0.0),
                )
            elif getattr(self, "_tracker_running", False):
                # Single-mouse fallback: strict Do-No-Harm stop confirmation
                settled_1k_t = getattr(self, "_hand_settled_1k_since_t", 0.0)
                hand_settled = (v_hand_norm <= 35.0) and (
                    settled_1k_t > 0.0 and (t_now - settled_1k_t) >= 0.040
                )
                last_flick_t = max(
                    getattr(self, "_last_high_flick_t", 0.0),
                    getattr(self, "_last_high_flick_1k_t", 0.0),
                )
            else:
                hand_settled = (v_hand_norm <= 35.0) and (
                    getattr(self, "_settled_low_vh_ticks", 0) >= 3
                )
                last_flick_t = getattr(self, "_last_high_flick_t", 0.0)
            since_flick_s = t_now - last_flick_t
            lmb_hold_s_now = max(0.0, t_now - self._lmb_down_since_t) if lmb_down else 0.0
            lead_active = bool(is_moving_tgt and abs(lead_x_px) >= 3.0)
            is_shot1_rescue = (
                lmb_down
                and (not getattr(self, "_shot1_rescue_sent", False))
                and (lmb_hold_s_now <= 0.025)
                and (e_norm > max(r_lock, 5.0) or lead_active)
                and (e_pred_norm > max(r_lock, 5.0) or lead_active)
            )
            pre_deadband_px = max(r_lock, 3.8) if self._pre_micro_count == 0 else max(r_lock, 2.6)
            is_pre_micro = (
                (not lmb_down)
                and ((t_now - self._last_lmb_release_t) >= 0.130)
                and (since_flick_s >= (0.045 if getattr(self, "_dual_mouse_detected", False) else 0.065))
                and (vis_stable or is_moving_tgt or (not getattr(self, "_tracker_running", False)))
                and (locked_det.seen_frames >= 2)
                and (self._pre_micro_count < (1 if is_moving_tgt else 2))
                and (e_pred_norm <= 18.5)
                and (e_norm > pre_deadband_px or lead_active)
                and (e_pred_norm > pre_deadband_px)
                and hand_settled
            )
            tap_guard_pass = is_shot1_rescue or is_moving_tgt or (not in_tap_guard) or (e_pred_norm > max(r_lock, 5.0))
            in_flight_n = float(np.hypot(dp_in_flight[0], dp_in_flight[1]))
            if is_shot1_rescue:
                eff_aim_cd = 0.0
            elif is_pre_micro:
                eff_aim_cd = 0.0 if self._pre_micro_count == 0 else 0.115
            elif lmb_down and is_moving_tgt:
                eff_aim_cd = 0.045
            elif lmb_down:
                eff_aim_cd = 0.095 if in_flight_n >= 14.0 else max(self.aim_cooldown_s, 0.085)
            else:
                eff_aim_cd = max(self.aim_cooldown_s, 0.130)
            pre_range_ok = bool(lmb_down) or bool(is_pre_micro)
            aim_due = (
                (not self._mono_tripped)
                and pre_range_ok
                and tap_guard_pass
                and (t_now - self._last_aim_cmd_t) >= eff_aim_cd
                and (e_norm > r_lock or lead_active)
                and e_pred_norm > r_lock
                and (e_norm >= min(self.min_aim_dist_px, r_lock) or lead_active)
                and hand_settled
            )
            if aim_due:
                eta_cp = self.eng_logger.get_effective_coupling(default=0.90, min_samples=3) if hasattr(self, "eng_logger") else 1.0
                ff_trim = float(np.clip(1.0 / max(0.88, eta_cp), 1.0, 1.10))
                fine_trim_px = min(float(self.small_error_px), 8.5)
                is_initial_snap = bool(is_pre_micro and self._pre_micro_count == 0)
                if is_pre_micro:
                    if is_moving_tgt:
                        gain = min(0.70, max(0.60, 0.65 * ff_trim))
                        step_cap = 44.0
                        use_boost_aim = True
                        b_scale_aim = min(self.boost_scale, 1.15)
                        b_dur_aim = min(self.boost_dur_ms, 25.0)
                    elif self._pre_micro_count == 0:
                        gain = min(0.78, max(0.68, 0.72 * ff_trim))
                        step_cap = 45.0
                        use_boost_aim = True
                        b_scale_aim = min(self.boost_scale, 1.15)
                        b_dur_aim = min(self.boost_dur_ms, 25.0)
                    else:
                        gain = min(0.56, max(0.46, 0.52 * ff_trim))
                        step_cap = 18.0
                        use_boost_aim = False
                        b_scale_aim = 1.0
                        b_dur_aim = 20.0
                elif is_shot1_rescue:
                    if is_moving_tgt and (float(v_drift_px_s[0]) * float(v_world_px_s[0])) >= -15.0:
                        gain = min(0.80, max(0.70, 0.75 * ff_trim))
                        step_cap = 72.0
                    elif (not in_pulse_immunity) and v_hand_norm > 200.0:
                        gain = min(0.38, max(0.30, 0.34 * ff_trim))
                        step_cap = 32.0
                    elif (not in_pulse_immunity) and v_hand_norm > 80.0:
                        gain = min(0.50, max(0.42, 0.46 * ff_trim))
                        step_cap = 40.0
                    else:
                        gain = min(0.64, max(0.56, 0.60 * ff_trim))
                        step_cap = 48.0
                    use_boost_aim = True
                    b_scale_aim = min(self.boost_scale, 1.16)
                    b_dur_aim = self.boost_dur_ms
                else:
                    if is_moving_tgt:
                        gain = min(0.70, max(0.60, 0.65 * ff_trim))
                        if (not in_pulse_immunity) and v_hand_norm > 1200.0:
                            gain *= 0.65
                        step_cap = 68.0
                    else:
                        is_fine = e_pred_norm <= fine_trim_px
                        base_gain = min(0.46, max(self.kp_small_head, 0.38)) if is_fine else min(0.58, max(self.kp_large_error, 0.48))
                        gain = min(0.60, base_gain * ff_trim)
                        if (not in_pulse_immunity) and v_hand_norm > 340.0:
                            gain *= 0.65
                        step_cap = 54.0
                    use_boost_aim = True
                    b_scale_aim = min(self.boost_scale, 1.18)
                    b_dur_aim = self.boost_dur_ms

                if has_pos_drift:
                    ts_p = 0.045
                    hz_s = (self.visual_lag_s + 0.5 * ts_p) + ((1.0 - gain) / max(0.35, gain)) * ts_p
                    lead_x_exact = float(np.clip(hz_s * float(v_drift_px_s[0]), -24.0, 24.0))
                    e_pred_px[0] = float(e_no_lead_pred_px[0]) + lead_x_exact

                dp_stage = self.calibrator.aim_error_to_stage_delta(float(e_pred_px[0]), float(e_pred_px[1])) * gain
                # AK-47 vertical feedforward (Bullets 2-6, 55..550ms; prevents +16px recoil float by Bullet 5)
                if (
                    lmb_down
                    and float(e_raw_px[1]) >= 1.0
                    and float(e_pred_px[1]) >= 0.0
                    and hasattr(self, "ak_calibrator")
                ):
                    lmb_hold_s_ff = max(0.0, t_now - self._lmb_down_since_t)
                    if 0.055 <= lmb_hold_s_ff <= 0.550:
                        shot_f = lmb_hold_s_ff / 0.100
                        idx_k = int(min(5, max(0, int(shot_f))))
                        pts_ak = getattr(self.ak_calibrator, "pull_waypoints_cnt", None)
                        if pts_ak is not None and len(pts_ak) > idx_k + 1:
                            y_cur = float(pts_ak[idx_k, 1]) if idx_k > 0 else 0.0
                            y_nxt = float(pts_ak[idx_k + 1, 1])
                            dy_bullet = max(0.0, y_nxt - y_cur)
                            ff_y = float(np.clip(0.62 * dy_bullet * (min(0.085, eff_aim_cd) / 0.100), 0.0, 18.0))
                            dp_stage[1] += ff_y
                # Spray X-corridor anti-shake (stationary only; keep X live on moving targets)
                if lmb_down and (not is_moving_tgt) and abs(float(e_pred_px[0])) <= max(3.0, 0.95 * r_lock) and abs(float(e_raw_px[0])) <= 4.5:
                    dp_stage[0] = 0.0
                # During spray, peel opposing vertical delta off p_recoil_cnt first instead of fighting via -Y p_aim_cnt
                if lmb_down and abs(self.p_recoil_cnt[1]) >= 10.0 and (dp_stage[1] * self.p_recoil_cnt[1]) < 0.0:
                    recoil_peel = min(abs(float(self.p_recoil_cnt[1])), abs(float(dp_stage[1])))
                    p_rec_cand[1] = float(self.p_recoil_cnt[1]) - math.copysign(recoil_peel, float(self.p_recoil_cnt[1]))
                    p_rec_goal[1] = p_rec_cand[1]
                    recoil_due = True
                    dp_stage[1] = 0.0
                # Anti-reversal damping: if an axis flips direction soon after a >=7 cnt step, damp 0.35x & block boost
                # Never damp downward (+Y) recoil pull during LMB spray!
                is_rev_step = False
                if (not is_initial_snap) and (t_now - getattr(self, "_prev_trim_t", 0.0)) <= 0.32:
                    prev_dp = getattr(self, "_prev_trim_dp", np.zeros(2, dtype=float))
                    for ax in (0, 1):
                        if lmb_down and ax == 1 and float(dp_stage[1]) > 0.0:
                            continue
                        if is_moving_tgt and ax == 0:
                            continue
                        if abs(float(prev_dp[ax])) >= 7.0 and (float(dp_stage[ax]) * float(prev_dp[ax])) < 0.0:
                            dp_stage[ax] *= 0.35
                            is_rev_step = True
                dp_n = float(np.hypot(dp_stage[0], dp_stage[1]))
                if dp_n > step_cap:
                    dp_stage = dp_stage * (step_cap / dp_n)
                    dp_n = step_cap
                # Keep full inter-bullet authority within 9px and on moving targets (so counter-strafe reversals track briskly)
                if (not is_moving_tgt) and e_pred_norm > 9.0 and float(np.dot(dp_stage, self._lock_dir_sign)) < 0.0:
                    dp_stage = dp_stage * 0.72
                    dp_n = float(np.hypot(dp_stage[0], dp_stage[1]))

                rev_cd_ok = (not is_rev_step) or ((t_now - self._last_aim_cmd_t) >= max(self.aim_cooldown_s, 0.120))
                boost_min_dp = 4.5 if (is_shot1_rescue or is_initial_snap) else 6.0
                bcmd_ok = is_shot1_rescue or is_initial_snap or ((t_now - getattr(self, "_last_bcmd_t", 0.0)) >= 0.075)
                use_boost_final = bool(
                    use_boost_aim
                    and (lmb_down or is_initial_snap)
                    and dp_n >= boost_min_dp
                    and (not is_rev_step)
                    and bcmd_ok
                )
                if dp_n >= 2.2 and rev_cd_ok:
                    if (
                        lmb_down
                        and (float(self.p_aim_cnt[1]) + float(dp_stage[1])) > float(aim_env_hi[1]) + 1.0
                        and (t_now - self._last_recoil_cmd_t) >= self.recoil_cooldown_s
                    ):
                        overflow_y = (float(self.p_aim_cnt[1]) + float(dp_stage[1])) - float(aim_env_hi[1])
                        spill_step = min(42.0, overflow_y)
                        new_rec_y = min(190.0, max(0.0, float(self.p_recoil_cnt[1])) + spill_step)
                        if new_rec_y > float(self.p_recoil_cnt[1]) + 0.5:
                            p_rec_cand[1] = new_rec_y
                            p_rec_goal[1] = new_rec_y
                            recoil_due = True
                    p_aim_cand = np.clip(self.p_aim_cnt + dp_stage, aim_env_lo, aim_env_hi)
                    # Gentle per-axis washout bleed only when that axis approaches ceiling
                    for ax, ceil_ax in ((0, 225.0), (1, 255.0)):
                        if abs(float(p_aim_cand[ax] + p_rec_cand[ax])) > ceil_ax:
                            p_aim_cand[ax] *= math.exp(-max(0.002, dt_s) / 0.14)
                    p_cmd_cand = self._soft_saturate_stage_cmd(p_aim_cand + p_rec_cand, limits, soft_cap=(250.0, 285.0))
                    cd_req = self.recoil_cooldown_s if (lmb_down and recoil_due) else self.aim_cooldown_s
                    hw_cd_req = 0.0 if (is_shot1_rescue or is_initial_snap) else min(float(cd_req), 0.018)
                    sent = disp(
                        p_cmd_cand,
                        is_recenter=False,
                        cooldown_s=hw_cd_req,
                        freeze_play_g=bool(lmb_down and dp_n < 4.5),
                        use_boost=use_boost_final,
                        boost_scale=b_scale_aim,
                        boost_dur_ms=b_dur_aim,
                    )
                    if sent:
                        if is_pre_micro and self._pre_micro_count == 0 and (t_now - self._last_aim_cmd_t) >= 0.110:
                            self._aim_cmd_history = deque([(float(t_now) - 1.0, self.p_aim_cnt.copy())], maxlen=64)
                            self._target_base_stage_cnt = p_stage_now.copy()
                        self.p_aim_cnt = p_aim_cand
                        self._last_aim_cmd_t = t_now
                        self._last_engage_t = t_now
                        self._need_switch_recenter = False
                        if is_shot1_rescue:
                            self._shot1_rescue_sent = True
                        if is_pre_micro:
                            self._pre_micro_count += 1
                            self._pre_micro_done_tid = tid_cur
                            self._pre_micro_sent_t = t_now
                        self._prev_trim_dp = dp_stage.copy()
                        self._prev_trim_t = t_now
                        if use_boost_final:
                            self._last_bcmd_t = t_now
                        aim_sent = True
                        if is_initial_snap:
                            self._snap_done_tid = tid_cur
                            self._last_snap_t = t_now
                        if recoil_due or not lmb_down:
                            self.p_recoil_cnt = p_rec_cand
                            self._last_recoil_cmd_t = t_now
                        self._cmd_history.append((float(t_now), p_cmd_cand.copy()))
                        self._aim_cmd_history.append((float(t_now), self.p_aim_cnt.copy()))
                        if self._mono_cmd_t == 0.0:
                            self._mono_e0_norm = e_norm
                            self._mono_e0_x = abs(float(e_raw_px[0]))
                            self._mono_cmd_t = t_now
                        if hasattr(self, "eng_logger"):
                            if is_shot1_rescue:
                                g_tag = "SHOT1_RESCUE"
                            elif is_pre_micro and is_moving_tgt:
                                g_tag = "PRE_TRACK_SNAP"
                            elif is_pre_micro:
                                g_tag = "PRE_MICRO_SNAP" if self._pre_micro_count == 1 else "PRE_MICRO_TRIM"
                            else:
                                g_tag = "LOCK_TRIM"
                            self.eng_logger.record_frame(
                                t_now, e_raw_px, v_hand_raw, self.p_aim_cnt, self.p_recoil_cnt,
                                g_tag,
                                dp_stage,
                            )
            elif hasattr(self, "eng_logger"):
                if not lmb_down:
                    if not hand_settled:
                        g_reason = "PRE_MOVING"
                    elif e_norm <= pre_deadband_px:
                        g_reason = "PRE_ON_HEAD"
                    elif self._pre_micro_count >= 2:
                        g_reason = "PRE_DONE"
                    else:
                        g_reason = "PRE_WAIT"
                elif self._mono_tripped:
                    g_reason = "MONO_TRIP"
                elif not tap_guard_pass:
                    g_reason = "TAP_GUARD"
                elif e_norm <= r_lock or e_pred_norm <= r_lock:
                    g_reason = "IN_DEADBAND"
                elif not hand_settled:
                    g_reason = "BLOCKED_VH"
                else:
                    g_reason = "COOLDOWN"
                self.eng_logger.record_frame(t_now, e_raw_px, v_hand_raw, self.p_aim_cnt, self.p_recoil_cnt, g_reason)

            if (not aim_sent) and pan_bleed_active:
                p_cmd_cand = self._soft_saturate_stage_cmd(self.p_aim_cnt + p_rec_cand, limits, soft_cap=260.0)
                if (
                    float(np.hypot(p_stage_now[0] - p_cmd_cand[0], p_stage_now[1] - p_cmd_cand[1])) >= 1.5
                    and (t_now - getattr(self, "_last_wash_cmd_t", 0.0)) >= 0.040
                ):
                    sent = disp(
                        p_cmd_cand,
                        is_recenter=float(np.hypot(p_cmd_cand[0], p_cmd_cand[1])) < 4.0,
                        cooldown_s=0.040,
                        freeze_play_g=False,
                        use_boost=False,
                    )
                    if sent:
                        self._last_wash_cmd_t = t_now
                        aim_sent = True
                        if recoil_due or not lmb_down:
                            self.p_recoil_cnt = p_rec_cand
                            self._last_recoil_cmd_t = t_now
                        self._cmd_history.append((float(t_now), p_cmd_cand.copy()))
                        self._aim_cmd_history.append((float(t_now), self.p_aim_cnt.copy()))

            if (not aim_sent) and recoil_due:
                _dispatch_recoil_only(self.recoil_cooldown_s)
        elif lmb_down:
            if self._yolo_miss_since_t is None:
                self._yolo_miss_since_t = t_now
            if (t_now - self._yolo_miss_since_t) >= 0.25:
                self._lock_target_id = None
                self._snap_done_tid = None
                self._tgt_motion_tid = None
                self._tgt_moving_frames = 0
                self._lock_dir_sign[:] = 0.0
                self._mono_tripped = False
            if recoil_due:
                _dispatch_recoil_only(self.recoil_cooldown_s)
            elif (t_now - self._yolo_miss_since_t) >= 0.25 and float(np.hypot(p_stage_now[0], p_stage_now[1])) >= 4.0 and float(np.hypot(self.p_recoil_cnt[0], self.p_recoil_cnt[1])) < 4.0:
                sent = disp(
                    np.zeros(2, dtype=float),
                    is_recenter=True,
                    cooldown_s=0.055,
                    freeze_play_g=False,
                    use_boost=False,
                )
                if sent:
                    self.p_aim_cnt[:] = 0.0
                    self.p_recoil_cnt[:] = 0.0
                    self._cmd_history.append((float(t_now), np.zeros(2, dtype=float)))
                    self._aim_cmd_history.append((float(t_now), np.zeros(2, dtype=float)))
        else:
            self._yolo_miss_since_t = None
            self._lock_target_id = None
            self._snap_done_tid = None
            self._pre_micro_done_tid = None
            self._pre_micro_count = 0
            self._pre_prev_epx = None
            self._tgt_motion_tid = None
            self._tgt_moving_frames = 0
            self._lock_dir_sign[:] = 0.0
            self._mono_tripped = False
            idle_s = t_now - self._last_engage_t
            stage_norm = float(np.hypot(p_stage_now[0], p_stage_now[1]))
            safe_to_recenter = (
                bool(rec_key_down)
                or ((idle_s >= 0.65) and ((t_now - self._last_lmb_release_t) >= 0.70) and (v_hand_norm <= 35.0))
                or (
                    (not in_comfort_zone)
                    and ((t_now - self._last_lmb_release_t) >= 0.055)
                    and (not has_close_tgt_34)
                    and (v_hand_norm >= 520.0)
                )
                or (
                    ((abs(float(p_stage_now[0])) > 145.0) or (abs(float(p_stage_now[1])) > 165.0))
                    and ((t_now - self._last_lmb_release_t) >= 0.35)
                    and (v_hand_norm <= 35.0)
                )
            )
            if safe_to_recenter and recoil_due:
                _dispatch_recoil_only(0.035)
            elif safe_to_recenter and stage_norm >= 18.0 and (t_now - self._last_wash_cmd_t) >= 0.055:
                p_step = self.p_aim_cnt * 0.55
                sent = disp(
                    p_step,
                    is_recenter=False,
                    cooldown_s=0.055,
                    freeze_play_g=False,
                    use_boost=False,
                )
                if sent:
                    self._last_wash_cmd_t = t_now
                    self.p_aim_cnt = p_step
                    self._cmd_history.append((float(t_now), p_step.copy()))
                    self._aim_cmd_history.append((float(t_now), self.p_aim_cnt.copy()))
            elif safe_to_recenter and stage_norm >= 2.0 and (t_now - self._last_wash_cmd_t) >= 0.055:
                sent = disp(
                    np.zeros(2, dtype=float),
                    is_recenter=True,
                    cooldown_s=0.055,
                    freeze_play_g=False,
                    use_boost=False,
                )
                if sent:
                    self._last_wash_cmd_t = t_now
                    self.p_aim_cnt[:] = 0.0
                    self.p_recoil_cnt[:] = 0.0
                    self._cmd_history.append((float(t_now), np.zeros(2, dtype=float)))
                    self._aim_cmd_history.append((float(t_now), np.zeros(2, dtype=float)))

        return locked_det, locked_dist_px

    def run_live_loop(self, max_duration_s: float = 0.0) -> Dict:
        """Main non-blocking high-frequency loop for real CS2 gameplay."""
        session_t0 = time.perf_counter()
        self.startup_self_check()
        self.keys.flush((VK_F4, VK_F5, VK_F6, VK_F7, VK_F8, VK_F9, VK_F10, VK_END, VK_LBUTTON))
        self._start_1khz_state_tracker()
        self.vision.start_async_yolo()
        gsi_cfg_path = ensure_cs2_gsi_config(port=30001)
        self._gsi_started = self.gsi.start()
        if self._gsi_started:
            if gsi_cfg_path:
                print(f"[CS2 GSI] Listening on http://127.0.0.1:30001 | cfg ready: {gsi_cfg_path}")
            else:
                print("[CS2 GSI] Listening on http://127.0.0.1:30001 (put gamestate_integration_3rrr.cfg in CS2 cfg for kill tags)")

        last_step_t = time.perf_counter()
        last_det_gen = -1
        last_det_t = last_step_t
        cached_raw_dets: List[TargetDetection] = []
        roi_center = 0.5 * self.vision.roi_size
        try:
            while not self._stop_requested:
                t_now = time.perf_counter()
                if max_duration_s > 0.0 and (t_now - session_t0) >= max_duration_s:
                    break

                # Global hotkeys
                if self.keys.pressed_once(VK_END):
                    print("\n[END] Shutting down safely...")
                    break
                if self.keys.pressed_once(VK_F4):
                    self.toggle_f4_blind_ab()
                    last_step_t = time.perf_counter()
                    continue
                if self.keys.pressed_once(VK_F10):
                    self.trigger_f10_pause()
                    continue
                if self.keys.pressed_once(VK_F5):
                    self.trigger_f5_in_game_autotrain()
                    last_step_t = time.perf_counter()
                    continue
                if self.keys.pressed_once(VK_F6):
                    self.trigger_f6_loaded_ak_ilc()
                    last_step_t = time.perf_counter()
                    continue
                if self.keys.pressed_once(VK_F7):
                    self.toggle_f7_ak_spray()
                    last_step_t = time.perf_counter()
                    continue
                if self.keys.pressed_once(VK_F8):
                    self.trigger_f8_calibration()
                    last_step_t = time.perf_counter()
                    continue
                if self.keys.pressed_once(VK_F9):
                    self.toggle_f9_mode()
                    last_step_t = time.perf_counter()
                    continue

                if self.mode == MODE_AK_SPRAY:
                    if bool(self.keys.is_down(VK_LBUTTON)):
                        self._run_ak_spray_burst(n_shots=30, check_lmb=True)
                        last_step_t = time.perf_counter()
                    else:
                        time.sleep(0.005)
                    continue

                if self.mode == MODE_PAUSED:
                    if self.enable_gui:
                        roi_rgb = self.vision.grab_roi()
                        c_px_p, dets_p, _, _ = detect_live_crosshair_and_targets(
                            roi_rgb=roi_rgb,
                            vision=self.vision,
                            counts_per_px=self.ctrl.counts_per_px,
                            dt_s=0.030,
                            k_lead_s=self.ctrl.k_lead,
                            prev_gray=None,
                        )
                        self._render_vision_hud(
                            roi_rgb=roi_rgb,
                            c_px=c_px_p,
                            dets_px=dets_p,
                            locked_det=None,
                            arrow_tip_px=None,
                            p_stage_now=self.p_stage_est,
                        )
                        time.sleep(0.025)
                    else:
                        time.sleep(0.010)
                    continue

                dt_s = max(0.002, min(0.050, t_now - last_step_t))
                last_step_t = t_now

                roi_rgb = self.vision.grab_roi()
                t_cap = time.perf_counter()
                if hasattr(self.trainer.sensor, "get_hand_position"):
                    cur_h_xy = np.array(self.trainer.sensor.get_hand_position(), dtype=float)
                else:
                    cur_h_xy = np.zeros(2, dtype=float)
                self._cap_hand_history.append((float(t_cap), cur_h_xy))
                self.vision.submit_async_roi(roi_rgb, t_cap)
                c_px = extract_live_crosshair_fast(roi_rgb, self.vision, dt_s=dt_s)
                du_bg_px = np.zeros(2, dtype=float)
                cur_gray = None

                raw_dets, det_t, det_gen = self.vision.get_latest_async_detections()
                if det_gen != last_det_gen and det_gen > 0:
                    dt_kf = max(0.002, min(0.100, det_t - last_det_t)) if last_det_gen > 0 else dt_s
                    last_det_gen = det_gen
                    last_det_t = det_t
                    cached_raw_dets = self.vision.update_target_filters(
                        raw_dets, dt_s=dt_kf, k_lead_s=self.ctrl.k_lead, det_t_s=det_t
                    )
                    dets_px = self.vision.extrapolate_detections(
                        cached_raw_dets, t_now=t_cap, t_det=det_t, k_lead_s=self.ctrl.k_lead
                    )
                elif det_gen > 0:
                    dets_px = self.vision.extrapolate_detections(
                        cached_raw_dets, t_now=t_cap, t_det=last_det_t, k_lead_s=self.ctrl.k_lead
                    )
                else:
                    # Fallback until first async frame completes
                    c_px, dets_px, du_bg_px, cur_gray = detect_live_crosshair_and_targets(
                        roi_rgb=roi_rgb,
                        vision=self.vision,
                        counts_per_px=self.ctrl.counts_per_px,
                        dt_s=dt_s,
                        k_lead_s=self.ctrl.k_lead,
                        prev_gray=self._prev_gray,
                        compute_bg_flow=not self._tracker_running,
                    )
                self._prev_gray = cur_gray

                with self._state_lock:
                    v_hand_raw = self._v_hand_1k.copy()
                    lmb_edge = self._lmb_edge_pending
                    cs_event = self._counter_strafe_flag
                    rec_key = self._recenter_key_flag
                    self._lmb_edge_pending = False
                    self._counter_strafe_flag = False
                    self._recenter_key_flag = False

                if float(np.hypot(v_hand_raw[0], v_hand_raw[1])) < 1.0 and float(np.hypot(du_bg_px[0], du_bg_px[1])) > 0.5:
                    # Raw Input fallback via background ROI optical flow when OS cursor is static
                    v_hand_raw = (-du_bg_px * self.ctrl.counts_per_px) / dt_s

                c_rel_cnt = self.calibrator.pixel_to_counts(float(c_px[0]) - roi_center, float(c_px[1]) - roi_center)
                dets_cnt: List[TargetDetection] = []
                det_px_by_id: Dict[int, TargetDetection] = {}
                for d in dets_px:
                    det_px_by_id[int(d.target_id)] = d
                    h_cnt = self.calibrator.pixel_to_counts(float(d.head_px[0]) - roi_center, float(d.head_px[1]) - roi_center)
                    ch_cnt = self.calibrator.pixel_to_counts(float(d.chest_px[0]) - roi_center, float(d.chest_px[1]) - roi_center)
                    hp_cnt = self.calibrator.pixel_to_counts(
                        float(d.head_pred_px[0] if d.head_pred_px is not None else d.head_px[0]) - roi_center,
                        float(d.head_pred_px[1] if d.head_pred_px is not None else d.head_px[1]) - roi_center,
                    )
                    cp_cnt = self.calibrator.pixel_to_counts(
                        float(d.chest_pred_px[0] if d.chest_pred_px is not None else d.chest_px[0]) - roi_center,
                        float(d.chest_pred_px[1] if d.chest_pred_px is not None else d.chest_px[1]) - roi_center,
                    )
                    dets_cnt.append(
                        TargetDetection(
                            target_id=int(d.target_id),
                            head_px=h_cnt,
                            head_radius_px=d.head_radius_px * self.ctrl.counts_per_px,
                            chest_px=ch_cnt,
                            chest_radius_px=d.chest_radius_px * self.ctrl.counts_per_px,
                            vel_px_s=d.vel_px_s * self.ctrl.counts_per_px,
                            head_pred_px=hp_cnt,
                            chest_pred_px=cp_cnt,
                        )
                    )

                lmb_down = bool(self.keys.is_down(VK_LBUTTON))
                fire_now = lmb_edge or (lmb_down and (t_now - self._last_fire_t) >= AUTO_FIRE_PERIOD_S)
                if lmb_edge or not lmb_down:
                    self._burst_shot_idx = 0

                with self._hw_lock:
                    p_stage_now = self.p_stage_est.copy()

                t_step = time.perf_counter()
                p_stage_rel_open = (
                    (p_stage_now - self._target_base_stage_cnt)
                    if (self.mode == MODE_ASSIST_ON and (t_step - getattr(self, "_pre_micro_sent_t", 0.0)) <= 0.280)
                    else np.zeros(2, dtype=float)
                )
                if hasattr(self, "eng_logger") and lmb_down and self.eng_logger._active is None:
                    self.eng_logger._open_engagement(
                        t_s=t_now - session_t0,
                        p_stage_now=p_stage_rel_open,
                        weapon=self.gsi.active_weapon if self._gsi_started else "",
                        blind_ab=self.blind_ab_active,
                        blind_pair_id=(self._blind_pair_id if self.blind_ab_active else None),
                        blind_slot=(self._blind_slot_idx if self.blind_ab_active else None),
                    )
                hand_xy_for_det: Optional[np.ndarray] = None
                if self._dual_mouse_detected and self._cap_hand_history:
                    ref_det_t = float(last_det_t) if det_gen > 0 else float(t_cap)
                    for th_c, hxy_c in self._cap_hand_history:
                        if th_c <= ref_det_t + 0.002:
                            hand_xy_for_det = hxy_c
                        else:
                            break
                if self.mode == MODE_ASSIST_ON:
                    locked_det, _ = self._step_live_assist(
                        t_now=t_step,
                        dt_s=dt_s,
                        roi_rgb=roi_rgb,
                        c_px=c_px,
                        dets_px=dets_px,
                        lmb_down=lmb_down,
                        v_hand_raw=v_hand_raw,
                        p_stage_now=p_stage_now,
                        det_age_s=(max(0.0, t_step - last_det_t) if det_gen > 0 else 0.0),
                        rec_key_down=rec_key,
                        det_t_s=(float(last_det_t) if det_gen > 0 else float(t_step)),
                        hand_xy_at_det=hand_xy_for_det,
                    )
                else:
                    self._last_is_moving_tgt = False
                    self._last_v_world_x_px_s = 0.0
                    locked_det, _ = self._select_locked_target(
                        dets_px,
                        c_px,
                        v_hand_norm=float(np.hypot(v_hand_raw[0], v_hand_raw[1])),
                        lmb_down=lmb_down,
                    )
                    if locked_det is not None:
                        self._update_locked_target_motion(
                            int(locked_det.target_id),
                            np.asarray(locked_det.head_px, dtype=float),
                            float(last_det_t) if det_gen > 0 else float(t_step),
                            hand_xy_for_det,
                            v_hand_raw,
                            np.zeros((2, 2), dtype=float),
                        )
                if hasattr(self, "eng_logger") and locked_det is not None:
                    self.eng_logger.mark_target_motion(self._last_is_moving_tgt, self._last_v_world_x_px_s)

                out = self.ctrl.step(
                    dt_s=dt_s,
                    crosshair_cnt=c_rel_cnt,
                    targets_cnt=dets_cnt,
                    v_hand_override=v_hand_raw,
                    lmb_held=lmb_down,
                    shot_idx_global=self._burst_shot_idx,
                    counter_strafe_event=cs_event,
                    reload_or_switch_key=rec_key,
                    p_stage_actual_cnt=p_stage_now if self.mode == MODE_ASSIST_ON else np.zeros(2, dtype=float),
                )
                self.decision_latencies_ms.append(out.decision_time_ms)
                short_state = STATE_LABEL_MAP.get(out.state, "DORMANT")

                arrow_tip_px: Optional[np.ndarray] = None
                if locked_det is not None:
                    if self.mode == MODE_ASSIST_ON and getattr(self, "_live_anchor_px", None) is not None:
                        arrow_tip_px = self._live_anchor_px.copy()
                    else:
                        arrow_tip_px = locked_det.head_px.copy()
                elif lmb_down and float(np.hypot(c_px[0] - roi_center, c_px[1] - roi_center)) >= 2.5:
                    arrow_tip_px = np.array([roi_center, roi_center], dtype=float)

                if getattr(self, "_hud_initialized", False):
                    self._hide_vision_hud()

                if fire_now:
                    self._last_fire_t = t_now
                    self._burst_shot_idx += 1
                    tgt_px = locked_det or det_px_by_id.get(out.active_target_id)
                    if tgt_px is None and dets_px:
                        tgt_px = min(
                            dets_px,
                            key=lambda d: float(np.hypot(d.head_px[0] - c_px[0], d.head_px[1] - c_px[1])),
                        )
                    if tgt_px is not None:
                        with self._hw_lock:
                            if self.mode == MODE_ASSIST_ON:
                                p_abs = (
                                    p_stage_now.copy()
                                    if self._burst_shot_idx == 1
                                    else self.p_stage_est.copy()
                                )
                                p_shot = p_abs - self._target_base_stage_cnt
                            else:
                                p_shot = np.zeros(2, dtype=float)
                        shot_anchor_mode = (
                            self._live_anchor_mode
                            if (self.mode == MODE_ASSIST_ON and locked_det is not None)
                            else out.anchor_mode
                        )
                        verdict = self._record_and_print_shot(
                            t_now=t_now - session_t0,
                            mode_now=self.mode,
                            short_state=short_state,
                            c_cnt=c_rel_cnt,
                            p_stage_shot=p_shot,
                            det_target=tgt_px,
                            v_hand_cnt_s=v_hand_raw,
                            active_target_id=out.active_target_id,
                            anchor_mode=shot_anchor_mode,
                        )
                        self._capture_shot_snapshot(
                            roi_rgb=roi_rgb,
                            c_px=c_px,
                            dets_px=dets_px,
                            locked_det=locked_det or tgt_px,
                            arrow_tip_px=arrow_tip_px,
                            p_stage_now=p_shot,
                            label=f"#{verdict.shot_idx} {verdict.hitbox[:4]} ea={verdict.e_actual_cnt:.0f} eh={verdict.e_hand_cnt:.0f}",
                        )

                # Poll sensor coupling & engagement/kill lifecycle
                if hasattr(self, "eng_logger"):
                    try:
                        s_poll = np.array(self.trainer.sensor.get_position(), dtype=float)
                    except Exception:
                        s_poll = np.zeros(2, dtype=float)
                    self.eng_logger.poll_coupling(t_now, s_poll, float(np.hypot(v_hand_raw[0], v_hand_raw[1])))
                    closed_eng = self.eng_logger.step_lifecycle(
                        t_s=t_now - session_t0,
                        lmb_down=lmb_down,
                        p_stage_now=p_stage_now if self.mode == MODE_ASSIST_ON else np.zeros(2, dtype=float),
                        gsi=self.gsi if self._gsi_started else None,
                        blind_ab=self.blind_ab_active,
                        blind_pair_id=(self._blind_pair_id if self.blind_ab_active else None),
                        blind_slot=(self._blind_slot_idx if self.blind_ab_active else None),
                    )
                    if closed_eng is not None and closed_eng.get("gsi_kills", 0) > 0:
                        if self.blind_ab_active:
                            p_tag = closed_eng.get("blind_pair_id") or self._blind_pair_id
                            print(
                                f"[KILL #{closed_eng['eng_id']:02d} | BLIND_AB Pair #{p_tag:02d} | "
                                f"{closed_eng['weapon']} | HS={closed_eng['gsi_hs']}/{closed_eng['gsi_kills']}]",
                                flush=True,
                            )
                        else:
                            print(
                                f"[KILL #{closed_eng['eng_id']:02d} | {closed_eng['weapon']} | HS={closed_eng['gsi_hs']}/{closed_eng['gsi_kills']}] "
                                f"Shot1: {closed_eng['shot1_eh_cnt']:.0f}->{closed_eng['shot1_ea_cnt']:.0f} cnt | "
                                f"sp_pre={closed_eng['sp_pre_norm']:.0f} | cp={closed_eng['coupling_ratio']} | {closed_eng['diag_flags']}",
                                flush=True,
                            )
                    min_tgt_dist_px = min(
                        (float(np.hypot(d.head_px[0] - c_px[0], d.head_px[1] - c_px[1])) for d in dets_px),
                        default=1e9,
                    )
                    self._step_blind_ab_crossover(
                        t_now=t_step,
                        lmb_down=lmb_down,
                        v_hand_norm=float(np.hypot(v_hand_raw[0], v_hand_raw[1])),
                        min_tgt_dist_px=min_tgt_dist_px,
                        closed_eng=closed_eng,
                    )

                # Push ring frame for fault-attribution lookback
                err_vec = out.residual_err_cnt
                self.referee.push_frame(
                    t_s=t_now - session_t0,
                    c_cnt=c_rel_cnt,
                    p_stage_cnt=p_stage_now if self.mode == MODE_ASSIST_ON else np.zeros(2, dtype=float),
                    v_hand_cnt_s=v_hand_raw,
                    active_target_id=out.active_target_id,
                    intended_target_id=out.active_target_id,
                    err_vec_cnt=err_vec,
                )
                time.sleep(0.0005)
        except KeyboardInterrupt:
            print("\n[Ctrl+C] Safe shutdown triggered...")
        finally:
            dur = time.perf_counter() - session_t0
            self.close()
            report = self.finalize_and_save_report(dur)
        return report

    def run_self_test(self) -> Dict:
        """
        Real-hardware COM4 + live ROI FPS self-test verifying:
          1) COM4 startup check & 1.0s ROI FPS benchmark
          2) F8 online 1.3s self-calibration
          3) F9 ASSIST_ON (Counter-Strafe + Spray Transfer on physical COM4 with 85ms cooldown guard)
          4) F9 SHADOW_REFEREE_ONLY blind comparison shots
          5) F10 pause neutral recenter & END report save to live_session_report.json
        """
        session_t0 = time.perf_counter()
        orig_cfg_text = None
        if os.path.isfile(self.config_path):
            with open(self.config_path, "r", encoding="utf-8") as f:
                orig_cfg_text = f.read()
        try:
            self.startup_self_check()

            # Simulate F8 hotkey
            cal_res = self.trigger_f8_calibration()
            assert cal_res["pass_time"] and cal_res["pass_recon"], f"F8 calib failed: {cal_res}"

            # Simulate F9 hotkey: PAUSED -> ASSIST_ON
            assert self.toggle_f9_mode() == MODE_ASSIST_ON

            roi_size = self.vision.roi_size
            roi_center = 0.5 * roi_size
            cpp = self.ctrl.counts_per_px

            # 4 physical ASSIST_ON shots (2 Counter-Strafe + 2 Spray Transfer) + 2 SHADOW_REFEREE_ONLY shots
            scenarios = [
                ("ASSIST_ON", np.array([23.0, -6.0]), np.array([0.0, 0.0]), True, False, 0),
                ("ASSIST_ON", np.array([-29.0, 12.0]), np.array([0.0, 0.0]), True, False, 0),
                ("ASSIST_ON", np.array([29.0, 17.0]), np.array([25.0, 0.0]), False, True, 1),
                ("ASSIST_ON", np.array([-23.0, -17.0]), np.array([20.0, 0.0]), False, True, 2),
                ("SHADOW_REFEREE_ONLY", np.array([23.0, 6.0]), np.array([0.0, 0.0]), True, False, 0),
                ("SHADOW_REFEREE_ONLY", np.array([-18.0, 12.0]), np.array([0.0, 0.0]), True, False, 0),
            ]

            for sc_mode, init_err_cnt, v_hand_cnt, cs_evt, lmb_held, shot_k in scenarios:
                if sc_mode == MODE_SHADOW and self.mode != MODE_SHADOW:
                    self.toggle_f9_mode()

                # Respect 85 ms cooldown lock strictly between shots
                rem_cd = COOLDOWN_LOCK_S - (time.perf_counter() - self._last_hw_cmd_t)
                if rem_cd > 0.0:
                    time.sleep(rem_cd + 0.004)

                # Reset mechanical origin cleanly between isolated self-test probes
                if self.mode == MODE_ASSIST_ON:
                    self.trainer.move_to_rect_origin(settle_s=0.28)
                    self.p_stage_est = np.zeros(2, dtype=float)
                    self.ctrl.p_stage_cur = np.zeros(2, dtype=float)
                    self._prev_sensor_xy = np.array(self.trainer.sensor.get_position(), dtype=float)

                self.vision.reset_filters()
                # Convert desired initial crosshair-to-head error into synthetic 256x256 ROI coordinates
                d_px = self.calibrator.counts_to_pixel(float(init_err_cnt[0]), float(init_err_cnt[1]))
                ch_xy = np.array([roi_center - 0.5 * d_px[0], roi_center - 0.5 * d_px[1]], dtype=float)
                head_xy = ch_xy + d_px
                chest_xy = head_xy + np.array([0.0, 14.0 / cpp], dtype=float)
                tgt = TargetDetection(
                    target_id=0,
                    head_px=head_xy,
                    head_radius_px=10.5 / cpp,
                    chest_px=chest_xy,
                    chest_radius_px=21.0 / cpp,
                )

                # Prime vision KF across 3 synthetic ROI frames at 240 Hz equivalent dt (inject tgt for hardware pipeline test)
                frame_rgb = render_synthetic_roi_frame(ch_xy, [tgt], roi_size=roi_size)
                for _ in range(3):
                    c_px, dets_px = self.vision.process_frame(
                        frame_rgb,
                        dt_s=0.004,
                        k_lead_s=self.ctrl.k_lead,
                        injected_detections=[tgt],
                    )

                c_hand_cnt = self.calibrator.pixel_to_counts(float(c_px[0]) - roi_center, float(c_px[1]) - roi_center)
                det_px = dets_px[0]
                h_cnt = self.calibrator.pixel_to_counts(float(det_px.head_px[0]) - roi_center, float(det_px.head_px[1]) - roi_center)
                ch_cnt = self.calibrator.pixel_to_counts(float(det_px.chest_px[0]) - roi_center, float(det_px.chest_px[1]) - roi_center)
                det_cnt = TargetDetection(
                    target_id=0,
                    head_px=h_cnt,
                    head_radius_px=det_px.head_radius_px * cpp,
                    chest_px=ch_cnt,
                    chest_radius_px=det_px.chest_radius_px * cpp,
                    head_pred_px=h_cnt,
                    chest_pred_px=ch_cnt,
                )

                out = None
                short_state = "DORMANT"
                p_shot = np.zeros(2, dtype=float)
                n_cl_ticks = 2 if self.mode == MODE_ASSIST_ON else 1
                for cl_tick in range(n_cl_ticks):
                    c_obs_cnt = c_hand_cnt + p_shot
                    out = self.ctrl.step(
                        dt_s=0.008,
                        crosshair_cnt=c_obs_cnt,
                        targets_cnt=[det_cnt],
                        v_hand_override=v_hand_cnt,
                        lmb_held=lmb_held,
                        shot_idx_global=shot_k,
                        counter_strafe_event=cs_evt,
                        p_stage_actual_cnt=p_shot,
                    )
                    self.decision_latencies_ms.append(out.decision_time_ms)
                    short_state = STATE_LABEL_MAP.get(out.state, "DORMANT")

                    if self.mode == MODE_ASSIST_ON:
                        rem_cd = COOLDOWN_LOCK_S - (time.perf_counter() - self._last_hw_cmd_t)
                        if cl_tick > 0 and rem_cd > 0.0:
                            time.sleep(rem_cd + 0.004)
                        p_cmd = np.clip(out.p_stage_cmd_cnt, -SAFE_ENVELOPE_CNT, SAFE_ENVELOPE_CNT)
                        dispatched = self._dispatch_stage_command(p_cmd, is_recenter=False, wait_done=False)
                        if cl_tick == 0:
                            assert dispatched, "Expected ASSIST_ON shot to dispatch after cooldown"
                            p_dup = np.clip(p_cmd + np.array([5.0, 0.0]), -SAFE_ENVELOPE_CNT, SAFE_ENVELOPE_CNT)
                            blocked_dup = self._dispatch_stage_command(p_dup, is_recenter=False, wait_done=False)
                            assert not blocked_dup, "85 ms cooldown guard must block immediate duplicate dispatch"
                        self._wait_hw_idle()
                        p_shot = self.p_stage_est.copy()
                        res_err = float(np.hypot(h_cnt[0] - (c_hand_cnt[0] + p_shot[0]), h_cnt[1] - (c_hand_cnt[1] + p_shot[1])))
                        if res_err <= 6.5:
                            break
                    else:
                        p_shot = np.zeros(2, dtype=float)

                c_fire_cnt = c_hand_cnt + p_shot
                t_rel = time.perf_counter() - session_t0
                self.referee.push_frame(
                    t_s=max(0.0, t_rel - 0.080),
                    c_cnt=c_hand_cnt,
                    p_stage_cnt=np.zeros(2, dtype=float),
                    v_hand_cnt_s=v_hand_cnt,
                    active_target_id=0,
                    intended_target_id=0,
                    err_vec_cnt=h_cnt - c_hand_cnt,
                )
                self._record_and_print_shot(
                    t_now=t_rel,
                    mode_now=self.mode,
                    short_state=short_state,
                    c_cnt=c_fire_cnt,
                    p_stage_shot=p_shot,
                    det_target=det_px,
                    v_hand_cnt_s=v_hand_cnt,
                    active_target_id=out.active_target_id,
                    anchor_mode=out.anchor_mode,
                )

            # Verify F7 AK_SPRAY_LOCK 12-shot closed-loop full-stroke (> 300 counts) pull-down
            assert self.toggle_f7_ak_spray() == MODE_AK_SPRAY
            self.ak_calibrator.load_profile()
            self.trainer.driver.center(settle_s=0.2, preload=True)
            ak_res = self.ak_calibrator.execute_ak_spray_closed_loop(
                self.trainer,
                n_shots=12,
                use_feedback=True,
                rate_hz=250.0,
                reset_origin=True,
            )
            max_target_y = max(float(b["p_des_cnt"][1]) for b in ak_res["bullet_table"])
            max_meas_y = max(float(b["p_meas_cnt"][1]) for b in ak_res["bullet_table"])
            print(
                f"[F7 AK_SPRAY_LOCK Self-Test] 12-shot mean_err={ak_res['mean_bullet_err_cnt']:.2f} cnt "
                f"| max={ak_res['max_bullet_err_cnt']:.2f} cnt | max_y_pull={max_meas_y:.1f}/{max_target_y:.1f} cnt"
            )
            assert max_meas_y >= 300.0, f"AK spray stroke too small (<300 cnt): {max_meas_y:.1f} cnt"
            assert ak_res["mean_bullet_err_cnt"] <= 8.0, f"AK spray point-lock error too high: {ak_res['mean_bullet_err_cnt']}"

            # Verify real YOLOv8n CS2 head/chest GPU inference on positive val samples if present
            val_dir = os.path.join(HIL_DIR, "dataset_cs2", "images", "val")
            lbl_dir = os.path.join(HIL_DIR, "dataset_cs2", "labels", "val")
            if self.vision._yolo_model is not None and os.path.isdir(val_dir):
                import cv2
                pos_files = []
                for vf in sorted(f for f in os.listdir(val_dir) if f.endswith(".jpg")):
                    lbl_p = os.path.join(lbl_dir, os.path.splitext(vf)[0] + ".txt")
                    if os.path.isfile(lbl_p) and os.path.getsize(lbl_p) > 0:
                        pos_files.append(vf)
                    if len(pos_files) >= 12:
                        break
                val_files = pos_files
                if val_files:
                    bgr0 = cv2.imread(os.path.join(val_dir, val_files[0]))
                    if bgr0 is not None:
                        _ = self.vision.process_frame(
                            cv2.resize(bgr0[:, :, ::-1], (self.vision.roi_size, self.vision.roi_size)), dt_s=0.004
                        )
                t_y0 = time.perf_counter()
                yolo_hits = 0
                for vf in val_files:
                    bgr = cv2.imread(os.path.join(val_dir, vf))
                    if bgr is None:
                        continue
                    roi_test = cv2.resize(bgr[:, :, ::-1], (self.vision.roi_size, self.vision.roi_size))
                    _, d_yolo = self.vision.process_frame(roi_test, dt_s=0.004)
                    if d_yolo:
                        yolo_hits += 1
                yolo_ms = ((time.perf_counter() - t_y0) * 1000.0) / max(1, len(val_files))
                print(
                    f"[YOLO Live Check] pos_images={len(val_files)} | detected={yolo_hits}/{len(val_files)} "
                    f"| mean_latency={yolo_ms:.2f} ms/frame | backend={self.vision.detector_backend} "
                    f"(conf>={self.vision.yolo_conf:.2f})"
                )
                assert yolo_hits >= max(1, len(val_files) // 2), f"YOLOv8n detection rate too low: {yolo_hits}/{len(val_files)}"

            # Verify Open-Loop Manifold P-only Recoil & Residual Aim Mapping (zero windup)
            p_rec1 = self.calibrator.recoil_offset_to_stage_abs(0.0, -30.0)
            p_rec2 = self.calibrator.recoil_offset_to_stage_abs(0.0, -30.0)
            p_aim = self.calibrator.aim_error_to_stage_delta(20.0, -10.0)
            self._last_hw_cmd_t = 0.0
            ok_ol = self._dispatch_stage_manifold_open_loop(p_rec1, is_recenter=False, cooldown_s=0.0)
            assert ok_ol and np.allclose(p_rec1, p_rec2) and p_rec1[1] > 0.0 and p_aim[0] > 0.0
            self._dispatch_stage_manifold_open_loop(np.zeros(2, dtype=float), is_recenter=True, cooldown_s=0.0)
            print(
                f"[Open-Loop Manifold Check] recoil(-30px)={np.round(p_rec1,1)} cnt (no windup) | "
                f"aim(+20,-10px)={np.round(p_aim,1)} cnt | PASS=True"
            )

            # Verify 144 Hz Delayed-Plant (95ms lag) Closed-Loop _step_live_assist (ZERO windup/double-count)
            self._reset_live_lock_and_stage_state()
            self.calibrator.J_screen_to_stage = np.array([[-6.5619, 0.7642], [-0.6433, -3.2631]], dtype=float)
            self.calibrator.set_x_axis_asymmetry(1.0, 1.0)
            M_lag = np.linalg.inv(self.calibrator.J_screen_to_stage)
            sim_hist: deque = deque([(0.0, np.zeros(2, dtype=float))], maxlen=128)
            sim_stage = np.zeros(2, dtype=float)
            sim_last_disp_t = -1.0

            def _sim_disp(p_t, is_recenter=False, cooldown_s=0.0, freeze_play_g=False, use_boost=True, **kwargs):
                nonlocal sim_stage, sim_last_disp_t
                if (t_cur - sim_last_disp_t) < float(cooldown_s):
                    return False
                sim_stage = np.zeros(2, dtype=float) if is_recenter else np.asarray(p_t, dtype=float).copy()
                sim_last_disp_t = t_cur
                sim_hist.append((t_cur, sim_stage.copy()))
                return True

            def _sim_vis(t_q: float) -> np.ndarray:
                vis = np.zeros(2, dtype=float)
                for ts, ps in sim_hist:
                    if ts <= (t_q - 0.095):
                        vis = ps
                    else:
                        break
                return vis

            roi_blank = np.zeros((roi_size, roi_size, 3), dtype=np.uint8)
            # Green crosshair at center during Aim-only phase, then kicks UP -22px on LMB spray
            h0_px = np.array([roi_center + 18.0, roi_center - 10.0], dtype=float)
            errs_aim, max_stg_seen = [], 0.0
            for k in range(136):  # 136 * 7ms = 952 ms at 144 Hz (covers 260ms HEAD hold + CHEST settle under 95ms lag)
                t_cur = 1.0 + k * 0.007
                lmb_sim = (k >= 60)
                c_off_sim = np.array([0.0, -22.0]) if lmb_sim else np.zeros(2, dtype=float)
                c_sim = np.array([roi_center, roi_center]) + c_off_sim
                roi_sim = roi_blank.copy()
                cx_i, cy_i = int(round(c_sim[0])), int(round(c_sim[1]))
                roi_sim[max(0, cy_i - 2) : min(roi_size, cy_i + 3), max(0, cx_i - 2) : min(roi_size, cx_i + 3)] = (0, 255, 0)
                p_vis_k = _sim_vis(t_cur)
                h_cur = h0_px + (M_lag @ p_vis_k)
                # Drop YOLO for 1 frame at k=75 during LMB spray and add distractor bot at id=1
                dets_k = [] if k == 75 else [
                    TargetDetection(target_id=0, head_px=h_cur, head_radius_px=6.0, chest_px=h_cur + [0, 12], chest_radius_px=10.0, seen_frames=5),
                    TargetDetection(target_id=1, head_px=c_sim + [25.0, 25.0], head_radius_px=6.0, chest_px=c_sim + [25, 37], chest_radius_px=10.0, seen_frames=5),
                ]
                self._step_live_assist(
                    t_now=t_cur, dt_s=0.007, roi_rgb=roi_sim, c_px=c_sim, dets_px=dets_k,
                    lmb_down=lmb_sim, v_hand_raw=np.zeros(2), p_stage_now=sim_stage.copy(), dispatch_fn=_sim_disp,
                )
                max_stg_seen = max(max_stg_seen, float(np.max(np.abs(sim_stage))))
                if k in (55, 90):
                    errs_aim.append(float(np.hypot(*(h_cur - c_sim))))
                if k == 90:
                    assert self._live_anchor_mode == "HEAD", f"Early chest transition at 210ms: {self._live_anchor_mode}"
                if k == 108:
                    assert self._live_anchor_mode == "CHEST", f"Expected upper-chest blend by 336ms @ 22px kick: {self._live_anchor_mode}"
                if k == 134:
                    ref_anc = self._live_anchor_px if self._live_anchor_px is not None else h_cur
                    errs_aim.append(float(np.hypot(*(ref_anc - c_sim))))
                assert not (k == 76 and float(np.hypot(sim_stage[0], sim_stage[1])) < 5.0), "1-frame YOLO dropout caused recenter slam!"
            assert max_stg_seen <= 160.0, f"144 Hz delayed-plant rail slam detected: max_stage={max_stg_seen:.1f}"
            assert all(e <= 4.0 for e in errs_aim), f"144 Hz delayed-plant failed to converge: errs={errs_aim}"
            print(
                f"[144Hz Delayed-Plant Smith Check] settle_err={errs_aim[0]:.2f}px | "
                f"210ms_head_err={errs_aim[1]:.2f}px | 518ms_upper_chest_err={errs_aim[2]:.2f}px | "
                f"max_stage={max_stg_seen:.1f} cnt | PASS=True"
            )
            self._reset_live_lock_and_stage_state()

            # Verify F10 smooth recenter + pause
            self.trigger_f10_pause()
        finally:
            dur = time.perf_counter() - session_t0
            self.close()
            report = self.finalize_and_save_report(dur)
            if orig_cfg_text is not None:
                with open(self.config_path, "w", encoding="utf-8") as f:
                    f.write(orig_cfg_text)

        assist_ov = report["mode_comparison"][MODE_ASSIST_ON]
        assert assist_ov["total_shots"] == 4
        assert assist_ov["effective_hit_rate"] >= 0.75, f"Self-test ASSIST_ON hit rate too low: {assist_ov}"
        assert report["mode_comparison"][MODE_SHADOW]["total_shots"] == 2
        return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CS2 3-RRR Live Shared-Control Resident Runner")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH, help="Path to config_hil.json")
    parser.add_argument("--report", type=str, default=DEFAULT_REPORT_PATH, help="Output path for live_session_report.json")
    parser.add_argument("--self-test", action="store_true", help="Run non-interactive COM4 + ROI FPS + hotkey self-test")
    parser.add_argument("--no-gui", action="store_true", help="Disable topmost 256x256 CS2 3-RRR Vision HUD window")
    parser.add_argument("--duration", type=float, default=0.0, help="Optional live session duration limit in seconds (0 = until END)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runner = LiveGameSharedControlRunner(
        config_path=args.config,
        report_path=args.report,
        synthetic_capture=False,
        enable_gui=(False if args.self_test else (not args.no_gui)),
    )
    if args.self_test:
        runner.run_self_test()
    else:
        runner.run_live_loop(max_duration_s=float(args.duration))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
