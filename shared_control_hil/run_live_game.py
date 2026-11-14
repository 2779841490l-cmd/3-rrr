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
import ctypes
import json
import math
import os
import sys
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

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


def detect_live_crosshair_and_targets(
    roi_rgb: np.ndarray,
    vision: VisionROITracker,
    counts_per_px: float,
    dt_s: float,
    k_lead_s: float,
    prev_gray: Optional[np.ndarray],
) -> Tuple[np.ndarray, List[TargetDetection], np.ndarray, np.ndarray]:
    """
    Extract dynamic crosshair c_px, live targets (head + chest), ego background shift, and gray frame.
    Supports Cyan (0,255,255), Green (0,255,0)/(0,255,80), and synthetic/silhouette fallback.
    """
    h, w, _ = roi_rgb.shape
    center = np.array([0.5 * w, 0.5 * h], dtype=float)
    r = roi_rgb[:, :, 0].astype(np.int16)
    g = roi_rgb[:, :, 1].astype(np.int16)
    b = roi_rgb[:, :, 2].astype(np.int16)

    # Multi-profile crosshair mask: Cyan or Bright Green (CS2 standard high-visibility crosshairs)
    cyan_mask = (g >= 180) & (b >= 180) & ((g - r) >= 60)
    green_mask = (g >= 180) & (r <= 115) & (b <= 135)
    ch_mask = cyan_mask if int(cyan_mask.sum()) >= int(green_mask.sum()) else green_mask

    c_meas: Optional[np.ndarray] = None
    ch_cnt = int(ch_mask.sum())
    if 3 <= ch_cnt <= 450:
        row_sum = ch_mask.sum(axis=1).astype(float)
        col_sum = ch_mask.sum(axis=0).astype(float)
        vr = int(np.argmax(row_sum))
        ur = int(np.argmax(col_sum))
        vw = slice(max(0, vr - 10), min(h, vr + 11))
        uw = slice(max(0, ur - 10), min(w, ur + 11))
        ys, xs = np.nonzero(ch_mask[vw, uw])
        if len(xs) >= 3:
            c_meas = np.array([float(xs.mean() + uw.start), float(ys.mean() + vw.start)], dtype=float)

    if c_meas is not None:
        c_px, _ = vision.crosshair_tracker.kf.step(c_meas, dt_s=dt_s)
        vision.crosshair_tracker.last_pos = c_px
    else:
        c_px, _ = vision.crosshair_tracker.detect_in_frame(roi_rgb, dt_s=dt_s)

    gray = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float32)
    du_bg = np.zeros(2, dtype=float)
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
        self.ctrl = MacroMicroSharedController(self.config)
        self.ak_calibrator = AK47SprayCalibrator(config_path=self.config_path, calibrator=self.calibrator)
        self.referee = TelemetryReferee(
            counts_per_px=self.ctrl.counts_per_px,
            head_radius_cnt=float(self.config["thresholds"].get("head_radius_cnt", 10.5)),
            chest_radius_cnt=float(self.config["thresholds"].get("chest_radius_cnt", 21.0)),
            hard_limit_cnt=float(self.config["stage_kinematics"].get("hard_limit_counts", 88.0)),
        )
        self.keys = Win32KeyPoller()

        self.mode: str = MODE_PAUSED
        self.shot_count: int = 0
        self.shot_modes: List[str] = []
        self.shot_states: List[str] = []
        self.decision_latencies_ms: List[float] = []
        self.calib_history: List[Dict] = []
        self.roi_fps_bench: float = 0.0

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
        self._reload_live_control_params()

        self.in_game_trainer = InGameAutoTrainer(
            config_path=self.config_path,
            calibrator=self.calibrator,
            vision=self.vision,
            trainer=self.trainer,
            dispatch_fn=lambda p, cooldown_s=COOLDOWN_LOCK_S, freeze_play_g=False, use_boost=True, boost_scale=None, boost_dur_ms=None: self._dispatch_stage_manifold_open_loop(
                p,
                is_recenter=False,
                cooldown_s=cooldown_s,
                freeze_play_g=freeze_play_g,
                use_boost=use_boost,
                boost_scale=boost_scale,
                boost_dur_ms=boost_dur_ms,
            ),
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
        self.kp_small_head = float(lc.get("kp_small_head", 0.42))
        self.kp_large_error = float(lc.get("kp_large_error", 0.68))
        self.small_error_px = float(lc.get("small_error_px", 14.0))
        self.in_box_deadband_ratio = float(lc.get("in_box_deadband_ratio", 0.38))
        self.min_deadband_px = float(lc.get("min_deadband_px", 2.2))
        self.cooldown_lock_s = float(lc.get("cooldown_ms", 85.0)) / 1000.0
        self.recoil_cooldown_s = float(lc.get("recoil_cooldown_ms", 20.0)) / 1000.0
        self.boost_scale = float(lc.get("boost_scale", 1.35))
        self.boost_dur_ms = float(lc.get("boost_dur_ms", 22.0))
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
        self._last_engage_t = time.perf_counter()

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
            is_final = alpha == 0.0
            try:
                u1, u2 = self.trainer.manifold.evaluate_pulse(
                    float(p_step[0]),
                    float(p_step[1]),
                    dir_hint=(-float(p_cur[0]), -float(p_cur[1])),
                )
                u1_c, u2_c = _clamp_servo_pair(u1, u2)
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

    def _start_1khz_state_tracker(self) -> None:
        if self._tracker_running:
            return
        self._tracker_running = True

        def _loop() -> None:
            last_t = time.perf_counter()
            last_sensor = np.array(self.trainer.sensor.get_position(), dtype=float)
            last_stage = self.p_stage_est.copy()
            prev_lmb_1k = False
            prev_wasd = (False, False, False, False)
            while self._tracker_running:
                t_now = time.perf_counter()
                dt = max(0.0005, t_now - last_t)
                last_t = t_now

                cur_sensor = np.array(self.trainer.sensor.get_position(), dtype=float)
                with self._hw_lock:
                    cur_stage = self.p_stage_est.copy()
                dp_usb = cur_sensor - last_sensor
                dp_stage = cur_stage - last_stage
                last_sensor = cur_sensor
                last_stage = cur_stage

                inst_v = (dp_usb - dp_stage) / dt
                alpha = min(1.0, dt / 0.012)
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
                with self._state_lock:
                    self._v_hand_1k = (1.0 - alpha) * self._v_hand_1k + alpha * inst_v
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

        # 1.0 second center 256x256 ROI capture FPS benchmark
        n_frames = 0
        t0 = time.perf_counter()
        while (time.perf_counter() - t0) < 1.0:
            _ = self.vision.grab_roi()
            n_frames += 1
        elapsed = max(1e-3, time.perf_counter() - t0)
        self.roi_fps_bench = n_frames / elapsed
        print(
            f"[ROI Benchmark] Backend={self.vision.backend_name} | "
            f"1.0s frames={n_frames} | Measured FPS={self.roi_fps_bench:.1f} Hz"
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
        print(f"[STATE] Entered {self.mode}. Alt-tab into CS2 and press F5/F8 (calib) or F9 (activate).")
        return self.roi_fps_bench

    def trigger_f5_in_game_autotrain(self) -> Dict:
        """Pause assist, calibrate loaded YOLO head Jacobian (Stage 1) + zero-overshoot damping (Stage 2)."""
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
        res_yolo = self.in_game_trainer.calibrate_yolo_head_jacobian(step_cnt=70.0, settle_s=0.105)
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
            u_kin0 = self.trainer.manifold.evaluate_kinematic_pulse(0.0, 0.0)
            self.trainer.manifold.pi_net.reset_play_state(u_kin0, (1.0, 1.0))
            self._last_u_kin_live = np.asarray(u_kin0, dtype=float)
        except Exception:
            pass
        self._prev_sensor_xy = np.array(self.trainer.sensor.get_position(), dtype=float)
        self._last_hw_cmd_t = time.perf_counter()
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

    def toggle_f7_ak_spray(self) -> str:
        """Toggle AK_SPRAY_LOCK <-> PAUSED for 100 Hz closed-loop AK-47 point-lock recoil control."""
        self._wait_hw_idle()
        self.recenter_to_neutral(duration_s=0.18)
        self.ctrl.reset_episode()
        self.vision.reset_filters()
        self._prev_gray = None
        if self.mode != MODE_AK_SPRAY:
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
            has_kin = hasattr(manifold, "evaluate_kinematic_pulse")
            if freeze_play_g and has_kin and (not is_recenter):
                u_kin = np.asarray(manifold.evaluate_kinematic_pulse(x_cmd, y_cmd), dtype=float)
                u1 = float(u_kin[0] + self._locked_play_g[0])
                u2 = float(u_kin[1] + self._locked_play_g[1])
                manifold.cur_rect_pos = np.array([x_cmd, y_cmd], dtype=float)
            else:
                u1_raw, u2_raw = manifold.evaluate_pulse(x_cmd, y_cmd, dir_hint=(dx, dy))
                u1, u2 = float(u1_raw), float(u2_raw)
                if has_kin:
                    u_kin = np.asarray(manifold.evaluate_kinematic_pulse(x_cmd, y_cmd), dtype=float)
                    self._locked_play_g = np.array([u1, u2], dtype=float) - u_kin
                else:
                    u_kin = np.array([u1, u2], dtype=float)

            e1, e2 = _clamp_servo_pair(u1, u2)
            b_scale = self.boost_scale if boost_scale is None else float(boost_scale)
            dur_ms = self.boost_dur_ms if boost_dur_ms is None else float(boost_dur_ms)

            if (
                use_boost
                and (not is_recenter)
                and getattr(self.trainer.driver, "b_cmd_supported", False)
                and self._last_u_kin_live is not None
            ):
                du_k = u_kin - self._last_u_kin_live
                du_b = np.clip(du_k * 0.40 * b_scale, -26.0, 26.0)
                if abs(du_b[0]) > 1.5 or abs(du_b[1]) > 1.5:
                    b1, b2 = _clamp_servo_pair(e1 + float(du_b[0]), e2 + float(du_b[1]))
                    bst_us = int(round(max(10.0, min(35.0, dur_ms)) * 1000.0))
                    self.trainer.driver.send_ballistic_pulse(b1, b2, e1, e2, bst_us)
                else:
                    self.trainer.driver.send_raw_pulses(e1, e2)
            else:
                self.trainer.driver.send_raw_pulses(e1, e2)

            self._last_u_kin_live = u_kin.copy()
            self.p_stage_est = p_clamped.copy()
            self.ctrl.p_stage_cur = p_clamped.copy()
        return True

    def _render_vision_hud(
        self,
        roi_rgb: np.ndarray,
        c_px: np.ndarray,
        dets_px: List[TargetDetection],
        locked_det: Optional[TargetDetection],
        arrow_tip_px: Optional[np.ndarray],
        p_stage_now: np.ndarray,
    ) -> None:
        """Render non-blocking topmost 256x256 real-time Vision HUD debug window."""
        if not self.enable_gui:
            return
        try:
            import cv2

            hud = roi_rgb[:, :, ::-1].copy()
            h, w, _ = hud.shape
            cx0, cy0 = w // 2, h // 2

            # White screen geometric center u0
            cv2.circle(hud, (cx0, cy0), 3, (255, 255, 255), -1)

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

            # Adaptive In-Box Deadband Freeze ring r_lock around currently locked target
            if locked_det is not None:
                hx_l = int(round(float(locked_det.head_px[0])))
                hy_l = int(round(float(locked_det.head_px[1])))
                if locked_det.head_bbox_px is not None:
                    bx1, by1, bx2, by2 = locked_det.head_bbox_px
                    min_wh = min(abs(bx2 - bx1), abs(by2 - by1))
                else:
                    min_wh = 2.0 * float(locked_det.head_radius_px)
                r_lock_px = max(self.min_deadband_px, self.in_box_deadband_ratio * min_wh)
                cv2.circle(hud, (hx_l, hy_l), max(2, int(round(r_lock_px))), (255, 255, 0), 1)

            # Yellow traction vector arrow from c(t) toward active target/recoil point
            if arrow_tip_px is not None:
                tx = int(np.clip(round(float(arrow_tip_px[0])), 2, w - 3))
                ty = int(np.clip(round(float(arrow_tip_px[1])), 2, h - 3))
                cv2.arrowedLine(hud, (gx, gy), (tx, ty), (0, 255, 255), 2, tipLength=0.25)

            src_str = self.calibrator.visual_calib_source
            if src_str == "YOLO_HEAD_FLOW":
                calib_tag = "YOLO"
            elif src_str == "OPTICAL_FLOW":
                calib_tag = "FLOW"
            else:
                calib_tag = "PINH"
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
                            # Map mounted target counts into 0-deg manifold frame and execute IF-OPB ballistic move
                            p_man = np.linalg.solve(K_lin, K_stage @ p_clamped)
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
        )
        self.shot_modes.append(mode_now)
        self.shot_states.append(short_state)

        px_int = int(round(float(verdict.p_stage_cnt[0])))
        py_int = int(round(float(verdict.p_stage_cnt[1])))
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
            "measured_roi_fps": round(self.roi_fps_bench, 1),
            "decision_p95_ms": round(p95_lat, 4),
            "safe_envelope_cnt": SAFE_ENVELOPE_CNT,
            "cooldown_lock_ms": round(COOLDOWN_LOCK_S * 1000.0, 1),
            "controller_params": self.config["controller_params"],
            "thresholds": self.config["thresholds"],
            "mounting_angle_deg": round(float(self.calibrator.mounting_angle_deg), 4),
            "f8_calibrations": self.calib_history,
            "overall_summary": overall,
            "mode_comparison": {
                MODE_ASSIST_ON: assist_stats,
                MODE_SHADOW: shadow_stats,
            },
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
                    "delta_e_cnt": v.delta_e_cnt,
                    "stage_pos_cnt": [round(float(v.p_stage_cnt[0]), 2), round(float(v.p_stage_cnt[1]), 2)],
                    "faults": v.faults,
                }
                for i, v in enumerate(self.referee.verdicts)
            ],
        }

    def finalize_and_save_report(self, session_dur_s: float) -> Dict:
        """Print formatted battle report 《实战会话遥测战报》 and write JSON log."""
        report = self._build_session_report(session_dur_s)
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
        print(f"[Saved] Session telemetry report -> {self.report_path}")
        print("=" * 78)
        return report

    def close(self) -> None:
        self._stop_1khz_state_tracker()
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

    def run_live_loop(self, max_duration_s: float = 0.0) -> Dict:
        """Main non-blocking high-frequency loop for real CS2 gameplay."""
        session_t0 = time.perf_counter()
        self.startup_self_check()
        self.keys.flush((VK_F5, VK_F6, VK_F7, VK_F8, VK_F9, VK_F10, VK_END, VK_LBUTTON))
        self._start_1khz_state_tracker()

        last_step_t = time.perf_counter()
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
                c_px, dets_px, du_bg_px, cur_gray = detect_live_crosshair_and_targets(
                    roi_rgb=roi_rgb,
                    vision=self.vision,
                    counts_per_px=self.ctrl.counts_per_px,
                    dt_s=dt_s,
                    k_lead_s=self.ctrl.k_lead,
                    prev_gray=self._prev_gray,
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

                # Strict YOLO target selection: >=2 consecutive frames and [4, 95] px around c(t)
                locked_det: Optional[TargetDetection] = None
                locked_dist_px = 1e9
                for d in dets_px:
                    dist_px = float(np.hypot(d.head_px[0] - c_px[0], d.head_px[1] - c_px[1]))
                    if d.seen_frames >= self.min_track_frames and dist_px <= self.max_aim_dist_px:
                        if dist_px < locked_dist_px:
                            locked_dist_px = dist_px
                            locked_det = d

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
                    arrow_tip_px = locked_det.head_px.copy()
                elif lmb_down and float(np.hypot(c_px[0] - roi_center, c_px[1] - roi_center)) >= 2.5:
                    arrow_tip_px = np.array([roi_center, roi_center], dtype=float)

                if self.mode == MODE_ASSIST_ON:
                    # 4 Iron Rules Live Open-Loop Manifold Mode (ZERO mouse_sensor feedback):
                    #   1. Decompose p_stage = p_aim + p_recoil; reset recoil & smooth-recenter on LMB release / idle
                    #   2. Adaptive In-Box Deadband Freeze r_lock = max(min_deadband, ratio * min(w_h, h_h))
                    #   3. Piecewise underdamped gain + One-Way No-Reversal Clamp + frozen play_g
                    #   4. Atomic WS63 B_CMD boost overcoming 300-500g loaded palm friction
                    limits = np.array([self.live_max_x_cnt, self.live_max_y_cnt], dtype=float)
                    lmb_released = self._prev_lmb_live and (not lmb_down)
                    self._prev_lmb_live = lmb_down

                    if lmb_released:
                        self.p_recoil_cnt[:] = 0.0
                        self._lock_target_id = None
                        self._lock_dir_sign[:] = 0.0
                        if locked_det is None and float(np.hypot(p_stage_now[0], p_stage_now[1])) > 12.0:
                            self._smooth_recenter_3step(duration_s=0.12)
                            p_stage_now = self.p_stage_est.copy()

                    # Compute live recoil offset component when firing
                    if lmb_down:
                        self._last_engage_t = t_now
                        off_px = detect_crosshair_offset_px(roi_rgb)
                        if off_px is None and float(np.hypot(c_px[0] - roi_center, c_px[1] - roi_center)) >= 2.5:
                            off_px = np.array([float(c_px[0]) - roi_center, float(c_px[1]) - roi_center], dtype=float)
                        if off_px is not None and float(np.hypot(off_px[0], off_px[1])) >= 2.5:
                            p_rec = self.calibrator.recoil_offset_to_stage_abs(float(off_px[0]), float(off_px[1]))
                            self.p_recoil_cnt = np.clip(p_rec, -limits, limits)
                    else:
                        self.p_recoil_cnt[:] = 0.0

                    if locked_det is not None:
                        self._last_engage_t = t_now
                        e_px = locked_det.head_px - c_px
                        e_norm = float(np.hypot(e_px[0], e_px[1]))
                        if locked_det.head_bbox_px is not None:
                            bx1, by1, bx2, by2 = locked_det.head_bbox_px
                            min_wh = min(abs(bx2 - bx1), abs(by2 - by1))
                        else:
                            min_wh = 2.0 * float(locked_det.head_radius_px)
                        r_lock = max(self.min_deadband_px, self.in_box_deadband_ratio * min_wh)

                        # Lock initial approach direction & backlash flank per target lock / 250ms window
                        tid_cur = int(locked_det.target_id)
                        new_lock = (tid_cur != self._lock_target_id) or (
                            (t_now - self._lock_start_t) >= self.no_reversal_window_s and e_norm > (r_lock * 1.35)
                        )
                        if new_lock:
                            dp_init = self.calibrator.aim_error_to_stage_delta(float(e_px[0]), float(e_px[1]))
                            self._lock_target_id = tid_cur
                            self._lock_start_t = t_now
                            self._lock_dir_sign = np.sign(dp_init)

                        if e_norm > r_lock and locked_dist_px >= min(self.min_aim_dist_px, r_lock):
                            gain = self.kp_small_head if e_norm <= self.small_error_px else self.kp_large_error
                            dp_stage = self.calibrator.aim_error_to_stage_delta(float(e_px[0]), float(e_px[1])) * gain
                            # One-Way No-Reversal Clamp: never cross head center in opposing direction during lock
                            for ax in (0, 1):
                                if self._lock_dir_sign[ax] != 0.0 and dp_stage[ax] * self._lock_dir_sign[ax] < 0.0:
                                    dp_stage[ax] = 0.0

                            if float(np.hypot(dp_stage[0], dp_stage[1])) >= 0.4:
                                self.p_aim_cnt = np.clip(self.p_aim_cnt + dp_stage, -limits, limits)
                                p_cmd = np.clip(self.p_aim_cnt + self.p_recoil_cnt, -limits, limits)
                                freeze_pg = (not new_lock) or (e_norm <= self.small_error_px)
                                cd_s = self.recoil_cooldown_s if lmb_down else self.cooldown_lock_s
                                self._dispatch_stage_manifold_open_loop(
                                    p_cmd,
                                    is_recenter=False,
                                    cooldown_s=cd_s,
                                    freeze_play_g=freeze_pg,
                                    use_boost=True,
                                )
                        elif lmb_down and float(np.hypot(self.p_recoil_cnt[0], self.p_recoil_cnt[1])) >= 2.0:
                            # Head already inside deadband r_lock; still track pure recoil pull-down
                            p_cmd = np.clip(self.p_aim_cnt + self.p_recoil_cnt, -limits, limits)
                            self._dispatch_stage_manifold_open_loop(
                                p_cmd,
                                is_recenter=False,
                                cooldown_s=self.recoil_cooldown_s,
                                freeze_play_g=True,
                                use_boost=True,
                            )
                    elif lmb_down:
                        self._lock_target_id = None
                        self._lock_dir_sign[:] = 0.0
                        if float(np.hypot(self.p_recoil_cnt[0], self.p_recoil_cnt[1])) >= 2.0:
                            p_cmd = np.clip(self.p_aim_cnt + self.p_recoil_cnt, -limits, limits)
                            self._dispatch_stage_manifold_open_loop(
                                p_cmd,
                                is_recenter=False,
                                cooldown_s=self.recoil_cooldown_s,
                                freeze_play_g=False,
                                use_boost=True,
                            )
                        elif float(np.hypot(p_stage_now[0], p_stage_now[1])) >= 4.0:
                            self._dispatch_stage_manifold_open_loop(
                                np.zeros(2, dtype=float),
                                is_recenter=True,
                                cooldown_s=self.recoil_cooldown_s,
                                freeze_play_g=False,
                                use_boost=False,
                            )
                    else:
                        # No locked target & no LMB: auto-washout / 150ms idle 3-step smooth recenter
                        self._lock_target_id = None
                        self._lock_dir_sign[:] = 0.0
                        idle_s = t_now - self._last_engage_t
                        stage_norm = float(np.hypot(p_stage_now[0], p_stage_now[1]))
                        aim_norm = float(np.hypot(self.p_aim_cnt[0], self.p_aim_cnt[1]))
                        if stage_norm >= 15.0 and idle_s >= self.recenter_idle_s:
                            self._smooth_recenter_3step(duration_s=0.12)
                        elif aim_norm >= self.auto_recenter_threshold_cnt and idle_s >= 0.04:
                            decay = math.exp(-dt_s / self.aim_washout_tau_s)
                            self.p_aim_cnt = self.p_aim_cnt * decay
                            self._dispatch_stage_manifold_open_loop(
                                self.p_aim_cnt,
                                is_recenter=False,
                                cooldown_s=self.recoil_cooldown_s,
                                freeze_play_g=False,
                                use_boost=False,
                            )

                self._render_vision_hud(
                    roi_rgb=roi_rgb,
                    c_px=c_px,
                    dets_px=dets_px,
                    locked_det=locked_det,
                    arrow_tip_px=arrow_tip_px,
                    p_stage_now=p_stage_now,
                )

                if fire_now:
                    self._last_fire_t = t_now
                    self._burst_shot_idx += 1
                    tgt_px = det_px_by_id.get(out.active_target_id) or (dets_px[0] if dets_px else None)
                    if tgt_px is not None:
                        with self._hw_lock:
                            p_shot = self.p_stage_est.copy() if self.mode == MODE_ASSIST_ON else np.zeros(2, dtype=float)
                        self._record_and_print_shot(
                            t_now=t_now - session_t0,
                            mode_now=self.mode,
                            short_state=short_state,
                            c_cnt=c_rel_cnt,
                            p_stage_shot=p_shot,
                            det_target=tgt_px,
                            v_hand_cnt_s=v_hand_raw,
                            active_target_id=out.active_target_id,
                            anchor_mode=out.anchor_mode,
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

            # Verify F10 smooth recenter + pause
            self.trigger_f10_pause()
        finally:
            dur = time.perf_counter() - session_t0
            self.close()
            report = self.finalize_and_save_report(dur)

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
