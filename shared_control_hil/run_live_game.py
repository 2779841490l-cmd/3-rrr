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
COOLDOWN_LOCK_S = 0.085
AUTO_FIRE_PERIOD_S = 0.095


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
    ) -> None:
        self.config_path = config_path
        self.report_path = report_path
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

        self.vision = VisionROITracker(
            screen_w=int(self.config["screen"]["width"]),
            screen_h=int(self.config["screen"]["height"]),
            roi_size=int(self.config["screen"].get("roi_size", 256)),
            yolo_model_path=resolved_yolo_path,
            yolo_conf=float(vision_cfg.get("yolo_conf", 0.35)),
            yolo_imgsz=int(vision_cfg.get("yolo_imgsz", 256)),
            synthetic_only=bool(synthetic_capture),
        )
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
        self._live_spray_pi_cnt = np.zeros(2, dtype=float)

        # Dedicated 1000 Hz state/input tracker
        self._state_lock = threading.Lock()
        self._v_hand_1k = np.zeros(2, dtype=float)
        self._lmb_edge_pending = False
        self._counter_strafe_flag = False
        self._recenter_key_flag = False
        self._tracker_running = False
        self._tracker_thread: Optional[threading.Thread] = None

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
        print("  [F6 ]  Collect 10-shot AK-47 cl_crosshair_recoil 1 burst -> ak47_recoil_profile.json")
        print("  [F7 ]  Toggle AK_SPRAY_LOCK (hold LMB: 100 Hz closed-loop pull-down, release: recenter)")
        print("  [F8 ]  1.3s COM4 Online Self-Calibration (updates mounting angle & Jacobian)")
        print("  [F9 ]  Toggle Mode: PAUSED -> ASSIST_ON <-> SHADOW_REFEREE_ONLY")
        print("  [F10]  Emergency Recenter to (1496, 1466) us & PAUSE")
        print("  [END]  Safe Exit (or Ctrl+C in terminal) -> Recenter, Release COM4 & Save report")
        print("-" * 78)
        self.mode = MODE_PAUSED
        print(f"[STATE] Entered {self.mode}. Alt-tab into CS2 and press F8 (calib) or F9 (activate).")
        return self.roi_fps_bench

    def trigger_f8_calibration(self) -> Dict:
        """Pause dispatching, run 1.3s 4-pulse COM4 self-calibration, and refresh controller."""
        prev_mode = self.mode
        self._wait_hw_idle()
        print("\n[F8] Running 1.3s COM4 4-pulse online self-calibration...")
        res = self.calibrator.run_online_self_calib()
        self.config = self.calibrator.config
        self.calib_history.append(res)
        self.p_stage_est = np.zeros(2, dtype=float)
        self.ctrl.p_stage_cur = np.zeros(2, dtype=float)
        self.ctrl.reset_episode()
        self.vision.reset_filters()
        self._prev_sensor_xy = np.array(self.trainer.sensor.get_position(), dtype=float)
        self._last_hw_cmd_t = time.perf_counter()
        self.mode = prev_mode
        passed = bool(res["pass_time"] and res["pass_recon"])
        print(
            f"[F8 Self-Calib] elapsed={res['elapsed_s']:.3f}s | angle={res['mounting_angle_deg']:+.2f} deg | "
            f"recon_err={res['recon_error_pct']:.2f}% | PASS={passed} | Mode={self.mode}"
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

    def trigger_f6_collect_crosshair(self) -> Dict:
        """Pause assist, recenter neutral, and collect 10-shot CS2 cl_crosshair_recoil 1 burst."""
        self._wait_hw_idle()
        self.mode = MODE_PAUSED
        self.recenter_to_neutral(duration_s=0.18)
        res = self.ak_calibrator.collect_crosshair_burst(n_shots=10, save_profile=True)
        self.ctrl.reset_episode()
        return res

    def _run_ak_spray_burst(self, n_shots: int = 30, check_lmb: bool = True) -> Dict:
        """Execute 100 Hz closed-loop AK-47 pull-down + full-range cl_crosshair_recoil 1 PI visual servo."""
        self._wait_hw_idle()
        self.recenter_to_neutral(duration_s=0.12)

        pi_integral_cnt = np.zeros(2, dtype=float)
        spray_limits = np.array([self.spray_max_x_cnt, self.spray_max_y_cnt], dtype=float)
        last_vis_t = time.perf_counter()

        def _visual_trim() -> np.ndarray:
            nonlocal pi_integral_cnt, last_vis_t
            now_t = time.perf_counter()
            dt_vis = float(np.clip(now_t - last_vis_t, 0.001, 0.04))
            last_vis_t = now_t
            roi = self.vision.grab_roi()
            off = detect_crosshair_offset_px(roi)
            if off is None:
                return np.clip(pi_integral_cnt, -spray_limits, spray_limits)
            p_err = self.calibrator.screen_offset_to_pull_counts(off[0], off[1])
            pi_integral_cnt = np.clip(
                pi_integral_cnt + 4.5 * p_err * dt_vis,
                -0.75 * spray_limits,
                0.75 * spray_limits,
            )
            return np.clip(0.65 * p_err + pi_integral_cnt, -spray_limits, spray_limits)

        res = self.ak_calibrator.execute_ak_spray_closed_loop(
            self.trainer,
            n_shots=n_shots,
            use_feedback=True,
            rate_hz=250.0,
            reset_origin=False,
            check_lmb=check_lmb,
            visual_trim_fn=_visual_trim,
        )
        self.recenter_to_neutral(duration_s=0.18)
        print(
            f"[AK_SPRAY_LOCK Burst] dur={res['duration_s']:.2f}s | ticks={res['ticks']} | "
            f"mean_err={res['mean_bullet_err_cnt']:.2f} cnt | max={res['max_bullet_err_cnt']:.2f} cnt"
        )
        return res

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
        self.keys.flush((VK_F8, VK_F9, VK_F10, VK_END, VK_LBUTTON))
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
                if self.keys.pressed_once(VK_F6):
                    self.trigger_f6_collect_crosshair()
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

                if self.mode == MODE_ASSIST_ON:
                    spray_limits = np.array([self.spray_max_x_cnt, self.spray_max_y_cnt], dtype=float)
                    if lmb_down:
                        # Continuous full-range Crosshair-to-Center PI visual recoil servo (cl_crosshair_recoil 1)
                        off_px = detect_crosshair_offset_px(roi_rgb)
                        if off_px is None and float(np.hypot(c_px[0] - roi_center, c_px[1] - roi_center)) > 1.5:
                            off_px = np.array([float(c_px[0]) - roi_center, float(c_px[1]) - roi_center], dtype=float)
                        if off_px is not None:
                            p_err_pull = self.calibrator.screen_offset_to_pull_counts(off_px[0], off_px[1])
                            self._live_spray_pi_cnt = np.clip(
                                self._live_spray_pi_cnt + 4.5 * p_err_pull * dt_s,
                                -0.75 * spray_limits,
                                0.75 * spray_limits,
                            )
                            p_vis_recoil = np.clip(
                                0.65 * p_err_pull + self._live_spray_pi_cnt,
                                -spray_limits,
                                spray_limits,
                            )
                        else:
                            p_vis_recoil = self._live_spray_pi_cnt.copy()

                        if out.state in (STATE_2_COUNTER_STRAFE_LOCK, STATE_3_SPRAY_TRANSFER):
                            p_cmd = np.clip(out.p_stage_cmd_cnt + p_vis_recoil, -spray_limits, spray_limits)
                            self._dispatch_stage_command(p_cmd, is_recenter=False, spray_active=True)
                        elif float(np.hypot(p_vis_recoil[0], p_vis_recoil[1])) >= 2.0:
                            # Keep compensating AK spray even if head leaves YOLO box mid-burst
                            self._dispatch_stage_command(p_vis_recoil, is_recenter=False, spray_active=True)
                    else:
                        self._live_spray_pi_cnt[:] = 0.0
                        if out.state == STATE_4_MASKED_RECENTER and float(np.hypot(p_stage_now[0], p_stage_now[1])) >= 4.0:
                            self._dispatch_stage_command(np.zeros(2, dtype=float), is_recenter=True)
                        elif out.state in (STATE_2_COUNTER_STRAFE_LOCK, STATE_3_SPRAY_TRANSFER):
                            p_cmd = np.clip(out.p_stage_cmd_cnt, -SAFE_ENVELOPE_CNT, SAFE_ENVELOPE_CNT)
                            self._dispatch_stage_command(p_cmd, is_recenter=False, spray_active=False)

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
                    f"| mean_latency={yolo_ms:.2f} ms/frame | backend={self.vision.detector_backend}"
                )
                assert yolo_hits >= max(1, (len(val_files) * 2) // 3), f"YOLOv8n detection rate too low: {yolo_hits}/{len(val_files)}"

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
    parser.add_argument("--duration", type=float, default=0.0, help="Optional live session duration limit in seconds (0 = until END)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runner = LiveGameSharedControlRunner(
        config_path=args.config,
        report_path=args.report,
        synthetic_capture=False,
    )
    if args.self_test:
        runner.run_self_test()
    else:
        runner.run_live_loop(max_duration_s=float(args.duration))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
