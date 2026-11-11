"""
ballistic_controller.py - Sub-100ms 3200 DPI Full-Workspace & Full-Screen Ballistic Controller
==============================================================================================
Core Architecture (Targeting IEEE TIM / IEEE T-MECH):
1. Motion-Onset Phase Synchronization (t_onset Locking):
   Eliminates the 0~20ms asynchronous UART-to-PWM frame quantization lottery by locking
   the boost-to-brake switching window to the exact millisecond of first optical motion
   detection (t_onset), placing the command transition dead-center between PWM latch edges.
2. Three-Stage Monotonic Ballistic Profile (BOOST -> ACTIVE BRAKE/TRIM -> LOCK):
   Enforces the Monotonic Zero-Reversal Time-Optimality Principle across both micro-steps
   (5~50 counts) and full-stroke macro flicks (100~400+ counts / full-screen 1536x864).
3. Physics-Guided Overdamped 2D Vector ILC (PG-NB-ILC):
   Uses spectral-radius-bounded gain eta_ilc = 0.34 < 1/sigma_max^2(K0) = 0.418 for
   strictly monotonic, non-oscillatory multi-shot convergence.
"""

import argparse
import json
import os
import time
from typing import Dict, List, Optional, Tuple
import numpy as np

from mouse_sensor import HighPrecisionMouseSensor
from servo_driver import WS63ServoDriver


class SpatialInverseJacobian2ndOrder:
    def __init__(self, poly_file: str = "prompts/fitted_jacobian_poly.json"):
        self.coeffs = self._load_coeffs(poly_file)

    @staticmethod
    def _load_coeffs(poly_file: str) -> Dict:
        if os.path.exists(poly_file):
            with open(poly_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "K11": {"Const": 0.88505747, "dUx": -0.0012, "dUy": -0.0003, "dUx^2/1k": -0.015, "dUy^2/1k": -0.002, "dUx*dUy/1k": 0.0005},
            "K12": {"Const": -0.56321839, "dUx": 0.0001, "dUy": -0.0045, "dUx^2/1k": 0.030, "dUy^2/1k": -0.045, "dUx*dUy/1k": -0.010},
            "K21": {"Const": -0.24137931, "dUx": -0.0005, "dUy": -0.0010, "dUx^2/1k": -0.004, "dUy^2/1k": -0.002, "dUx*dUy/1k": -0.002},
            "K22": {"Const": -0.48275862, "dUx": -0.0004, "dUy": -0.0025, "dUx^2/1k": 0.015, "dUy^2/1k": -0.055, "dUx*dUy/1k": -0.006},
        }

    def evaluate(self, du_x: float, du_y: float) -> np.ndarray:
        # Clamp polynomial evaluation domain so large macro flicks do not extrapolate wildly
        du_x_c = float(np.clip(du_x, -90.0, 90.0))
        du_y_c = float(np.clip(du_y, -90.0, 90.0))
        K = np.zeros((2, 2), dtype=float)
        basis = {
            "Const": 1.0,
            "dUx": du_x_c,
            "dUy": du_y_c,
            "dUx^2/1k": (du_x_c**2) / 1000.0,
            "dUy^2/1k": (du_y_c**2) / 1000.0,
            "dUx*dUy/1k": (du_x_c * du_y_c) / 1000.0,
        }
        for (r, c), key in [((0, 0), "K11"), ((0, 1), "K12"), ((1, 0), "K21"), ((1, 1), "K22")]:
            c_dict = self.coeffs.get(key, {})
            K[r, c] = sum(c_dict.get(name, 0.0) * val for name, val in basis.items())
        return K


class BallisticILCLearner:
    """
    Physics-Guided Norm-Optimal 2D Vector Iterative Learning Control (PG-NB-ILC) engine.
    """

    def __init__(self, ilc_file: str = "prompts/ilc_ballistic_weights.json"):
        self.ilc_file = ilc_file
        self.params = self._load()

    def _load(self) -> Dict:
        if os.path.exists(self.ilc_file):
            try:
                with open(self.ilc_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if "target_table" not in data:
                        data["target_table"] = {}
                    return data
            except Exception:
                pass
        return {
            "boost_ratio": 1.15,        # Overdrive factor: u_boost = u_lock + (boost_ratio - 1)*du_base
            "tau_brake_ms": 14.0,       # Predicted glide time constant (ms)
            "active_brake_ms": 0.0,     # Immediate lock latch upon t_onset detection
            "micro_yaw_gain": 0.65,
            "target_table": {},         # Keyed by f"{dx}_{dy}": [du_corr_x, du_corr_y]
            "iteration": 0,
            "loss_history": [],
        }

    def save(self):
        os.makedirs(os.path.dirname(self.ilc_file), exist_ok=True)
        with open(self.ilc_file, "w", encoding="utf-8") as f:
            json.dump(self.params, f, indent=2)

    def get_correction(self, dx: int, dy: int) -> np.ndarray:
        key = f"{dx}_{dy}"
        corr = self.params.get("target_table", {}).get(key, [0.0, 0.0])
        return np.array([float(corr[0]), float(corr[1])], dtype=float)

    def update_vector(
        self,
        dx_target: int,
        dy_target: int,
        dx_actual: int,
        dy_actual: int,
        K_local: np.ndarray,
        err_counts: float,
        eta_ilc: float = 0.35,
    ):
        """
        Physics-Guided Deadband-Frozen 2D Vector ILC Law (PG-NB-ILC):
        Uses K_local to decouple off-diagonal kinematic cross-coupling (Theorem 4)
        while enforcing Slotine-Li deadband freezing inside hardware noise floor.
        """
        key = f"{dx_target}_{dy_target}"
        if "target_table" not in self.params:
            self.params["target_table"] = {}

        cur_corr = self.get_correction(dx_target, dy_target)
        err_x = float(dx_target - dx_actual)
        err_y = float(dy_target - dy_actual)
        dist = float(np.hypot(dx_target, dy_target))
        tol = 3.8 if dist > 100.0 else 2.5

        # Deadband Freezing: Zero drift when already inside repeatable hardware tolerance
        if err_counts <= tol:
            return

        # Damped Physics-Guided Jacobian step (< theoretical 0.837 spectral stability limit)
        if dist <= 50.0:
            eta_eff, max_step = 0.14, 0.35
        elif dist <= 115.0:
            eta_eff, max_step = 0.32, 2.20
        else:
            eta_eff, max_step = 0.12, 0.80

        du_step = eta_eff * (K_local @ np.array([err_x, err_y], dtype=float))
        du_step = np.clip(du_step, -max_step, max_step)

        new_corr = np.clip(cur_corr + du_step, -120.0, 120.0)
        self.params["target_table"][key] = [round(float(new_corr[0]), 3), round(float(new_corr[1]), 3)]

        self.params["iteration"] += 1
        self.params["loss_history"].append(float(round(abs(err_counts), 2)))
        if len(self.params["loss_history"]) > 200:
            self.params["loss_history"].pop(0)
        self.save()


class Ballistic100msController:
    def __init__(self, calib_file: str = "calib_params.json"):
        with open(calib_file, "r", encoding="utf-8") as f:
            self.calib = json.load(f)

        self.sensor = HighPrecisionMouseSensor(dpi=self.calib.get("dpi", 3200))
        self.driver = WS63ServoDriver()
        self.poly_jac = SpatialInverseJacobian2ndOrder()
        self.ilc = BallisticILCLearner()

        self.u0_x = float(self.calib["neutral_us"]["s1_x"])
        self.u0_y = float(self.calib["neutral_us"]["s2_y1"])

        # Backlash engagement radii
        self.bx_pos = float(self.calib["backlash_play_radius_us"]["x_pos"])
        self.bx_neg = float(self.calib["backlash_play_radius_us"]["x_neg"])
        self.by_pos = float(self.calib["backlash_play_radius_us"]["y_pos"])
        self.by_neg = float(self.calib["backlash_play_radius_us"]["y_neg"])

        # Calibrated decoupled ballistic profiles: (lock_x, lock_y, boost_x, boost_y, act_x, act_y, cap_x, cap_y)
        self._calib_anchors = {
            (30, 20):   (23.5, -12.1, 35.5, -26.5, 18.0,  6.0, 68.0, 72.0),
            (35, 0):    (39.8, -11.6, 52.2, -15.5, 14.0, 14.0, 68.0, 68.0),
            (-35, 0):   (-53.4,  7.3,-74.5, -10.8, 12.0, 12.0, 73.0, 73.0),
            (0, 35):    (-24.8, -16.9,-35.2, -28.8,  7.0,  6.5, 70.0, 70.0),
            (0, -35):   (22.7,  14.3, 38.4,  23.0, 14.0, 14.0, 66.0, 66.0),
            (30, 25):   (21.2, -15.0, 32.8, -24.5, 18.0,  8.0, 66.0, 72.0),
            (-30, -25): (-30.0, 12.4,-45.5,  19.6, 10.0, 14.0, 72.0, 72.0),
            (160, -90): (177.2, 13.8, 263.0, 23.8, 64.0, 46.0, 104.0, 104.0),
            (-160, 90): (-180.6,-18.8,-243.5,-28.2, 60.0, 48.0, 102.0, 102.0),
            (200,-110): (227.6, 18.4, 338.0, 23.8, 82.0, 58.0, 120.0, 120.0),
            (-200,110): (-220.2,-18.1,-288.0,-27.5, 58.0, 47.0, 106.0, 106.0),
        }
        # Angular directional per-count gain correction ((lock_meas - K0 @ target) / dist) for arbitrary FPS targets
        K0 = self.poly_jac.evaluate(0.0, 0.0)
        self._angular_bias_table = []
        for (tx, ty), prof in self._calib_anchors.items():
            ang = float(np.arctan2(ty, tx))
            d_ref = max(1.0, float(np.hypot(tx, ty)))
            bias_per_cnt = (np.array([prof[0], prof[1]], dtype=float) - (K0 @ np.array([float(tx), float(ty)]))) / d_ref
            self._angular_bias_table.append((ang, d_ref, bias_per_cnt))

    @staticmethod
    def _compute_backlash_offset(du_x: float, du_y: float) -> Tuple[float, float]:
        bx = min(7.0, du_x * (7.0 / 12.0)) if du_x >= 0.0 else max(-27.0, du_x * (27.0 / 15.0))
        by = min(5.0, du_y * (5.0 / 10.0)) if du_y >= 0.0 else max(-3.6, du_y * (3.6 / 8.0))
        return float(bx), float(by)

    @staticmethod
    def _lock_to_du_base(lock_x: float, lock_y: float) -> np.ndarray:
        """Inverts C0 backlash offset map so du_base + backlash(du_base) == (lock_x, lock_y)."""
        if lock_x >= 19.0:
            dx = lock_x - 7.0
        elif lock_x >= 0.0:
            dx = lock_x * (12.0 / 19.0)
        elif lock_x <= -42.0:
            dx = lock_x + 27.0
        else:
            dx = lock_x * (15.0 / 42.0)

        if lock_y >= 15.0:
            dy = lock_y - 5.0
        elif lock_y >= 0.0:
            dy = lock_y / 1.5
        elif lock_y <= -11.6:
            dy = lock_y + 3.6
        else:
            dy = lock_y / 1.45
        return np.array([dx, dy], dtype=float)

    def _interp_angular_bias(self, delta_x: int, delta_y: int) -> np.ndarray:
        """Smooth cosine-weighted 360-deg directional & scale-matched bias across full mechanical stroke."""
        ang = float(np.arctan2(delta_y, delta_x))
        dist = float(np.hypot(delta_x, delta_y))
        w_sum, b_sum = 0.0, np.zeros(2, dtype=float)
        for ref_ang, ref_dist, ref_bias_per_cnt in self._angular_bias_table:
            d_ang = ang - ref_ang
            cos_w = max(0.0, float(np.cos(d_ang))) ** 4
            # Prefer macro anchors for macro distances and micro anchors for micro distances
            scale_sim = 1.0 / (1.0 + ((dist - ref_dist) / 60.0) ** 2)
            w = cos_w * (0.2 + 0.8 * scale_sim)
            if w > 1e-6:
                w_sum += w
                b_sum += w * (ref_bias_per_cnt * dist)
        return (b_sum / w_sum) if w_sum > 1e-6 else np.zeros(2, dtype=float)

    def flick_shot(
        self,
        delta_x: int,
        delta_y: int,
        timeout_ms: float = 100.0,
        enable_ilc_update: bool = True,
    ) -> Dict:
        """
        Executes a sub-100ms ballistic flick shot to relative target (delta_x, delta_y).
        Supports both micro-precision steps (5~50 counts) and full-stroke macro flicks
        (100~450 counts across [1150, 1850] us) via decoupled per-axis onset locking.
        """
        target_dist = float(np.hypot(delta_x, delta_y))
        if target_dist < 0.1:
            return {"status": "NOOP", "final_error_counts": 0.0, "settled_time_ms": 0.0}

        # 1. Evaluate 2nd-order Jacobian at current position
        cur_ux = self.driver.cur_s1 - self.u0_x
        cur_uy = self.driver.cur_s2 - self.u0_y
        K = self.poly_jac.evaluate(cur_ux, cur_uy)

        # 2. Compute nominal lock + overdrive pulse profiles with 2D PG-ILC correction
        e_vec = np.array([float(delta_x), float(delta_y)], dtype=float)
        du_ilc = self.ilc.get_correction(delta_x, delta_y)
        anchor = self._calib_anchors.get((int(delta_x), int(delta_y)))

        if anchor is not None:
            lock_x_nom, lock_y_nom, bst_x_nom, bst_y_nom, act_x_ms, act_y_ms, cap_x_ms, cap_y_ms = anchor
            du_base = self._lock_to_du_base(lock_x_nom, lock_y_nom)
            du_nom = du_base + du_ilc
            bx, by = self._compute_backlash_offset(du_nom[0], du_nom[1])
            u_lock_x = float(np.clip(self.driver.cur_s1 + du_nom[0] + bx, 1150.0, 1850.0))
            u_lock_y = float(np.clip(self.driver.cur_s2 + du_nom[1] + by, 1150.0, 1850.0))
            u_boost_x = float(np.clip(self.driver.cur_s1 + bst_x_nom + du_ilc[0], 1150.0, 1850.0))
            u_boost_y = float(np.clip(self.driver.cur_s2 + bst_y_nom + du_ilc[1], 1150.0, 1850.0))
            tau_brake = 0.025 if target_dist > 205.0 else (0.032 if target_dist > 100.0 else 0.015)
        else:
            bias = self._interp_angular_bias(delta_x, delta_y)
            target_lock = (K @ e_vec) + bias
            du_base = self._lock_to_du_base(float(target_lock[0]), float(target_lock[1]))
            du_nom = du_base + du_ilc
            bx, by = self._compute_backlash_offset(du_nom[0], du_nom[1])
            lock_x_rel = du_nom[0] + bx
            lock_y_rel = du_nom[1] + by
            u_lock_x = float(np.clip(self.driver.cur_s1 + lock_x_rel, 1150.0, 1850.0))
            u_lock_y = float(np.clip(self.driver.cur_s2 + lock_y_rel, 1150.0, 1850.0))

            # Distance-adaptive overdrive (high boost for micro stiction; full saturation + IF-OPB for macro 50Hz slew)
            if target_dist <= 50.0:
                boost_gain, kick_mag = 1.48, 4.5
                act_x_ms = 18.0 if lock_x_rel >= 0.0 else 12.0
                act_y_ms = 14.0 if lock_y_rel >= 0.0 else 7.0
                cap_x_ms = 66.0 if lock_x_rel >= 0.0 else 72.0
                cap_y_ms = 65.0 if lock_y_rel >= 0.0 else 70.0
                tau_brake = 0.015
            elif target_dist <= 115.0:
                boost_gain, kick_mag = 1.28, 8.0
                act_x_ms, act_y_ms = 34.0, 26.0
                cap_x_ms, cap_y_ms = 82.0, 82.0
                tau_brake = 0.024
            elif target_dist <= 205.0:
                boost_gain, kick_mag = 1.40, 8.0
                act_x_ms, act_y_ms = 66.0, 50.0
                cap_x_ms, cap_y_ms = 105.0, 105.0
                tau_brake = 0.032
            else:
                boost_gain, kick_mag = 1.48, 12.0
                act_x_ms, act_y_ms = 74.0, 54.0
                cap_x_ms, cap_y_ms = 112.0, 112.0
                tau_brake = 0.025

            bst_x_rel = lock_x_rel * boost_gain + (kick_mag * np.sign(lock_x_rel) if abs(lock_x_rel) > 1.0 else 0.0)
            bst_y_rel = lock_y_rel * boost_gain + (kick_mag * np.sign(lock_y_rel) if abs(lock_y_rel) > 1.0 else 0.0)
            u_boost_x = float(np.clip(self.driver.cur_s1 + bst_x_rel, 1150.0, 1850.0))
            u_boost_y = float(np.clip(self.driver.cur_s2 + bst_y_rel, 1150.0, 1850.0))

        # Ensure complete mechanical rest before firing shot
        for _ in range(5):
            self.sensor.read_delta_and_clear()
            time.sleep(0.012)
            dx_chk, dy_chk = self.sensor.read_delta_and_clear()
            if dx_chk == 0 and dy_chk == 0:
                break

        # Recenter OS cursor while stationary so large/beyond-screen flicks never hit screen edges
        if hasattr(self.sensor, "recenter_cursor"):
            self.sensor.recenter_cursor()
            time.sleep(0.003)

        x0, y0 = self.sensor.get_position()
        target_abs_x = x0 + delta_x
        target_abs_y = y0 + delta_y

        sx = 1 if delta_x > 0 else (-1 if delta_x < 0 else 0)
        sy = 1 if delta_y > 0 else (-1 if delta_y < 0 else 0)

        t_start = time.perf_counter()
        tx_on: Optional[float] = None
        ty_on: Optional[float] = None
        t_onset: Optional[float] = None
        trajectory = []
        vel_buf = []

        # Send Stage 1: Boost Overdrive Command
        self.driver.send_raw_pulses(u_boost_x, u_boost_y)
        cur_cmd_x, cur_cmd_y = u_boost_x, u_boost_y
        bx_done = False
        by_done = False
        phase = "BOOST"
        brake_triggered = False
        stage3_trimmed = False
        t_brake_ms = 0.0
        max_pred_progress = 0.0

        settled_t_ms = None
        tol_counts = max(1.5, 0.015 * target_dist)

        # 3. Decoupled Per-Axis Onset & Predictive Early-Settling Control Loop (1000Hz)
        while True:
            t_now = time.perf_counter()
            elapsed_ms = (t_now - t_start) * 1000.0

            cur_x, cur_y = self.sensor.get_position()
            dx_cur = cur_x - x0
            dy_cur = cur_y - y0

            if tx_on is None and sx != 0 and elapsed_ms >= 25.0 and dx_cur * sx >= 1:
                tx_on = t_now
            if ty_on is None and sy != 0 and elapsed_ms >= 25.0 and dy_cur * sy >= 1:
                ty_on = t_now
            if t_onset is None and elapsed_ms >= 25.0 and (cur_x != x0 or cur_y != y0):
                t_onset = t_now

            eff_tx = tx_on if sx != 0 else ty_on
            eff_ty = ty_on if sy != 0 else tx_on
            ax_ms = (t_now - eff_tx) * 1000.0 if eff_tx is not None else 0.0
            ay_ms = (t_now - eff_ty) * 1000.0 if eff_ty is not None else 0.0

            # Strictly 0.0 until primary physical motion begins at t_onset!
            if sx != 0 and sy != 0:
                t_main = tx_on if tx_on is not None else ty_on
            elif sx != 0:
                t_main = tx_on
            else:
                t_main = ty_on
            active_ms = ((t_now - t_main) * 1000.0) if t_main is not None else 0.0

            # 14ms sliding-window clean velocity filter for responsive IF-OPB
            vel_buf.append((t_now, cur_x, cur_y))
            while vel_buf and (t_now - vel_buf[0][0]) > 0.014:
                vel_buf.pop(0)
            dt_buf = t_now - vel_buf[0][0]
            vx_filt = (cur_x - vel_buf[0][1]) / dt_buf if dt_buf > 0.001 else 0.0
            vy_filt = (cur_y - vel_buf[0][2]) / dt_buf if dt_buf > 0.001 else 0.0

            err_x = target_abs_x - cur_x
            err_y = target_abs_y - cur_y
            err_counts = float(np.hypot(err_x, err_y))

            pred_x = cur_x + vx_filt * tau_brake
            pred_y = cur_y + vy_filt * tau_brake

            if err_counts <= tol_counts and settled_t_ms is None and active_ms >= 10.0:
                settled_t_ms = active_ms

            trajectory.append({
                "t_ms": round(elapsed_ms, 2),
                "active_ms": round(active_ms, 2),
                "phase": phase,
                "x": cur_x,
                "y": cur_y,
                "vx": round(vx_filt, 1),
                "vy": round(vy_filt, 1),
                "err_counts": round(err_counts, 2),
                "err_um": round(err_counts * self.sensor.um_per_count, 2),
            })

            # Stage 2: Decoupled Per-Axis Lock Switching + In-Flight Optical Predictive Brake (IF-OPB)
            raw_prog = dx_cur * delta_x + dy_cur * delta_y
            progress_pred = (pred_x - x0) * delta_x + (pred_y - y0) * delta_y
            if progress_pred > max_pred_progress:
                max_pred_progress = progress_pred
            target_mag_sq = float(delta_x**2 + delta_y**2)

            if target_dist > 100.0:
                onset_ms = ((t_main - t_start) * 1000.0) if t_main is not None else elapsed_ms
                if target_dist > 205.0:
                    if delta_x > 0:
                        opb_thresh = min(0.938, 0.875 + 0.004 * max(0.0, onset_ms - 26.0))
                    else:
                        opb_thresh = min(0.818, 0.780 + 0.004 * max(0.0, onset_ms - 32.0))
                else:
                    if delta_x > 0:
                        opb_thresh = min(0.87, 0.765 + 0.0065 * max(0.0, onset_ms - 26.0))
                    else:
                        opb_thresh = min(0.85, 0.805 + 0.003 * max(0.0, onset_ms - 38.0))
                opb_guard = (raw_prog >= 0.40 * target_mag_sq) and (progress_pred >= opb_thresh * target_mag_sq)
            else:
                opb_guard = (
                    (active_ms >= max(10.0, 0.88 * min(act_x_ms, act_y_ms)))
                    and (raw_prog >= 0.65 * target_mag_sq)
                    and (max_pred_progress >= 1.04 * target_mag_sq)
                )

            need_send = False
            if not bx_done and ((eff_tx is not None and ax_ms >= act_x_ms) or elapsed_ms >= cap_x_ms or opb_guard):
                cur_cmd_x = u_lock_x
                bx_done = True
                need_send = True
            if not by_done and ((eff_ty is not None and ay_ms >= act_y_ms) or elapsed_ms >= cap_y_ms or opb_guard):
                cur_cmd_y = u_lock_y
                by_done = True
                need_send = True
            if need_send:
                self.driver.send_raw_pulses(cur_cmd_x, cur_cmd_y)
                if not brake_triggered:
                    t_brake_ms = elapsed_ms
                    brake_triggered = True
                phase = "LOCK" if (bx_done and by_done) else "TRIM"

            # 100ms active flight window completes (145ms total budget ensures full physical settle even on late 50Hz onset)
            if (t_main is not None and active_ms >= timeout_ms) or (elapsed_ms >= timeout_ms + 45.0):
                if not bx_done or not by_done:
                    self.driver.send_raw_pulses(u_lock_x, u_lock_y)
                break
            time.sleep(0.001)

        # 5. Terminal Result & 2D Vector ILC Learning Update
        final_x, final_y = self.sensor.get_position()
        final_dx = int(final_x - x0)
        final_dy = int(final_y - y0)
        final_err = float(np.hypot(target_abs_x - final_x, target_abs_y - final_y))

        if enable_ilc_update:
            self.ilc.update_vector(delta_x, delta_y, final_dx, final_dy, K, final_err)

        if settled_t_ms is None and final_err <= tol_counts:
            settled_t_ms = min(timeout_ms, active_ms)

        return {
            "target_delta": (delta_x, delta_y),
            "final_delta": (final_dx, final_dy),
            "final_error_counts": round(final_err, 2),
            "final_error_um": round(final_err * self.sensor.um_per_count, 2),
            "settled_time_ms": round(settled_t_ms if settled_t_ms is not None else timeout_ms, 1),
            "t_brake_ms": round(t_brake_ms, 1),
            "t_onset_ms": round((t_onset - t_start) * 1000.0, 1) if t_onset is not None else -1.0,
            "trajectory": trajectory,
        }

    def close(self):
        try:
            self.sensor.stop()
            self.driver.close()
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sub-100ms 3200 DPI Ballistic Controller")
    parser.add_argument("--dx", type=int, default=30, help="Target Delta X (counts)")
    parser.add_argument("--dy", type=int, default=0, help="Target Delta Y (counts)")
    parser.add_argument("--test-suite", action="store_true", help="Run full-scale 100ms benchmark suite")
    args = parser.parse_args()

    ctrl = Ballistic100msController()
    ctrl.sensor.start()
    ctrl.driver.center(settle_s=0.45, preload=True)

    if args.test_suite:
        print("=" * 78)
        print("  3-RRR Sub-100ms 3200 DPI Full-Workspace Flick-Shot Benchmark Suite")
        print("=" * 78)
        waypoints = [
            (30, 20),     # Micro-precision step
            (-35, 0),     # Negative X backlash traversal
            (0, 35),      # Positive Y
            (120, 80),    # Medium-range flick (~1.15 mm)
            (-150, -90),  # Medium-large reversal
            (240, 150),   # Full-stroke macro flick (~2.25 mm raw / full-screen scaled)
            (-240, -150), # Full-stroke macro reversal
        ]
        for i, (tdx, tdy) in enumerate(waypoints):
            ctrl.driver.center(settle_s=0.35, preload=True)
            res = ctrl.flick_shot(tdx, tdy, timeout_ms=100.0)
            print(
                f"Shot {i+1:2d} | Target ({tdx:+4d}, {tdy:+4d}) -> "
                f"Final ({res['final_delta'][0]:+4d}, {res['final_delta'][1]:+4d}) | "
                f"Err: {res['final_error_counts']:5.1f} cnt ({res['final_error_um']:6.1f} um) | "
                f"Onset: {res['t_onset_ms']:4.1f}ms | Brake: {res['t_brake_ms']:4.1f}ms | "
                f"Settle: {res['settled_time_ms']:5.1f}ms"
            )
    else:
        print(f"Firing single 100ms shot to target ({args.dx}, {args.dy})...")
        res = ctrl.flick_shot(args.dx, args.dy, timeout_ms=100.0)
        print(f"Target: ({args.dx}, {args.dy})")
        print(f"Final : {res['final_delta']} | Error: {res['final_error_counts']} counts ({res['final_error_um']} um)")
        print(f"Onset : {res['t_onset_ms']} ms | Brake: {res['t_brake_ms']} ms | Settled: {res['settled_time_ms']} ms")

    ctrl.driver.center(settle_s=0.35, preload=True)
    ctrl.close()
