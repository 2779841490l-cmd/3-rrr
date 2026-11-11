"""
ballistic_controller.py - Full-Workspace 200Hz Guided Straight-Line 3200 DPI Controller
======================================================================================
Replaces legacy narrow-range polynomial & 11-point dictionary lookup with:
1. True Hardware 3200 DPI Raw Input sensing (`7.9375 um/count`, immune to screen clipping
   and Windows Pointer Acceleration).
2. Full-Envelope 2D Inverse Kinematic Manifold (`WorkspaceField2D` across [-947..+1080,
   -365..+1184] counts) with real position-dependent Jacobian K(x, y).
3. 200Hz Synchronized Straight-Line Guidance + Cross-Track Steering (CTEC) + In-Flight
   Optical Predictive Braking (IF-OPB): actively eliminates S-curve banana arcs caused
   by asymmetric X/Y gear backlash and J_12=-1.7 cross-coupling.
4. Spatially Continuous Direction-Gated Gaussian-Kernel ILC (`SpatialResidualILC`) that
   generalizes learned trims to arbitrary unseen coordinates across all 360 degrees.
"""

import argparse
import time
from typing import Dict, List, Optional, Tuple
import numpy as np

from mouse_sensor import HighPrecisionMouseSensor
from servo_driver import WS63ServoDriver
from workspace_model import SpatialResidualILC, WorkspaceField2D


class Ballistic100msController:
    def __init__(
        self,
        calib_file: str = "calib_params.json",
        map_file: str = "experiments/workspace_map_full.json",
        ilc_file: str = "prompts/ilc_ballistic_weights.json",
    ):
        self.workspace = WorkspaceField2D(map_file=map_file, calib_file=calib_file)
        self.ilc = SpatialResidualILC(ilc_file=ilc_file)
        self.sensor = HighPrecisionMouseSensor(dpi=3200)
        self.driver = WS63ServoDriver()
        self.driver.set_pwm_frequency(200)

        self.u0_x = float(self.workspace.u0_x)
        self.u0_y = float(self.workspace.u0_y)

    def _clamp_pulses(self, du_x: float, du_y: float) -> Tuple[float, float]:
        ux = float(np.clip(self.u0_x + du_x, self.driver.SAFE_MIN_US, self.driver.SAFE_MAX_US))
        uy = float(np.clip(self.u0_y + du_y, self.driver.SAFE_MIN_US, self.driver.SAFE_MAX_US))
        # Also guarantee dual-Y S3 stays inside [SAFE_MIN_US, SAFE_MAX_US]
        s3 = 3000.0 - uy
        if s3 < self.driver.SAFE_MIN_US:
            uy = 3000.0 - self.driver.SAFE_MIN_US
        elif s3 > self.driver.SAFE_MAX_US:
            uy = 3000.0 - self.driver.SAFE_MAX_US
        return ux, uy

    def _compute_lock_command(self, dx: float, dy: float) -> Tuple[float, float, np.ndarray, np.ndarray]:
        du_eq = self.workspace.target_to_equilibrium_du(dx, dy)
        du_ilc = self.ilc.get_correction(dx, dy)
        K_end = self.workspace.local_inverse_jacobian(dx, dy)
        ux_lock, uy_lock = self._clamp_pulses(du_eq[0] + du_ilc[0], du_eq[1] + du_ilc[1])
        return ux_lock, uy_lock, du_eq, K_end

    def _compute_guided_command(
        self,
        p_vec: np.ndarray,
        v_vec: np.ndarray,
        u_kin_target: np.ndarray,
        K_end: np.ndarray,
        L: float,
        d_hat: np.ndarray,
        n_hat: np.ndarray,
        du_ilc: np.ndarray,
        elapsed_ms: float,
    ) -> Tuple[float, float]:
        """
        Contouring-Error-Controlled (CEC) 200Hz Straight-Line Overdrive Guidance:
        - Holds full kinematic equilibrium `u_kin_target` (never starves setpoint below target)
        - Adds tangential overdrive purely along `d_hat` attenuated when `|e_perp|` grows
        - Applies pure orthogonal cross-track PD steering along `n_hat` mapped via `K_end`
          so fast axes hold back while lagging axes get accelerated across backlash.
        """
        p_par = float(np.dot(p_vec, d_hat))
        e_perp = float(np.dot(p_vec, n_hat))
        v_perp = float(np.dot(v_vec, n_hat))
        rem = max(0.0, L - p_par)

        # Hold back extra overdrive if sideways deviation grows so lagging axis catches up
        dev_ratio = abs(e_perp) / max(14.0, 0.08 * L)
        boost_scale = 1.0 / (1.0 + 0.85 * (dev_ratio ** 2))

        if L <= 90.0:
            raw_boost = min(50.0, 0.40 * rem + 15.0)
        elif L <= 300.0:
            raw_boost = min(120.0, 0.42 * rem + 25.0)
        else:
            raw_boost = min(180.0, 0.40 * rem + 35.0)
        tang_boost = boost_scale * raw_boost

        # Orthogonal Cross-Track PD steering directly in Cartesian space
        kp_perp = 1.35 if L > 90.0 else 1.05
        kd_perp = 0.020
        perp_cap = min(160.0, 0.70 * L)
        perp_corr = -float(np.clip(kp_perp * e_perp + kd_perp * v_perp, -perp_cap, perp_cap))

        # Local Jacobian around mid-path or endpoint
        K_loc = self.workspace.local_inverse_jacobian(
            float(min(L, max(0.0, p_par) + 0.35 * rem) * d_hat[0]),
            float(min(L, max(0.0, p_par) + 0.35 * rem) * d_hat[1]),
        )
        u_kin_cmd = u_kin_target + K_loc @ (tang_boost * d_hat + perp_corr * n_hat)
        du_cmd = self.workspace._unstrip_u(u_kin_cmd)

        # Initial breakaway pre-kick before motion onset along actual motor actuation direction
        if float(np.linalg.norm(p_vec)) < 6.0 and elapsed_ms < 50.0:
            if abs(u_kin_target[0]) > 2.0:
                du_cmd[0] += np.sign(u_kin_target[0]) * 12.0
            if abs(u_kin_target[1]) > 2.0:
                du_cmd[1] += np.sign(u_kin_target[1]) * 6.0

        ilc_frac = float(np.clip(max(0.40, p_par / max(1.0, L)), 0.40, 1.0))
        return self._clamp_pulses(du_cmd[0] + ilc_frac * du_ilc[0], du_cmd[1] + ilc_frac * du_ilc[1])

    def flick_shot(
        self,
        delta_x: int,
        delta_y: int,
        timeout_ms: float = 100.0,
        enable_ilc_update: bool = True,
    ) -> Dict:
        """
        Executes a synchronized straight-line 200Hz ballistic flick to arbitrary reachable
        (delta_x, delta_y) in true 3200 DPI counts, enforcing anti-edge clipping and full
        workspace safety bounds.
        """
        req_dist = float(np.hypot(delta_x, delta_y))
        if req_dist < 1.0:
            return {
                "target_delta": (delta_x, delta_y),
                "final_delta": (0, 0),
                "final_error_counts": 0.0,
                "final_error_um": 0.0,
                "settled_time_ms": 0.0,
                "t_brake_ms": 0.0,
                "t_onset_ms": 0.0,
                "straightness_max_dev_cnt": 0.0,
                "straightness_rms_cnt": 0.0,
                "straightness_pct": 0.0,
                "pos_at_100ms": (0, 0),
                "err_at_100ms": 0.0,
                "trajectory": [],
            }

        # Safe workspace clamp inside empirical 95% convex hull
        tx, ty = self.workspace.clamp_to_workspace(float(delta_x), float(delta_y), safety_ratio=0.95)
        target_vec = np.array([tx, ty], dtype=float)
        L = float(np.linalg.norm(target_vec))
        d_hat = target_vec / L
        n_hat = np.array([-d_hat[1], d_hat[0]], dtype=float)

        u_kin_target = self.workspace.predict_kinematic_u(tx, ty)
        u_lock_x, u_lock_y, du_eq, K_end = self._compute_lock_command(tx, ty)
        du_ilc = self.ilc.get_correction(tx, ty)

        # Ensure complete rest before firing shot
        for _ in range(5):
            self.sensor.read_delta_and_clear()
            time.sleep(0.010)
            dx_c, dy_c = self.sensor.read_delta_and_clear()
            if dx_c == 0 and dy_c == 0:
                break

        self.sensor.recenter_cursor()
        x0, y0 = self.sensor.get_position()

        # Distance-adaptive IF-OPB parameters & physical settling budget
        tau_brake = 0.028 if L > 220.0 else (0.024 if L > 90.0 else 0.018)
        opb_thresh = 0.91 if L > 220.0 else (0.88 if L > 90.0 else 0.84)
        boost_cap_ms = 185.0 if L > 450.0 else (145.0 if L > 200.0 else 115.0)
        max_loop_ms = max(timeout_ms + 90.0, 190.0 + 0.25 * L)

        # Stage 1 initial launch command: break backlash/stiction along straight tangent
        u_launch_x, u_launch_y = self._compute_guided_command(
            np.zeros(2), np.zeros(2), u_kin_target, K_end, L, d_hat, n_hat, du_ilc, 0.0
        )

        t_start = time.perf_counter()
        last_tx_t = t_start
        self.driver.send_raw_pulses(u_launch_x, u_launch_y)

        t_onset: Optional[float] = None
        braked = False
        t_brake_ms = 0.0
        settled_t_ms: Optional[float] = None
        tol_counts = max(2.5, 0.015 * L)

        vel_buf: List[Tuple[float, int, int]] = []
        trajectory: List[Dict] = []
        perp_devs: List[float] = []
        pos_100ms: Optional[Tuple[int, int]] = None
        trim_count = 0
        last_trim_ms = 0.0
        horn_dir_x = float(np.sign(du_eq[0])) if abs(du_eq[0]) > 1.0 else 0.0
        horn_dir_y = float(np.sign(du_eq[1])) if abs(du_eq[1]) > 1.0 else 0.0

        while True:
            t_now = time.perf_counter()
            elapsed_ms = (t_now - t_start) * 1000.0
            cur_x, cur_y = self.sensor.get_position()
            dx_cur = cur_x - x0
            dy_cur = cur_y - y0
            p_vec = np.array([float(dx_cur), float(dy_cur)], dtype=float)
            p_norm = float(np.linalg.norm(p_vec))

            if t_onset is None and p_norm >= 3.0:
                t_onset = t_now

            # 18ms sliding window velocity (immune to single-packet USB polling jitter!)
            vel_buf.append((t_now, cur_x, cur_y))
            while len(vel_buf) > 1 and (t_now - vel_buf[0][0]) > 0.018:
                vel_buf.pop(0)
            dt_v = t_now - vel_buf[0][0]
            vx = (cur_x - vel_buf[0][1]) / dt_v if dt_v > 0.004 else 0.0
            vy = (cur_y - vel_buf[0][2]) / dt_v if dt_v > 0.004 else 0.0
            v_vec = np.array([vx, vy], dtype=float)

            err_vec = target_vec - p_vec
            err_cnt = float(np.linalg.norm(err_vec))

            p_par = float(np.dot(p_vec, d_hat))
            e_perp = float(np.dot(p_vec, n_hat))
            if p_norm >= 10.0 and p_par <= L * 1.05:
                perp_devs.append(abs(e_perp))

            if pos_100ms is None and elapsed_ms >= 100.0:
                pos_100ms = (int(dx_cur), int(dy_cur))

            if settled_t_ms is None and t_onset is not None and err_cnt <= tol_counts and np.linalg.norm(v_vec) < 400.0:
                settled_t_ms = elapsed_ms

            trajectory.append({
                "t_ms": round(elapsed_ms, 2),
                "phase": "LOCK" if braked else ("GUIDED" if t_onset is not None else "LAUNCH"),
                "x": int(cur_x),
                "y": int(cur_y),
                "dx": int(dx_cur),
                "dy": int(dy_cur),
                "e_perp": round(e_perp, 2),
                "vx": round(vx, 1),
                "vy": round(vy, 1),
                "err_counts": round(err_cnt, 2),
            })

            # Stage 2: Synchronized 200Hz Straight-Line Guidance & Predictive Brake
            if not braked:
                p_pred_par = float(np.dot(p_vec + tau_brake * v_vec, d_hat))
                progress_raw = p_par / L
                progress_pred = p_pred_par / L
                min_raw = 0.45 if L <= 90.0 else 0.52
                opb_fire = (progress_raw >= min_raw) and (progress_pred >= opb_thresh)

                if opb_fire or elapsed_ms >= boost_cap_ms:
                    self.driver.send_raw_pulses(u_lock_x, u_lock_y)
                    braked = True
                    t_brake_ms = elapsed_ms
                    last_trim_ms = elapsed_ms
                elif (t_now - last_tx_t) >= 0.0048:
                    u_gx, u_gy = self._compute_guided_command(
                        p_vec, v_vec, u_kin_target, K_end, L, d_hat, n_hat, du_ilc, elapsed_ms
                    )
                    self.driver.send_raw_pulses(u_gx, u_gy)
                    last_tx_t = t_now
            else:
                # Stage 3: Anti-backlash-aware Jacobian closed-loop trim once speed drops
                v_speed = float(np.linalg.norm(v_vec))
                if trim_count < 2 and (elapsed_ms - last_trim_ms) >= 45.0 and elapsed_ms <= max_loop_ms - 40.0 and v_speed < 240.0:
                    if err_cnt > max(3.5, 0.012 * L):
                        du_trim = 0.78 * (K_end @ err_vec)
                        # Traverse full backlash play if reversing axis horn engagement direction
                        if abs(du_trim[0]) > 0.3:
                            sx = float(np.sign(du_trim[0]))
                            if horn_dir_x * sx < -0.5:
                                du_trim[0] += sx * 16.0
                            else:
                                du_trim[0] += sx * 3.0
                            horn_dir_x = sx
                        if abs(du_trim[1]) > 0.3:
                            sy = float(np.sign(du_trim[1]))
                            if horn_dir_y * sy < -0.5:
                                du_trim[1] += sy * 9.0
                            else:
                                du_trim[1] += sy * 2.5
                            horn_dir_y = sy
                        du_trim = np.clip(du_trim, -36.0, 36.0)
                        ux_t = float(np.clip(self.driver.cur_s1 + du_trim[0], self.driver.SAFE_MIN_US, self.driver.SAFE_MAX_US))
                        uy_t = float(np.clip(self.driver.cur_s2 + du_trim[1], self.driver.SAFE_MIN_US, self.driver.SAFE_MAX_US))
                        self.driver.send_raw_pulses(ux_t, uy_t)
                    trim_count += 1
                    last_trim_ms = elapsed_ms

            # Settle window: finish once velocity stops and either tolerance met or trims completed
            active_ms = ((t_now - t_onset) * 1000.0) if t_onset is not None else 0.0
            v_speed_now = float(np.linalg.norm(v_vec))
            tol_trim = max(4.0, 0.015 * L)
            done_settle = (
                braked
                and (elapsed_ms - t_brake_ms) >= 50.0
                and v_speed_now < 120.0
                and active_ms >= timeout_ms
                and (err_cnt <= tol_trim or (trim_count >= 2 and (elapsed_ms - last_trim_ms) >= 45.0))
            )
            if done_settle or elapsed_ms >= max_loop_ms:
                break
            time.sleep(0.001)

        final_x, final_y = self.sensor.get_position()
        final_dx = int(final_x - x0)
        final_dy = int(final_y - y0)
        final_err = float(np.hypot(delta_x - final_dx, delta_y - final_dy))
        if pos_100ms is None:
            pos_100ms = (final_dx, final_dy)
        err_100ms = float(np.hypot(delta_x - pos_100ms[0], delta_y - pos_100ms[1]))

        max_dev = float(max(perp_devs)) if perp_devs else 0.0
        rms_dev = float(np.sqrt(np.mean(np.square(perp_devs)))) if perp_devs else 0.0
        straight_pct = 100.0 * max_dev / max(1.0, L)

        if enable_ilc_update:
            self.ilc.update_vector(tx, ty, float(final_dx), float(final_dy), K_end, final_err)

        if settled_t_ms is None:
            settled_t_ms = elapsed_ms

        return {
            "target_delta": (int(delta_x), int(delta_y)),
            "clamped_target": (round(tx, 1), round(ty, 1)),
            "final_delta": (final_dx, final_dy),
            "final_error_counts": round(final_err, 2),
            "final_error_um": round(final_err * self.sensor.um_per_count, 2),
            "pos_at_100ms": pos_100ms,
            "err_at_100ms": round(err_100ms, 2),
            "settled_time_ms": round(settled_t_ms, 1),
            "t_brake_ms": round(t_brake_ms, 1),
            "t_onset_ms": round((t_onset - t_start) * 1000.0, 1) if t_onset is not None else -1.0,
            "straightness_max_dev_cnt": round(max_dev, 2),
            "straightness_rms_cnt": round(rms_dev, 2),
            "straightness_pct": round(straight_pct, 2),
            "trajectory": trajectory,
        }

    def close(self):
        try:
            self.sensor.stop()
            self.driver.close()
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full-Workspace 200Hz Straight-Line 3200 DPI Ballistic Controller")
    parser.add_argument("--dx", type=int, default=200, help="Target Delta X (counts)")
    parser.add_argument("--dy", type=int, default=0, help="Target Delta Y (counts)")
    parser.add_argument("--test-suite", action="store_true", help="Run 8-direction + unseen random target test suite")
    args = parser.parse_args()

    ctrl = Ballistic100msController()
    ctrl.ilc.reset_spatial_field()
    ctrl.sensor.start()
    ctrl.driver.center(settle_s=0.45, preload=True)

    if args.test_suite:
        print("=" * 94)
        print("  3-RRR FULL-WORKSPACE 8-DIRECTION + STRAIGHTNESS + RANDOM GENERALIZATION TEST (3200 DPI)")
        print("=" * 94)
        # 1. Test 8 directions across 70% of each direction's real mechanical boundary!
        print("\n[Phase A] 8-Direction Straight-Line Flicks (70% Max Mechanical Envelope per Angle):")
        degs = [0, 45, 90, 135, 180, -135, -90, -45]
        for deg in degs:
            rad = np.deg2rad(deg)
            rmax = ctrl.workspace.max_reach_at_angle(rad)
            tdx = int(round(0.70 * rmax * np.cos(rad)))
            tdy = int(round(0.70 * rmax * np.sin(rad)))
            # Fire 2 warmups (feeds spatial field) then evaluate 3rd shot
            for _ in range(2):
                ctrl.driver.center(settle_s=0.38, preload=True)
                ctrl.flick_shot(tdx, tdy, timeout_ms=100.0, enable_ilc_update=True)
            ctrl.driver.center(settle_s=0.38, preload=True)
            res = ctrl.flick_shot(tdx, tdy, timeout_ms=100.0, enable_ilc_update=True)
            print(
                f"  Dir {deg:+4d}° | Target ({tdx:+4d},{tdy:+4d}) -> Final ({res['final_delta'][0]:+4d},{res['final_delta'][1]:+4d}) | "
                f"Err:{res['final_error_counts']:5.1f} cnt ({res['final_error_um']:5.0f}um) | "
                f"100msErr:{res['err_at_100ms']:5.1f} | StraightDev:{res['straightness_max_dev_cnt']:5.1f} cnt ({res['straightness_pct']:4.1f}%) | "
                f"On:{res['t_onset_ms']:4.1f} Brk:{res['t_brake_ms']:4.1f}ms"
            )

        # 2. Blind zero-shot random targets inside workspace (never explicitly trained!)
        print("\n[Phase B] Zero-Shot Unseen Random Coordinates (Direct Generalization from Spatial Field):")
        rng = np.random.default_rng(42)
        for idx in range(1, 9):
            ang = rng.uniform(-np.pi, np.pi)
            rmax = ctrl.workspace.max_reach_at_angle(ang)
            frac = rng.uniform(0.25, 0.85)
            tdx = int(round(frac * rmax * np.cos(ang)))
            tdy = int(round(frac * rmax * np.sin(ang)))
            ctrl.driver.center(settle_s=0.38, preload=True)
            res = ctrl.flick_shot(tdx, tdy, timeout_ms=100.0, enable_ilc_update=False)
            print(
                f"  Rand #{idx} | Target ({tdx:+4d},{tdy:+4d}) -> Final ({res['final_delta'][0]:+4d},{res['final_delta'][1]:+4d}) | "
                f"Err:{res['final_error_counts']:5.1f} cnt ({res['final_error_um']:5.0f}um) | "
                f"StraightDev:{res['straightness_max_dev_cnt']:5.1f} cnt ({res['straightness_pct']:4.1f}%)"
            )
    else:
        ctrl.driver.center(settle_s=0.38, preload=True)
        res = ctrl.flick_shot(args.dx, args.dy, timeout_ms=100.0)
        print(
            f"Target ({args.dx},{args.dy}) -> Final {res['final_delta']} | "
            f"Err: {res['final_error_counts']} cnt ({res['final_error_um']} um) | "
            f"StraightDev: {res['straightness_max_dev_cnt']} cnt ({res['straightness_pct']}%)"
        )

    ctrl.driver.center(settle_s=0.35, preload=True)
    ctrl.close()
