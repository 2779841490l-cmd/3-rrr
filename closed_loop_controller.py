"""
closed_loop_controller.py - 3200 DPI Physics-Guided Closed-Loop Positioning Engine
==================================================================================
Combines:
1. 2nd-Order Spatial Inverse Jacobian Tensor + Online RLS residual adaptation
2. Continuous Asymmetric Play-Operator Backlash State Machine
3. Sub-Deadband Impulse-Retract Micro-Stepping (Kinetic Nudge)
4. Variable-Gain Anti-Windup PI Control (Discrete Lyapunov stable)
"""

import argparse
import json
import os
import time
from typing import Dict, List, Tuple
import numpy as np

from mouse_sensor import HighPrecisionMouseSensor
from servo_driver import WS63ServoDriver


class SpatialInverseJacobian2ndOrder:
    """
    Evaluates the 2nd-order polynomial Inverse Jacobian matrix K(du_x, du_y)
    at the current operating point to eliminate cross-axis coupling.
    """

    def __init__(self, poly_file: str = "prompts/fitted_jacobian_poly.json"):
        self.coeffs = self._load_coeffs(poly_file)

    @staticmethod
    def _load_coeffs(poly_file: str) -> Dict:
        if os.path.exists(poly_file):
            with open(poly_file, "r", encoding="utf-8") as f:
                return json.load(f)
        # Default fallback
        return {
            "K11": {"Const": 1.0251, "dUx": -0.00238, "dUy": -0.00181, "dUx^2/1k": -0.0126, "dUy^2/1k": 0.0119, "dUx*dUy/1k": 0.0202},
            "K12": {"Const": -0.9797, "dUx": 0.00102, "dUy": -0.00545, "dUx^2/1k": 0.0065, "dUy^2/1k": -0.0910, "dUx*dUy/1k": -0.0287},
            "K21": {"Const": -0.1074, "dUx": -0.00037, "dUy": -0.00205, "dUx^2/1k": -0.0057, "dUy^2/1k": -0.0074, "dUx*dUy/1k": 0.0045},
            "K22": {"Const": -0.5546, "dUx": -0.00033, "dUy": -0.00291, "dUx^2/1k": -0.0056, "dUy^2/1k": -0.0914, "dUx*dUy/1k": -0.0061},
        }

    def evaluate(self, du_x: float, du_y: float) -> np.ndarray:
        K = np.zeros((2, 2), dtype=float)
        basis = {
            "Const": 1.0,
            "dUx": float(du_x),
            "dUy": float(du_y),
            "dUx^2/1k": float(du_x**2) / 1000.0,
            "dUy^2/1k": float(du_y**2) / 1000.0,
            "dUx*dUy/1k": float(du_x * du_y) / 1000.0,
        }
        for (r, c), key in [((0, 0), "K11"), ((0, 1), "K12"), ((1, 0), "K21"), ((1, 1), "K22")]:
            c_dict = self.coeffs.get(key, {})
            K[r, c] = sum(c_dict.get(name, 0.0) * val for name, val in basis.items())
        return K


class OnlineInverseJacobianRLS:
    """
    Online Recursive Least Squares (RLS) estimator refining residual gains.
    """

    def __init__(self, forgetting_factor: float = 0.985):
        self.dK = np.zeros((2, 2), dtype=float)
        self.lam = float(forgetting_factor)
        self.P = np.eye(2, dtype=float) * 0.04

    def update(self, dx_vec: np.ndarray, du_res_vec: np.ndarray):
        norm_dx = float(np.linalg.norm(dx_vec))
        if norm_dx < 3.0 or norm_dx > 250.0:
            return
        x = dx_vec.reshape(2, 1)
        u = du_res_vec.reshape(2, 1)
        Px = self.P @ x
        denom = self.lam + float((x.T @ Px)[0, 0])
        gain = Px / denom
        err = u - (self.dK @ x)
        self.dK = self.dK + (err @ gain.T)
        self.P = (self.P - gain @ x.T @ self.P) / self.lam


class ClosedLoop3RRRController:
    def __init__(self, calib_file: str = "calib_params.json", enable_rls: bool = True):
        self.calib = self._load_calib(calib_file)
        self.enable_rls = enable_rls

        bl = self.calib["backlash_play_radius_us"]
        db = self.calib["deadband_us"]

        self.driver = WS63ServoDriver(
            backlash_x=(bl["x_pos"], bl["x_neg"]),
            backlash_y=(bl["y_pos"], bl["y_neg"]),
            deadband_x=(db["x_pos"], db["x_neg"]),
            deadband_y=(db["y_pos"], db["y_neg"]),
        )
        self.sensor = HighPrecisionMouseSensor(dpi=self.calib.get("dpi", 3200))

        self.poly_jac = SpatialInverseJacobian2ndOrder()
        self.rls = OnlineInverseJacobianRLS()

        self.u0_x = float(self.calib["neutral_us"]["s1_x"])
        self.u0_y = float(self.calib["neutral_us"]["s2_y1"])
        self.kin_u = np.array([self.u0_x, self.u0_y], dtype=float)

    @staticmethod
    def _load_calib(calib_file: str) -> Dict:
        if os.path.exists(calib_file):
            with open(calib_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "dpi": 3200,
            "neutral_us": {"s1_x": 1510, "s2_y1": 1545, "s3_y2": 1455, "s4_aux": 1500},
            "deadband_us": {"x_pos": 13.0, "x_neg": 10.0, "y_pos": 8.0, "y_neg": 8.0},
            "backlash_play_radius_us": {"x_pos": 23.0, "x_neg": 23.0, "y_pos": 18.0, "y_neg": 19.0},
            "inverse_jacobian_center_us_per_px": [[1.0341, -1.0967], [-0.1222, -0.7358]],
        }

    @staticmethod
    def _variable_gain(err_mag_counts: float) -> float:
        """Lyapunov-stable variable proportional gain schedule."""
        if err_mag_counts <= 1.0:
            return 0.0  # Deadband lock zone (7.9 um)
        elif err_mag_counts <= 5.0:
            return 0.35  # Sub-deadband fine nudge
        elif err_mag_counts <= 25.0:
            return 0.55  # Medium convergence
        return 0.72  # Fast tracking

    def move_to_target_counts(
        self,
        target_x: int,
        target_y: int,
        tolerance_counts: int = 1,
        max_steps: int = 40,
        step_dt: float = 0.035,
    ) -> Dict:
        """Execute closed-loop positioning to target (x, y) counts."""
        trajectory = []
        t_start = time.perf_counter()

        for step in range(max_steps):
            cur_x, cur_y = self.sensor.get_position()
            ex = float(target_x - cur_x)
            ey = float(target_y - cur_y)
            err_mag = float(np.hypot(ex, ey))

            trajectory.append(
                {
                    "step": step,
                    "t_ms": round((time.perf_counter() - t_start) * 1000.0, 1),
                    "x": cur_x,
                    "y": cur_y,
                    "err_counts": round(err_mag, 2),
                    "err_um": round(err_mag * self.sensor.um_per_count, 2),
                }
            )

            if err_mag <= tolerance_counts:
                break

            # 1. 2nd-order dynamic inverse Jacobian
            du_x_curr = self.kin_u[0] - self.u0_x
            du_y_curr = self.kin_u[1] - self.u0_y
            K_local = self.poly_jac.evaluate(du_x_curr, du_y_curr)
            if self.enable_rls:
                K_local = K_local + self.rls.dK

            kp = self._variable_gain(err_mag)
            e_vec = np.array([ex, ey], dtype=float)
            du_vec = kp * (K_local @ e_vec)

            # Slew rate limit
            du_norm = float(np.linalg.norm(du_vec))
            max_du_step = 40.0
            if du_norm > max_du_step:
                du_vec = du_vec * (max_du_step / du_norm)

            self.kin_u = np.clip(self.kin_u + du_vec, 1200.0, 1800.0)

            enable_nudge = err_mag <= 6.0
            self.driver.step_compensated_with_nudge(
                self.kin_u[0], self.kin_u[1], enable_nudge=enable_nudge
            )
            time.sleep(step_dt)

            new_x, new_y = self.sensor.get_position()
            actual_dx = np.array([float(new_x - cur_x), float(new_y - cur_y)])
            if self.enable_rls:
                pred_du = K_local @ actual_dx
                res_du = du_vec - pred_du
                self.rls.update(actual_dx, res_du)

        final_x, final_y = self.sensor.get_position()
        final_err = float(np.hypot(target_x - final_x, target_y - final_y))
        return {
            "target": (target_x, target_y),
            "final": (final_x, final_y),
            "final_error_counts": round(final_err, 2),
            "final_error_um": round(final_err * self.sensor.um_per_count, 2),
            "steps_taken": len(trajectory),
            "elapsed_ms": round((time.perf_counter() - t_start) * 1000.0, 1),
            "trajectory": trajectory,
        }

    def run_benchmark(self) -> List[Dict]:
        print("=" * 68)
        print("  3-RRR 3200 DPI Closed-Loop Target Benchmark (2nd-Order + Nudge)")
        print("=" * 68)
        self.sensor.start()
        self.driver.center(settle_s=0.6)
        self.sensor.reset_origin()

        waypoints = [
            (0, 0),
            (30, 0),     # Pure +X (~238 um)
            (-30, 0),    # Pure -X reversal
            (0, 35),     # Pure +Y (~278 um)
            (0, -35),    # Pure -Y reversal
            (4, -3),     # Sub-deadband micro step (~40 um)
            (0, 0),      # Return to center
        ]

        results = []
        for tx, ty in waypoints:
            res = self.move_to_target_counts(tx, ty, tolerance_counts=1)
            results.append(res)
            print(
                f"Target ({tx:+4d}, {ty:+4d}) -> Final ({res['final'][0]:+4d}, {res['final'][1]:+4d}) | "
                f"Err: {res['final_error_counts']:4.1f} cnt ({res['final_error_um']:5.1f} um) | "
                f"{res['steps_taken']:2d} steps ({res['elapsed_ms']:.0f} ms)"
            )
            time.sleep(0.20)

        self.driver.center(settle_s=0.4)
        self.sensor.stop()
        self.driver.close()
        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3-RRR 3200 DPI Closed-Loop Controller")
    parser.add_argument("--mode", choices=["benchmark", "center"], default="benchmark")
    args = parser.parse_args()

    ctrl = ClosedLoop3RRRController()
    if args.mode == "center":
        ctrl.driver.center(settle_s=0.5)
        ctrl.driver.close()
    else:
        ctrl.run_benchmark()
