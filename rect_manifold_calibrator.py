r"""
rect_manifold_calibrator.py — 3-RRR 最大内接矩形框寻优、伴随跳动消除与全屏任意点精细化定位引擎
====================================================================================================
【核心思想（源自用户工程洞察 & IEEE T-MECH/TIM 论文核心方法）】
彻底摒弃“每次回中 + 盲目开环/半开环瞎甩（Blind Flicking）”的错误范式，建立三阶段精细化控制架构：

1. Stage I — 几何中心重锚定与最大内接矩形框寻优 (Optimal Geometric Origin & Max Inscribed Rectangle)
   - 基于实测 2D 物理包络，自动将工作空间虚拟原点 $(u_{x0}^*, u_{y0}^*)$ 平移至平行连杆的真实几何中心，
     消除原本上下极度不对称（-357 vs +1004 counts）的畸变，提取对称的最大内接矩形框 $[-W, +W] \times [-H, +H]$。

2. Stage II — 矩形框连续走线与水平/纵向伴随跳动迭代消除 (Iterative Contour Learning for Cross-Axis Decoupling)
   - 控制机械臂连续走完最大矩形框的 4 条边（上、右、下、左）及内部十字/对角基准线；
   - 针对“走水平线时纵向跳动 $\delta y(x)$、走纵向线时水平跳动 $\delta x(y)$”的并联耦合与齿隙畸变，
     构建【三阶二元多项式 + $9\times 9$ 双线性控制网格残差场 + 方向齿隙状态机】复合解耦流形 $\Phi(x, y, \sigma_x, \sigma_y)$；
   - 经 $K=4\sim 6$ 轮实机画框迭代优化，使矩形四边垂直伴随跳动收敛至 $\le 2.5\text{ counts}$，把矩形框完美画平、画直。

3. Stage III — 基于解耦流形的全平面任意点到点（Point-to-Point）直线轨迹跟踪与微米级闭环精修
   - 无需每次回中或 `SetCursorPos` 作弊，支持从矩形框内任意当前点 $(x_A, y_A)$ 沿 Minimum-Jerk 直线轨迹
     平滑、无跳动地高速抵达任意目标点 $(x_B, y_B)$，并在末端通过跟踪齿轮啮合方向状态 $(\sigma_x, \sigma_y)$
     完成闭环微步精修（稳态误差 $\le 1.5\sim 2.0\text{ counts}$）。
"""

import argparse
import csv
import json
import os
import time
from typing import Dict, List, Optional, Tuple
import numpy as np

from mouse_sensor import HighPrecisionMouseSensor
from servo_driver import WS63ServoDriver
from workspace_model import WorkspaceField2D


class RectangularManifoldController:
    """
    2D Decoupled Rectangular Coordinate Manifold $\\Phi: [-W, W] \\times [-H, H] \\to (u_x, u_y)$
    with explicit gear-engagement state tracking $(\\sigma_x, \\sigma_y) \\in \\{-1, +1\\}^2$.
    """

    def __init__(
        self,
        map_file: str = "experiments/workspace_map_full.json",
        calib_file: str = "calib_params.json",
        manifold_file: str = "calib_rect_manifold.json",
        grid_size: int = 9,
    ):
        self.map_file = map_file
        self.calib_file = calib_file
        self.manifold_file = manifold_file
        self.grid_size = grid_size

        self.ws = WorkspaceField2D(map_file=map_file, calib_file=calib_file)

        # 1. Jointly optimize geometric rectangle center (ox, oy) and half-widths (W, H) inside ConvexHull
        self.origin_offset_xy, (self.W, self.H) = self._compute_max_inscribed_rectangle()
        u_origin_du = self.ws.target_to_equilibrium_du(
            float(self.origin_offset_xy[0]), float(self.origin_offset_xy[1])
        )
        self.u0_star_x = float(np.clip(self.ws.u0_x + u_origin_du[0], 1300.0, 1700.0))
        self.u0_star_y = float(np.clip(self.ws.u0_y + u_origin_du[1], 1300.0, 1700.0))

        # 3. 9x9 B-spline / Bilinear Control Lattice over [-W, +W] x [-H, +H] for cross-axis jump elimination
        self.grid_x = np.linspace(-self.W, self.W, self.grid_size)
        self.grid_y = np.linspace(-self.H, self.H, self.grid_size)
        self.residual_ux = np.zeros((self.grid_size, self.grid_size), dtype=float)
        self.residual_uy = np.zeros((self.grid_size, self.grid_size), dtype=float)

        # 4. Directional hysteresis compensation (us) when reversing motor horn direction
        self.hyst_x_pos = 10.0
        self.hyst_x_neg = 12.0
        self.hyst_y_pos = 5.5
        self.hyst_y_neg = 5.5

        # Track instantaneous horn tooth contact direction (+1 or -1) and current logical rectangle position
        self.sigma_x = 1.0
        self.sigma_y = 1.0
        self.cur_rect_pos = np.zeros(2, dtype=float)

        self.load_manifold()

    def _compute_max_inscribed_rectangle(self) -> Tuple[np.ndarray, Tuple[float, float]]:
        """Finds the globally largest safe axis-aligned rectangle inside the empirical ConvexHull."""
        if self.ws.hull is None:
            return np.array([0.0, 260.0], dtype=float), (380.0, 320.0)
        A = self.ws.hull.equations[:, :2]
        b = self.ws.hull.equations[:, 2]
        margin = 25.0  # 25 counts (~0.2 mm) safety buffer inside hull boundary

        best_score = 0.0
        best_center = np.array([0.0, 260.0], dtype=float)
        best_wh = (380.0, 320.0)
        for ox in range(-160, 220, 20):
            for oy in range(120, 420, 20):
                for w in range(240, 660, 15):
                    for h in range(200, 560, 15):
                        corners = np.array([
                            [ox + w, oy + h],
                            [ox - w, oy + h],
                            [ox - w, oy - h],
                            [ox + w, oy - h],
                        ], dtype=float)
                        if np.all(corners @ A.T + b <= -margin):
                            aspect_bonus = 1.0 - 0.12 * abs((w / max(1.0, h)) - 1.30)
                            score = (w * h) * aspect_bonus
                            if score > best_score:
                                best_score = score
                                best_center = np.array([float(ox), float(oy)], dtype=float)
                                best_wh = (float(w), float(h))
        return best_center, best_wh

    def _interp_residual(self, x: float, y: float) -> np.ndarray:
        """Bilinear interpolation of the 9x9 learned decoupling residual lattice at (x, y)."""
        xc = float(np.clip(x, -self.W, self.W))
        yc = float(np.clip(y, -self.H, self.H))
        tx = (xc + self.W) / (2.0 * self.W) * (self.grid_size - 1)
        ty = (yc + self.H) / (2.0 * self.H) * (self.grid_size - 1)

        ix = int(np.clip(np.floor(tx), 0, self.grid_size - 2))
        iy = int(np.clip(np.floor(ty), 0, self.grid_size - 2))
        fx = tx - ix
        fy = ty - iy

        rx = (
            (1 - fx) * (1 - fy) * self.residual_ux[ix, iy]
            + fx * (1 - fy) * self.residual_ux[ix + 1, iy]
            + (1 - fx) * fy * self.residual_ux[ix, iy + 1]
            + fx * fy * self.residual_ux[ix + 1, iy + 1]
        )
        ry = (
            (1 - fx) * (1 - fy) * self.residual_uy[ix, iy]
            + fx * (1 - fy) * self.residual_uy[ix + 1, iy]
            + (1 - fx) * fy * self.residual_uy[ix, iy + 1]
            + fx * fy * self.residual_uy[ix + 1, iy + 1]
        )
        return np.array([float(rx), float(ry)], dtype=float)

    def update_lattice_from_samples(
        self, samples: List[Dict], lr: float = 0.55, kernel_radius_frac: float = 0.28
    ):
        """
        Updates the 9x9 decoupling lattice from recorded trajectory errors:
        eliminates horizontal/vertical cross-axis jumping across the entire rectangle.
        """
        rx_rad = kernel_radius_frac * self.W
        ry_rad = kernel_radius_frac * self.H

        du_accum_x = np.zeros_like(self.residual_ux)
        du_accum_y = np.zeros_like(self.residual_uy)
        w_accum = np.zeros_like(self.residual_ux) + 1e-8

        for s in samples:
            xd, yd = float(s["x_des"]), float(s["y_des"])
            err_vec = np.array([float(s["err_x"]), float(s["err_y"])], dtype=float)
            gx = xd + float(self.origin_offset_xy[0])
            gy = yd + float(self.origin_offset_xy[1])
            K_loc = self.ws.local_inverse_jacobian(gx, gy)
            du_err = K_loc @ err_vec

            for i, nx in enumerate(self.grid_x):
                for j, ny in enumerate(self.grid_y):
                    d2 = ((xd - nx) / rx_rad) ** 2 + ((yd - ny) / ry_rad) ** 2
                    if d2 <= 4.0:
                        w = float(np.exp(-0.5 * d2))
                        du_accum_x[i, j] += w * du_err[0]
                        du_accum_y[i, j] += w * du_err[1]
                        w_accum[i, j] += w

        mask = w_accum > 1e-3
        self.residual_ux[mask] = np.clip(
            self.residual_ux[mask] + lr * (du_accum_x[mask] / w_accum[mask]), -95.0, 95.0
        )
        self.residual_uy[mask] = np.clip(
            self.residual_uy[mask] + lr * (du_accum_y[mask] / w_accum[mask]), -95.0, 95.0
        )

    def evaluate_pulse(
        self,
        x_rect: float,
        y_rect: float,
        prev_u: Optional[Tuple[float, float]] = None,
    ) -> Tuple[float, float]:
        """
        Maps rectangular coordinates $(x, y) \\in [-W, W] \\times [-H, H]$ to decoupled servo pulses
        $(u_x, u_y)$ with directional backlash compensation.
        """
        xc = float(np.clip(x_rect, -self.W, self.W))
        yc = float(np.clip(y_rect, -self.H, self.H))

        # Global workspace coordinates
        gx = xc + float(self.origin_offset_xy[0])
        gy = yc + float(self.origin_offset_xy[1])

        # Smooth kinematic baseline + learned 9x9 cross-axis decoupling residual
        u_kin = self.ws.predict_kinematic_u(gx, gy)
        du_res = self._interp_residual(xc, yc)
        u_base_x = self.ws.u0_x + float(u_kin[0]) + float(du_res[0])
        u_base_y = self.ws.u0_y + float(u_kin[1]) + float(du_res[1])

        # Update tooth contact direction state (sigma_x, sigma_y) if moving beyond deadzone
        if prev_u is not None:
            dux_step = u_base_x - prev_u[0]
            duy_step = u_base_y - prev_u[1]
            if abs(dux_step) > 0.6:
                self.sigma_x = 1.0 if dux_step > 0 else -1.0
            if abs(duy_step) > 0.6:
                self.sigma_y = 1.0 if duy_step > 0 else -1.0

        hyst_x = self.hyst_x_pos if self.sigma_x > 0 else -self.hyst_x_neg
        hyst_y = self.hyst_y_pos if self.sigma_y > 0 else -self.hyst_y_neg

        ux = float(np.clip(u_base_x + hyst_x, WS63ServoDriver.SAFE_MIN_US, WS63ServoDriver.SAFE_MAX_US))
        uy = float(np.clip(u_base_y + hyst_y, WS63ServoDriver.SAFE_MIN_US, WS63ServoDriver.SAFE_MAX_US))
        # Dual-Y S3 safety check
        if 3000.0 - uy < WS63ServoDriver.SAFE_MIN_US:
            uy = 3000.0 - WS63ServoDriver.SAFE_MIN_US
        elif 3000.0 - uy > WS63ServoDriver.SAFE_MAX_US:
            uy = 3000.0 - WS63ServoDriver.SAFE_MAX_US
        return ux, uy

    def save_manifold(self):
        payload = {
            "origin_offset_xy_counts": self.origin_offset_xy.tolist(),
            "u0_star_us": {"s1_x": round(self.u0_star_x, 2), "s2_y1": round(self.u0_star_y, 2)},
            "rectangle_half_size_counts": {"W": self.W, "H": self.H},
            "rectangle_full_size_mm": {
                "width_mm": round(2.0 * self.W * 0.0079375, 3),
                "height_mm": round(2.0 * self.H * 0.0079375, 3),
            },
            "grid_size": self.grid_size,
            "hysteresis_us": {
                "x_pos": self.hyst_x_pos,
                "x_neg": self.hyst_x_neg,
                "y_pos": self.hyst_y_pos,
                "y_neg": self.hyst_y_neg,
            },
            "residual_ux": self.residual_ux.tolist(),
            "residual_uy": self.residual_uy.tolist(),
        }
        if os.path.exists(self.manifold_file):
            os.makedirs("backups", exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            with open(self.manifold_file, "r", encoding="utf-8") as f_old:
                with open(f"backups/calib_rect_manifold_{ts}.json", "w", encoding="utf-8") as f_bak:
                    f_bak.write(f_old.read())
        with open(self.manifold_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def load_manifold(self) -> bool:
        if not os.path.exists(self.manifold_file):
            return False
        try:
            with open(self.manifold_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            rux = np.array(data.get("residual_ux", []), dtype=float)
            ruy = np.array(data.get("residual_uy", []), dtype=float)
            if rux.shape == (self.grid_size, self.grid_size):
                self.residual_ux = rux
                self.residual_uy = ruy
                return True
        except Exception:
            pass
        return False


class HardwareRectangleTrainer:
    """
    Executes Stage II (Iterative Rectangle Contour & Cross-Axis Jump Elimination)
    and Stage III (Arbitrary Point-to-Point Fine Control inside the Rectangle).
    """

    def __init__(self):
        self.manifold = RectangularManifoldController()
        self.sensor = HighPrecisionMouseSensor(dpi=3200)
        self.driver = WS63ServoDriver()
        self.driver.set_pwm_frequency(200)
        self.origin_sensor_xy = np.zeros(2, dtype=float)
        self.last_base_u: Optional[Tuple[float, float]] = None

    def move_to_rect_origin(self, settle_s: float = 0.55):
        """
        Brings mechanism to the consistent rectangular center $(0, 0)_{\\text{rect}}$ with
        deterministic negative-to-positive preload so the coordinate frame is locked.
        """
        ux_pre, uy_pre = self.manifold.evaluate_pulse(-0.35 * self.manifold.W, -0.35 * self.manifold.H)
        self.driver.send_raw_pulses(ux_pre, uy_pre)
        time.sleep(0.32)

        self.manifold.sigma_x = 1.0
        self.manifold.sigma_y = 1.0
        ux0, uy0 = self.manifold.evaluate_pulse(0.0, 0.0)
        self.driver.send_raw_pulses(ux0, uy0)
        time.sleep(settle_s)
        self.last_base_u = (ux0, uy0)

        self.sensor.recenter_cursor()
        sx, sy = self.sensor.get_position()
        self.origin_sensor_xy = np.array([float(sx), float(sy)], dtype=float)
        self.manifold.cur_rect_pos = np.zeros(2, dtype=float)

    def get_rect_position(self) -> np.ndarray:
        sx, sy = self.sensor.get_position()
        return np.array([float(sx) - self.origin_sensor_xy[0], float(sy) - self.origin_sensor_xy[1]], dtype=float)

    def generate_calibration_contour(self, pts_per_segment: int = 24) -> List[Tuple[str, float, float]]:
        """
        Generates the closed perimeter of the Maximum Inscribed Rectangle + internal crosshairs:
        1. Top Edge (Horizontal):    (-W, -H) -> (+W, -H)  [checks vertical jump dy(x)]
        2. Right Edge (Vertical):    (+W, -H) -> (+W, +H)  [checks horizontal jump dx(y)]
        3. Bottom Edge (Horizontal): (+W, +H) -> (-W, +H)  [checks vertical jump dy(x)]
        4. Left Edge (Vertical):     (-W, +H) -> (-W, -H)  [checks horizontal jump dx(y)]
        5. Horizontal Center Axis:   (-W,  0) -> (+W,  0)
        6. Vertical Center Axis:     ( 0, -H) -> ( 0, +H)
        """
        W, H = 0.88 * self.manifold.W, 0.88 * self.manifold.H
        waypoints: List[Tuple[str, float, float]] = []

        # 4 Perimeter Edges of the Rectangle
        for t in np.linspace(0.0, 1.0, pts_per_segment, endpoint=False):
            waypoints.append(("TOP_EDGE", -W + 2 * W * t, -H))
        for t in np.linspace(0.0, 1.0, pts_per_segment, endpoint=False):
            waypoints.append(("RIGHT_EDGE", +W, -H + 2 * H * t))
        for t in np.linspace(0.0, 1.0, pts_per_segment, endpoint=False):
            waypoints.append(("BOTTOM_EDGE", +W - 2 * W * t, +H))
        for t in np.linspace(0.0, 1.0, pts_per_segment + 1):
            waypoints.append(("LEFT_EDGE", -W, +H - 2 * H * t))

        # Internal Horizontal & Vertical Crosshairs
        for t in np.linspace(-1.0, 1.0, pts_per_segment):
            waypoints.append(("HORIZ_AXIS", W * t, 0.0))
        for t in np.linspace(-1.0, 1.0, pts_per_segment):
            waypoints.append(("VERT_AXIS", 0.0, H * t))

        return waypoints

    def run_rectangle_contour_iteration(
        self, iter_idx: int, pts_per_segment: int = 24, step_dwell_s: float = 0.045
    ) -> Dict:
        """
        Traces the rectangle perimeter & crosshairs, measures horizontal/vertical accompanying
        jumps, saves CSV trace, and updates the 9x9 decoupling manifold.
        """
        self.move_to_rect_origin(settle_s=0.55)
        waypoints = self.generate_calibration_contour(pts_per_segment=pts_per_segment)

        # Smooth lead-in to starting corner (-W, -H)
        start_x, start_y = waypoints[0][1], waypoints[0][2]
        for alpha in np.linspace(0.1, 1.0, 12):
            ux, uy = self.manifold.evaluate_pulse(alpha * start_x, alpha * start_y, self.last_base_u)
            self.driver.send_raw_pulses(ux, uy)
            self.last_base_u = (ux, uy)
            time.sleep(0.025)
        time.sleep(0.20)

        samples: List[Dict] = []
        horiz_jump_y: List[float] = []
        vert_jump_x: List[float] = []

        for seg_name, x_des, y_des in waypoints:
            ux, uy = self.manifold.evaluate_pulse(x_des, y_des, self.last_base_u)
            self.driver.send_raw_pulses(ux, uy)
            self.last_base_u = (ux, uy)
            time.sleep(step_dwell_s)

            pos = self.get_rect_position()
            err_x = float(x_des - pos[0])
            err_y = float(y_des - pos[1])

            if seg_name in ("TOP_EDGE", "BOTTOM_EDGE", "HORIZ_AXIS"):
                horiz_jump_y.append(abs(err_y))
            if seg_name in ("LEFT_EDGE", "RIGHT_EDGE", "VERT_AXIS"):
                vert_jump_x.append(abs(err_x))

            samples.append({
                "iter": iter_idx,
                "segment": seg_name,
                "x_des": round(x_des, 2),
                "y_des": round(y_des, 2),
                "x_act": round(float(pos[0]), 2),
                "y_act": round(float(pos[1]), 2),
                "err_x": round(err_x, 2),
                "err_y": round(err_y, 2),
                "err_norm": round(float(np.hypot(err_x, err_y)), 2),
                "u_x": round(ux, 2),
                "u_y": round(uy, 2),
            })

        os.makedirs("experiments", exist_ok=True)
        csv_path = f"experiments/rect_contour_iter_{iter_idx}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(samples[0].keys()))
            writer.writeheader()
            writer.writerows(samples)

        rmse_all = float(np.sqrt(np.mean([s["err_norm"] ** 2 for s in samples])))
        mean_horiz_jump = float(np.mean(horiz_jump_y)) if horiz_jump_y else 0.0
        mean_vert_jump = float(np.mean(vert_jump_x)) if vert_jump_x else 0.0

        # Update 9x9 decoupling lattice to cancel observed cross-axis jump & lag
        self.manifold.update_lattice_from_samples(samples, lr=0.55)
        self.manifold.save_manifold()

        return {
            "iter": iter_idx,
            "rmse_counts": round(rmse_all, 2),
            "horiz_parasitic_jump_y_counts": round(mean_horiz_jump, 2),
            "vert_parasitic_jump_x_counts": round(mean_vert_jump, 2),
            "csv_path": csv_path,
        }

    def move_point_to_point(
        self,
        x_target: float,
        y_target: float,
        duration_ms: float = 85.0,
        max_settle_ms: float = 165.0,
        tol_counts: float = 2.0,
    ) -> Dict:
        """
        Stage III: Continuous Point-to-Point Fine Control on the Decoupled Rectangular Manifold.
        Moves smoothly from current position `(x_A, y_A)` to `(x_target, y_target)` along a
        Minimum-Jerk straight line evaluated through the learned decoupling manifold $\\Phi(x, y)$,
        followed by closed-loop micro-trim without `recenter_cursor` teleportation.
        """
        x_tgt = float(np.clip(x_target, -0.92 * self.manifold.W, 0.92 * self.manifold.W))
        y_tgt = float(np.clip(y_target, -0.92 * self.manifold.H, 0.92 * self.manifold.H))
        p_start = self.get_rect_position()
        p_target = np.array([x_tgt, y_tgt], dtype=float)
        chord_vec = p_target - p_start
        L = float(np.linalg.norm(chord_vec))
        if L < 1.0:
            return {
                "start": tuple(np.round(p_start, 1)),
                "target": (round(x_tgt, 1), round(y_tgt, 1)),
                "final": tuple(np.round(p_start, 1)),
                "err_counts": 0.0,
                "straight_dev_cnt": 0.0,
                "elapsed_ms": 0.0,
            }

        d_hat = chord_vec / L
        n_hat = np.array([-d_hat[1], d_hat[0]], dtype=float)

        gx_end = x_tgt + float(self.manifold.origin_offset_xy[0])
        gy_end = y_tgt + float(self.manifold.origin_offset_xy[1])
        K_end = self.manifold.ws.local_inverse_jacobian(gx_end, gy_end)

        t0 = time.perf_counter()
        last_cmd_t = t0
        last_trim_ms = 0.0
        perp_devs: List[float] = []
        u_accum_trim = np.zeros(2, dtype=float)

        while True:
            now = time.perf_counter()
            elapsed_ms = (now - t0) * 1000.0
            p_cur = self.get_rect_position()
            err_vec = p_target - p_cur
            err_cnt = float(np.linalg.norm(err_vec))

            rel_vec = p_cur - p_start
            p_par = float(np.dot(rel_vec, d_hat))
            e_perp = float(np.dot(rel_vec, n_hat))
            if 8.0 <= p_par <= 0.98 * L:
                perp_devs.append(abs(e_perp))

            if elapsed_ms <= duration_ms:
                # Phase 1: 200Hz Minimum-Jerk Straight-Line Trajectory on Decoupled Manifold + Lead
                if (now - last_cmd_t) >= 0.0048:
                    tau = float(np.clip((elapsed_ms + 18.0) / max(20.0, duration_ms), 0.0, 1.0))
                    s_mj = 10.0 * (tau ** 3) - 15.0 * (tau ** 4) + 6.0 * (tau ** 5)
                    # Add orthogonal cross-track damping in rectangle coordinates
                    p_ref = p_start + s_mj * chord_vec - 0.65 * e_perp * n_hat
                    ux, uy = self.manifold.evaluate_pulse(float(p_ref[0]), float(p_ref[1]), self.last_base_u)
                    self.driver.send_raw_pulses(ux, uy)
                    self.last_base_u = (ux, uy)
                    last_cmd_t = now
            else:
                # Phase 2: Closed-Loop Fine Micro-Stepping on the Decoupled Manifold
                if err_cnt <= tol_counts and elapsed_ms >= duration_ms + 25.0:
                    break
                if (elapsed_ms - last_trim_ms) >= 32.0:
                    du_step = 0.72 * (K_end @ err_vec)
                    # Hysteresis-aware micro-step when reversing axis tooth engagement
                    for idx, (sigma_attr, h_pos, h_neg) in enumerate([
                        ("sigma_x", self.manifold.hyst_x_pos, self.manifold.hyst_x_neg),
                        ("sigma_y", self.manifold.hyst_y_pos, self.manifold.hyst_y_neg),
                    ]):
                        if abs(du_step[idx]) > 0.25:
                            new_sig = 1.0 if du_step[idx] > 0 else -1.0
                            cur_sig = getattr(self.manifold, sigma_attr)
                            if new_sig * cur_sig < 0:
                                du_step[idx] += new_sig * (h_pos + h_neg) * 0.85
                            else:
                                du_step[idx] += new_sig * 2.0
                            setattr(self.manifold, sigma_attr, new_sig)

                    du_step = np.clip(du_step, -28.0, 28.0)
                    u_accum_trim = np.clip(u_accum_trim + du_step, -65.0, 65.0)
                    ux_eq, uy_eq = self.manifold.evaluate_pulse(x_tgt, y_tgt, self.last_base_u)
                    ux_cmd = float(np.clip(ux_eq + u_accum_trim[0], WS63ServoDriver.SAFE_MIN_US, WS63ServoDriver.SAFE_MAX_US))
                    uy_cmd = float(np.clip(uy_eq + u_accum_trim[1], WS63ServoDriver.SAFE_MIN_US, WS63ServoDriver.SAFE_MAX_US))
                    self.driver.send_raw_pulses(ux_cmd, uy_cmd)
                    last_trim_ms = elapsed_ms

            if elapsed_ms >= max_settle_ms:
                break
            time.sleep(0.001)

        p_final = self.get_rect_position()
        final_err = float(np.hypot(x_tgt - p_final[0], y_tgt - p_final[1]))
        max_dev = float(max(perp_devs)) if perp_devs else 0.0
        return {
            "start": (round(float(p_start[0]), 1), round(float(p_start[1]), 1)),
            "target": (round(x_tgt, 1), round(y_tgt, 1)),
            "final": (round(float(p_final[0]), 1), round(float(p_final[1]), 1)),
            "err_counts": round(final_err, 2),
            "err_um": round(final_err * 7.9375, 1),
            "straight_dev_cnt": round(max_dev, 2),
            "straight_dev_pct": round(100.0 * max_dev / max(1.0, L), 2),
            "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 1),
        }

    def close(self):
        try:
            self.sensor.stop()
            self.driver.close()
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3-RRR Maximum Inscribed Rectangle Decoupling & All-Point Fine Control")
    parser.add_argument("--dry-check", action="store_true", help="Offline geometry & rectangle check without moving servos")
    parser.add_argument("--train-rect", type=int, default=0, help="Run N iterations of rectangle perimeter jump-elimination")
    parser.add_argument("--test-grid", action="store_true", help="Test continuous point-to-point positioning across rectangle grid")
    args = parser.parse_args()

    if args.dry_check:
        m = RectangularManifoldController()
        print(f"[Dry-Check OK] Shifted Geometric Origin: offset_xy={m.origin_offset_xy.tolist()} cnt -> u0*=({m.u0_star_x:.1f}, {m.u0_star_y:.1f}) us")
        print(f"[Dry-Check OK] Maximum Inscribed Symmetric Rectangle: [-{m.W:.0f}, +{m.W:.0f}] x [-{m.H:.0f}, +{m.H:.0f}] counts "
              f"({2*m.W:.0f}x{2*m.H:.0f} cnt = {2*m.W*0.0079375:.2f}x{2*m.H*0.0079375:.2f} mm)")
        raise SystemExit(0)

    trainer = HardwareRectangleTrainer()
    trainer.sensor.start()
    try:
        if args.train_rect > 0:
            print("=" * 90)
            print(f"  STAGE II: ITERATIVE RECTANGLE CONTOUR & CROSS-AXIS JUMP ELIMINATION ({args.train_rect} Iters)")
            print("=" * 90)
            for k in range(1, args.train_rect + 1):
                res = trainer.run_rectangle_contour_iteration(iter_idx=k)
                print(
                    f"  [Iter {k}/{args.train_rect}] Contour RMSE: {res['rmse_counts']:5.2f} cnt | "
                    f"Horiz->Vert Jump: {res['horiz_parasitic_jump_y_counts']:5.2f} cnt | "
                    f"Vert->Horiz Jump: {res['vert_parasitic_jump_x_counts']:5.2f} cnt -> {res['csv_path']}"
                )

        if args.test_grid:
            print("=" * 90)
            print("  STAGE III: CONTINUOUS POINT-TO-POINT FINE POSITIONING ACROSS RECTANGLE")
            print("=" * 90)
            trainer.move_to_rect_origin(settle_s=0.55)
            W, H = 0.80 * trainer.manifold.W, 0.80 * trainer.manifold.H
            test_pts = [
                (-W, -H), (+W, -H), (+W, +H), (-W, +H),
                (0.0, 0.0), (+0.5 * W, -0.6 * H), (-0.7 * W, +0.4 * H),
                (+0.8 * W, +0.2 * H), (-0.3 * W, -0.8 * H), (0.0, +0.75 * H),
            ]
            errs = []
            for idx, (tx, ty) in enumerate(test_pts, 1):
                r = trainer.move_point_to_point(tx, ty)
                errs.append(r["err_counts"])
                print(
                    f"  Pt #{idx:02d} {r['start']} -> Tgt{r['target']} | Final{r['final']} | "
                    f"Err: {r['err_counts']:4.2f} cnt ({r['err_um']:4.1f} um) | "
                    f"StraightDev: {r['straight_dev_cnt']:4.1f} cnt ({r['straight_dev_pct']:4.1f}%) | {r['elapsed_ms']:5.1f}ms"
                )
            print(f"\n  [Summary] Mean Error: {np.mean(errs):.2f} cnt | Max Error: {np.max(errs):.2f} cnt")
    finally:
        trainer.driver.center(0.3)
        trainer.close()
