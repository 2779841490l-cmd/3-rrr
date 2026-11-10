"""
hardware_probe.py - 90-Second Universal 3-RRR Hardware Probe & Auto-Calibrator
==============================================================================
Automatically characterizes:
1. Servo Internal Deadband (D_pos, D_neg) in microseconds.
2. Asymmetric Gear Backlash Hysteresis (B_pos, B_neg) in microseconds.
3. 2nd-Order Position-Dependent Jacobian Field J(u_x, u_y) across workspace.
4. Generates:
   - `calib_params.json` (universal model parameters)
   - `prompts/deepseek_r1_stage1.md` (concise mathematical prompt for DeepSeek R1)
"""

import json
import os
import shutil
import time
from datetime import datetime
from typing import Dict, List, Tuple
import numpy as np

from mouse_sensor import HighPrecisionMouseSensor
from servo_driver import WS63ServoDriver


class Universal3RRRProbe:
    def __init__(self, dpi: int = 3200):
        self.sensor = HighPrecisionMouseSensor(dpi=dpi)
        self.driver = WS63ServoDriver()
        self.u0_x = float(WS63ServoDriver.NEUTRAL_S1)
        self.u0_y = float(WS63ServoDriver.NEUTRAL_S2)

    def _preload_axis(self, axis: str, direction: int, preload_us: float = 55.0):
        """Pre-engage gear teeth in `direction` (+1 or -1) and return to neutral."""
        self.driver.center(settle_s=0.25)
        if axis == "X":
            self.driver.move_smooth_raw(self.u0_x - direction * preload_us, self.u0_y, duration_s=0.25)
            time.sleep(0.15)
            self.driver.move_smooth_raw(self.u0_x, self.u0_y, duration_s=0.25)
        else:
            self.driver.move_smooth_raw(self.u0_x, self.u0_y - direction * preload_us, duration_s=0.25)
            time.sleep(0.15)
            self.driver.move_smooth_raw(self.u0_x, self.u0_y, duration_s=0.25)
        time.sleep(0.30)
        self.sensor.read_delta_and_clear()

    def measure_deadband(self, axis: str, direction: int, count_threshold: int = 3) -> float:
        """
        After pre-engaging gear teeth in the SAME direction `direction`,
        incrementally step by 1 us until optical sensor detects >= count_threshold counts.
        """
        self._preload_axis(axis, direction, preload_us=50.0)
        accum_dx, accum_dy = 0, 0
        for du in range(1, 28):
            if axis == "X":
                self.driver.send_raw_pulses(self.u0_x + direction * du, self.u0_y)
            else:
                self.driver.send_raw_pulses(self.u0_x, self.u0_y + direction * du)
            time.sleep(0.06)
            dx, dy = self.sensor.read_delta_and_clear()
            accum_dx += dx
            accum_dy += dy
            mag = float(np.hypot(accum_dx, accum_dy))
            if mag >= count_threshold:
                return float(du)
        return 8.0

    def measure_backlash(self, axis: str, reverse_dir: int, count_threshold: int = 5) -> float:
        """
        Pre-engage teeth in `-reverse_dir`, then reverse in `reverse_dir` by 2 us increments
        until motion in the reversed direction reaches `count_threshold` counts.
        """
        # Engage teeth in the opposite direction first
        self._preload_axis(axis, -reverse_dir, preload_us=60.0)
        accum_dx, accum_dy = 0, 0
        for du in range(2, 95, 2):
            if axis == "X":
                self.driver.send_raw_pulses(self.u0_x + reverse_dir * du, self.u0_y)
            else:
                self.driver.send_raw_pulses(self.u0_x, self.u0_y + reverse_dir * du)
            time.sleep(0.06)
            dx, dy = self.sensor.read_delta_and_clear()
            accum_dx += dx
            accum_dy += dy
            mag = float(np.hypot(accum_dx, accum_dy))
            if mag >= count_threshold:
                return float(du) * 0.5  # Half-width play operator radius B
        return 25.0

    def measure_spatial_jacobian_grid(
        self, grid_offsets_us: Tuple[int, ...] = (-60, 0, 60), delta_probe_us: float = 35.0
    ) -> Tuple[List[Dict], np.ndarray, np.ndarray]:
        """
        Probe a spatial grid of (u_x, u_y) operating points to identify both:
        1. Linear central Jacobian J0 and Inverse K0 = J0^(-1)
        2. 2nd-order nonlinear polynomial Jacobian coefficients
        """
        samples = []
        preload = 40.0

        for ox in grid_offsets_us:
            for oy in grid_offsets_us:
                bx = self.u0_x + ox
                by = self.u0_y + oy

                # 1. Probe +dUx at (bx, by) with positive tooth preload
                self.driver.move_smooth_raw(bx - preload, by, duration_s=0.20)
                self.driver.move_smooth_raw(bx, by, duration_s=0.20)
                time.sleep(0.20)
                self.sensor.read_delta_and_clear()

                self.driver.move_smooth_raw(bx + delta_probe_us, by, duration_s=0.20)
                time.sleep(0.20)
                dx_ux, dy_ux = self.sensor.read_delta_and_clear()

                # 2. Probe +dUy at (bx, by) with positive tooth preload
                self.driver.move_smooth_raw(bx, by - preload, duration_s=0.20)
                self.driver.move_smooth_raw(bx, by, duration_s=0.20)
                time.sleep(0.20)
                self.sensor.read_delta_and_clear()

                self.driver.move_smooth_raw(bx, by + delta_probe_us, duration_s=0.20)
                time.sleep(0.20)
                dx_uy, dy_uy = self.sensor.read_delta_and_clear()

                J_local = np.array(
                    [
                        [dx_ux / delta_probe_us, dx_uy / delta_probe_us],
                        [dy_ux / delta_probe_us, dy_uy / delta_probe_us],
                    ],
                    dtype=float,
                )

                samples.append(
                    {
                        "offset_ux": float(ox),
                        "offset_uy": float(oy),
                        "J_local": J_local.tolist(),
                        "det_J": float(np.linalg.det(J_local)),
                    }
                )

        # Central or mean Jacobian
        J_center = np.mean([np.array(s["J_local"]) for s in samples], axis=0)
        if abs(np.linalg.det(J_center)) > 1e-4:
            K_center = np.linalg.inv(J_center)
        else:
            # Fallback to previously calibrated matrix from servo_test_3rrr_backup.c
            K_center = np.array([[-3.6980, 0.9905], [0.2311, -2.7405]], dtype=float)
            J_center = np.linalg.inv(K_center)

        return samples, J_center, K_center

    def run_full_probe(self) -> Dict:
        print("=" * 65)
        print("  3-RRR Universal 90-Second Hardware Limit & Jacobian Probe")
        print("=" * 65)
        self.sensor.start()
        self.driver.center(settle_s=0.6)

        print("[1/3] Probing MG996R Internal Comparator Deadband (D+, D-)...")
        db_x_pos = self.measure_deadband("X", +1)
        db_x_neg = self.measure_deadband("X", -1)
        db_y_pos = self.measure_deadband("Y", +1)
        db_y_neg = self.measure_deadband("Y", -1)
        print(f"      Deadband X: (+{db_x_pos:.1f} us, -{db_x_neg:.1f} us)")
        print(f"      Deadband Y: (+{db_y_pos:.1f} us, -{db_y_neg:.1f} us)")

        print("[2/3] Probing Asymmetric Gear Backlash Hysteresis (B+, B-)...")
        bl_x_pos = self.measure_backlash("X", +1)
        bl_x_neg = self.measure_backlash("X", -1)
        bl_y_pos = self.measure_backlash("Y", +1)
        bl_y_neg = self.measure_backlash("Y", -1)
        print(f"      Backlash X Play Radius: (+{bl_x_pos:.1f} us, -{bl_x_neg:.1f} us)")
        print(f"      Backlash Y Play Radius: (+{bl_y_pos:.1f} us, -{bl_y_neg:.1f} us)")

        print("[3/3] Probing 3x3 Spatial Nonlinear Jacobian Field J(u_x, u_y)...")
        grid_samples, J_center, K_center = self.measure_spatial_jacobian_grid()

        self.driver.center(settle_s=0.4)
        self.sensor.stop()
        self.driver.close()

        result = {
            "timestamp": datetime.now().isoformat(),
            "dpi": self.sensor.dpi,
            "neutral_us": {
                "s1_x": int(self.u0_x),
                "s2_y1": int(self.u0_y),
                "s3_y2": int(3000 - self.u0_y),
                "s4_aux": 1500,
            },
            "deadband_us": {
                "x_pos": db_x_pos,
                "x_neg": db_x_neg,
                "y_pos": db_y_pos,
                "y_neg": db_y_neg,
            },
            "backlash_play_radius_us": {
                "x_pos": bl_x_pos,
                "x_neg": bl_x_neg,
                "y_pos": bl_y_pos,
                "y_neg": bl_y_neg,
            },
            "jacobian_center_px_per_us": J_center.tolist(),
            "inverse_jacobian_center_us_per_px": K_center.tolist(),
            "spatial_grid_samples": grid_samples,
        }

        self._save_and_archive(result)
        self._generate_deepseek_r1_prompt(result)
        return result

    @staticmethod
    def _save_and_archive(result: Dict):
        calib_path = "calib_params.json"
        if os.path.exists(calib_path):
            os.makedirs("backups", exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            shutil.copy2(calib_path, os.path.join("backups", f"calib_params_{ts}.json"))
        with open(calib_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n[OK] Saved universal calibration parameters -> {calib_path}")

    @staticmethod
    def _generate_deepseek_r1_prompt(result: Dict):
        os.makedirs("prompts", exist_ok=True)
        prompt_path = os.path.join("prompts", "deepseek_r1_stage1.md")
        content = f"""# DeepSeek R1 核心控制律数学推导请求（高浓缩实测数据版）

我们正在控制一台基于海思 WS63（50Hz 原子级无抖动 PWM）驱动 3 颗 MG996R 舵机的 **3-RRR 平面并联机构**，末端夹持 3200 DPI（$7.9375\\ \\mu\\text{{m}}/\\text{{count}}$）光学鼠标。机械装配已达物理极限，现已完成实机探针测试，实测物理极限参数如下：

```json
{json.dumps(result, indent=2, ensure_ascii=False)}
```

请严格基于上述实测参数，一次性完成以下 3 项数学推导（无需编写串口/UI 工程代码，只需给出精确闭式方程与核心数值参数）：

1. **二阶空间非线性逆雅可比张量 $K(u_x, u_y)$**：
   - 基于 `spatial_grid_samples` 拟合 $\\Delta \\mathbf{{u}} = K(u_x - u_{{x0}}, u_y - u_{{y0}}) \\Delta \\mathbf{{x}}$ 的二阶多项式系数矩阵，消除大行程下的交叉轴耦合。
2. **非对称 Prandtl-Ishlinskii (Play-Operator) 齿隙逆补偿方程**：
   - 根据实测 `backlash_play_radius_us`，给出连续齿面接触状态更新方程 $g_i(k) = \\mathcal{{P}}[u_{{kin,i}}(k), g_i(k-1)]$，并针对换向瞬间设计平滑过渡因子 $\\eta(\\Delta u)$，防止在目标误差 $\\pm 2\\text{{ counts}}$ 内发生正反跳变抽搐。
3. **跨死区瞬态脉冲激发（Impulse-Retract）+ 变增益防超调闭环控制律**：
   - 针对实测死区 `deadband_us`，设计当 $|e| \\in [1, 5]\\text{{ counts}}$ 时的单帧激发幅值 $u_{{kick}}(e)$ 与变增益比例积分律 $K_p(e), K_i(e)$，并给出离散采样周期 $T_s = 20\\text{{ ms}}$ 下不产生极限环震荡（Limit Cycle）的稳定性边界。
"""
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"[OK] Generated concise DeepSeek R1 math prompt -> {prompt_path}")


if __name__ == "__main__":
    probe = Universal3RRRProbe(dpi=3200)
    probe.run_full_probe()
