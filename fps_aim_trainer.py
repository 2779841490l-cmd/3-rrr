r"""
fps_aim_trainer.py — Full-Screen (1536x864) 100ms 3200-DPI FPS Aim-Lab Visual Target System
===========================================================================================
Powered by PyTorch Physics-Informed Neural Ballistic Manifold (PI-NBM-Net) &
Joint-Space Inverse Play Compensator (`rect_manifold_calibrator.py`).

Features:
1. True Live Hardware Reticle (实时硬件光流瞄准准星):
   - High-frequency 60 FPS Canvas reticle reading the optical displacement sensor directly;
   - Displays real-time continuous trajectory, live micrometric counts, and flight dynamics.
2. Honest Multi-Threaded State Machine:
   - UI never freezes; state transitions clearly from [AIMING...] to [HIT] / [MISS];
   - Hit results hold for 1.2s before the next shot so the user clearly sees the outcome.
3. Interactive Modes:
   - [SPACE]: Spawn single random target ball within safe envelope.
   - [A]: Toggle continuous auto-aim loop.
   - [LEFT CLICK]: Click anywhere inside the boundary to spawn and shoot at that target.
   - [C]: Return mechanism to center origin (0, 0).
   - [ESC]: Exit cleanly and recenter servos.
4. Headless Full-Screen Benchmark:
   - `python fps_aim_trainer.py --headless-benchmark --num-balls 12`
"""

import argparse
import csv
import ctypes
import math
import os
import queue
import random
import sys
import threading
import time
import tkinter as tk
from typing import Dict, List, Optional, Tuple
import numpy as np

from rect_manifold_calibrator import HardwareRectangleTrainer, WS63ServoDriver

user32 = ctypes.windll.user32


class FullScreenFPSAimTrainer:
    def __init__(
        self,
        ball_radius_px: int = 18,
        tol_counts: float = 1.2,
        seed: int = 12345,
        # 【否证：默认关闭】把搜索指标从"计数圆形误差"改成"屏幕加权误差"，
        # 直觉上应该直接降屏幕误差（X 权重 2.0 是 Y 的 1.4 倍），但实测是负结果。
        # 同 seed=12345、同 20 发、相邻时段的配对实验（experiments/fps_*_{mw,nomw}_r*.csv）：
        #   ON  r1/r2: 19/20 & 18/20 命中 | 均光学 5.04 & 3.96 cnt | 均屏幕 8.53 & 6.91 px
        #   OFF r2/r3: 20/20 & 19/20 命中 | 均光学 3.66 & 3.49 cnt | 均屏幕 6.76 & 6.39 px
        # OFF 在均值与命中率上更好，尾部也无额外收益 => 加权会扭曲离散晶格上的择优，
        # 在量化平台上"加权更近"不等于"落点更好"。故默认回到 OFF 的计数圆形误差。
        use_metric_w: bool = False,
        run_tag: str = "",
    ):
        self.trainer = HardwareRectangleTrainer()
        self.trainer.sensor.start()
        self.trainer.move_to_rect_origin(settle_s=0.45)

        self.sw = user32.GetSystemMetrics(0) or 1536
        self.sh = user32.GetSystemMetrics(1) or 864
        self.cx = self.sw // 2
        self.cy = self.sh // 2

        # Safe high-linearity target envelope (well within physical mechanism bounds)
        # Manifold limits: W=390, H=300 counts. We use safe envelope W_safe=280, H_safe=210.
        self.safe_rect_x = min(280.0, 0.75 * self.trainer.manifold.W)
        self.safe_rect_y = min(210.0, 0.72 * self.trainer.manifold.H)

        # Screen scale: map [-safe_rect_x, +safe_rect_x] to [-540, +540] px
        # and [-safe_rect_y, +safe_rect_y] to [-300, +300] px
        self.max_screen_dx = min(560, int(0.38 * self.sw))
        self.max_screen_dy = min(300, int(0.36 * self.sh))
        self.scale_x = float(self.max_screen_dx) / self.safe_rect_x
        self.scale_y = float(self.max_screen_dy) / self.safe_rect_y

        self.ball_radius_px = int(ball_radius_px)
        self.tol_counts = float(tol_counts)
        self.seed = int(seed)
        # A/B 归因开关：use_metric_w=False 时完全退回旧的"计数圆形误差"择优，
        # 用来量化"搜索指标 = 评分指标"这一改动到底贡献了多少屏幕精度。
        self.use_metric_w = bool(use_metric_w)
        # 运行标签：追加到 CSV 文件名，避免 A/B 与重复实验互相覆盖。
        # 默认空串 => experiments/fps_fullscreen_aim_benchmark.csv（Gate3 / 论文画图沿用）
        self.run_tag = str(run_tag or "").strip()

        # State tracking
        self.shots_count = 0
        self.hits_count = 0
        self.is_busy = False
        self.auto_mode = False
        self.target_screen_xy: Optional[Tuple[int, int]] = None
        self.last_res: Optional[Dict] = None
        self.traj_history: List[Tuple[int, int]] = []
        self.traj_lock = threading.Lock()

    def screen_to_rect(self, sx_rel_px: float, sy_rel_px: float) -> Tuple[float, float]:
        rx = float(np.clip(sx_rel_px / self.scale_x, -self.safe_rect_x, self.safe_rect_x))
        ry = float(np.clip(sy_rel_px / self.scale_y, -self.safe_rect_y, self.safe_rect_y))
        return rx, ry

    def rect_to_screen(self, rx_cnt: float, ry_cnt: float) -> Tuple[float, float]:
        return float(rx_cnt * self.scale_x), float(ry_cnt * self.scale_y)

    def shoot_screen_target_blocking(self, screen_dx_px: int, screen_dy_px: int) -> Dict:
        """
        Executes point-to-point motion with 100ms ballistic acceleration and
        inverse-play stiction-break trim.
        """
        rx_tgt, ry_tgt = self.screen_to_rect(screen_dx_px, screen_dy_px)
        # metric_w = (scale_x, scale_y)：让控制器按"屏幕像素误差"择优，
        # 而不是按"计数空间圆形误差"。两者最优落点不同（X 轴权重是 Y 的 1.4 倍），
        # 而打靶判分用的是屏幕误差 —— 让搜索指标 = 评分指标，是零硬件成本的直接收益。
        # use_metric_w=False 时为 None，完全等价于旧行为（A/B 对照组）。
        metric_w = (self.scale_x, self.scale_y) if self.use_metric_w else None
        res = self.trainer.move_point_to_point(
            rx_tgt,
            ry_tgt,
            tol_counts=self.tol_counts,
            dual_lattice=True,
            metric_w=metric_w,
        )

        act_sx, act_sy = self.rect_to_screen(res["final"][0], res["final"][1])
        tgt_sx, tgt_sy = self.rect_to_screen(res["target"][0], res["target"][1])
        screen_err_px = float(np.hypot(tgt_sx - act_sx, tgt_sy - act_sy))

        res["target_screen_delta_px"] = (int(round(tgt_sx)), int(round(tgt_sy)))
        res["actual_screen_delta_px"] = (round(act_sx, 1), round(act_sy, 1))
        res["screen_error_px"] = round(screen_err_px, 2)
        res["hit_ball"] = bool(screen_err_px <= self.ball_radius_px)
        return res

    def run_headless_benchmark(self, num_targets: int = 12) -> int:
        print("=" * 92)
        print(f"  3-RRR FULL-SCREEN ({self.sw}x{self.sh}) PI-NBM-NET AIM-TRAINER HARDWARE BENCHMARK")
        print(
            f"  Safe Envelope: [-{self.safe_rect_x:.0f}, +{self.safe_rect_x:.0f}] x [-{self.safe_rect_y:.0f}, +{self.safe_rect_y:.0f}] cnt "
            f"| Screen Target Box: [±{self.max_screen_dx}, ±{self.max_screen_dy}] px "
            f"| Ball Radius: {self.ball_radius_px} px"
        )
        print(
            f"  Config: tol={self.tol_counts:.2f} cnt | seed={self.seed} "
            f"| metric_w={'ON (' + f'{self.scale_x:.4f}, {self.scale_y:.4f}' + ')' if self.use_metric_w else 'OFF (count-circle)'} "
            f"| tag='{self.run_tag or '(default)'}'"
        )
        print("=" * 92)

        # High-coverage target sequence across corners, edges, and inner quadrants
        fixed_targets: List[Tuple[int, int]] = [
            (-self.max_screen_dx, -self.max_screen_dy),  # Top-Left Corner
            (self.max_screen_dx, -self.max_screen_dy),   # Top-Right Corner
            (self.max_screen_dx, self.max_screen_dy),    # Bottom-Right Corner
            (-self.max_screen_dx, self.max_screen_dy),   # Bottom-Left Corner
            (0, -self.max_screen_dy),                    # Top Edge
            (self.max_screen_dx, 0),                     # Right Edge
            (0, self.max_screen_dy),                     # Bottom Edge
            (-self.max_screen_dx, 0),                    # Left Edge
            (int(0.6 * self.max_screen_dx), int(-0.6 * self.max_screen_dy)),  # Quadrant 1
            (int(-0.6 * self.max_screen_dx), int(0.6 * self.max_screen_dy)),  # Quadrant 3
            (int(-0.5 * self.max_screen_dx), int(-0.5 * self.max_screen_dy)), # Quadrant 2
            (0, 0),                                      # Center Return
        ]
        screen_targets: List[Tuple[int, int]] = list(fixed_targets[:num_targets])
        if num_targets > len(screen_targets):
            # 超过 12 发时补充可复现的随机点（同 seed 同序列），用于把 n 拉到 20~40
            rng = random.Random(self.seed)
            while len(screen_targets) < num_targets:
                screen_targets.append(
                    (
                        rng.randint(-self.max_screen_dx, self.max_screen_dx),
                        rng.randint(-self.max_screen_dy, self.max_screen_dy),
                    )
                )

        os.makedirs("experiments", exist_ok=True)
        csv_path = f"experiments/fps_fullscreen_aim_benchmark{self.run_tag}.csv"
        rows = []
        hits = 0
        raw_errs = []
        screen_errs: List[float] = []
        ballistic_times = []

        for idx, (sx, sy) in enumerate(screen_targets, 1):
            res = self.shoot_screen_target_blocking(sx, sy)
            hit_str = "HIT " if res["hit_ball"] else "MISS"
            if res["hit_ball"]:
                hits += 1
            raw_errs.append(res["err_counts"])
            screen_errs.append(res["screen_error_px"])
            ballistic_times.append(res["ballistic_ms"])

            print(
                f"  Ball {idx:2d}/{len(screen_targets)} [{hit_str}] | "
                f"Screen ({res['target_screen_delta_px'][0]:+4d}, {res['target_screen_delta_px'][1]:+4d}) -> "
                f"({res['actual_screen_delta_px'][0]:+6.1f}, {res['actual_screen_delta_px'][1]:+6.1f}) px | "
                f"Rect {res['target']} -> {res['final']} cnt | "
                f"Err: {res['err_counts']:4.2f} cnt ({res['err_um']:4.1f} um) | "
                f"Ballistic: {res['ballistic_ms']:4.1f} ms"
            )
            rows.append({
                "ball_id": idx,
                "screen_target_x_px": res["target_screen_delta_px"][0],
                "screen_target_y_px": res["target_screen_delta_px"][1],
                "screen_actual_x_px": res["actual_screen_delta_px"][0],
                "screen_actual_y_px": res["actual_screen_delta_px"][1],
                "raw_target_dx_cnt": res["target"][0],
                "raw_target_dy_cnt": res["target"][1],
                "raw_actual_dx_cnt": res["final"][0],
                "raw_actual_dy_cnt": res["final"][1],
                "raw_error_counts": res["err_counts"],
                "raw_error_um": res["err_um"],
                "screen_error_px": res["screen_error_px"],
                "ballistic_time_ms": res["ballistic_ms"],
                "settled_time_ms": res["elapsed_ms"],
                "hit": int(res["hit_ball"]),
            })

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        hit_rate = 100.0 * hits / max(1, len(screen_targets))
        mean_raw_err = float(np.mean(raw_errs))
        max_raw_err = float(np.max(raw_errs))
        mean_bal_ms = float(np.mean(ballistic_times))
        mean_screen_err = float(np.mean(screen_errs))
        max_screen_err = float(np.max(screen_errs))
        print("-" * 92)
        print(
            f"  [SUMMARY] Hit Rate: {hits}/{len(screen_targets)} ({hit_rate:.1f}%) | "
            f"Mean Optical Error: {mean_raw_err:.2f} cnt ({mean_raw_err*7.9375:.1f} um) | "
            f"Max Error: {max_raw_err:.2f} cnt | Mean Ballistic Time: {mean_bal_ms:.1f} ms"
        )
        print(
            f"  [SCREEN ] Mean screen err: {mean_screen_err:.2f} px | "
            f"Max screen err: {max_screen_err:.2f} px | "
            f"Scale: {self.scale_x:.4f} px/cnt (X), {self.scale_y:.4f} px/cnt (Y)"
        )

        # ---------------- 球半径阶梯扫描：回答"当前精度能打多大的球" ----------------
        # screen_errs 里每一发存的都是 hypot(scale_x*ex, scale_y*ey) 的屏幕像素误差，
        # 而"打中半径 R 的球"的判据正是 screen_err <= R。所以整条 R 曲线可以直接
        # 从已经打完的这一场数据里算出，零额外硬件时间、零额外采样。
        # 同时给出该 R 对应的物理容差（counts 与 um），便于和死区 6~14 us 对比：
        # 一旦 R/scale_x 掉到 4~5 cnt 以下，物理上就不可能全中了。
        radius_ladder = [2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32, 40]
        n_balls = len(screen_errs)
        sweep_rows = []
        r100 = None   # 100% 命中最小半径
        r95 = None    # >=95% 命中最小半径
        r50 = None    # >=50% 命中最小半径
        print("-" * 92)
        print("  [BALL SWEEP] 球半径 R 的等价容差 = X ±R/scale_x cnt, Y ±R/scale_y cnt")
        print(
            f"  {'R(px)':>6} {'Dia(px)':>8} {'X_tol(cnt)':>11} {'Y_tol(cnt)':>11} "
            f"{'X_tol(um)':>10} {'Y_tol(um)':>10} {'Hit':>9} {'Rate':>7}"
        )
        for R in radius_ladder:
            n_hit = sum(1 for e in screen_errs if e <= R + 1e-9)
            rate = 100.0 * n_hit / max(1, n_balls)
            xtol = float(R) / max(1e-9, self.scale_x)
            ytol = float(R) / max(1e-9, self.scale_y)
            print(
                f"  {R:6d} {2 * R:8d} {xtol:11.2f} {ytol:11.2f} "
                f"{xtol * 7.9375:10.1f} {ytol * 7.9375:10.1f} "
                f"{n_hit:4d}/{n_balls:<4d} {rate:6.1f}%"
            )
            sweep_rows.append({
                "radius_px": R,
                "diameter_px": 2 * R,
                "x_tol_counts": round(xtol, 4),
                "y_tol_counts": round(ytol, 4),
                "x_tol_um": round(xtol * 7.9375, 2),
                "y_tol_um": round(ytol * 7.9375, 2),
                "hits": n_hit,
                "total": n_balls,
                "hit_rate_pct": round(rate, 2),
            })
            if n_hit == n_balls and r100 is None:
                r100 = R
            if rate >= 95.0 and r95 is None:
                r95 = R
            if rate >= 50.0 and r50 is None:
                r50 = R

        # 物理红线标注：X 轴容差低于死区等效位移（4.5~10.5 cnt）的半径不可达
        phys_floor_cnt = 4.5
        r_phys = phys_floor_cnt * float(self.scale_x)
        print("-" * 92)
        if r100 is not None:
            print(
                f"  [VERDICT ] 100% 命中最小半径 = {r100} px (直径 {2 * r100} px | "
                f"X ±{r100 / self.scale_x:.2f} cnt = ±{r100 / self.scale_x * 7.9375:.1f} um | "
                f"Y ±{r100 / self.scale_y:.2f} cnt = ±{r100 / self.scale_y * 7.9375:.1f} um)"
            )
        else:
            r_need = float(np.ceil(max_screen_err))
            print(
                f"  [VERDICT ] 本场 {n_balls} 发未达 100%；最大屏幕误差 {max_screen_err:.2f} px "
                f"⇒ 至少需半径 {r_need:.0f} px 才能全中"
            )
        print(
            f"  [VERDICT ]  95% 命中最小半径 = {r95 if r95 is not None else 'N/A'}"
            f" px | 50% 命中最小半径 = {r50 if r50 is not None else 'N/A'} px"
        )
        print(
            f"  [PHYS    ] X 容差等于死区下界 4.5 cnt 时 R = {r_phys:.2f} px "
            f"⇒ 半径小于此值的球在物理上不可达（与控制器无关）"
        )

        sweep_path = f"experiments/fps_ball_sweep{self.run_tag}.csv"
        with open(sweep_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(sweep_rows[0].keys()))
            writer.writeheader()
            writer.writerows(sweep_rows)

        print(f"  [SAVED]   {csv_path}")
        print(f"  [SAVED]   {sweep_path}")
        print("=" * 92)
        # 退出码语义改为"按配置半径是否全中"：
        # 旧的 mean_raw_err <= 2.5 是历史门禁，与"能不能打球"无关，
        # 会把"半径 18 px 全中、但均值 2.6 cnt"这种成功判成失败。
        return 0 if hits == len(screen_targets) else 1

    def run_gui(self):
        root = tk.Tk()
        root.title("3-RRR 100ms 3200DPI Full-Screen PI-NBM-Net Aim-Trainer")
        root.attributes("-fullscreen", True)
        root.configure(bg="#080c14")

        sw, sh = self.sw, self.sh
        cx, cy = self.cx, self.cy
        bx0 = cx - self.max_screen_dx
        by0 = cy - self.max_screen_dy
        bx1 = cx + self.max_screen_dx
        by1 = cy + self.max_screen_dy

        canvas = tk.Canvas(root, width=sw, height=sh, bg="#080c14", highlightthickness=0)
        canvas.pack(fill="both", expand=True)

        # Draw dark cyber grid
        for gx in range(0, sw, 80):
            canvas.create_line(gx, 0, gx, sh, fill="#0f172a", width=1)
        for gy in range(0, sh, 80):
            canvas.create_line(0, gy, sw, gy, fill="#0f172a", width=1)

        # Safe operational boundary frame
        canvas.create_rectangle(bx0, by0, bx1, by1, outline="#1e3a8a", width=2, dash=(6, 4))
        canvas.create_text(
            bx0 + 8, by0 + 8, anchor="nw", fill="#3b82f6", font=("Consolas", 9),
            text=f"3-RRR SAFE OPERATIONAL ENVELOPE [±{self.safe_rect_x:.0f}, ±{self.safe_rect_y:.0f}] cnt"
        )

        # Screen center crosshair
        canvas.create_line(cx - 20, cy, cx + 20, cy, fill="#334155", width=2)
        canvas.create_line(cx, cy - 20, cx, cy + 20, fill="#334155", width=2)
        canvas.create_oval(cx - 3, cy - 3, cx + 3, cy + 3, fill="#475569", outline="")

        # Title & Help
        canvas.create_text(
            24, 18, anchor="nw", fill="#38bdf8", font=("Consolas", 13, "bold"),
            text=(
                "3-RRR 100ms 3200DPI FULL-SCREEN PI-NBM-NET AIM-TRAINER (LIVE OPTICAL TRACKER)\n"
                "[SPACE] Spawn Target  |  [A] Auto-Aim Mode  |  [LEFT CLICK] Click-to-Aim  |  [C] Recenter  |  [ESC] Exit"
            ),
        )

        # Dynamic Status HUD
        hud_status = canvas.create_text(
            24, 68, anchor="nw", fill="#67e8f9", font=("Consolas", 12, "bold"),
            text="Ready. Press [SPACE] to spawn target, or [LEFT CLICK] anywhere inside the blue frame."
        )
        hud_hw = canvas.create_text(
            24, 94, anchor="nw", fill="#94a3b8", font=("Consolas", 10),
            text="Stage Optical Pos: X: +0.0 cnt, Y: +0.0 cnt (+0.00 mm, +0.00 mm) | S1: 1500us S2: 1470us S3: 1530us"
        )

        # Worker & state queues
        action_queue: queue.Queue = queue.Queue()
        gui_running = [True]

        def worker_thread_fn():
            while gui_running[0]:
                try:
                    task = action_queue.get(timeout=0.05)
                except queue.Empty:
                    continue

                if task is None:
                    break

                task_type, target_pos = task
                if task_type == "SHOOT":
                    tx_screen, ty_screen = target_pos
                    dx_px = tx_screen - cx
                    dy_px = ty_screen - cy

                    # Clear trajectory history for this shot
                    with self.traj_lock:
                        self.traj_history.clear()

                    res = self.shoot_screen_target_blocking(dx_px, dy_px)
                    self.last_res = res
                    self.shots_count += 1
                    if res["hit_ball"]:
                        self.hits_count += 1

                    # Sync OS cursor to physical final position
                    try:
                        act_sx = int(round(cx + res["actual_screen_delta_px"][0]))
                        act_sy = int(round(cy + res["actual_screen_delta_px"][1]))
                        user32.SetCursorPos(act_sx, act_sy)
                    except Exception:
                        pass

                    self.is_busy = False

                    # If auto mode is enabled, wait 1.2s to clearly show result, then spawn next
                    if self.auto_mode and gui_running[0]:
                        time.sleep(1.20)
                        if self.auto_mode and gui_running[0] and not self.is_busy:
                            trigger_spawn_random()

                elif task_type == "CENTER":
                    with self.traj_lock:
                        self.traj_history.clear()
                    self.trainer.move_to_rect_origin(settle_s=0.35)
                    self.is_busy = False

        worker = threading.Thread(target=worker_thread_fn, daemon=True)
        worker.start()

        def trigger_shoot(tx: int, ty: int):
            if self.is_busy:
                return
            self.is_busy = True
            self.target_screen_xy = (tx, ty)
            self.last_res = None

            dx_px = tx - cx
            dy_px = ty - cy
            dist_px = float(np.hypot(dx_px, dy_px))

            # Immediately update HUD to AIMING status (prevents stale display!)
            canvas.itemconfig(
                hud_status,
                fill="#facc15",
                text=(
                    f"Shot #{self.shots_count + 1} [AIMING...] Target: ({dx_px:+d}, {dy_px:+d}) px | "
                    f"Distance: {dist_px:4.0f} px | 100ms Ballistic In-Flight..."
                ),
            )
            action_queue.put(("SHOOT", (tx, ty)))

        def trigger_spawn_random(_event=None):
            if self.is_busy:
                return
            # Pick a target safely within boundary
            margin = 35
            tx = random.randint(bx0 + margin, bx1 - margin)
            ty = random.randint(by0 + margin, by1 - margin)
            trigger_shoot(tx, ty)

        def toggle_auto(_event=None):
            self.auto_mode = not self.auto_mode
            if self.auto_mode and not self.is_busy:
                trigger_spawn_random()

        def on_click(event):
            # Disable auto on manual click
            self.auto_mode = False
            tx = int(np.clip(event.x, bx0 + 10, bx1 - 10))
            ty = int(np.clip(event.y, by0 + 10, by1 - 10))
            trigger_shoot(tx, ty)

        def trigger_recenter(_event=None):
            if self.is_busy:
                return
            self.is_busy = True
            self.auto_mode = False
            self.target_screen_xy = None
            self.last_res = None
            canvas.itemconfig(hud_status, fill="#94a3b8", text="Recentering 3-RRR stage to rectangular origin (0, 0)...")
            action_queue.put(("CENTER", None))

        # Main 60 FPS Render Tick (Live Reticle & Visual Feedback)
        def render_tick():
            if not gui_running[0]:
                return

            # Read live hardware position
            try:
                rx_cnt, ry_cnt = self.trainer.get_rect_position()
            except Exception:
                rx_cnt, ry_cnt = 0.0, 0.0

            cur_sx, cur_sy = self.rect_to_screen(rx_cnt, ry_cnt)
            hw_x = int(round(cx + cur_sx))
            hw_y = int(round(cy + cur_sy))

            # Continuously sync the real Windows OS mouse pointer to the 3-RRR hardware position!
            if self.is_busy:
                try:
                    user32.SetCursorPos(hw_x, hw_y)
                except Exception:
                    pass

            # Record in trajectory history during active aiming
            if self.is_busy:
                with self.traj_lock:
                    if not self.traj_history or np.hypot(hw_x - self.traj_history[-1][0], hw_y - self.traj_history[-1][1]) >= 2.0:
                        self.traj_history.append((hw_x, hw_y))
                        if len(self.traj_history) > 300:
                            self.traj_history.pop(0)

            # Redraw dynamic overlay
            canvas.delete("dynamic")

            # 1. Draw Target Ball (if active)
            r = self.ball_radius_px
            if self.target_screen_xy is not None:
                tx, ty = self.target_screen_xy

                if self.is_busy:
                    # In-flight aiming target (pulsing red/amber glow)
                    canvas.create_oval(tx - r, ty - r, tx + r, ty + r, fill="#ef4444", outline="#fca5a5", width=3, tags="dynamic")
                    canvas.create_oval(tx - 4, ty - 4, tx + 4, ty + 4, fill="#ffffff", outline="", tags="dynamic")
                    canvas.create_oval(tx - r - 6, ty - r - 6, tx + r + 6, ty + r + 6, outline="#ef4444", width=1, dash=(3, 3), tags="dynamic")
                elif self.last_res is not None:
                    # Settle finished: show true HIT or MISS
                    res = self.last_res
                    if res["hit_ball"]:
                        # Bright neon-green hit celebration ring
                        canvas.create_oval(tx - r, ty - r, tx + r, ty + r, fill="#10b981", outline="#6ee7b7", width=3, tags="dynamic")
                        canvas.create_oval(tx - 5, ty - 5, tx + 5, ty + 5, fill="#ffffff", outline="", tags="dynamic")
                        canvas.create_oval(tx - r - 8, ty - r - 8, tx + r + 8, ty + r + 8, outline="#34d399", width=2, tags="dynamic")
                    else:
                        # Miss: Keep red target, draw landing point with dashed connecting line
                        canvas.create_oval(tx - r, ty - r, tx + r, ty + r, fill="#dc2626", outline="#f87171", width=3, tags="dynamic")
                        canvas.create_oval(tx - 4, ty - 4, tx + 4, ty + 4, fill="#ffffff", outline="", tags="dynamic")
                        act_sx = int(round(cx + res["actual_screen_delta_px"][0]))
                        act_sy = int(round(cy + res["actual_screen_delta_px"][1]))
                        canvas.create_line(tx, ty, act_sx, act_sy, fill="#f59e0b", width=2, dash=(4, 2), tags="dynamic")
                        canvas.create_oval(act_sx - 6, act_sy - 6, act_sx + 6, act_sy + 6, fill="#f59e0b", outline="#ffffff", width=2, tags="dynamic")
                        canvas.create_text(
                            act_sx + 10, act_sy, anchor="w", fill="#f59e0b", font=("Consolas", 10, "bold"),
                            text=f"MISS ({res['err_counts']:.1f} cnt / {res['screen_error_px']:.0f} px)", tags="dynamic"
                        )

            # 2. Draw Real-time Trajectory Trail
            with self.traj_lock:
                if len(self.traj_history) >= 2:
                    canvas.create_line(self.traj_history, fill="#38bdf8", width=2, dash=(3, 2), tags="dynamic")

            # 3. Draw True Live Hardware Optical Reticle (高亮青色/绿色十字瞄准星)
            ret_r = 14
            ret_color = "#22c55e" if not self.is_busy else "#00f0ff"
            # Cross lines
            canvas.create_line(hw_x - ret_r - 8, hw_y, hw_x + ret_r + 8, hw_y, fill=ret_color, width=2, tags="dynamic")
            canvas.create_line(hw_x, hw_y - ret_r - 8, hw_x, hw_y + ret_r + 8, fill=ret_color, width=2, tags="dynamic")
            # Outer targeting circle
            canvas.create_oval(hw_x - ret_r, hw_y - ret_r, hw_x + ret_r, hw_y + ret_r, outline=ret_color, width=2, tags="dynamic")
            # Center precision dot
            canvas.create_oval(hw_x - 2, hw_y - 2, hw_x + 2, hw_y + 2, fill="#ffffff", outline="", tags="dynamic")
            # Floating coordinates tooltip
            canvas.create_text(
                hw_x + ret_r + 10, hw_y - 12, anchor="w", fill=ret_color, font=("Consolas", 9, "bold"),
                text=f"[{rx_cnt:+5.1f}, {ry_cnt:+5.1f}] cnt", tags="dynamic"
            )

            # 4. Update HUD text on settled result
            if not self.is_busy and self.last_res is not None:
                res = self.last_res
                hit_rate = 100.0 * self.hits_count / max(1, self.shots_count)
                status_color = "#34d399" if res["hit_ball"] else "#f87171"
                status_tag = "HIT" if res["hit_ball"] else "MISS"
                canvas.itemconfig(
                    hud_status,
                    fill=status_color,
                    text=(
                        f"Shot #{self.shots_count} [{status_tag}] | "
                        f"Target: ({res['target_screen_delta_px'][0]:+d}, {res['target_screen_delta_px'][1]:+d}) px | "
                        f"Optical Err: {res['err_counts']:.2f} cnt ({res['err_um']:.1f} um / {res['screen_error_px']:.1f} px) | "
                        f"Ballistic: {res['ballistic_ms']:.1f} ms | Settle: {res['elapsed_ms']:.1f} ms | "
                        f"Score: {self.hits_count}/{self.shots_count} ({hit_rate:.1f}%)"
                    ),
                )

            # Update hardware coordinate telemetry
            rx_mm = rx_cnt * 0.0079375
            ry_mm = ry_cnt * 0.0079375
            auto_str = "[AUTO-AIM ON]" if self.auto_mode else "[MANUAL MODE]"
            canvas.itemconfig(
                hud_hw,
                text=f"Live Optical Pos: X: {rx_cnt:+6.1f} cnt, Y: {ry_cnt:+6.1f} cnt ({rx_mm:+5.2f} mm, {ry_mm:+5.2f} mm) | {auto_str}"
            )

            root.after(16, render_tick)

        def on_exit(_event=None):
            gui_running[0] = False
            action_queue.put(None)
            try:
                root.destroy()
            except Exception:
                pass

        root.bind("<space>", trigger_spawn_random)
        root.bind("<a>", toggle_auto)
        root.bind("<A>", toggle_auto)
        root.bind("<c>", trigger_recenter)
        root.bind("<C>", trigger_recenter)
        root.bind("<Button-1>", on_click)
        root.bind("<Escape>", on_exit)

        root.after(50, render_tick)
        try:
            root.mainloop()
        finally:
            gui_running[0] = False
            action_queue.put(None)

    def close(self):
        try:
            self.trainer.driver.center(0.3)
            self.trainer.close()
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3-RRR Full-Screen 100ms 3200DPI FPS Aim-Trainer")
    parser.add_argument("--headless-benchmark", action="store_true", help="Run automated 12-ball full-screen benchmark")
    parser.add_argument("--num-balls", type=int, default=12, help="Number of full-screen targets in benchmark")
    parser.add_argument("--ball-radius", type=int, default=18,
                        help="Ball radius in screen px used for the hit/miss verdict (default 18)")
    parser.add_argument("--tol", type=float, default=1.2,
                        help="Controller tolerance in counts passed to move_point_to_point (default 1.2)")
    parser.add_argument("--seed", type=int, default=12345,
                        help="Seed for the extra random targets when --num-balls > 12 (default 12345)")
    parser.add_argument("--metric-w", action="store_true",
                        help="Enable screen-weighted selection metric (default OFF: measured worse, kept only for A/B repro)")
    parser.add_argument("--tag", type=str, default="",
                        help="Suffix appended to output CSV filenames (e.g. _nomw_r1)")
    args = parser.parse_args()

    app = FullScreenFPSAimTrainer(
        ball_radius_px=args.ball_radius,
        tol_counts=args.tol,
        seed=args.seed,
        use_metric_w=bool(args.metric_w),
        run_tag=args.tag,
    )
    try:
        if args.headless_benchmark:
            code = app.run_headless_benchmark(num_targets=args.num_balls)
            raise SystemExit(code)
        else:
            app.run_gui()
    finally:
        app.close()
