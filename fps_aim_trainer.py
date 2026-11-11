r"""
fps_aim_trainer.py — Full-Screen (1536x864) 100ms 3200-DPI FPS Aim-Lab Visual Target System
===========================================================================================
Powered by the PyTorch Physics-Informed Neural Ballistic Manifold (PI-NBM-Net) &
Joint-Space Inverse Play Compensator (`rect_manifold_calibrator.py`).

Features:
1. Full-Screen 1536x864 Visual Aim-Trainer GUI (`python fps_aim_trainer.py`):
   - Spawns glowing target balls at random locations across the ENTIRE screen (including corners & edges).
   - Auto-Continuous Mode (`[A]` key) or Single Random Spawn (`[SPACE]`) or Click-Anywhere (`[Left Click]`).
   - Physically actuates the 3-RRR parallel robot over COM4 from the current position to the new ball
     (true continuous point-to-point without resetting to center), displaying the live 3200 DPI optical
     trajectory, ballistic flight time, and micrometric accuracy in real time.
2. Headless Full-Screen Benchmark (`python fps_aim_trainer.py --headless-benchmark`):
   - Executes 12 full-screen edge-to-edge target transitions across all 4 quadrants and screen borders,
     saving reproducible results to `experiments/fps_fullscreen_aim_benchmark.csv`.
"""

import argparse
import csv
import ctypes
import os
import random
import time
import tkinter as tk
from typing import Dict, List, Tuple
import numpy as np

from rect_manifold_calibrator import HardwareRectangleTrainer

user32 = ctypes.windll.user32


class FullScreenFPSAimTrainer:
    def __init__(self, ball_radius_px: int = 16):
        self.trainer = HardwareRectangleTrainer()
        self.trainer.sensor.start()
        self.trainer.move_to_rect_origin(settle_s=0.45)

        self.sw = user32.GetSystemMetrics(0) or 1536
        self.sh = user32.GetSystemMetrics(1) or 864
        self.cx = self.sw // 2
        self.cy = self.sh // 2

        # Map full usable rectangle [-0.92*W, +0.92*W] x [-0.92*H, +0.92*H] to full screen [-700, +700] x [-390, +390] px
        self.max_rect_x = 0.92 * self.trainer.manifold.W
        self.max_rect_y = 0.92 * self.trainer.manifold.H
        self.scale_x = (0.46 * self.sw) / self.max_rect_x
        self.scale_y = (0.45 * self.sh) / self.max_rect_y
        self.ball_radius_px = int(ball_radius_px)

    def screen_to_rect(self, sx_rel_px: float, sy_rel_px: float) -> Tuple[float, float]:
        rx = float(np.clip(sx_rel_px / self.scale_x, -self.max_rect_x, self.max_rect_x))
        ry = float(np.clip(sy_rel_px / self.scale_y, -self.max_rect_y, self.max_rect_y))
        return rx, ry

    def rect_to_screen(self, rx_cnt: float, ry_cnt: float) -> Tuple[float, float]:
        return float(rx_cnt * self.scale_x), float(ry_cnt * self.scale_y)

    def shoot_screen_target(self, screen_dx_px: int, screen_dy_px: int) -> Dict:
        """
        Moves continuously from current position to the full-screen target `(screen_dx_px, screen_dy_px)`
        using the PyTorch PI-NBM-Net + 100ms Ballistic Predictive Brake + Inverse-Play Trim.
        """
        rx_tgt, ry_tgt = self.screen_to_rect(screen_dx_px, screen_dy_px)
        res = self.trainer.move_point_to_point(rx_tgt, ry_tgt, tol_counts=2.0)

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
            f"  Rectangle Envelope: [-{self.max_rect_x:.0f}, +{self.max_rect_x:.0f}] x [-{self.max_rect_y:.0f}, +{self.max_rect_y:.0f}] cnt "
            f"| Scale: ({self.scale_x:.2f}x, {self.scale_y:.2f}x) | Ball Radius: {self.ball_radius_px} px"
        )
        print("=" * 92)

        # Full-screen targets covering all 4 screen corners, edges, and interior quadrants
        screen_targets: List[Tuple[int, int]] = [
            (-680, -370),  # Top-Left Screen Corner
            (680, -370),   # Top-Right Screen Corner
            (680, 370),    # Bottom-Right Screen Corner
            (-680, 370),   # Bottom-Left Screen Corner
            (0, -380),     # Top Screen Edge Center
            (700, 0),      # Right Screen Edge Center
            (0, 380),      # Bottom Screen Edge Center
            (-700, 0),     # Left Screen Edge Center
            (460, -240),   # Upper-Right Quadrant
            (-480, 250),   # Lower-Left Quadrant
            (-360, -210),  # Upper-Left Quadrant
            (0, 0),        # Return to Screen Center
        ][:num_targets]

        os.makedirs("experiments", exist_ok=True)
        csv_path = "experiments/fps_fullscreen_aim_benchmark.csv"
        rows = []
        hits = 0
        raw_errs = []
        ballistic_times = []

        for idx, (sx, sy) in enumerate(screen_targets, 1):
            res = self.shoot_screen_target(sx, sy)
            hit_str = "HIT " if res["hit_ball"] else "MISS"
            if res["hit_ball"]:
                hits += 1
            raw_errs.append(res["err_counts"])
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
        print("-" * 92)
        print(
            f"  [SUMMARY] Hit Rate: {hits}/{len(screen_targets)} ({hit_rate:.1f}%) | "
            f"Mean Optical Error: {mean_raw_err:.2f} cnt ({mean_raw_err*7.9375:.1f} um) | "
            f"Max Error: {max_raw_err:.2f} cnt | Mean Ballistic Time: {mean_bal_ms:.1f} ms"
        )
        print(f"  [SAVED]   {csv_path}")
        print("=" * 92)
        return 0 if (hits == len(screen_targets) and mean_raw_err <= 2.5) else 1

    def run_gui(self):
        root = tk.Tk()
        root.title("3-RRR 100ms 3200DPI Full-Screen PI-NBM-Net Aim-Trainer")
        root.attributes("-fullscreen", True)
        root.configure(bg="#0b0f19")

        sw, sh = self.sw, self.sh
        cx, cy = self.cx, self.cy

        canvas = tk.Canvas(root, width=sw, height=sh, bg="#0b0f19", highlightthickness=0)
        canvas.pack(fill="both", expand=True)

        # Draw full-screen boundary frame & grid
        bx0 = cx - int(self.max_rect_x * self.scale_x)
        by0 = cy - int(self.max_rect_y * self.scale_y)
        bx1 = cx + int(self.max_rect_x * self.scale_x)
        by1 = cy + int(self.max_rect_y * self.scale_y)

        for gx in range(0, sw, 120):
            canvas.create_line(gx, 0, gx, sh, fill="#162032", width=1)
        for gy in range(0, sh, 120):
            canvas.create_line(0, gy, sw, gy, fill="#162032", width=1)

        canvas.create_rectangle(bx0, by0, bx1, by1, outline="#1e3a8a", width=2, dash=(6, 4))
        canvas.create_line(cx - 25, cy, cx + 25, cy, fill="#334155", width=2)
        canvas.create_line(cx, cy - 25, cx, cy + 25, fill="#334155", width=2)

        canvas.create_text(
            24,
            20,
            anchor="nw",
            fill="#38bdf8",
            font=("Consolas", 14, "bold"),
            text=(
                "3-RRR 3200DPI FULL-SCREEN PI-NBM-NET AIM-TRAINER (CONTINUOUS POINT-TO-POINT)\n"
                "[SPACE] Spawn Random Ball  |  [A] Toggle Continuous Auto-Aim  |  [LEFT CLICK] Spawn Ball Here  |  [ESC] Exit"
            ),
        )
        stats_text = canvas.create_text(
            24, 72, anchor="nw", fill="#a7f3d0", font=("Consolas", 13), text="Ready. Press [SPACE] or [A] to start full-screen aiming..."
        )

        state = {"hits": 0, "shots": 0, "busy": False, "auto": False, "cur_screen": (cx, cy)}

        def fire_at_screen_pos(target_x: int, target_y: int):
            if state["busy"]:
                return
            state["busy"] = True
            canvas.delete("dynamic")

            r = self.ball_radius_px
            start_sx, start_sy = state["cur_screen"]
            canvas.create_oval(
                target_x - r, target_y - r, target_x + r, target_y + r,
                fill="#ef4444", outline="#fca5a5", width=3, tags="dynamic"
            )
            root.update()

            dx_px = target_x - cx
            dy_px = target_y - cy
            res = self.shoot_screen_target(dx_px, dy_px)

            act_x = int(round(cx + res["actual_screen_delta_px"][0]))
            act_y = int(round(cy + res["actual_screen_delta_px"][1]))
            state["cur_screen"] = (act_x, act_y)

            # Sync visible OS cursor to the physical optical position on screen
            try:
                user32.SetCursorPos(act_x, act_y)
            except Exception:
                pass

            state["shots"] += 1
            if res["hit_ball"]:
                state["hits"] += 1

            color = "#22c55e" if res["hit_ball"] else "#f59e0b"
            canvas.create_line(start_sx, start_sy, act_x, act_y, fill="#38bdf8", width=2, dash=(4, 2), tags="dynamic")
            canvas.create_oval(act_x - 6, act_y - 6, act_x + 6, act_y + 6, fill=color, outline="#ffffff", width=2, tags="dynamic")

            hit_rate = 100.0 * state["hits"] / max(1, state["shots"])
            canvas.itemconfig(
                stats_text,
                text=(
                    f"Shot #{state['shots']} [{'HIT' if res['hit_ball'] else 'NEAR'}] | "
                    f"Target: ({dx_px:+d}, {dy_px:+d}) px | "
                    f"Optical Err: {res['err_counts']:.2f} cnt ({res['err_um']:.1f} um) | "
                    f"Ballistic: {res['ballistic_ms']:.1f} ms | "
                    f"Hit Rate: {state['hits']}/{state['shots']} ({hit_rate:.1f}%)"
                ),
            )
            root.update()
            state["busy"] = False

            if state["auto"]:
                root.after(180, spawn_random)

        def spawn_random(_event=None):
            tx = random.randint(bx0 + 20, bx1 - 20)
            ty = random.randint(by0 + 20, by1 - 20)
            fire_at_screen_pos(tx, ty)

        def toggle_auto(_event=None):
            state["auto"] = not state["auto"]
            if state["auto"] and not state["busy"]:
                spawn_random()

        def on_click(event):
            state["auto"] = False
            tx = int(np.clip(event.x, bx0 + 10, bx1 - 10))
            ty = int(np.clip(event.y, by0 + 10, by1 - 10))
            fire_at_screen_pos(tx, ty)

        root.bind("<space>", spawn_random)
        root.bind("<a>", toggle_auto)
        root.bind("<A>", toggle_auto)
        root.bind("<Button-1>", on_click)
        root.bind("<Escape>", lambda _e: root.destroy())
        root.mainloop()

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
    args = parser.parse_args()

    app = FullScreenFPSAimTrainer()
    try:
        if args.headless_benchmark:
            code = app.run_headless_benchmark(num_targets=args.num_balls)
            raise SystemExit(code)
        else:
            app.run_gui()
    finally:
        app.close()
