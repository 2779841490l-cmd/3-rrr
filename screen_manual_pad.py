r"""
screen_manual_pad.py — 3-RRR 全屏幕边界矩形巡检与键盘手动控制台
===================================================================
功能：
1. 【走屏幕边界矩形】：光标与机械臂全自动沿着显示器四条物理边缘（上->右->下->左）流畅走完完整大矩形；
2. 【键盘手动控制】：支持键盘方向键 [↑] [↓] [←] [→] 或 [W] [A] [S] [D] 实时控制光标在全屏幕任意移动；
3. 【实时数字看板】：显示屏幕坐标 (X, Y) px、流形坐标 (rx, ry) counts、三轴舵机脉宽 (ux, uy, 3000-uy) us；
4. 【多档步长调节】：5px（微调）、20px（常规）、50px（快速）、100px（大步跨越）。
"""

import ctypes
import os
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk
import numpy as np

from rect_manifold_calibrator import RectangularManifoldController
from servo_driver import WS63ServoDriver

user32 = ctypes.windll.user32


class ScreenManualPadApp:
    def __init__(self):
        # 1. Screen resolution & geometric center
        self.sw = user32.GetSystemMetrics(0) or 1536
        self.sh = user32.GetSystemMetrics(1) or 864
        self.cx = self.sw // 2
        self.cy = self.sh // 2

        # Safe screen boundary margins (leaving 32px from physical screen edge)
        self.margin_x = 32
        self.margin_y = 24
        self.min_sx = self.margin_x
        self.max_sx = self.sw - self.margin_x
        self.min_sy = self.margin_y
        self.max_sy = self.sh - self.margin_y

        # 2. Hardware Controller & Drivers
        self.manifold = RectangularManifoldController()
        self.driver = WS63ServoDriver()
        self.driver.set_pwm_frequency(200)

        # Scale factor between screen pixels and 3-RRR rectangle manifold counts
        self.max_rx = 0.92 * self.manifold.W
        self.max_ry = 0.92 * self.manifold.H
        self.scale_x = (self.max_sx - self.cx) / self.max_rx
        self.scale_y = (self.max_sy - self.cy) / self.max_ry

        # Current screen state
        self.cur_sx = self.cx
        self.cur_sy = self.cy
        self.step_px = 20
        self.is_patrolling = False
        self.patrol_loop = False
        self.lock = threading.Lock()

        # Initialize to center
        self.move_to_screen_xy(self.cx, self.cy)

        # 3. GUI Construction
        self.root = tk.Tk()
        self.root.title("3-RRR 全屏幕边界矩形与手动控制台")
        self.root.geometry("460x560+1020+220")
        self.root.attributes("-topmost", True)
        self.root.configure(bg="#0b1120")
        self.root.resizable(False, False)

        self._build_ui()
        self._bind_keys()

    def _build_ui(self):
        pad_bg = "#0b1120"
        card_bg = "#1e293b"
        accent_blue = "#38bdf8"
        text_white = "#f8fafc"
        text_gray = "#94a3b8"

        # Header Title
        title_frame = tk.Frame(self.root, bg=pad_bg)
        title_frame.pack(fill="x", padx=16, pady=(12, 6))

        tk.Label(
            title_frame,
            text="3-RRR 100ms 3200DPI 屏幕控制台",
            font=("Microsoft YaHei UI", 13, "bold"),
            fg=accent_blue,
            bg=pad_bg,
        ).pack(anchor="w")

        tk.Label(
            title_frame,
            text=f"屏幕分辨率: {self.sw}x{self.sh} | 边界范围: [{self.min_sx}, {self.max_sx}]x[{self.min_sy}, {self.max_sy}]",
            font=("Consolas", 9),
            fg=text_gray,
            bg=pad_bg,
        ).pack(anchor="w")

        # Dashboard Card
        dash_frame = tk.Frame(self.root, bg=card_bg, highlightbackground="#334155", highlightthickness=1)
        dash_frame.pack(fill="x", padx=16, pady=6)

        self.lbl_screen_pos = tk.Label(
            dash_frame,
            text=f"屏幕光标: ({self.cur_sx:4d}, {self.cur_sy:4d}) px",
            font=("Consolas", 12, "bold"),
            fg="#22c55e",
            bg=card_bg,
        )
        self.lbl_screen_pos.pack(anchor="w", padx=12, pady=(8, 2))

        self.lbl_robot_pos = tk.Label(
            dash_frame,
            text="机械坐标: (  +0.0,   +0.0) cnt",
            font=("Consolas", 10),
            fg="#cbd5e1",
            bg=card_bg,
        )
        self.lbl_robot_pos.pack(anchor="w", padx=12, pady=1)

        self.lbl_pulse_pos = tk.Label(
            dash_frame,
            text="舵机脉宽: S1=1500.0 | S2=1466.0 | S3=1534.0 us",
            font=("Consolas", 9),
            fg="#94a3b8",
            bg=card_bg,
        )
        self.lbl_pulse_pos.pack(anchor="w", padx=12, pady=(1, 8))

        # Main Action Buttons (Walk Rectangle & Center)
        btn_frame = tk.Frame(self.root, bg=pad_bg)
        btn_frame.pack(fill="x", padx=16, pady=8)

        self.btn_walk_rect = tk.Button(
            btn_frame,
            text="🚀 立即走屏幕边界矩形 (B)",
            font=("Microsoft YaHei UI", 11, "bold"),
            fg="#ffffff",
            bg="#2563eb",
            activebackground="#1d4ed8",
            activeforeground="#ffffff",
            relief="flat",
            padx=12,
            pady=8,
            cursor="hand2",
            command=self.start_boundary_patrol,
        )
        self.btn_walk_rect.pack(fill="x", pady=3)

        ctrl_row = tk.Frame(btn_frame, bg=pad_bg)
        ctrl_row.pack(fill="x", pady=3)

        self.btn_center = tk.Button(
            ctrl_row,
            text="🎯 归中复位 (C)",
            font=("Microsoft YaHei UI", 10, "bold"),
            fg="#ffffff",
            bg="#475569",
            activebackground="#334155",
            relief="flat",
            cursor="hand2",
            command=self.center_cursor,
        )
        self.btn_center.pack(side="left", expand=True, fill="x", padx=(0, 4))

        self.loop_var = tk.BooleanVar(value=False)
        self.chk_loop = tk.Checkbutton(
            ctrl_row,
            text="循环巡屏",
            font=("Microsoft YaHei UI", 9),
            fg="#94a3b8",
            bg=pad_bg,
            selectcolor="#1e293b",
            activebackground=pad_bg,
            activeforeground="#38bdf8",
            variable=self.loop_var,
        )
        self.chk_loop.pack(side="right", padx=(4, 0))

        # Step Size Selector
        step_frame = tk.LabelFrame(
            self.root,
            text=" 步长微调 (键盘数字键 1~4) ",
            font=("Microsoft YaHei UI", 9),
            fg=text_gray,
            bg=pad_bg,
            labelanchor="nw",
        )
        step_frame.pack(fill="x", padx=16, pady=4)

        self.step_var = tk.IntVar(value=20)
        steps = [("微调 5px (1)", 5), ("常规 20px (2)", 20), ("快速 50px (3)", 50), ("大步 100px (4)", 100)]
        for label, val in steps:
            rb = tk.Radiobutton(
                step_frame,
                text=label,
                value=val,
                variable=self.step_var,
                font=("Microsoft YaHei UI", 9),
                fg=text_white,
                bg=pad_bg,
                selectcolor="#1e293b",
                activebackground=pad_bg,
                activeforeground=accent_blue,
                command=self._on_step_change,
            )
            rb.pack(side="left", expand=True, pady=4)

        # Directional D-Pad (Virtual Buttons)
        dpad_frame = tk.Frame(self.root, bg=pad_bg)
        dpad_frame.pack(pady=8)

        btn_style = {
            "font": ("Consolas", 11, "bold"),
            "width": 6,
            "height": 1,
            "bg": "#334155",
            "fg": "#ffffff",
            "activebackground": "#475569",
            "relief": "flat",
            "cursor": "hand2",
        }

        tk.Button(dpad_frame, text="▲ 上 (W)", **btn_style, command=lambda: self.jog(0, -self.step_px)).grid(
            row=0, column=1, padx=4, pady=3
        )
        tk.Button(dpad_frame, text="◀ 左 (A)", **btn_style, command=lambda: self.jog(-self.step_px, 0)).grid(
            row=1, column=0, padx=4, pady=3
        )
        tk.Button(dpad_frame, text="● 中", **btn_style, bg="#1e293b", command=self.center_cursor).grid(
            row=1, column=1, padx=4, pady=3
        )
        tk.Button(dpad_frame, text="▶ 右 (D)", **btn_style, command=lambda: self.jog(self.step_px, 0)).grid(
            row=1, column=2, padx=4, pady=3
        )
        tk.Button(dpad_frame, text="▼ 下 (S)", **btn_style, command=lambda: self.jog(0, self.step_px)).grid(
            row=2, column=1, padx=4, pady=3
        )

        # Keyboard Cheat-Sheet Footer
        footer = tk.Label(
            self.root,
            text="提示: 焦点在此窗口时直接按方向键 / WASD 移动光标\n按住 Shift 移动速度加快 2.5 倍 | 按 [ESC] 退出",
            font=("Microsoft YaHei UI", 8),
            fg="#64748b",
            bg=pad_bg,
            justify="center",
        )
        footer.pack(side="bottom", pady=8)

    def _bind_keys(self):
        # Keyboard movement bindings
        self.root.bind("<Up>", lambda e: self._handle_key(0, -1, e))
        self.root.bind("<Down>", lambda e: self._handle_key(0, 1, e))
        self.root.bind("<Left>", lambda e: self._handle_key(-1, 0, e))
        self.root.bind("<Right>", lambda e: self._handle_key(1, 0, e))

        self.root.bind("<w>", lambda e: self._handle_key(0, -1, e))
        self.root.bind("<s>", lambda e: self._handle_key(0, 1, e))
        self.root.bind("<a>", lambda e: self._handle_key(-1, 0, e))
        self.root.bind("<d>", lambda e: self._handle_key(1, 0, e))
        self.root.bind("<W>", lambda e: self._handle_key(0, -1, e))
        self.root.bind("<S>", lambda e: self._handle_key(0, 1, e))
        self.root.bind("<A>", lambda e: self._handle_key(-1, 0, e))
        self.root.bind("<D>", lambda e: self._handle_key(1, 0, e))

        # Hotkeys
        self.root.bind("<b>", lambda e: self.start_boundary_patrol())
        self.root.bind("<B>", lambda e: self.start_boundary_patrol())
        self.root.bind("<c>", lambda e: self.center_cursor())
        self.root.bind("<C>", lambda e: self.center_cursor())

        self.root.bind("1", lambda e: self._set_step(5))
        self.root.bind("2", lambda e: self._set_step(20))
        self.root.bind("3", lambda e: self._set_step(50))
        self.root.bind("4", lambda e: self._set_step(100))

        self.root.bind("<Escape>", lambda e: self.close())
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def _set_step(self, val: int):
        self.step_var.set(val)
        self.step_px = val

    def _on_step_change(self):
        self.step_px = self.step_var.get()

    def _handle_key(self, dir_x: int, dir_y: int, event):
        if self.is_patrolling:
            self.is_patrolling = False
            return
        # Shift key speeds up by 2.5x
        multiplier = 2.5 if (event.state & 0x0001) != 0 else 1.0
        step = int(round(self.step_px * multiplier))
        self.jog(dir_x * step, dir_y * step)

    def jog(self, dx_px: int, dy_px: int):
        with self.lock:
            target_sx = int(np.clip(self.cur_sx + dx_px, self.min_sx, self.max_sx))
            target_sy = int(np.clip(self.cur_sy + dy_px, self.min_sy, self.max_sy))
            self.move_to_screen_xy(target_sx, target_sy)

    def center_cursor(self):
        self.is_patrolling = False
        with self.lock:
            self.move_to_screen_xy(self.cx, self.cy)

    def move_to_screen_xy(self, sx: int, sy: int) -> tuple:
        """Translates screen pixel (sx, sy) to 3-RRR pulse and commands servos."""
        self.cur_sx = int(sx)
        self.cur_sy = int(sy)

        # Map to rectangular manifold coordinates
        rx = (self.cur_sx - self.cx) / self.scale_x
        ry = (self.cur_sy - self.cy) / self.scale_y

        # Compute exact decoupled pulse
        ux, uy = self.manifold.evaluate_pulse(rx, ry)
        self.driver.send_raw_pulses(ux, uy)

        # Move visible Windows mouse cursor
        try:
            user32.SetCursorPos(self.cur_sx, self.cur_sy)
        except Exception:
            pass

        self._update_hud(rx, ry, ux, uy)
        return rx, ry, ux, uy

    def _update_hud(self, rx: float, ry: float, ux: float, uy: float):
        try:
            self.lbl_screen_pos.config(text=f"屏幕光标: ({self.cur_sx:4d}, {self.cur_sy:4d}) px")
            self.lbl_robot_pos.config(
                text=f"机械坐标: ({rx:+6.1f}, {ry:+6.1f}) cnt  ({rx*7.9375/1000:+4.2f}, {ry*7.9375/1000:+4.2f}) mm"
            )
            self.lbl_pulse_pos.config(
                text=f"舵机脉宽: S1_X={ux:6.1f} | S2_Y1={uy:6.1f} | S3_Y2={3000-uy:6.1f} us"
            )
        except Exception:
            pass

    def start_boundary_patrol(self):
        """Launches smooth continuous screen boundary tracing in a background thread."""
        if self.is_patrolling:
            self.is_patrolling = False
            self.btn_walk_rect.config(text="🚀 立即走屏幕边界矩形 (B)", bg="#2563eb")
            return

        self.is_patrolling = True
        self.btn_walk_rect.config(text="⏹ 停止走边界巡屏 (按任意键停止)", bg="#dc2626")
        threading.Thread(target=self._patrol_worker, daemon=True).start()

    def _patrol_worker(self):
        """Walks the full physical rectangle perimeter of the monitor screen."""
        min_x, max_x = self.min_sx, self.max_sx
        min_y, max_y = self.min_sy, self.max_sy

        # Density of waypoints along edges
        num_h = 24  # horizontal edge steps
        num_v = 16  # vertical edge steps

        def generate_path():
            path = []
            # 1. Top Edge: Left -> Right
            for x in np.linspace(min_x, max_x, num_h):
                path.append((int(x), min_y))
            # 2. Right Edge: Top -> Bottom
            for y in np.linspace(min_y, max_y, num_v):
                path.append((max_x, int(y)))
            # 3. Bottom Edge: Right -> Left
            for x in np.linspace(max_x, min_x, num_h):
                path.append((int(x), max_y))
            # 4. Left Edge: Bottom -> Top
            for y in np.linspace(max_y, min_y, num_v):
                path.append((min_x, int(y)))
            return path

        loop_count = 0
        while self.is_patrolling:
            loop_count += 1
            # Move to start corner smoothly
            with self.lock:
                self.move_to_screen_xy(min_x, min_y)
            time.sleep(0.20)

            for sx, sy in generate_path():
                if not self.is_patrolling:
                    break
                with self.lock:
                    self.move_to_screen_xy(sx, sy)
                time.sleep(0.040)

            if not self.loop_var.get() or not self.is_patrolling:
                break
            time.sleep(0.25)

        self.is_patrolling = False
        try:
            self.root.after(0, lambda: self.btn_walk_rect.config(text="🚀 立即走屏幕边界矩形 (B)", bg="#2563eb"))
        except Exception:
            pass

    def run(self):
        self.root.mainloop()

    def close(self):
        self.is_patrolling = False
        try:
            self.driver.center(0.3)
            self.driver.close()
        except Exception:
            pass
        try:
            self.root.destroy()
        except Exception:
            pass
        sys.exit(0)


if __name__ == "__main__":
    app = ScreenManualPadApp()
    app.run()
