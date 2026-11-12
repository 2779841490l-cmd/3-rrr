# -*- coding: utf-8 -*-
"""
analyze_aim_bias.py — 打靶落点误差的交叉统计（零实机成本，纯离线）

回答三个问题：
  1) 误差是"零均值随机散布"还是"系统性同号偏差"？（决定性：能否用软件补偿）
  2) 误差是否与"接近方向"相关？（X 负向接近是否过冲 -> 可修 backlash/早刹）
  3) 逐球跨场一致性：同一个目标在多次独立运行中是否重复同一个偏差？
     （若一致 => 是晶格节点确定性，不是噪声 => 可用残差网格吸收，或至少写进论文）

输入：experiments/fps_fullscreen_aim_benchmark*.csv
输出：experiments/analysis_aim_bias.txt

约定：dx = actual - target（counts，带符号；正值=落在目标右侧/下侧）
不修改任何标定文件，不碰硬件。
"""

import csv
import glob
import os
from collections import OrderedDict, defaultdict

import numpy as np

BENCH_GLOB = os.path.join("experiments", "fps_fullscreen_aim_benchmark*.csv")
OUT_TXT = os.path.join("experiments", "analysis_aim_bias.txt")


def load_runs():
    """读入所有 benchmark CSV，剔除固件挂死产生的死数据。"""
    runs = OrderedDict()
    for path in sorted(glob.glob(BENCH_GLOB)):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                try:
                    rows.append(
                        dict(
                            bid=int(r["ball_id"]),
                            tx=float(r["raw_target_dx_cnt"]),
                            ty=float(r["raw_target_dy_cnt"]),
                            ax=float(r["raw_actual_dx_cnt"]),
                            ay=float(r["raw_actual_dy_cnt"]),
                            hit=int(r["hit"]),
                            ms=float(r.get("ballistic_time_ms") or 0.0),
                        )
                    )
                except (KeyError, ValueError):
                    continue
        if not rows:
            continue
        # 死数据判据：绝大多数行 (actual == 0,0) 而 target != (0,0)
        n_zero = sum(1 for r in rows if r["ax"] == 0.0 and r["ay"] == 0.0
                     and (abs(r["tx"]) > 5 or abs(r["ty"]) > 5))
        dead = n_zero >= max(3, int(0.5 * len(rows)))
        name = os.path.basename(path).replace("fps_fullscreen_aim_benchmark", "").replace(".csv", "") or "_base"
        runs[name] = dict(rows=rows, dead=dead, n_zero=n_zero)
    return runs


def main():
    runs = load_runs()
    if not runs:
        print("[FATAL] 没有找到 benchmark CSV")
        return 1

    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    out("=" * 96)
    out("  FPS AIM-TRAINER 落点误差交叉统计（离线分析，不改硬件）")
    out("=" * 96)
    out("  场次清单 / 死活判据：")
    for name, rr in runs.items():
        tag = "DEAD(固件挂死,已剔除)" if rr["dead"] else "OK"
        hits = sum(r["hit"] for r in rr["rows"])
        out(f"    {name:<14} rows={len(rr['rows']):3d} hits={hits:3d} zero_rows={rr['n_zero']:3d}  -> {tag}")

    live = {k: v for k, v in runs.items() if not v["dead"]}
    if not live:
        out("\n[FATAL] 全部场次都是死数据，无法分析。")
        return 2

    # ---------- 1. 总体符号偏差 ----------
    out("\n" + "-" * 96)
    out("[1] 总体符号偏差（全部活场次合并）：dx=actual-target，counts")
    out("-" * 96)
    ex_all, ey_all, ehyp, hit_all, ms_all = [], [], [], [], []
    for name, rr in live.items():
        for r in rr["rows"]:
            if abs(r["tx"]) < 1e-9 and abs(r["ty"]) < 1e-9 and r["ax"] == 0.0 and r["ay"] == 0.0:
                continue  # 尚未移动过的空行
            ex = r["ax"] - r["tx"]
            ey = r["ay"] - r["ty"]
            ex_all.append(ex)
            ey_all.append(ey)
            ehyp.append(np.hypot(ex, ey))
            hit_all.append(r["hit"])
            ms_all.append(r["ms"])
    ex_a = np.array(ex_all)
    ey_a = np.array(ey_all)
    eh_a = np.array(ehyp)
    n = len(ex_a)
    out(f"  n = {n}  命中率 = {100.0*np.mean(hit_all):.1f}%   均弹道 = {np.mean(ms_all):.1f} ms")
    out(f"  dx: mean={ex_a.mean():+.2f} sd={ex_a.std(ddof=1):.2f}  median={np.median(ex_a):+.2f}  "
        f"se={ex_a.std(ddof=1)/np.sqrt(n):.2f}  t={ex_a.mean()/(ex_a.std(ddof=1)/np.sqrt(n)):+.2f}")
    out(f"  dy: mean={ey_a.mean():+.2f} sd={ey_a.std(ddof=1):.2f}  median={np.median(ey_a):+.2f}  "
        f"se={ey_a.std(ddof=1)/np.sqrt(n):.2f}  t={ey_a.mean()/(ey_a.std(ddof=1)/np.sqrt(n)):+.2f}")
    out(f"  |e| = {eh_a.mean():.2f} cnt ({eh_a.mean()*7.9375:.1f} um)   max = {eh_a.max():.2f} cnt "
        f"({eh_a.max()*7.9375:.1f} um)")
    out(f"  理论零均值下 E[hypot] = 1.2533*sqrt(sx^2+sy^2) = "
        f"{1.2533*np.sqrt(ex_a.var(ddof=1)+ey_a.var(ddof=1)):.2f} cnt（对照实测 {eh_a.mean():.2f}）")

    # ---------- 2. 按接近方向分桶 ----------
    out("\n" + "-" * 96)
    out("[2] 按接近方向分桶（approach = target - 上一发 actual）")
    out("    若某方向的 mean(dx) 明显异号/同号 => 早刹/齿隙方向不对称，可软件修")
    out("-" * 96)
    buckets = defaultdict(lambda: dict(ex=[], ey=[], eh=[]))
    for name, rr in live.items():
        prev = None
        for r in rr["rows"]:
            if prev is None:
                prev = (r["ax"], r["ay"])
                continue
            adx = r["tx"] - prev[0]
            ady = r["ty"] - prev[1]
            if abs(adx) < 3.0 and abs(ady) < 3.0:
                prev = (r["ax"], r["ay"])
                continue
            sx = "+" if adx > 3.0 else ("-" if adx < -3.0 else "0")
            sy = "+" if ady > 3.0 else ("-" if ady < -3.0 else "0")
            key = f"X{sx} Y{sy}"
            ex = r["ax"] - r["tx"]
            ey = r["ay"] - r["ty"]
            buckets[key]["ex"].append(ex)
            buckets[key]["ey"].append(ey)
            buckets[key]["eh"].append(np.hypot(ex, ey))
            prev = (r["ax"], r["ay"])

    out(f"  {'approach':<10} {'n':>4} {'mean_dx':>9} {'mean_dy':>9} {'mean|e|':>9}")
    for key in sorted(buckets.keys(), key=lambda k: -len(buckets[k]["ex"])):
        b = buckets[key]
        if len(b["ex"]) < 3:
            continue
        out(f"  {key:<10} {len(b['ex']):>4d} {np.mean(b['ex']):>+9.2f} {np.mean(b['ey']):>+9.2f} "
            f"{np.mean(b['eh']):>9.2f}")

    xp = [np.mean(buckets[k]["ex"]) for k in buckets if k.startswith("X+") and len(buckets[k]["ex"]) >= 3]
    xn = [np.mean(buckets[k]["ex"]) for k in buckets if k.startswith("X-") and len(buckets[k]["ex"]) >= 3]
    if xp and xn:
        out(f"\n  X 正向接近 mean(dx) 合计均值 = {np.mean(xp):+.2f} cnt   (n_dir={len(xp)})")
        out(f"  X 负向接近 mean(dx) 合计均值 = {np.mean(xn):+.2f} cnt   (n_dir={len(xn)})")
        out(f"  方向对称性差异 = {abs(np.mean(xp)-np.mean(xn)):.2f} cnt "
            f"（< 1 cnt 视为方向无关 => 早刹/齿隙不是主因）")

    # ---------- 3. 逐目标跨场一致性 ----------
    out("\n" + "-" * 96)
    out("[3] 逐目标跨场一致性（同一 target 在多次运行中是否重复同一偏差）")
    out("    一致 => 晶格节点确定性偏差（可补偿/可写论文）；不一致 => 随机散布")
    out("-" * 96)
    per_tgt = defaultdict(list)
    for name, rr in live.items():
        for r in rr["rows"]:
            per_tgt[(round(r["tx"], 1), round(r["ty"], 1))].append(
                (r["ax"] - r["tx"], r["ay"] - r["ty"], name)
            )
    out(f"  {'target(cnt)':<20} {'n':>3} {'mean_dx':>9} {'mean_dy':>9} {'sd_x':>7} {'同号':>6} {'mean|e|':>8}")
    rows_out = []
    for tgt, lst in per_tgt.items():
        if len(lst) < 2:
            continue
        exs = np.array([v[0] for v in lst])
        eys = np.array([v[1] for v in lst])
        same = 0
        if len(exs) >= 2:
            s = np.sign(exs)
            same = max(int(np.sum(s > 0)), int(np.sum(s < 0)))
        ehs = np.hypot(exs, eys)
        rows_out.append((np.mean(ehs), tgt, len(lst), exs.mean(), eys.mean(), exs.std(ddof=1), same, ehs.mean()))
    rows_out.sort(reverse=True)
    for eh_m, tgt, nn, mx, my, sx, same, eh2 in rows_out:
        out(f"  ({tgt[0]:+7.1f},{tgt[1]:+7.1f}) {nn:>3d} {mx:>+9.2f} {my:>+9.2f} {sx:>7.2f} "
            f"{same:>3d}/{nn:<2d} {eh2:>8.2f}")

    worst = [r for r in rows_out if r[2] >= 3]
    if worst:
        out(f"\n  跨场>=3 次的目标共 {len(worst)} 个，按平均|e|降序前 5：")
        for eh_m, tgt, nn, mx, my, sx, same, eh2 in worst[:5]:
            verdict = "确定性偏差" if same >= max(3, nn - 1) else "散布"
            out(f"    ({tgt[0]:+7.1f},{tgt[1]:+7.1f}) mean|e|={eh2:.2f} cnt  mean(dx)={mx:+.2f} "
                f"mean(dy)={my:+.2f}  同号 {same}/{nn}  => {verdict}")

    # ---------- 4. 头部与尾部 ----------
    out("\n" + "-" * 96)
    out("[4] 误差尾部（最差的 10% 决定 100% 命中半径）")
    out("-" * 96)
    q90 = float(np.percentile(eh_a, 90))
    q95 = float(np.percentile(eh_a, 95))
    out(f"  mean={eh_a.mean():.2f}  median={np.median(eh_a):.2f}  p75={np.percentile(eh_a,75):.2f}  "
        f"p90={q90:.2f}  p95={q95:.2f}  max={eh_a.max():.2f} cnt")
    x_only = np.abs(ex_a)
    y_only = np.abs(ey_a)
    out(f"  |dx|: mean={x_only.mean():.2f} p90={np.percentile(x_only,90):.2f} max={x_only.max():.2f}")
    out(f"  |dy|: mean={y_only.mean():.2f} p90={np.percentile(y_only,90):.2f} max={y_only.max():.2f}")
    out(f"  屏幕权重 2.0:1.4286 => X 1 cnt 等价于 {2.0/1.4286:.2f} cnt 的 Y 误差")
    out(f"  若把全部 |dx| 压到 0（理想补偿），mean|e| 将从 {eh_a.mean():.2f} 降到 "
        f"{y_only.mean():.2f} cnt —— 这是 X 轴残余的损失上界")

    txt = "\n".join(lines)
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    out(f"\n[SAVED] {OUT_TXT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
