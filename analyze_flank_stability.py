"""analyze_flank_stability.py -- 定案0 的后处理(零成本，不动机台)。

输入: experiments/flank_stability_*.csv (由 probe_flank_stability.py 产生)

回答三个问题，且每个都会被换算成一个可以直接和基线 5.15 cnt 比较的数:
  Q1 相位彩票有多大?  同一 offset、同一方向，两轮之间的位置差
        -> mean/max |pos_R2 - pos_R1|，这是"整张晶格在时间上跑了多少"
  Q2 迟滞是系统性的还是随机的?
        hyst(off) = pos_UP(off) - pos_DN(off)
        若 R1 与 R2 的 hyst 曲线形状一致(corr 高、逐点差小) => 系统性 => 并集可用
        若两轮 hyst 曲线互不相关              => 随机   => 并集无意义
  Q3 并集到底能把误差压到多少?
        可达集合 A = {实测位置}，对中段稠密目标计算 min_a |T - a|
        分别给出 A=UP 单格 / UP|DN 并集 / 两轮全部并集 的 mean/median/p95/max
        这个数的 2 倍就是"单次定位误差"的数量级估计。

用法:
  set PYTHONIOENCODING=utf-8 && chcp 65001 >nul
  "E:\\anaconda\\envs\\pytorch\\python.exe" analyze_flank_stability.py [csv路径]
"""

import csv
import glob
import os
import statistics
import sys
from collections import defaultdict

MID_LO, MID_HI = 0.20, 0.80   # 取行程中段，避开端点伪影
DENSE_STEP = 0.1              # 目标稠密采样步长(count)


def load(path):
    """-> {(repeat, axis, dir): {offset: pos}}"""
    table = defaultdict(dict)
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = (int(r["repeat_id"]), r["axis"], r["approach_dir"])
            idx = "pos_x_cnt" if r["axis"] == "X" else "pos_y_cnt"
            table[key][float(r["u_offset_us"])] = float(r[idx])
    return table


def _corr(a, b):
    if len(a) < 3:
        return None
    ma, mb = statistics.mean(a), statistics.mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = sum((x - ma) ** 2 for x in a) ** 0.5
    db = sum((y - mb) ** 2 for y in b) ** 0.5
    if da < 1e-9 or db < 1e-9:
        return None
    return num / (da * db)


def _min_dist_stats(achievable, lo, hi):
    """对 lo..hi 稠密目标，返回到最近可达点的距离统计。"""
    a = sorted(achievable)
    if len(a) < 2:
        return None
    n = int((hi - lo) / DENSE_STEP) + 1
    ds = []
    j = 0
    for k in range(n):
        t = lo + k * DENSE_STEP
        while j + 1 < len(a) and abs(a[j + 1] - t) <= abs(a[j] - t):
            j += 1
        ds.append(abs(t - a[j]))
    ds.sort()
    return {
        "n": len(ds),
        "mean": statistics.mean(ds),
        "median": statistics.median(ds),
        "p95": ds[int(0.95 * (len(ds) - 1))],
        "max": ds[-1],
    }


def _fmt_d(d):
    if d is None:
        return "n/a"
    return (
        f"mean={d['mean']:.2f} median={d['median']:.2f} "
        f"p95={d['p95']:.2f} max={d['max']:.2f}"
    )


def main(path=None):
    if path is None:
        cands = sorted(glob.glob("experiments/flank_stability_*.csv"))
        if not cands:
            print("[ERR] 找不到 experiments/flank_stability_*.csv")
            return 3
        path = cands[-1]
    print("=" * 96)
    print(f"[输入] {os.path.abspath(path)}")
    print("=" * 96)

    table = load(path)
    axes = sorted({k[1] for k in table})
    repeats = sorted({k[0] for k in table})

    # ------------------------------------------------------------ Q1 相位彩票
    print("\n=== Q1 相位彩票：同一 offset + 同一方向，两轮之间的位置差 (counts) ===")
    print("     （这就是『整张晶格在时间上跑了多少』，直接决定『标定一次永久用』是否可行）")
    print("-" * 96)
    phase = {}
    for axis in axes:
        for dname in ("UP", "DN"):
            t1 = table.get((1, axis, dname), {})
            t2 = table.get((2, axis, dname), {})
            common = sorted(set(t1) & set(t2))
            if len(common) < 3:
                continue
            diffs = [abs(t2[o] - t1[o]) for o in common]
            signed = [t2[o] - t1[o] for o in common]
            phase[(axis, dname)] = statistics.mean(diffs)
            print(
                f"{axis}/{dname:<2}  n={len(common):>2}  "
                f"|dpos| mean={statistics.mean(diffs):6.2f}  median={statistics.median(diffs):6.2f}  "
                f"max={max(diffs):6.2f}   signed mean={statistics.mean(signed):+7.2f} cnt"
            )
    print("-" * 96)
    if phase:
        worst = max(phase.values())
        print(
            f"相位漂移最大 = {worst:.2f} cnt  "
            f"(对比: 单轮 25 点网格 mean 误差 5.15 cnt / 单点 RMS 8.7 cnt)"
        )

    # -------- Q1b: 把"恒定平移"和"散布"分开（这才是晶格相位噪声的真值） --------
    print("\n=== Q1b 分解：恒定平移(可靠重新定基准消掉) vs 残余散布(不可消) ===")
    print("     残余散布 = dpos 去掉其自身均值后的 mean|.|；只有它才决定晶格能否复用")
    print("-" * 96)
    resid = {}
    for axis in axes:
        for dname in ("UP", "DN"):
            t1 = table.get((1, axis, dname), {})
            t2 = table.get((2, axis, dname), {})
            common = sorted(set(t1) & set(t2))
            if len(common) < 3:
                continue
            signed = [t2[o] - t1[o] for o in common]
            mu = statistics.mean(signed)
            r = [abs(s - mu) for s in signed]
            resid[(axis, dname)] = statistics.mean(r)
            print(
                f"{axis}/{dname:<2}  平移={mu:+7.2f} cnt   "
                f"残余散布 mean={statistics.mean(r):5.2f}  median={statistics.median(r):5.2f}  "
                f"max={max(r):5.2f} cnt"
            )
    print("-" * 96)
    if resid:
        print(
            f"残余散布最大 = {max(resid.values()):.2f} cnt   "
            f"(若 <=2 cnt 则晶格本身稳定，问题只在定基准)"
        )
        # 最差 offset 定位（只在最坏那一条 pass 上做，避免刷屏）
        worst_key = max(resid, key=resid.get)
        axis, dname = worst_key
        t1 = table[(1, axis, dname)]
        t2 = table[(2, axis, dname)]
        common = sorted(set(t1) & set(t2))
        signed = [t2[o] - t1[o] for o in common]
        mu = statistics.mean(signed)
        pairs = sorted(zip(common, signed), key=lambda kv: -abs(kv[1] - mu))[:8]
        print(f"    {axis}/{dname} 最差 8 个 offset (offset_us, dpos-mu):")
        for o, s in pairs:
            print(f"      {o:+6.0f} us   {s - mu:+7.2f} cnt")

    # ------------------------------------------------------------ Q2 迟滞系统性
    print("\n=== Q2 迟滞 hyst(off)=pos_UP-pos_DN 是系统性的还是随机的? ===")
    print("     判据: R1/R2 两条 hyst 曲线 corr>0.6 且逐点差 <3 cnt => 系统性(并集可用)")
    print("-" * 96)
    hyst_info = {}
    for axis in axes:
        curves = {}
        for r in repeats:
            up = table.get((r, axis, "UP"), {})
            dn = table.get((r, axis, "DN"), {})
            common = sorted(set(up) & set(dn))
            if len(common) < 3:
                continue
            curves[r] = [up[o] - dn[o] for o in common]
        if len(curves) < 2:
            print(f"{axis}: 轮次不足，跳过")
            continue
        rs = sorted(curves)
        a, b = curves[rs[0]], curves[rs[1]]
        m = min(len(a), len(b))
        a, b = a[:m], b[:m]
        corr = _corr(a, b)
        mismatch = [abs(x - y) for x, y in zip(a, b)]
        hyst_info[axis] = (corr, statistics.mean(mismatch))
        print(
            f"{axis}: corr={('n/a' if corr is None else f'{corr:+.3f}')}  "
            f"|hyst_R1-hyst_R2| mean={statistics.mean(mismatch):.2f} max={max(mismatch):.2f} cnt  "
            f"hyst_R1 mean={statistics.mean(a):+.2f}  hyst_R2 mean={statistics.mean(b):+.2f}"
        )
    print("-" * 96)

    # ------------------------------------------------------------ Q3 并集收益
    print("\n=== Q3 并集能压到多少? 对中段稠密目标的最小可达距离 (counts) ===")
    print("     （2x 该值 ~ 单次命中误差量级；基线实测 25 点网格 mean=5.15 cnt）")
    print("-" * 96)
    proj = {}
    for axis in axes:
        axis_pos = []
        for r in repeats:
            for dname in ("UP", "DN"):
                axis_pos += list(table.get((r, axis, dname), {}).values())
        if len(axis_pos) < 5:
            continue
        pmin, pmax = min(axis_pos), max(axis_pos)
        span = pmax - pmin
        if span < 10:
            print(f"{axis}: 行程 {span:.1f} cnt 过小，跳过")
            continue
        lo = pmin + MID_LO * span
        hi = pmax - (1.0 - MID_HI) * span

        sets = {
            "UP单一格": table.get((1, axis, "UP"), {}).values(),
            "UP|DN 并集(R1)": list(table.get((1, axis, "UP"), {}).values())
            + list(table.get((1, axis, "DN"), {}).values()),
            "四遍全并集(R1+R2)": [
                v
                for r in repeats
                for d in ("UP", "DN")
                for v in table.get((r, axis, d), {}).values()
            ],
        }
        print(f"  [{axis}] 中段范围 [{lo:.1f}, {hi:.1f}] cnt  (span={span:.1f})")
        for name, vals in sets.items():
            acc = [v for v in vals if lo <= v <= hi]
            acc = sorted({round(v, 1) for v in acc})
            st = _min_dist_stats(acc, lo, hi)
            proj[(axis, name)] = st
            print(f"      {name:<18} 可达点数={len(acc):>3}  {_fmt_d(st)}")

    # ------------------------------------------------------------ 总结
    print("\n" + "=" * 96)
    print("=== 判读 ===")
    ok_q1 = phase and max(phase.values()) <= 3.0
    ok_q2 = all(
        (c is not None and c > 0.6 and m < 3.0) for (c, m) in hyst_info.values()
    ) if hyst_info else False

    print(
        f"Q1 相位稳定(<=3 cnt): {'PASS' if ok_q1 else 'FAIL'}   "
        f"Q2 迟滞系统性: {'PASS' if ok_q2 else 'FAIL'}"
    )
    for axis in axes:
        s1 = proj.get((axis, "UP单一格"))
        s2 = proj.get((axis, "UP|DN 并集(R1)"))
        if s1 and s2:
            gain = s1["mean"] - s2["mean"]
            print(
                f"  {axis}: 并集把平均最小可达距离从 {s1['mean']:.2f} 降到 {s2['mean']:.2f} cnt "
                f"(收益 {gain:+.2f} cnt)"
            )
    if ok_q1 and ok_q2:
        print("[结论] 相位稳定 + 迟滞系统性 => 【路1 双向并集晶格】成立，可以免费实现四象限选优")
    elif ok_q2 and not ok_q1:
        print("[结论] 迟滞是系统性的但相位会跑 => 静态晶格表会过期，但【同一次 visit 内自参照】仍可用")
    else:
        print("[结论] 迟滞不系统性 => 并集只是随机散布，没有可学习结构 => 路1 不成立")
    print("=" * 96)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else None))
