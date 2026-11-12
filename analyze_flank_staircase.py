"""analyze_flank_staircase.py — 把 flank_probe.csv 的阶梯自动切分成平台段。
用法：
    python analyze_flank_staircase.py experiments/flank_probe_3preload.csv
    python analyze_flank_staircase.py experiments/flank_probe_1band.csv
"""
import csv
import sys
from collections import OrderedDict

TOL = 0.51        # 判定"读数未变"的阈值（counts）
BIG_JUMP = 5.0    # 判定"大跳变"的阈值（counts）
CORE_W = 6        # 判定"内部平台"的最小宽度（us），用于排除端点碎片


def load(path):
    groups = OrderedDict()
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            k = (int(r["run_id"]), r["axis"])
            groups.setdefault(k, []).append(
                (int(r["u_offset_us"]), float(r["pos_x_cnt"]), float(r["pos_y_cnt"]))
            )
    for k in groups:
        groups[k].sort(key=lambda t: t[0])
    return groups


def segments(recs, axis):
    """返回 [(offset_start, offset_end, value)]，宽度 = end - start。"""
    idx = 1 if axis == "X" else 2
    segs = []
    start = recs[0][0]
    val = recs[0][idx]
    prev = recs[0][0]
    for off, px, py in recs[1:]:
        v = px if axis == "X" else py
        if abs(v - val) < TOL:
            prev = off
        else:
            segs.append((start, prev, val))
            start, val, prev = off, v, off
    segs.append((start, prev, val))
    return segs


def main(path):
    groups = load(path)
    print("=" * 96)
    print(f"[文件] {path}")
    print("=" * 96)

    widths_all = {"X": [], "Y": []}
    jumps_all = {"X": [], "Y": []}

    for (run_id, axis), recs in groups.items():
        segs = segments(recs, axis)
        o0, o1 = recs[0][0], recs[-1][0]
        travel = recs[-1][1 if axis == "X" else 2] - recs[0][1 if axis == "X" else 2]
        gain = travel / float(o1 - o0) if o1 != o0 else 0.0

        print(f"\n--- Run {run_id}  axis={axis}  ---")
        print(f"    span={o0}..{o1} us   travel={travel:+.1f} cnt   gain={gain:.4f} cnt/us")
        print(f"    {'seg':<4}{'offset range':<16}{'width(us)':<12}{'pos(cnt)':<12}")
        for i, (a, b, v) in enumerate(segs, 1):
            print(f"    {i:<4}{f'{a}..{b}':<16}{b - a:<12}{v:<12.1f}")

        jumps = []
        for i in range(len(segs) - 1):
            j = segs[i + 1][2] - segs[i][2]
            jumps.append(j)
            if abs(j) >= BIG_JUMP:
                jumps_all[axis].append(abs(j))
        if jumps:
            print("    jumps(cnt): " + ", ".join(f"{j:+.1f}" for j in jumps))
        for a, b, v in segs:
            if (b - a) >= CORE_W:
                widths_all[axis].append(b - a)

    def _stat(v):
        if not v:
            return "N/A"
        v = sorted(v)
        med = v[len(v) // 2] if len(v) % 2 else 0.5 * (v[len(v) // 2 - 1] + v[len(v) // 2])
        return f"n={len(v)}  median={med:.1f}  max={max(v):.1f}  min={min(v):.1f}"

    print("\n" + "=" * 96)
    print("[汇总]  内部平台宽度(>= {} us)   X: {}".format(CORE_W, _stat(widths_all["X"])))
    print("                              Y: {}".format(_stat(widths_all["Y"])))
    print("[汇总]  大跳变幅度(>= {} cnt)    X: {}".format(BIG_JUMP, _stat(jumps_all["X"])))
    print("                              Y: {}".format(_stat(jumps_all["Y"])))
    print("=" * 96)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "experiments/flank_probe.csv")

