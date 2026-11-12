"""test_sensor_noise_floor.py — 噪声地板三段测量（量具 + 平台 + 舵机保持抖动）。

Phase A  舵机未通电（limp）  : 传感器自身噪声 + 平台自由漂移
Phase B  舵机通电保持中立位    : **真实读数噪声地板**（含 MG996R 保持抖动）
Phase C  12 us 微推自校验      : 证明量具能读到非零，否则前面的 0 无意义

判据（写死，事后不改）：
  单次 110 ms 静置窗内抖动 < 0.5 cnt  ->  量具与保持抖动都不是瓶颈
  >= 0.5 cnt                          ->  必须从误差预算里扣除后再谈控制器

用法：
    python test_sensor_noise_floor.py                # 默认 1500/2000 样本 + 自校验
    python test_sensor_noise_floor.py 3000 4000      # 自定义 Phase A / B 样本数
    python test_sensor_noise_floor.py 1500 2000 noself
"""
import sys
import time
from mouse_sensor import HighPrecisionMouseSensor


def _stats(vals):
    if not vals:
        return "n=0"
    s = sorted(vals)
    n = len(s)
    med = s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])
    return "n={} median={:.1f} max={:.0f} min={:.0f}".format(n, med, max(s), min(s))


def _collect(sensor, n_samples, hard_timeout=40.0):
    """返回 (dxs, dys, dur, cum_x, cum_y)。cum_* 为累计位置轨迹。"""
    dxs, dys = [], []
    cx = cy = 0
    cxs, cys = [0], [0]
    t0 = time.time()
    while len(dxs) < n_samples:
        if time.time() - t0 > hard_timeout:
            print("[WARN] hard timeout {:.0f}s, stopped early".format(hard_timeout))
            break
        dx, dy = sensor.read_delta_and_clear()
        dx = int(dx)
        dy = int(dy)
        dxs.append(dx)
        dys.append(dy)
        cx += dx
        cy += dy
        cxs.append(cx)
        cys.append(cy)
        time.sleep(0.001)
    return dxs, dys, time.time() - t0, cxs, cys


def _report(tag, dxs, dys, dur, cxs, cys):
    n = len(dxs)
    nz_x = [abs(v) for v in dxs if v != 0]
    nz_y = [abs(v) for v in dys if v != 0]
    path_x = sum(abs(v) for v in dxs)
    path_y = sum(abs(v) for v in dys)
    jx = path_x / dur if dur > 0 else 0.0
    jy = path_y / dur if dur > 0 else 0.0

    print("=" * 78)
    print("[{}]".format(tag))
    print("[samples] n={}  duration={:.2f} s  avg interval={:.2f} ms".format(n, dur, 1000.0 * dur / n))
    print("[nonzero sample ratio]  X: {:.1f}%   Y: {:.1f}%".format(100.0 * len(nz_x) / n, 100.0 * len(nz_y) / n))
    print("[per-sample nonzero]    X: {}".format(_stats(nz_x)))
    print("                        Y: {}".format(_stats(nz_y)))
    print("[cumulative path]  X={} cnt  Y={} cnt   (total jitter)".format(path_x, path_y))
    print("[net drift]        X={:+d} cnt  Y={:+d} cnt".format(cxs[-1], cys[-1]))
    print("[peak-to-peak]     X={} cnt  Y={} cnt   (window wander)".format(max(cxs) - min(cxs), max(cys) - min(cys)))
    print("[jitter rate]      X={:.2f} cnt/s  Y={:.2f} cnt/s".format(jx, jy))
    w = 0.110
    print("[one 110 ms window] X={:.2f} cnt  Y={:.2f} cnt".format(jx * w, jy * w))
    return jx, jy


def main(n_a: int = 1500, n_b: int = 2000, do_selftest: bool = True) -> int:
    sensor = HighPrecisionMouseSensor()
    sensor.start()
    time.sleep(0.6)

    drv = None
    try:
        from servo_driver import WS63ServoDriver
        drv = WS63ServoDriver()
        try:
            drv.set_pwm_frequency(200)
        except Exception as e:
            print("[WARN] set_pwm_frequency failed: {!r}".format(e))
    except Exception as e:
        print("[WARN] servo driver unavailable: {!r}  -> Phase B will be skipped".format(e))

    # ---------------- Phase A: 舵机未驱动（limp） ----------------
    sensor.reset_origin()
    ax, ay, adur, acx, acy = _collect(sensor, n_a)
    jax, jay = _report("PHASE A / LIMP  -- servos NOT driven (sensor floor + free drift)", ax, ay, adur, acx, acy)

    # ---------------- Phase B: 通电保持中立位 ----------------
    jbx = jby = None
    if drv is not None:
        try:
            drv.center(settle_s=0.50, preload=True)
            time.sleep(0.35)
            sensor.read_delta_and_clear()
            sensor.reset_origin()
            bx, by, bdur, bcx, bcy = _collect(sensor, n_b)
            jbx, jby = _report("PHASE B / HOLD NEUTRAL  -- servos powered & holding (REAL reading floor)",
                               bx, by, bdur, bcx, bcy)
        except Exception as e:
            print("[WARN] Phase B failed: {!r}".format(e))

    # ---------------- Phase C: 自校验（量具必须读到非零） ----------------
    live = None
    if do_selftest and drv is not None:
        print("-" * 78)
        print("[PHASE C / LIVENESS  -- +12 us on S1, -12 us on S2 (expect tens of cnt)]")
        try:
            drv.center(settle_s=0.45, preload=True)
            time.sleep(0.25)
            sensor.read_delta_and_clear()
            time.sleep(0.08)
            sensor.read_delta_and_clear()
            drv.send_raw_pulses(drv.NEUTRAL_S1 + 12.0, drv.NEUTRAL_S2 - 12.0)
            time.sleep(0.25)
            ldx, ldy = sensor.read_delta_and_clear()
            live = (abs(ldx) + abs(ldy)) >= 3
            print("[selftest] measured shift = ({:+d}, {:+d}) cnt  -> liveness={}".format(
                ldx, ldy, "OK" if live else "FAIL (sensor blind)"))
            drv.center(settle_s=0.45, preload=True)
        except Exception as e:
            print("[selftest] failed: {!r}".format(e))

    if drv is not None:
        try:
            drv.close()
        except Exception:
            pass
    sensor.stop()

    # ---------------- 判定 ----------------
    print("-" * 78)
    if live is False:
        print("[VERDICT] INVALID -- a ~20 cnt nudge was not seen; Phase A/B zeros are meaningless.")
        print("=" * 78)
        return 2
    jx = jbx if jbx is not None else jax
    jy = jby if jby is not None else jay
    w = 0.110
    if jx * w < 0.5 and jy * w < 0.5:
        print("[VERDICT] floor < 0.5 cnt per reading  ->  neither instrument nor holding jitter is the bottleneck.")
        print("          1-2 counts KPI is within measurable capability; remaining error is mechanics/actuator.")
    else:
        print("[VERDICT] floor >= 0.5 cnt per reading ->  this component must be removed from the error budget")
        print("          before attributing the 5.15 cnt mean to mechanics or control.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    na, nb = 1500, 2000
    args = sys.argv[1:]
    if len(args) >= 1:
        try:
            na = int(args[0])
        except ValueError:
            pass
    if len(args) >= 2:
        try:
            nb = int(args[1])
        except ValueError:
            pass
    selftest = not any(a.lower() == "noself" for a in args)
    sys.exit(main(na, nb, selftest))
