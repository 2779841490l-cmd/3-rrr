import csv
import os
import time
from typing import Dict, List, Optional, Tuple

from rect_manifold_calibrator import HardwareRectangleTrainer, RectangularManifoldController


def estimate_inflection_offset(
    records: List[Tuple[int, float, float]], axis: str
) -> Optional[int]:
    """Return the first offset where cumulative displacement along `axis` exceeds 1 count."""
    if not records:
        return None
    idx = 1 if axis == "X" else 2
    p0 = records[0][idx]
    for offset, px, py in records[1:]:
        p = px if axis == "X" else py
        if abs(p - p0) > 1.0:
            return offset
    return None


def main():
    _ = RectangularManifoldController  # ensure class availability per spec
    trainer = HardwareRectangleTrainer()
    manifold = trainer.manifold
    driver = trainer.driver
    sensor = trainer.sensor

    sensor.start()
    offsets = list(range(-40, 41, 2))
    rows: List[Dict] = []
    inflections: List[Tuple[int, Optional[int], Optional[int]]] = []

    try:
        trainer.move_to_rect_origin()
        sensor.recenter_cursor()

        os.makedirs("experiments", exist_ok=True)
        csv_path = "experiments/flank_probe.csv"

        for run_id in range(1, 6):
            trainer.move_to_rect_origin()
            sensor.recenter_cursor()

            u0_kin = manifold.evaluate_kinematic_pulse(0.0, 0.0)
            play_g = manifold.pi_net.play_g.copy()
            print(
                f"[Run {run_id}/5] play_g={play_g.tolist()}, "
                f"hyst_x_pos={manifold.hyst_x_pos}, hyst_y_pos={manifold.hyst_y_pos}, "
                f"u0_kin={u0_kin.tolist()}"
            )

            # X axis sweep
            x_records: List[Tuple[int, float, float]] = []
            for offset in offsets:
                u = [float(u0_kin[0] + offset), float(u0_kin[1] + manifold.hyst_y_pos)]
                driver.send_raw_pulses(*u)
                time.sleep(0.25)
                pos = trainer.get_rect_position()
                px, py = float(pos[0]), float(pos[1])
                x_records.append((offset, px, py))
                rows.append(
                    {
                        "run_id": run_id,
                        "axis": "X",
                        "u_offset_us": offset,
                        "pos_x_cnt": round(px, 2),
                        "pos_y_cnt": round(py, 2),
                        "play_g_x": round(float(manifold.pi_net.play_g[0]), 4),
                        "play_g_y": round(float(manifold.pi_net.play_g[1]), 4),
                    }
                )

            # Y axis sweep
            y_records: List[Tuple[int, float, float]] = []
            for offset in offsets:
                u = [float(u0_kin[0] + manifold.hyst_x_pos), float(u0_kin[1] + offset)]
                driver.send_raw_pulses(*u)
                time.sleep(0.25)
                pos = trainer.get_rect_position()
                px, py = float(pos[0]), float(pos[1])
                y_records.append((offset, px, py))
                rows.append(
                    {
                        "run_id": run_id,
                        "axis": "Y",
                        "u_offset_us": offset,
                        "pos_x_cnt": round(px, 2),
                        "pos_y_cnt": round(py, 2),
                        "play_g_x": round(float(manifold.pi_net.play_g[0]), 4),
                        "play_g_y": round(float(manifold.pi_net.play_g[1]), 4),
                    }
                )

            inf_x = estimate_inflection_offset(x_records, "X")
            inf_y = estimate_inflection_offset(y_records, "Y")
            inflections.append((run_id, inf_x, inf_y))

        fieldnames = [
            "run_id",
            "axis",
            "u_offset_us",
            "pos_x_cnt",
            "pos_y_cnt",
            "play_g_x",
            "play_g_y",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print("\n=== Inflection Offset Estimates (|disp| > 1.0 count) ===")
        for run_id, inf_x, inf_y in inflections:
            print(f"  Run {run_id}: X_offset={inf_x} us, Y_offset={inf_y} us")
        print(f"CSV saved: {os.path.abspath(csv_path)}")
    finally:
        trainer.close()


if __name__ == "__main__":
    main()

