import csv
with open('experiments/rect_contour_iter_6.csv', 'r') as f:
    rows = list(csv.DictReader(f))

for seg in ['TOP_EDGE', 'RIGHT_EDGE', 'BOTTOM_EDGE', 'LEFT_EDGE']:
    sub = [r for r in rows if r['segment'] == seg]
    print(f"=== {seg} (N={len(sub)}) ===")
    for i, r in enumerate(sub):
        perp = float(r['err_y']) if seg in ('TOP_EDGE', 'BOTTOM_EDGE') else float(r['err_x'])
        print(f"  #{i:02d}: des=({r['x_des']:>7s}, {r['y_des']:>7s}) act=({r['x_act']:>7s}, {r['y_act']:>7s}) perp_err={perp:6.2f}")

