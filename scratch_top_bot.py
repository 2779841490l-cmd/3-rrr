import csv
with open('experiments/rect_contour_iter_6.csv') as f:
    rows = list(csv.DictReader(f))
for seg in ['TOP_EDGE', 'BOTTOM_EDGE']:
    sub = [r for r in rows if r['segment'] == seg]
    print(f"=== {seg} ===")
    for r in sub[::3]:
        print(f"des=({r['x_des']:>7s}, {r['y_des']:>7s}) act=({r['x_act']:>7s}, {r['y_act']:>7s}) err_y={r['err_y']:>6s} | uy={r['u_y']:>6s}")

