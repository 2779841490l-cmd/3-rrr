import json
import numpy as np

with open("experiments/real_rect_grid_samples.json", "r") as f:
    samples = json.load(f)

# Extract dux, duy, x, y
dux = np.array([s["dux"] for s in samples])
duy = np.array([s["duy"] for s in samples])
x = np.array([s["x"] for s in samples])
y = np.array([s["y"] for s in samples])

print(f"Total samples: {len(samples)}")
print(f"X range: [{x.min():.1f}, {x.max():.1f}], span={x.max()-x.min():.1f} counts")
print(f"Y range: [{y.min():.1f}, {y.max():.1f}], span={y.max()-y.min():.1f} counts")

# Build 2nd degree bivariate polynomial features for (x, y) -> dux and (x, y) -> duy
# Features: [1, x, y, x^2, x*y, y^2]
A = np.column_stack([np.ones_like(x), x, y, x**2, x*y, y**2])

# Fit least squares
coeffs_ux, res_ux, _, _ = np.linalg.lstsq(A, dux, rcond=None)
coeffs_uy, res_uy, _, _ = np.linalg.lstsq(A, duy, rcond=None)

pred_dux = A @ coeffs_ux
pred_duy = A @ coeffs_uy

rmse_ux = np.sqrt(np.mean((dux - pred_dux)**2))
rmse_uy = np.sqrt(np.mean((duy - pred_duy)**2))

print(f"\n2nd Order Bivariate Fit RMSE: du_x = {rmse_ux:.2f} us, du_y = {rmse_uy:.2f} us")
print("\nPolynomial coefficients for du_x [1, x, y, x^2, x*y, y^2]:")
print(np.round(coeffs_ux, 6))
print("\nPolynomial coefficients for du_y [1, x, y, x^2, x*y, y^2]:")
print(np.round(coeffs_uy, 6))

# Let's inspect linear Jacobian from the fit at (0, 0)
# du_x ≈ c0 + c1*x + c2*y
# du_y ≈ d0 + d1*x + d2*y
K_center = np.array([
    [coeffs_ux[1], coeffs_ux[2]],
    [coeffs_uy[1], coeffs_uy[2]]
])
print("\nEstimated K at (0,0) [us/count]:\n", np.round(K_center, 4))
print("Estimated J at (0,0) [count/us]:\n", np.round(np.linalg.inv(K_center), 4))

