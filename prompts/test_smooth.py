import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np

def smooth_backlash_x(du: float) -> float:
    if du >= 0:
        return min(7.0, du * (7.0 / 12.0))
    else:
        return max(-20.5, du * (20.5 / 15.0))

def smooth_backlash_y(du: float) -> float:
    if du >= 0:
        return min(5.0, du * (5.0 / 10.0))
    else:
        return max(-6.5, du * (6.5 / 10.0))

print("Backlash smoothly tested at:")
for d in [-20, -10, -5, -2, 0, 2, 5, 10, 20]:
    print(f"du={d:3d} -> bx={smooth_backlash_x(d):+5.2f}, by={smooth_backlash_y(d):+5.2f}")

