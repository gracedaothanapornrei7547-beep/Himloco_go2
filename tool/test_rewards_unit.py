import torch
import math
from himloco_lab.tasks.locomotion.mdp.rewards import _bernstein_torch, _bezier_curve_torch, _bezier_curve_derivative_torch

t = torch.linspace(0, 1, 10)
tau = 0.0
psi = 0.05
ctrl = torch.tensor([
    [-tau, 0.0],
    [-0.95 * tau, 0.80 * psi],
    [-0.55 * tau, 1.00 * psi],
    [0.55 * tau, 1.00 * psi],
    [0.95 * tau, 0.80 * psi],
    [tau, 0.0],
])

curve = _bezier_curve_torch(ctrl, t)
deriv = _bezier_curve_derivative_torch(ctrl, t)
print("Bezier test success!")
print(f"Curve shape: {curve.shape}, max z: {curve[:, 1].max().item():.4f}")
assert curve.shape == (10, 2)
assert abs(curve[:, 1].max().item() - 0.05) < 1e-4
print("Unit test verified perfectly!")
