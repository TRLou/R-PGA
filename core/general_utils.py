import torch
import numpy as np


def inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
	return torch.log(x / (1 - x))


def strip_lowerdiag(L: torch.Tensor) -> torch.Tensor:
	uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device=L.device)
	uncertainty[:, 0] = L[:, 0, 0]
	uncertainty[:, 1] = L[:, 0, 1]
	uncertainty[:, 2] = L[:, 0, 2]
	uncertainty[:, 3] = L[:, 1, 1]
	uncertainty[:, 4] = L[:, 1, 2]
	uncertainty[:, 5] = L[:, 2, 2]
	return uncertainty


def strip_symmetric(sym: torch.Tensor) -> torch.Tensor:
	return strip_lowerdiag(sym)


def build_rotation(r: torch.Tensor) -> torch.Tensor:
	norm = torch.sqrt(r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3])
	q = r / norm[:, None]
	R = torch.zeros((q.size(0), 3, 3), device=r.device)
	rw = q[:, 0]
	rx = q[:, 1]
	ry = q[:, 2]
	rz = q[:, 3]
	R[:, 0, 0] = 1 - 2 * (ry * ry + rz * rz)
	R[:, 0, 1] = 2 * (rx * ry - rw * rz)
	R[:, 0, 2] = 2 * (rx * rz + rw * ry)
	R[:, 1, 0] = 2 * (rx * ry + rw * rz)
	R[:, 1, 1] = 1 - 2 * (rx * rx + rz * rz)
	R[:, 1, 2] = 2 * (ry * rz - rw * rx)
	R[:, 2, 0] = 2 * (rx * rz - rw * ry)
	R[:, 2, 1] = 2 * (ry * rz + rw * rx)
	R[:, 2, 2] = 1 - 2 * (rx * rx + ry * ry)
	return R


def build_scaling_rotation(s: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
	L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device=s.device)
	R = build_rotation(r)
	L[:, 0, 0] = s[:, 0]
	L[:, 1, 1] = s[:, 1]
	L[:, 2, 2] = s[:, 2]
	L = R @ L
	return L


