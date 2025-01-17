import torch

from .typing import Tensor


def hinge_real_loss(tensor : Tensor) -> Tensor:
	return torch.relu(1 - tensor)


def hinge_fake_loss(tensor : Tensor) -> Tensor:
	return torch.relu(tensor + 1)
