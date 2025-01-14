import configparser
from typing import Tuple

import torch

from .typing import Tensor

CONFIG = configparser.ConfigParser()
CONFIG.read('config.ini')

L2_loss = torch.nn.MSELoss()


def transform_points(points : Tensor, rotation_matrix : Tensor, expression : Tensor, scale : Tensor, translation : Tensor) -> Tensor:
	points_transformed = points.view(-1, 21, 3) @ rotation_matrix + expression.view(-1, 21, 3)
	points_transformed *= scale[..., None]
	points_transformed[:, :, 0:2] += translation[:, None, 0:2]
	return points_transformed


def hinge_loss(tensor : Tensor, is_positive : bool) -> Tensor:
	if is_positive:
		return torch.relu(1 - tensor)
	else:
		return torch.relu(tensor + 1)


def calc_distance_ratio(landmarks : Tensor, indices : Tuple[int, int, int, int]) -> Tensor:
	distance_horizontal = torch.norm(landmarks[:, indices[0]] - landmarks[:, indices[1]], p = 2, dim = 1, keepdim = True)
	distance_vertical = torch.norm(landmarks[:, indices[2]] - landmarks[:, indices[3]], p=2, dim = 1, keepdim = True)
	return distance_horizontal / (distance_vertical + 1e-4)
