import configparser
from typing import Tuple

import torch
from .typing import Tensor
import numpy
import torch.nn.functional as F

CONFIG = configparser.ConfigParser()
CONFIG.read('config.ini')

if CONFIG.getboolean('preparing.augmentation', 'expression'):
	from LivePortrait.src.utils.camera import headpose_pred_to_degree, get_rotation_matrix

L2_loss = torch.nn.MSELoss()
EXPRESSION_MIN = numpy.array(
[
	[
		[-2.88067125e-02, -8.12731311e-02, -1.70541159e-03],
		[-4.88598682e-02, -3.32196616e-02, -1.67431499e-04],
		[-6.75425082e-02, -4.28681746e-02, -1.98950816e-04],
		[-7.23103955e-02, -3.28503326e-02, -7.31324719e-04],
		[-3.87073644e-02, -6.01546466e-02, -5.50269964e-04],
		[-6.38048723e-02, -2.23840728e-01, -7.13261834e-04],
		[-3.02710701e-02, -3.93195450e-02, -8.24086510e-06],
		[-2.95799859e-02, -5.39318882e-02, -1.74219604e-04],
		[-2.92359516e-02, -1.53050944e-02, -6.30460854e-05],
		[-5.56493877e-03, -2.34344602e-02, -1.26858242e-04],
		[-4.37593013e-02, -2.77768299e-02, -2.70503685e-02],
		[-1.76926646e-02, -1.91676542e-02, -1.15090821e-04],
		[-8.34268332e-03, -3.99775570e-03, -3.27481248e-05],
		[-3.40162888e-02, -2.81868968e-02, -1.96679524e-04],
		[-2.91855410e-02, -3.97511162e-02, -2.81230678e-05],
		[-1.50395725e-02, -2.49494594e-02, -9.42573533e-05],
		[-1.67938769e-02, -2.00953931e-02, -4.00750607e-04],
		[-1.86435618e-02, -2.48535164e-02, -2.74416432e-02],
		[-4.61211195e-03, -1.21660791e-02, -2.93173041e-04],
		[-4.10017073e-02, -7.43824020e-02, -4.42762971e-02],
		[-1.90370996e-02, -3.74363363e-02, -1.34740388e-02]
	]
]).astype(numpy.float32)
EXPRESSION_MAX = numpy.array(
[
	[
		[4.46682945e-02, 7.08772913e-02, 4.08344204e-04],
		[2.14308221e-02, 6.15894832e-02, 4.85319615e-05],
		[3.02363783e-02, 4.45043296e-02, 1.28298725e-05],
		[3.05869691e-02, 3.79812494e-02, 6.57040102e-04],
		[4.45670523e-02, 3.97259220e-02, 7.10966764e-04],
		[9.43699256e-02, 9.85926315e-02, 2.02551950e-04],
		[1.61131397e-02, 2.92906128e-02, 3.44733417e-06],
		[5.23825921e-02, 1.07065082e-01, 6.61510974e-04],
		[2.85718683e-03, 8.32320191e-03, 2.39314613e-04],
		[2.57947259e-02, 1.60935968e-02, 2.41853559e-05],
		[4.90833223e-02, 3.43903080e-02, 3.22353356e-02],
		[1.44766076e-02, 3.39248963e-02, 1.42291479e-04],
		[8.75749043e-04, 6.82212645e-03, 2.76097053e-05],
		[1.86958015e-02, 3.84016186e-02, 7.33085908e-05],
		[2.01714113e-02, 4.90544215e-02, 2.34028921e-05],
		[2.46518422e-02, 3.29151377e-02, 3.48571630e-05],
		[2.22457591e-02, 1.21796541e-02, 1.56396593e-04],
		[1.72109623e-02, 3.01626958e-02, 1.36556877e-02],
		[1.83460284e-02, 1.61141958e-02, 2.87440169e-04],
		[3.57594155e-02, 1.80554688e-01, 2.75554154e-02],
		[2.17450950e-02, 8.66811201e-02, 3.34241726e-02]
	]
]).astype(numpy.float32)


def randomize_expression(face_tensor, feature_extractor, motion_extractor, warping_network, spade_generator):
	with torch.no_grad():
		face_tensor_norm = (face_tensor + 1) * 0.5
		input_device = face_tensor.device
		feature_volume = feature_extractor(face_tensor_norm)
		motion_extractor_dict = motion_extractor(face_tensor_norm)

		translation = motion_extractor_dict.get('t')
		expression = motion_extractor_dict.get('exp')
		scale = motion_extractor_dict.get('scale')
		points = motion_extractor_dict.get('kp')

		pitch = headpose_pred_to_degree(motion_extractor_dict.get('pitch'))[:, None]
		yaw = headpose_pred_to_degree(motion_extractor_dict.get('yaw'))[:, None]
		roll = headpose_pred_to_degree(motion_extractor_dict.get('roll'))[:, None]
		rotation_matrix = get_rotation_matrix(pitch, yaw, roll)
		random_expression = get_random_expression_blend(expression)

		points_transformed = transform_points(points, rotation_matrix, expression, scale, translation)
		points_driv = transform_points(points, rotation_matrix, random_expression, scale, translation)

		data = warping_network(feature_volume, points_driv, points_transformed).get('out')
		output = spade_generator(data)
		output = output.to(input_device)
		output = F.interpolate(output.clamp(0, 1), [256, 256], mode='bilinear', align_corners=False)
		output = (output - 0.5) * 2
	return output


def get_random_expression_blend(expression : Tensor) -> Tensor:
	blend = 0.35
	expression = expression.view(-1, 21, 3)
	min_array = torch.from_numpy(EXPRESSION_MIN).to(expression.device).to(expression.dtype).expand(expression.shape[0], -1, -1)
	max_array = torch.from_numpy(EXPRESSION_MAX).to(expression.device).to(expression.dtype).expand(expression.shape[0], -1, -1)
	random_batch = torch.rand_like(min_array).to(expression.device) * (max_array - min_array) + min_array
	random_batch[:, [0, 1, 8, 6, 9, 4, 5, 10]] = expression[:, [0, 1, 8, 6, 9, 4, 5, 10]]
	random_batch[:, [3, 7]] = random_batch[:, [13, 16]] * 0.1 + expression[:, [13, 16]] * 0.9
	random_batch[:, [3, 7]] = random_batch[:, [3, 7]] * 0.5 + expression[:, [3, 7]] * 0.5
	return random_batch * 0.8 * blend + expression * (1 - blend)


def transform_points(points : Tensor, rotation_matrix : Tensor, expression : Tensor, scale : Tensor, translation : Tensor):
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
