import configparser
from typing import List, Tuple

import torch
from pytorch_msssim import ssim
from torch import Tensor, nn
from torchvision import transforms

from ..helper import calc_embedding
from ..types import Attributes, EmbedderModule, Gaze, GazerModule, MotionExtractorModule

CONFIG = configparser.ConfigParser()
CONFIG.read('config.ini')


class DiscriminatorLoss(nn.Module):
	def __init__(self) -> None:
		super().__init__()

	def forward(self, discriminator_source_tensors : List[Tensor], discriminator_output_tensors : List[Tensor]) -> Tensor:
		positive_tensors = []
		negative_tensors = []

		for discriminator_output_tensor in discriminator_output_tensors:
			positive_tensor = torch.relu(discriminator_output_tensor[0] + 1).mean(dim = [ 1, 2, 3 ])
			positive_tensors.append(positive_tensor)

		for discriminator_source_tensor in discriminator_source_tensors:
			negative_tensor = torch.relu(1 - discriminator_source_tensor[0]).mean(dim = [ 1, 2, 3 ])
			negative_tensors.append(negative_tensor)

		positive_loss = torch.stack(positive_tensors).mean()
		negative_loss = torch.stack(negative_tensors).mean()
		discriminator_loss = (positive_loss + negative_loss) * 0.5
		return discriminator_loss


class AdversarialLoss(nn.Module):
	def __init__(self) -> None:
		super().__init__()

	def forward(self, discriminator_output_tensors : List[Tensor]) -> Tuple[Tensor, Tensor]:
		adversarial_weight = CONFIG.getfloat('training.losses', 'adversarial_weight')
		temp_tensors = []

		for discriminator_output_tensor in discriminator_output_tensors:
			temp_tensor = torch.relu(1 - discriminator_output_tensor[0]).mean(dim = [ 1, 2, 3 ]).mean()
			temp_tensors.append(temp_tensor)

		adversarial_loss = torch.stack(temp_tensors).mean()
		weighted_adversarial_loss = adversarial_loss * adversarial_weight
		return adversarial_loss, weighted_adversarial_loss


class AttributeLoss(nn.Module):
	def __init__(self) -> None:
		super().__init__()

	def forward(self, target_attributes : Attributes, output_attributes : Attributes) -> Tuple[Tensor, Tensor]:
		batch_size = CONFIG.getint('training.loader', 'batch_size')
		attribute_weight = CONFIG.getfloat('training.losses', 'attribute_weight')
		temp_tensors = []

		for target_attribute, output_attribute in zip(target_attributes, output_attributes):
			temp_tensor = torch.mean(torch.pow(output_attribute - target_attribute, 2).reshape(batch_size, -1), dim = 1).mean()
			temp_tensors.append(temp_tensor)

		attribute_loss = torch.stack(temp_tensors).mean() * 0.5
		weighted_attribute_loss = attribute_loss * attribute_weight
		return attribute_loss, weighted_attribute_loss


class ReconstructionLoss(nn.Module):
	def __init__(self, embedder : EmbedderModule) -> None:
		super().__init__()
		self.embedder = embedder
		self.mse_loss = nn.MSELoss()

	def forward(self, source_tensor : Tensor, target_tensor : Tensor, output_tensor : Tensor) -> Tuple[Tensor, Tensor]:
		reconstruction_weight = CONFIG.getfloat('training.losses', 'reconstruction_weight')
		source_embedding = calc_embedding(self.embedder, source_tensor, (0, 0, 0, 0))
		target_embedding = calc_embedding(self.embedder, target_tensor, (0, 0, 0, 0))
		has_similar_identity = torch.cosine_similarity(source_embedding, target_embedding) > 0.8

		reconstruction_loss = torch.mean((source_tensor - target_tensor) ** 2, dim = (1, 2, 3))
		reconstruction_loss = (reconstruction_loss * has_similar_identity).mean() * 0.5

		data_range = float(torch.max(output_tensor) - torch.min(output_tensor))
		visual_loss = 1 - ssim(output_tensor, target_tensor, data_range = data_range).mean()
		reconstruction_loss = (reconstruction_loss + visual_loss) * 0.5
		weighted_reconstruction_loss = reconstruction_loss * reconstruction_weight
		return reconstruction_loss, weighted_reconstruction_loss


class IdentityLoss(nn.Module):
	def __init__(self, embedder : EmbedderModule) -> None:
		super().__init__()
		self.embedder = embedder

	def forward(self, source_tensor : Tensor, output_tensor : Tensor) -> Tuple[Tensor, Tensor]:
		identity_weight = CONFIG.getfloat('training.losses', 'identity_weight')
		output_embedding = calc_embedding(self.embedder, output_tensor, (30, 0, 10, 10))
		source_embedding = calc_embedding(self.embedder, source_tensor, (30, 0, 10, 10))
		identity_loss = (1 - torch.cosine_similarity(source_embedding, output_embedding)).mean()
		weighted_identity_loss = identity_loss * identity_weight
		return identity_loss, weighted_identity_loss


class MotionLoss(nn.Module):
	def __init__(self, motion_extractor : MotionExtractorModule):
		super().__init__()
		self.motion_extractor = motion_extractor
		self.mse_loss = nn.MSELoss()

	def forward(self, target_tensor : Tensor, output_tensor : Tensor) -> Tuple[Tensor, ...]:
		target_poses, target_expression = self.get_motions(target_tensor)
		output_poses, output_expression = self.get_motions(output_tensor)
		pose_loss, weighted_pose_loss = self.calc_pose_loss(target_poses, output_poses)
		expression_loss, weighted_expression_loss = self.calc_expression_loss(target_expression, output_expression)
		return pose_loss, weighted_pose_loss, expression_loss, weighted_expression_loss

	def calc_pose_loss(self, target_poses : Tuple[Tensor, ...], output_poses : Tuple[Tensor, ...]) -> Tuple[Tensor, Tensor]:
		pose_weight = CONFIG.getfloat('training.losses', 'pose_weight')
		temp_tensors = []

		for target_pose, output_pose in zip(target_poses, output_poses):
			temp_tensor = self.mse_loss(target_pose, output_pose)
			temp_tensors.append(temp_tensor)

		pose_loss = torch.stack(temp_tensors).mean()
		weighted_pose_loss = pose_loss * pose_weight
		return pose_loss, weighted_pose_loss

	def calc_expression_loss(self, target_expression : Tensor, output_expression : Tensor) -> Tuple[Tensor, Tensor]:
		expression_weight = CONFIG.getfloat('training.losses', 'expression_weight')
		expression_loss = (1 - torch.cosine_similarity(target_expression, output_expression)).mean()
		weighted_expression_loss = expression_loss * expression_weight
		return expression_loss, weighted_expression_loss

	def get_motions(self, input_tensor : Tensor) -> Tuple[Tuple[Tensor, ...], Tensor]:
		input_tensor = (input_tensor + 1) * 0.5
		pitch, yaw, roll, translation, expression, scale, motion_points = self.motion_extractor(input_tensor)
		rotation = torch.cat([ pitch, yaw, roll ], dim = 1)
		pose = translation, scale, rotation, motion_points
		return pose, expression


class GazeLoss(nn.Module):
	def __init__(self, gazer : GazerModule) -> None:
		super().__init__()
		self.gazer = gazer
		self.mae_loss = nn.L1Loss()
		self.transform = transforms.Compose(
		[
			transforms.Resize(448),
			transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ], std = [ 0.229, 0.224, 0.225 ])
		])

	def forward(self, target_tensor : Tensor, output_tensor : Tensor) -> Tuple[Tensor, Tensor]:
		gaze_weight = CONFIG.getfloat('training.losses', 'gaze_weight')
		output_pitch_tensor, output_yaw_tensor = self.detect_gaze(output_tensor)
		target_pitch_tensor, target_yaw_tensor = self.detect_gaze(target_tensor)

		pitch_gaze_loss = self.mae_loss(output_pitch_tensor, target_pitch_tensor)
		yaw_gaze_loss = self.mae_loss(output_yaw_tensor, target_yaw_tensor)

		gaze_loss = (pitch_gaze_loss + yaw_gaze_loss) * 0.5
		weighted_gaze_loss = gaze_loss * gaze_weight
		return gaze_loss, weighted_gaze_loss

	def detect_gaze(self, input_tensor : Tensor) -> Gaze:
		scale_factor = CONFIG.getint('training.losses', 'gaze_scale_factor')
		y_min = int(60 * scale_factor)
		y_max = int(224 * scale_factor)
		x_min = int(16 * scale_factor)
		x_max = int(205 * scale_factor)

		crop_tensor = input_tensor[:, :, y_min:y_max, x_min:x_max]
		crop_tensor = (crop_tensor + 1) * 0.5
		crop_tensor = transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ], std = [ 0.229, 0.224, 0.225 ])(crop_tensor)
		crop_tensor = nn.functional.interpolate(crop_tensor, size = 448, mode = 'bicubic')
		pitch_tensor, yaw_tensor = self.gazer(crop_tensor)
		return pitch_tensor, yaw_tensor
