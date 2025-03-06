from configparser import ConfigParser
from typing import List, Tuple

import torch
from pytorch_msssim import ssim
from torch import Tensor, nn
from torchvision import transforms

from ..helper import calc_embedding
from ..types import Attributes, EmbedderModule, Gaze, GazerModule, MotionExtractorModule


class DiscriminatorLoss(nn.Module):
	def __init__(self) -> None:
		super().__init__()

	def forward(self, discriminator_source_tensors : List[Tensor], discriminator_output_tensors : List[Tensor]) -> Tensor:
		positive_tensors = []
		negative_tensors = []

		for discriminator_source_tensor in discriminator_source_tensors:
			positive_tensor = torch.relu(discriminator_source_tensor + 1).mean(dim = [ 1, 2, 3 ])
			positive_tensors.append(positive_tensor)

		for discriminator_output_tensor in discriminator_output_tensors:
			negative_tensor = torch.relu(1 - discriminator_output_tensor).mean(dim = [ 1, 2, 3 ])
			negative_tensors.append(negative_tensor)

		positive_loss = torch.stack(positive_tensors).mean()
		negative_loss = torch.stack(negative_tensors).mean()
		discriminator_loss = (positive_loss + negative_loss) * 0.5
		return discriminator_loss


class AdversarialLoss(nn.Module):
	def __init__(self, config_parser : ConfigParser) -> None:
		super().__init__()
		self.config =\
		{
			'adversarial_weight': config_parser.getfloat('training.losses', 'adversarial_weight')
		}

	def forward(self, discriminator_output_tensors : List[Tensor]) -> Tuple[Tensor, Tensor]:
		temp_tensors = []

		for discriminator_output_tensor in discriminator_output_tensors:
			temp_tensor = torch.relu(1 - discriminator_output_tensor).mean(dim = [ 1, 2, 3 ]).mean()
			temp_tensors.append(temp_tensor)

		adversarial_loss = torch.stack(temp_tensors).mean()
		weighted_adversarial_loss = adversarial_loss * self.config.get('adversarial_weight')
		return adversarial_loss, weighted_adversarial_loss


class AttributeLoss(nn.Module):
	def __init__(self, config_parser : ConfigParser) -> None:
		super().__init__()
		self.config =\
		{
			'batch_size': config_parser.getint('training.loader', 'batch_size'),
			'attribute_weight': config_parser.getfloat('training.losses', 'attribute_weight')
		}

	def forward(self, target_attributes : Attributes, output_attributes : Attributes) -> Tuple[Tensor, Tensor]:
		temp_tensors = []

		for target_attribute, output_attribute in zip(target_attributes, output_attributes):
			temp_tensor = torch.mean(torch.pow(output_attribute - target_attribute, 2).reshape(self.config.get('batch_size'), -1), dim = 1).mean()
			temp_tensors.append(temp_tensor)

		attribute_loss = torch.stack(temp_tensors).mean() * 0.5
		weighted_attribute_loss = attribute_loss * self.config.get('attribute_weight')
		return attribute_loss, weighted_attribute_loss


class ReconstructionLoss(nn.Module):
	def __init__(self, config_parser : ConfigParser, embedder : EmbedderModule) -> None:
		super().__init__()
		self.config =\
		{
			'reconstruction_weight': config_parser.getfloat('training.losses', 'reconstruction_weight')
		}
		self.embedder = embedder
		self.mse_loss = nn.MSELoss()

	def forward(self, source_tensor : Tensor, target_tensor : Tensor, output_tensor : Tensor) -> Tuple[Tensor, Tensor]:
		source_embedding = calc_embedding(self.embedder, source_tensor, (0, 0, 0, 0))
		target_embedding = calc_embedding(self.embedder, target_tensor, (0, 0, 0, 0))
		has_similar_identity = torch.cosine_similarity(source_embedding, target_embedding) > 0.8

		reconstruction_loss = torch.mean((source_tensor - target_tensor) ** 2, dim = (1, 2, 3))
		reconstruction_loss = (reconstruction_loss * has_similar_identity).mean() * 0.5

		data_range = float(torch.max(output_tensor) - torch.min(output_tensor))
		visual_loss = 1 - ssim(output_tensor, target_tensor, data_range = data_range).mean()
		reconstruction_loss = (reconstruction_loss + visual_loss) * 0.5
		weighted_reconstruction_loss = reconstruction_loss * self.config.get('reconstruction_weight')
		return reconstruction_loss, weighted_reconstruction_loss


class IdentityLoss(nn.Module):
	def __init__(self, config_parser : ConfigParser, embedder : EmbedderModule) -> None:
		super().__init__()
		self.config =\
		{
			'identity_weight': config_parser.getfloat('training.losses', 'identity_weight')
		}
		self.embedder = embedder

	def forward(self, source_tensor : Tensor, output_tensor : Tensor) -> Tuple[Tensor, Tensor]:
		output_embedding = calc_embedding(self.embedder, output_tensor, (30, 0, 10, 10))
		source_embedding = calc_embedding(self.embedder, source_tensor, (30, 0, 10, 10))
		identity_loss = (1 - torch.cosine_similarity(source_embedding, output_embedding)).mean()
		weighted_identity_loss = identity_loss * self.config.get('identity_weight')
		return identity_loss, weighted_identity_loss


class MotionLoss(nn.Module):
	def __init__(self, config_parser : ConfigParser, motion_extractor : MotionExtractorModule):
		super().__init__()
		self.config =\
		{
			'pose_weight': config_parser.getfloat('training.losses', 'pose_weight'),
			'expression_weight': config_parser.getfloat('training.losses', 'expression_weight')
		}
		self.motion_extractor = motion_extractor
		self.mse_loss = nn.MSELoss()

	def forward(self, target_tensor : Tensor, output_tensor : Tensor) -> Tuple[Tensor, ...]:
		target_poses, target_expression = self.get_motions(target_tensor)
		output_poses, output_expression = self.get_motions(output_tensor)
		pose_loss, weighted_pose_loss = self.calc_pose_loss(target_poses, output_poses)
		expression_loss, weighted_expression_loss = self.calc_expression_loss(target_expression, output_expression)
		return pose_loss, weighted_pose_loss, expression_loss, weighted_expression_loss

	def calc_pose_loss(self, target_poses : Tuple[Tensor, ...], output_poses : Tuple[Tensor, ...]) -> Tuple[Tensor, Tensor]:
		temp_tensors = []

		for target_pose, output_pose in zip(target_poses, output_poses):
			temp_tensor = self.mse_loss(target_pose, output_pose)
			temp_tensors.append(temp_tensor)

		pose_loss = torch.stack(temp_tensors).mean()
		weighted_pose_loss = pose_loss * self.config.get('pose_weight')
		return pose_loss, weighted_pose_loss

	def calc_expression_loss(self, target_expression : Tensor, output_expression : Tensor) -> Tuple[Tensor, Tensor]:
		expression_loss = (1 - torch.cosine_similarity(target_expression, output_expression)).mean()
		weighted_expression_loss = expression_loss * self.config.get('expression_weight')
		return expression_loss, weighted_expression_loss

	def get_motions(self, input_tensor : Tensor) -> Tuple[Tuple[Tensor, ...], Tensor]:
		input_tensor = (input_tensor + 1) * 0.5

		with torch.no_grad():
			pitch, yaw, roll, translation, expression, scale, motion_points = self.motion_extractor(input_tensor)
		rotation = torch.cat([ pitch, yaw, roll ], dim = 1)
		pose = translation, scale, rotation, motion_points
		return pose, expression


class GazeLoss(nn.Module):
	def __init__(self, config_parser : ConfigParser, gazer : GazerModule) -> None:
		super().__init__()
		self.config =\
		{
			'gaze_weight': config_parser.getfloat('training.losses', 'gaze_weight'),
			'output_size': config_parser.getint('training.model.generator', 'output_size')
		}
		self.gazer = gazer
		self.l1_loss = nn.L1Loss()

	def forward(self, target_tensor : Tensor, output_tensor : Tensor) -> Tuple[Tensor, Tensor]:
		output_pitch, output_yaw = self.detect_gaze(output_tensor)
		target_pitch, target_yaw = self.detect_gaze(target_tensor)

		pitch_loss = self.l1_loss(output_pitch, target_pitch)
		yaw_loss = self.l1_loss(output_yaw, target_yaw)

		gaze_loss = (pitch_loss + yaw_loss) * 0.5
		weighted_gaze_loss = gaze_loss * self.config.get('gaze_weight')
		return gaze_loss, weighted_gaze_loss

	def detect_gaze(self, input_tensor : Tensor) -> Gaze:
		crop_sizes = (torch.tensor([ 0.235, 0.875, 0.0625, 0.8 ]) * self.config.get('output_size')).int()
		crop_tensor = input_tensor[:, :, crop_sizes[0]:crop_sizes[1], crop_sizes[2]:crop_sizes[3]]
		crop_tensor = (crop_tensor + 1) * 0.5
		crop_tensor = transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ], std = [ 0.229, 0.224, 0.225 ])(crop_tensor)
		crop_tensor = nn.functional.interpolate(crop_tensor, size = 448, mode = 'bicubic')

		with torch.no_grad():
			pitch, yaw = self.gazer(crop_tensor)
		return pitch, yaw
