from configparser import ConfigParser
from typing import List, Tuple

import torch
from pytorch_msssim import ssim
from torch import Tensor, nn
from torchvision import transforms

from ..helper import calc_embedding
from ..types import EmbedderModule, FaceMaskerModule, Feature, GazerModule, Loss, Mask


class DiscriminatorLoss(nn.Module):
	def __init__(self) -> None:
		super().__init__()

	def forward(self, discriminator_source_tensors : List[Tensor], discriminator_output_tensors : List[Tensor]) -> Loss:
		positive_tensors = []
		negative_tensors = []

		for discriminator_source_tensor in discriminator_source_tensors:
			positive_tensor = torch.relu(1 - discriminator_source_tensor).mean(dim = [ 1, 2, 3 ])
			positive_tensors.append(positive_tensor)

		for discriminator_output_tensor in discriminator_output_tensors:
			negative_tensor = torch.relu(discriminator_output_tensor + 1).mean(dim = [ 1, 2, 3 ])
			negative_tensors.append(negative_tensor)

		positive_loss = torch.stack(positive_tensors).mean()
		negative_loss = torch.stack(negative_tensors).mean()
		discriminator_loss = (positive_loss + negative_loss) * 0.5
		return discriminator_loss


class AdversarialLoss(nn.Module):
	def __init__(self, config_parser : ConfigParser) -> None:
		super().__init__()
		self.config_adversarial_weight = config_parser.getfloat('training.losses', 'adversarial_weight')

	def forward(self, discriminator_output_tensors : List[Tensor]) -> Tuple[Loss, Loss]:
		temp_tensors = []

		for discriminator_output_tensor in discriminator_output_tensors:
			temp_tensor = torch.relu(1 - discriminator_output_tensor).mean(dim = [ 1, 2, 3 ]).mean()
			temp_tensors.append(temp_tensor)

		adversarial_loss = torch.stack(temp_tensors).mean()
		weighted_adversarial_loss = adversarial_loss * self.config_adversarial_weight
		return adversarial_loss, weighted_adversarial_loss


class CycleLoss(nn.Module):
	def __init__(self, config_parser : ConfigParser) -> None:
		super().__init__()
		self.config_batch_size = config_parser.getint('training.loader', 'batch_size')
		self.config_cycle_weight = config_parser.getfloat('training.losses', 'cycle_weight')
		self.l1_loss = nn.L1Loss()

	def forward(self, target_tensor : Tensor, cycle_tensor : Tensor, target_features : Tuple[Feature, ...], cycle_features : Tuple[Feature, ...]) -> Tuple[Loss, Loss]:
		temp_tensors = []

		for target_feature, output_feature in zip(target_features, cycle_features):
			temp_tensor = torch.mean(torch.pow(output_feature - target_feature, 2).reshape(self.config_batch_size, -1), dim = 1).mean()
			temp_tensors.append(temp_tensor)

		feature_loss = torch.stack(temp_tensors).mean()
		reconstruction_loss = self.l1_loss(target_tensor, cycle_tensor)
		cycle_loss = (feature_loss + reconstruction_loss) * 0.5
		weighted_feature_loss = cycle_loss * self.config_cycle_weight
		return cycle_loss, weighted_feature_loss


class FeatureLoss(nn.Module):
	def __init__(self, config_parser : ConfigParser) -> None:
		super().__init__()
		self.config_batch_size = config_parser.getint('training.loader', 'batch_size')
		self.config_feature_weight = config_parser.getfloat('training.losses', 'feature_weight')

	def forward(self, target_features : Tuple[Feature, ...], output_features : Tuple[Feature, ...]) -> Tuple[Loss, Loss]:
		temp_tensors = []

		for target_feature, output_feature in zip(target_features, output_features):
			temp_tensor = torch.mean(torch.pow(output_feature - target_feature, 2).reshape(self.config_batch_size, -1), dim = 1).mean()
			temp_tensors.append(temp_tensor)

		feature_loss = torch.stack(temp_tensors).mean() * 0.5
		weighted_feature_loss = feature_loss * self.config_feature_weight
		return feature_loss, weighted_feature_loss


class ReconstructionLoss(nn.Module):
	def __init__(self, config_parser : ConfigParser, embedder : EmbedderModule) -> None:
		super().__init__()
		self.config_reconstruction_weight = config_parser.getfloat('training.losses', 'reconstruction_weight')
		self.embedder = embedder
		self.mse_loss = nn.MSELoss()

	def forward(self, source_tensor : Tensor, target_tensor : Tensor, output_tensor : Tensor) -> Tuple[Loss, Loss]:
		with torch.no_grad():
			source_embedding = calc_embedding(self.embedder, source_tensor, (0, 0, 0, 0))
			target_embedding = calc_embedding(self.embedder, target_tensor, (0, 0, 0, 0))

		has_similar_identity = torch.cosine_similarity(source_embedding, target_embedding) > 0.8

		reconstruction_loss = torch.mean((source_tensor - target_tensor) ** 2, dim = (1, 2, 3))
		reconstruction_loss = (reconstruction_loss * has_similar_identity).mean() * 0.5

		data_range = float(torch.max(output_tensor) - torch.min(output_tensor))
		visual_loss = 1 - ssim(output_tensor, target_tensor, data_range = data_range).mean()
		reconstruction_loss = (reconstruction_loss + visual_loss) * 0.5
		weighted_reconstruction_loss = reconstruction_loss * self.config_reconstruction_weight
		return reconstruction_loss, weighted_reconstruction_loss


class IdentityLoss(nn.Module):
	def __init__(self, config_parser : ConfigParser, embedder : EmbedderModule) -> None:
		super().__init__()
		self.config_identity_weight = config_parser.getfloat('training.losses', 'identity_weight')
		self.embedder = embedder

	def forward(self, source_tensor : Tensor, output_tensor : Tensor) -> Tuple[Loss, Loss]:
		output_embedding = calc_embedding(self.embedder, output_tensor, (30, 0, 10, 10))
		source_embedding = calc_embedding(self.embedder, source_tensor, (30, 0, 10, 10))
		identity_loss = (1 - torch.cosine_similarity(source_embedding, output_embedding)).mean()
		weighted_identity_loss = identity_loss * self.config_identity_weight
		return identity_loss, weighted_identity_loss


class GazeLoss(nn.Module):
	def __init__(self, config_parser : ConfigParser, gazer : GazerModule) -> None:
		super().__init__()
		self.config_gaze_weight = config_parser.getfloat('training.losses', 'gaze_weight')
		self.config_output_size = config_parser.getint('training.model.generator', 'output_size')
		self.gazer = gazer
		self.l1_loss = nn.L1Loss()

	def forward(self, target_tensor : Tensor, output_tensor : Tensor) -> Tuple[Loss, Loss]:
		output_pitch, output_yaw = self.detect_gaze(output_tensor)
		target_pitch, target_yaw = self.detect_gaze(target_tensor)

		pitch_loss = self.l1_loss(output_pitch, target_pitch)
		yaw_loss = self.l1_loss(output_yaw, target_yaw)

		gaze_loss = (pitch_loss + yaw_loss) * 0.5
		weighted_gaze_loss = gaze_loss * self.config_gaze_weight
		return gaze_loss, weighted_gaze_loss

	def detect_gaze(self, input_tensor : Tensor) -> Tuple[Tensor, Tensor]:
		crop_sizes = (torch.tensor([ 0.235, 0.875, 0.0625, 0.8 ]) * self.config_output_size).int()
		crop_tensor = input_tensor[:, :, crop_sizes[0]:crop_sizes[1], crop_sizes[2]:crop_sizes[3]]
		crop_tensor = (crop_tensor + 1) * 0.5
		crop_tensor = transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ], std = [ 0.229, 0.224, 0.225 ])(crop_tensor)
		crop_tensor = nn.functional.interpolate(crop_tensor, size = 448, mode = 'bicubic')

		with torch.no_grad():
			pitch, yaw = self.gazer(crop_tensor)

		return pitch, yaw


class MaskLoss(nn.Module):
	def __init__(self, config_parser : ConfigParser, face_masker : FaceMaskerModule) -> None:
		super().__init__()
		self.config_mask_weight = config_parser.getfloat('training.losses', 'mask_weight')
		self.config_output_size = config_parser.getint('training.model.generator', 'output_size')
		self.face_masker = face_masker
		self.mse_loss = nn.MSELoss()

	def forward(self, target_tensor : Tensor, output_mask : Mask) -> Tuple[Loss, Loss]:
		target_mask = self.calc_mask(target_tensor)
		target_mask = target_mask.view(-1, self.config_output_size, self.config_output_size)
		output_mask = output_mask.view(-1, self.config_output_size, self.config_output_size)
		mask_loss = self.mse_loss(target_mask, output_mask)
		weighted_mask_loss = mask_loss * self.config_mask_weight
		return mask_loss, weighted_mask_loss

	def calc_mask(self, target_tensor : Tensor) -> Tensor:
		target_tensor = torch.nn.functional.interpolate(target_tensor, (256, 256), mode = 'bilinear')
		target_tensor = (target_tensor.clip(-1, 1) + 1) * 0.5

		with torch.no_grad():
			output_tensor = self.face_masker(target_tensor)
			output_tensor = output_tensor.clamp(0, 1)
			output_tensor = torch.nn.functional.interpolate(output_tensor, (self.config_output_size, self.config_output_size), mode = 'bilinear')

		return output_tensor
