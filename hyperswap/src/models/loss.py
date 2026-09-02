from configparser import ConfigParser
from typing import List, Tuple

import torch
from pytorch_msssim import ssim
from torch import Tensor, nn

from ..helper import calculate_face_embedding, dilate_mask
from ..types import EmbedderModule, FaceMaskerModule, Feature, Loss, Mask


class DiscriminatorLoss(nn.Module):
	def __init__(self) -> None:
		super().__init__()

	def forward(self, discriminator_real_tensors : List[Tensor], discriminator_fake_tensors : List[Tensor]) -> Loss:
		positive_tensors = []
		negative_tensors = []

		for discriminator_real_tensor in discriminator_real_tensors:
			positive_tensor = torch.relu(1 - discriminator_real_tensor).mean(dim = [ 1, 2, 3 ])
			positive_tensors.append(positive_tensor)

		for discriminator_fake_tensor in discriminator_fake_tensors:
			negative_tensor = torch.relu(discriminator_fake_tensor + 1).mean(dim = [ 1, 2, 3 ])
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
			source_embedding = calculate_face_embedding(self.embedder, source_tensor, (0, 0, 0, 0))
			target_embedding = calculate_face_embedding(self.embedder, target_tensor, (0, 0, 0, 0))

		has_similar_identity = torch.cosine_similarity(source_embedding, target_embedding) > 0.8

		pixel_loss = torch.mean((output_tensor - target_tensor) ** 2, dim = (1, 2, 3))
		pixel_loss = (pixel_loss * has_similar_identity).mean()

		visual_loss = 1 - ssim(output_tensor, target_tensor, data_range = 2.0)
		visual_loss = (visual_loss * has_similar_identity).mean()

		reconstruction_loss = (pixel_loss + visual_loss) * 0.5
		weighted_reconstruction_loss = reconstruction_loss * self.config_reconstruction_weight
		return reconstruction_loss, weighted_reconstruction_loss


class IdentityLoss(nn.Module):
	def __init__(self, config_parser : ConfigParser, embedder : EmbedderModule) -> None:
		super().__init__()
		self.config_identity_weight = config_parser.getfloat('training.losses', 'identity_weight')
		self.embedder = embedder

	def forward(self, source_tensor : Tensor, output_tensor : Tensor) -> Tuple[Loss, Loss]:
		output_embedding = calculate_face_embedding(self.embedder, output_tensor, (30, 0, 10, 10))
		source_embedding = calculate_face_embedding(self.embedder, source_tensor, (30, 0, 10, 10))
		identity_loss = (1 - torch.cosine_similarity(source_embedding, output_embedding)).mean()
		weighted_identity_loss = identity_loss * self.config_identity_weight
		return identity_loss, weighted_identity_loss


class MaskLoss(nn.Module):
	def __init__(self, config_parser : ConfigParser, face_masker : FaceMaskerModule) -> None:
		super().__init__()
		self.config_mask_weight = config_parser.getfloat('training.losses', 'mask_weight')
		self.config_mask_factor = config_parser.getfloat('training.modifier', 'mask_factor')
		self.config_output_size = config_parser.getint('training.model.generator', 'output_size')
		self.face_masker = face_masker
		self.mse_loss = nn.MSELoss()

	def forward(self, target_tensor : Tensor, output_mask : Mask) -> Tuple[Loss, Loss]:
		target_mask = self.calculate_mask(target_tensor)

		if self.config_mask_factor > 0:
			target_mask = dilate_mask(target_mask, self.config_mask_factor)

		target_mask = target_mask.view(-1, self.config_output_size, self.config_output_size)
		output_mask = output_mask.view(-1, self.config_output_size, self.config_output_size)
		mask_loss = self.mse_loss(target_mask, output_mask)
		weighted_mask_loss = mask_loss * self.config_mask_weight
		return mask_loss, weighted_mask_loss

	def calculate_mask(self, target_tensor : Tensor) -> Tensor:
		target_tensor = torch.nn.functional.interpolate(target_tensor, (256, 256), mode = 'bilinear')
		target_tensor = (target_tensor.clip(-1, 1) + 1) * 0.5

		with torch.no_grad():
			output_tensor = self.face_masker(target_tensor)
			output_tensor = output_tensor.clamp(0, 1)
			output_tensor = torch.nn.functional.interpolate(output_tensor, (self.config_output_size, self.config_output_size), mode = 'bilinear')

		return output_tensor
