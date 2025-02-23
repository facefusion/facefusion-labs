import torch
from torch import Tensor, nn

from ..types import Attributes, Embedding


class AAD(nn.Module):
	def __init__(self, identity_channels : int, output_channels : int, num_blocks : int) -> None:
		super().__init__()
		self.pixel_shuffle_up_sample = PixelShuffleUpSample(identity_channels, output_channels)
		self.layers = self.create_layers(identity_channels, num_blocks)

	@staticmethod
	def create_layers(identity_channels : int, num_blocks : int) -> nn.ModuleList:
		return nn.ModuleList(
		[
			AADResBlock(1024, 1024, 1024, identity_channels, num_blocks),
			AADResBlock(1024, 1024, 2048, identity_channels, num_blocks),
			AADResBlock(1024, 1024, 1024, identity_channels, num_blocks),
			AADResBlock(1024, 512, 512, identity_channels, num_blocks),
			AADResBlock(512, 256, 256, identity_channels, num_blocks),
			AADResBlock(256, 128, 128, identity_channels, num_blocks),
			AADResBlock(128, 64, 64, identity_channels, num_blocks),
			AADResBlock(64, 3, 64, identity_channels, num_blocks)
		])

	def forward(self, source_embedding : Embedding, target_attributes : Attributes) -> Tensor:
		temp_tensors = self.pixel_shuffle_up_sample(source_embedding)

		for index, layer in enumerate(self.layers[:-1]):
			temp_tensor = layer(temp_tensors, target_attributes[index], source_embedding)
			temp_tensors = nn.functional.interpolate(temp_tensor, scale_factor = 2, mode = 'bilinear', align_corners = False)

		temp_tensors = self.layers[-1](temp_tensors, target_attributes[-1], source_embedding)
		output_tensor = torch.tanh(temp_tensors)
		return output_tensor


class AADResBlock(nn.Module):
	def __init__(self, input_channels : int, output_channels : int, attribute_channels : int, identity_channels : int, num_blocks : int) -> None:
		super().__init__()
		self.input_channels = input_channels
		self.output_channels = output_channels
		self.prepare_primary_add_blocks(input_channels, attribute_channels, identity_channels, output_channels, num_blocks)
		self.prepare_auxiliary_add_blocks(input_channels, attribute_channels, identity_channels, output_channels)

	def prepare_primary_add_blocks(self, input_channels : int, attribute_channels : int, identity_channels : int, output_channels : int, num_blocks : int) -> None:
		primary_add_blocks = []

		for index in range(num_blocks):
			intermediate_channels = input_channels if index < (num_blocks - 1) else output_channels
			primary_add_blocks.extend(
				[
					AADLayer(input_channels, attribute_channels, identity_channels),
					nn.ReLU(inplace = True),
					nn.Conv2d(input_channels, intermediate_channels, kernel_size = 3, padding = 1, bias = False)
				]
			)
		self.primary_add_blocks = AADSequential(*primary_add_blocks)

	def prepare_auxiliary_add_blocks(self, input_channels : int, attribute_channels : int, identity_channels : int, output_channels : int) -> None:
		if input_channels > output_channels:
			auxiliary_add_blocks = AADSequential(
				AADLayer(input_channels, attribute_channels, identity_channels),
				nn.ReLU(inplace = True),
				nn.Conv2d(input_channels, output_channels, kernel_size = 3, padding = 1, bias = False)
			)
			self.auxiliary_add_blocks = auxiliary_add_blocks

	def forward(self, feature_map : Tensor, attribute_embedding : Embedding, identity_embedding : Embedding) -> Tensor:
		primary_feature = self.primary_add_blocks(feature_map, attribute_embedding, identity_embedding)

		if self.input_channels > self.output_channels:
			feature_map = self.auxiliary_add_blocks(feature_map, attribute_embedding, identity_embedding)

		output_feature = primary_feature + feature_map
		return output_feature


class AADSequential(nn.Module):
	def __init__(self, *args : nn.Module) -> None:
		super().__init__()
		self.layers = nn.ModuleList(args)

	def forward(self, feature_map : Tensor, attribute_embedding : Embedding, identity_embedding : Embedding) -> Tensor:
		for layer in self.layers:
			if isinstance(layer, AADLayer):
				feature_map = layer(feature_map, attribute_embedding, identity_embedding)
			else:
				feature_map = layer(feature_map)
		return feature_map


class AADLayer(nn.Module):
	def __init__(self, input_channels : int, attribute_channels : int, identity_channels : int) -> None:
		super().__init__()
		self.input_channels = input_channels
		self.conv_beta = nn.Conv2d(attribute_channels, input_channels, kernel_size = 1)
		self.conv_gamma = nn.Conv2d(attribute_channels, input_channels, kernel_size = 1)
		self.fc_beta = nn.Linear(identity_channels, input_channels)
		self.fc_gamma = nn.Linear(identity_channels, input_channels)
		self.instance_norm = nn.InstanceNorm2d(input_channels)
		self.conv_mask = nn.Conv2d(input_channels, 1, kernel_size = 1)

	def forward(self, feature_map : Tensor, attribute_embedding : Embedding, identity_embedding : Embedding) -> Tensor:
		feature_map = self.instance_norm(feature_map)
		gamma_attribute = self.conv_gamma(attribute_embedding)
		beta_attribute = self.conv_beta(attribute_embedding)
		attribute_modulation = gamma_attribute * feature_map + beta_attribute
		identity_gamma = self.fc_gamma(identity_embedding).reshape(feature_map.shape[0], self.input_channels, 1, 1).expand_as(feature_map)
		identity_beta = self.fc_beta(identity_embedding).reshape(feature_map.shape[0], self.input_channels, 1, 1).expand_as(feature_map)
		identity_modulation = identity_gamma * feature_map + identity_beta
		feature_mask = torch.sigmoid(self.conv_mask(feature_map))
		feature_blend = (1 - feature_mask) * attribute_modulation + feature_mask * identity_modulation
		return feature_blend


class PixelShuffleUpSample(nn.Module):
	def __init__(self, input_channels : int, output_channels : int) -> None:
		super().__init__()
		self.conv = nn.Conv2d(input_channels, output_channels, kernel_size = 3, padding = 1)
		self.pixel_shuffle = nn.PixelShuffle(upscale_factor = 2)

	def forward(self, input_tensor : Tensor) -> Tensor:
		temp_tensor = self.conv(input_tensor.view(input_tensor.shape[0], -1, 1, 1))
		temp_tensor = self.pixel_shuffle(temp_tensor)
		return temp_tensor
