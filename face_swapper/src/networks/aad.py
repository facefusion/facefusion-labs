from configparser import ConfigParser
from typing import Tuple

import torch
from torch import Tensor, nn

from ..types import Embedding, Feature


class AAD(nn.Module):
	def __init__(self, config_parser : ConfigParser) -> None:
		super().__init__()
		self.config_source_channels = config_parser.getint('training.model.generator', 'source_channels')
		self.config_output_channels = config_parser.getint('training.model.generator', 'output_channels')
		self.config_output_size = config_parser.getint('training.model.generator', 'output_size')
		self.config_num_blocks = config_parser.getint('training.model.generator', 'num_blocks')
		self.pixel_shuffle_up_sample = PixelShuffleUpSample(self.config_source_channels, self.config_output_channels)
		self.layers = self.create_layers()

	def create_layers(self) -> nn.ModuleList:
		layers = nn.ModuleList()

		if self.config_output_size == 128:
			layers.extend(
			[
				AdaptiveFeatureModulation(512, 512, 512, self.config_source_channels, self.config_num_blocks),
				AdaptiveFeatureModulation(512, 512, 1024, self.config_source_channels, self.config_num_blocks),
				AdaptiveFeatureModulation(512, 512, 512, self.config_source_channels, self.config_num_blocks)
			])

		if self.config_output_size == 256:
			layers.extend(
			[
				AdaptiveFeatureModulation(1024, 1024, 1024, self.config_source_channels, self.config_num_blocks),
				AdaptiveFeatureModulation(1024, 1024, 2048, self.config_source_channels, self.config_num_blocks),
				AdaptiveFeatureModulation(1024, 1024, 1024, self.config_source_channels, self.config_num_blocks),
				AdaptiveFeatureModulation(1024, 512, 512, self.config_source_channels, self.config_num_blocks)
			])

		if self.config_output_size == 512:
			layers.extend(
			[
				AdaptiveFeatureModulation(2048, 2048, 2048, self.config_source_channels, self.config_num_blocks),
				AdaptiveFeatureModulation(2048, 2048, 4096, self.config_source_channels, self.config_num_blocks),
				AdaptiveFeatureModulation(2048, 2048, 2048, self.config_source_channels, self.config_num_blocks),
				AdaptiveFeatureModulation(2048, 1024, 1024, self.config_source_channels, self.config_num_blocks),
				AdaptiveFeatureModulation(1024, 512, 512, self.config_source_channels, self.config_num_blocks)
			])

		layers.extend(
		[
			AdaptiveFeatureModulation(512, 256, 256, self.config_source_channels, self.config_num_blocks),
			AdaptiveFeatureModulation(256, 128, 128, self.config_source_channels, self.config_num_blocks),
			AdaptiveFeatureModulation(128, 64, 64, self.config_source_channels, self.config_num_blocks),
			AdaptiveFeatureModulation(64, 3, 64, self.config_source_channels, self.config_num_blocks)
		])

		return layers

	def forward(self, source_embedding : Embedding, target_features : Tuple[Feature, ...]) -> Tensor:
		temp_tensors = self.pixel_shuffle_up_sample(source_embedding)

		for index, layer in enumerate(self.layers[:-1]):
			target_feature = target_features[index]
			temp_tensor = layer(temp_tensors, source_embedding, target_feature)
			temp_tensors = nn.functional.interpolate(temp_tensor, scale_factor = 2, mode = 'bilinear', align_corners = False)

		target_feature = target_features[-1]
		temp_tensors = self.layers[-1](temp_tensors, source_embedding, target_feature)
		output_tensor = torch.tanh(temp_tensors)
		return output_tensor


class AdaptiveFeatureModulation(nn.Module):
	def __init__(self, input_channels : int, output_channels : int, target_channels : int, source_channels : int, num_blocks : int) -> None:
		super().__init__()
		self.context_input_channels = input_channels
		self.context_output_channels = output_channels
		self.context_target_channels = target_channels
		self.context_source_channels = source_channels
		self.context_num_blocks = num_blocks
		self.primary_layers = self.create_primary_layers()
		self.shortcut_layers = self.create_shortcut_layers()

	def create_primary_layers(self) -> nn.ModuleList:
		primary_layers = nn.ModuleList()

		for index in range(self.context_num_blocks):
			primary_layers.extend(
			[
				FeatureModulation(self.context_input_channels, self.context_target_channels, self.context_source_channels),
				nn.ReLU(inplace = True)
			])

			if index < self.context_num_blocks - 1:
				primary_layers.append(nn.Conv2d(self.context_input_channels, self.context_input_channels, kernel_size = 3, padding = 1, bias = False))
			else:
				primary_layers.append(nn.Conv2d(self.context_input_channels, self.context_output_channels, kernel_size = 3, padding = 1, bias = False))

		return primary_layers

	def create_shortcut_layers(self) -> nn.ModuleList:
		shortcut_layers = nn.ModuleList()

		if self.context_input_channels > self.context_output_channels:
			shortcut_layers.extend(
			[
				FeatureModulation(self.context_input_channels, self.context_target_channels, self.context_source_channels),
				nn.ReLU(inplace = True),
				nn.Conv2d(self.context_input_channels, self.context_output_channels, kernel_size = 3, padding = 1, bias = False)
			])

		return shortcut_layers

	def forward(self, input_tensor : Tensor, source_embedding : Embedding, target_feature : Feature) -> Tensor:
		primary_tensor = input_tensor

		for primary_layer in self.primary_layers:
			if isinstance(primary_layer, FeatureModulation):
				primary_tensor = primary_layer(primary_tensor, source_embedding, target_feature)
			else:
				primary_tensor = primary_layer(primary_tensor)

		if self.context_input_channels > self.context_output_channels:
			shortcut_tensor = input_tensor

			for shortcut_layer in self.shortcut_layers:
				if isinstance(shortcut_layer, FeatureModulation):
					shortcut_tensor = shortcut_layer(shortcut_tensor, source_embedding, target_feature)
				else:
					shortcut_tensor = shortcut_layer(shortcut_tensor)

			input_tensor = shortcut_tensor

		return primary_tensor + input_tensor


class FeatureModulation(nn.Module):
	def __init__(self, input_channels : int, target_channels : int, source_channels : int) -> None:
		super().__init__()
		self.context_input_channels = input_channels
		self.conv1 = nn.Conv2d(target_channels, input_channels, kernel_size = 1)
		self.conv2 = nn.Conv2d(target_channels, input_channels, kernel_size = 1)
		self.conv3 = nn.Conv2d(input_channels, 1, kernel_size = 1)
		self.linear1 = nn.Linear(source_channels, input_channels)
		self.linear2 = nn.Linear(source_channels, input_channels)
		self.instance_norm = nn.InstanceNorm2d(input_channels)

	def forward(self, input_tensor : Tensor, source_embedding : Embedding, target_feature : Feature) -> Tensor:
		temp_tensor = self.instance_norm(input_tensor)

		source_scale = self.linear2(source_embedding).reshape(temp_tensor.shape[0], self.context_input_channels, 1, 1).expand_as(temp_tensor)
		source_shift = self.linear1(source_embedding).reshape(temp_tensor.shape[0], self.context_input_channels, 1, 1).expand_as(temp_tensor)
		source_modulation = source_scale * temp_tensor + source_shift

		target_scale = self.conv1(target_feature)
		target_shift = self.conv2(target_feature)
		target_modulation = target_scale * temp_tensor + target_shift

		temp_mask = torch.sigmoid(self.conv3(temp_tensor))
		output_tensor = (1 - temp_mask) * target_modulation + temp_mask * source_modulation
		return output_tensor


class PixelShuffleUpSample(nn.Module):
	def __init__(self, input_channels : int, output_channels : int) -> None:
		super().__init__()
		self.sequences = self.create_sequences(input_channels, output_channels)

	@staticmethod
	def create_sequences(input_channels : int, output_channels : int) -> nn.Sequential:
		return nn.Sequential(
			nn.Conv2d(input_channels, output_channels, kernel_size = 3, padding = 1),
			nn.PixelShuffle(upscale_factor = 2)
		)

	def forward(self, input_tensor : Tensor) -> Tensor:
		temp_tensor = input_tensor.view(input_tensor.shape[0], -1, 1, 1)
		output_tensor = self.sequences(temp_tensor)
		return output_tensor
