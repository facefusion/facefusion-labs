import torch
from torch import Tensor, nn

from ..types import Embedding, TargetAttributes


class AADGenerator(nn.Module):
	def __init__(self, id_channels : int, num_blocks : int) -> None:
		super(AADGenerator, self).__init__()
		output_channels = 1024 * 4
		self.pixel_shuffle_up = PixelShuffleUp(id_channels, output_channels)
		self.add_res_blocks = self.create_add_res_blocks(id_channels, num_blocks)

	@staticmethod
	def create_add_res_blocks(id_channels : int, num_blocks : int) -> nn.ModuleList:
		return nn.ModuleList(
		[
			AADResBlock(1024, 1024, 1024, id_channels, num_blocks),
			AADResBlock(1024, 1024, 2048, id_channels, num_blocks),
			AADResBlock(1024, 1024, 1024, id_channels, num_blocks),
			AADResBlock(1024, 512, 512, id_channels, num_blocks),
			AADResBlock(512, 256, 256, id_channels, num_blocks),
			AADResBlock(256, 128, 128, id_channels, num_blocks),
			AADResBlock(128, 64, 64, id_channels, num_blocks),
			AADResBlock(64, 3, 64, id_channels, num_blocks)
		])

	def forward(self, target_attributes : TargetAttributes, source_embedding : Embedding) -> torch.Tensor:
		feature_map = self.pixel_shuffle_up(source_embedding)

		for index, add_res_block in enumerate(self.add_res_blocks[:-1]):
			feature = add_res_block(feature_map, target_attributes[index], source_embedding)
			feature_map = nn.functional.interpolate(feature, scale_factor = 2, mode = 'bilinear', align_corners = False)

		output = self.add_res_blocks[-1](feature_map, target_attributes[-1], source_embedding)
		return torch.tanh(output)


class AADLayer(nn.Module):
	def __init__(self, input_channels : int, attr_channels : int, id_channels : int) -> None:
		super(AADLayer, self).__init__()
		self.input_channels = input_channels
		self.conv_beta = nn.Conv2d(attr_channels, input_channels, kernel_size = 1)
		self.conv_gamma = nn.Conv2d(attr_channels, input_channels, kernel_size = 1)
		self.fc_beta = nn.Linear(id_channels, input_channels)
		self.fc_gamma = nn.Linear(id_channels, input_channels)
		self.instance_norm = nn.InstanceNorm2d(input_channels)
		self.conv_mask = nn.Conv2d(input_channels, 1, kernel_size = 1)

	def forward(self, feature_map : Tensor, attribute_embedding : Embedding, id_embedding : Embedding) -> Tensor:
		feature_map = self.instance_norm(feature_map)
		gamma_attribute = self.conv_gamma(attribute_embedding)
		beta_attribute = self.conv_beta(attribute_embedding)
		attribute_modulation = gamma_attribute * feature_map + beta_attribute
		id_gamma = self.fc_gamma(id_embedding).reshape(feature_map.shape[0], self.input_channels, 1, 1).expand_as(feature_map)
		id_beta = self.fc_beta(id_embedding).reshape(feature_map.shape[0], self.input_channels, 1, 1).expand_as(feature_map)
		id_modulation = id_gamma * feature_map + id_beta
		feature_mask = torch.sigmoid(self.conv_mask(feature_map))
		feature_blend = (1 - feature_mask) * attribute_modulation + feature_mask * id_modulation
		return feature_blend


class AADSequential(nn.Module):
	def __init__(self, *args : nn.Module) -> None:
		super(AADSequential, self).__init__()
		self.layers = nn.ModuleList(args)

	def forward(self, feature_map : Tensor, attribute_embedding : Embedding, id_embedding : Embedding) -> Tensor:
		for layer in self.layers:
			if isinstance(layer, AADLayer):
				feature_map = layer(feature_map, attribute_embedding, id_embedding)
			else:
				feature_map = layer(feature_map)
		return feature_map


class AADResBlock(nn.Module):
	def __init__(self, input_channels : int, output_channels : int, attribute_channels : int, id_channels : int, num_blocks : int) -> None:
		super(AADResBlock, self).__init__()
		self.input_channels = input_channels
		self.output_channels = output_channels
		self.primary_add_blocks = self.prepare_primary_add_blocks(input_channels, attribute_channels, id_channels, output_channels, num_blocks)
		self.auxiliary_add_blocks = self.prepare_auxiliary_add_blocks(input_channels, attribute_channels, id_channels, output_channels)

	@staticmethod
	def prepare_primary_add_blocks(input_channels : int, attribute_channels : int, id_channels : int, output_channels : int, num_blocks : int) -> AADSequential:
		primary_add_blocks = []

		for index in range(num_blocks):
			intermediate_channels = input_channels if index < (num_blocks - 1) else output_channels
			primary_add_blocks.extend(
			[
				AADLayer(input_channels, attribute_channels, id_channels),
				nn.ReLU(inplace = True),
				nn.Conv2d(input_channels, intermediate_channels, kernel_size = 3, padding = 1, bias = False)
			])
		return AADSequential(*primary_add_blocks)

	@staticmethod
	def prepare_auxiliary_add_blocks(input_channels : int, attribute_channels : int, id_channels : int, output_channels : int) -> AADSequential:
		return AADSequential(
			AADLayer(input_channels, attribute_channels, id_channels),
			nn.ReLU(inplace = True),
			nn.Conv2d(input_channels, output_channels, kernel_size = 3, padding = 1, bias = False)
		)

	def forward(self, feature_map : Tensor, attribute_embedding : Embedding, id_embedding : Embedding) -> Tensor:
		primary_feature = self.primary_add_blocks(feature_map, attribute_embedding, id_embedding)

		if self.input_channels > self.output_channels:
			feature_map = self.auxiliary_add_blocks(feature_map, attribute_embedding, id_embedding)

		output_feature = primary_feature + feature_map
		return output_feature


class PixelShuffleUp(nn.Module):
	def __init__(self, input_channels : int, output_channels : int) -> None:
		super(PixelShuffleUp, self).__init__()
		self.conv = nn.Conv2d(in_channels = input_channels, out_channels = output_channels, kernel_size = 3, padding = 1)
		self.pixel_shuffle = nn.PixelShuffle(upscale_factor = 2)

	def forward(self, input_tensor : Tensor) -> Tensor:
		temp_tensor = self.conv(input_tensor.view(input_tensor.shape[0], -1, 1, 1))
		temp_tensor = self.pixel_shuffle(temp_tensor)
		return temp_tensor
