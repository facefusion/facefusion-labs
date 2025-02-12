import torch
from torch import Tensor, nn

from face_swapper.src.types import Embedding, TargetAttributes


class AADGenerator(nn.Module):
	def __init__(self, id_channels : int, num_blocks : int) -> None:
		super(AADGenerator, self).__init__()
		self.upsample = PixelShuffleUpsample(id_channels, 1024 * 4)
		self.res_block_1 = AADResBlock(1024, 1024, 1024, id_channels, num_blocks)
		self.res_block_2 = AADResBlock(1024, 1024, 2048, id_channels, num_blocks)
		self.res_block_3 = AADResBlock(1024, 1024, 1024, id_channels, num_blocks)
		self.res_block_4 = AADResBlock(1024, 512, 512, id_channels, num_blocks)
		self.res_block_5 = AADResBlock(512, 256, 256, id_channels, num_blocks)
		self.res_block_6 = AADResBlock(256, 128, 128, id_channels, num_blocks)
		self.res_block_7 = AADResBlock(128, 64, 64, id_channels, num_blocks)
		self.res_block_8 = AADResBlock(64, 3, 64, id_channels, num_blocks)

	def forward(self, target_attributes : TargetAttributes, source_embedding : Embedding) -> Tensor:
		feature_map = self.upsample(source_embedding)
		feature_map_1 = nn.functional.interpolate(self.res_block_1(feature_map, target_attributes[0], source_embedding), scale_factor = 2, mode = 'bilinear', align_corners = False)
		feature_map_2 = nn.functional.interpolate(self.res_block_2(feature_map_1, target_attributes[1], source_embedding), scale_factor = 2, mode = 'bilinear', align_corners = False)
		feature_map_3 = nn.functional.interpolate(self.res_block_3(feature_map_2, target_attributes[2], source_embedding), scale_factor = 2, mode = 'bilinear', align_corners = False)
		feature_map_4 = nn.functional.interpolate(self.res_block_4(feature_map_3, target_attributes[3], source_embedding), scale_factor = 2, mode = 'bilinear', align_corners = False)
		feature_map_5 = nn.functional.interpolate(self.res_block_5(feature_map_4, target_attributes[4], source_embedding), scale_factor = 2, mode = 'bilinear', align_corners = False)
		feature_map_6 = nn.functional.interpolate(self.res_block_6(feature_map_5, target_attributes[5], source_embedding), scale_factor = 2, mode = 'bilinear', align_corners = False)
		feature_map_7 = nn.functional.interpolate(self.res_block_7(feature_map_6, target_attributes[6], source_embedding), scale_factor = 2, mode = 'bilinear', align_corners = False)
		output = self.res_block_8(feature_map_7, target_attributes[7], source_embedding)
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
		self.prepare_primary_add_blocks(input_channels, attribute_channels, id_channels, output_channels, num_blocks)
		self.prepare_auxiliary_add_blocks(input_channels, attribute_channels, id_channels, output_channels)

	def prepare_primary_add_blocks(self, input_channels : int, attribute_channels : int, id_channels : int, output_channels : int, num_blocks : int) -> None:
		primary_add_blocks = []

		for index in range(num_blocks):
			intermediate_channels = input_channels if index < (num_blocks - 1) else output_channels
			primary_add_blocks.extend(
				[
					AADLayer(input_channels, attribute_channels, id_channels),
					nn.ReLU(inplace = True),
					nn.Conv2d(input_channels, intermediate_channels, kernel_size = 3, padding = 1, bias = False)
				]
			)
		self.primary_add_blocks = AADSequential(*primary_add_blocks)

	def prepare_auxiliary_add_blocks(self, input_channels : int, attribute_channels : int, id_channels : int, output_channels : int) -> None:
		if input_channels > output_channels:
			auxiliary_add_blocks = AADSequential(
				AADLayer(input_channels, attribute_channels, id_channels),
				nn.ReLU(inplace = True),
				nn.Conv2d(input_channels, output_channels, kernel_size = 3, padding = 1, bias = False)
			)
			self.auxiliary_add_blocks = auxiliary_add_blocks

	def forward(self, feature_map : Tensor, attribute_embedding : Embedding, id_embedding : Embedding) -> Tensor:
		primary_feature = self.primary_add_blocks(feature_map, attribute_embedding, id_embedding)

		if self.input_channels > self.output_channels:
			feature_map = self.auxiliary_add_blocks(feature_map, attribute_embedding, id_embedding)

		output_feature = primary_feature + feature_map
		return output_feature


class PixelShuffleUpsample(nn.Module):
	def __init__(self, input_channels : int, output_channels : int) -> None:
		super(PixelShuffleUpsample, self).__init__()
		self.conv = nn.Conv2d(in_channels = input_channels, out_channels = output_channels, kernel_size = 3, padding = 1)
		self.pixel_shuffle = nn.PixelShuffle(upscale_factor = 2)

	def forward(self, input_tensor : Tensor) -> Tensor:
		temp_tensor = self.conv(input_tensor.view(input_tensor.shape[0], -1, 1, 1))
		temp_tensor = self.pixel_shuffle(temp_tensor)
		return temp_tensor
