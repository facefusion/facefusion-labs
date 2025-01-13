import torch
import torch.nn as nn

from .typing import IDEmbedding, TargetAttributes, Tensor, Tuple


class AdaptiveEmbeddingIntegrationNetwork(nn.Module):
	def __init__(self, id_channels : int, num_blocks : int) -> None:
		super(AdaptiveEmbeddingIntegrationNetwork, self).__init__()
		self.encoder = UNet()
		self.generator = AADGenerator(id_channels, num_blocks)

	def forward(self, target : Tensor, source_embedding : IDEmbedding) -> Tuple[Tensor, TargetAttributes]:
		target_attributes = self.get_attributes(target)
		swap = self.generator(target_attributes, source_embedding)
		return swap, target_attributes

	def get_attributes(self, target : Tensor) -> TargetAttributes:
		return self.encoder(target)


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
		self.apply(initialize_weight)

	def forward(self, target_attributes : TargetAttributes, source_embedding : IDEmbedding) -> Tensor:
		feature_map = self.upsample(source_embedding)
		feature_map_1 = torch.nn.functional.interpolate(self.res_block_1(feature_map, target_attributes[0], source_embedding), scale_factor = 2, mode = 'bilinear', align_corners = False)
		feature_map_2 = torch.nn.functional.interpolate(self.res_block_2(feature_map_1, target_attributes[1], source_embedding), scale_factor = 2, mode = 'bilinear', align_corners = False)
		feature_map_3 = torch.nn.functional.interpolate(self.res_block_3(feature_map_2, target_attributes[2], source_embedding), scale_factor = 2, mode = 'bilinear', align_corners = False)
		feature_map_4 = torch.nn.functional.interpolate(self.res_block_4(feature_map_3, target_attributes[3], source_embedding), scale_factor = 2, mode = 'bilinear', align_corners = False)
		feature_map_5 = torch.nn.functional.interpolate(self.res_block_5(feature_map_4, target_attributes[4], source_embedding), scale_factor = 2, mode = 'bilinear', align_corners = False)
		feature_map_6 = torch.nn.functional.interpolate(self.res_block_6(feature_map_5, target_attributes[5], source_embedding), scale_factor = 2, mode = 'bilinear', align_corners = False)
		feature_map_7 = torch.nn.functional.interpolate(self.res_block_7(feature_map_6, target_attributes[6], source_embedding), scale_factor = 2, mode = 'bilinear', align_corners = False)
		output = self.res_block_8(feature_map_7, target_attributes[7], source_embedding)
		return torch.tanh(output)


class UNet(nn.Module):
	def __init__(self) -> None:
		super(UNet, self).__init__()
		self.downsampler_1 = DownSample(3, 32)
		self.downsampler_2 = DownSample(32, 64)
		self.downsampler_3 = DownSample(64, 128)
		self.downsampler_4 = DownSample(128, 256)
		self.downsampler_5 = DownSample(256, 512)
		self.downsampler_6 = DownSample(512, 1024)
		self.bottleneck = DownSample(1024, 1024)
		self.upsampler_1 = Upsample(1024, 1024)
		self.upsampler_2 = Upsample(2048, 512)
		self.upsampler_3 = Upsample(1024, 256)
		self.upsampler_4 = Upsample(512, 128)
		self.upsampler_5 = Upsample(256, 64)
		self.upsampler_6 = Upsample(128, 32)
		self.apply(initialize_weight)

	def forward(self, target : Tensor) -> TargetAttributes:
		downsample_feature_1 = self.downsampler_1(target)
		downsample_feature_2 = self.downsampler_2(downsample_feature_1)
		downsample_feature_3 = self.downsampler_3(downsample_feature_2)
		downsample_feature_4 = self.downsampler_4(downsample_feature_3)
		downsample_feature_5 = self.downsampler_5(downsample_feature_4)
		downsample_feature_6 = self.downsampler_6(downsample_feature_5)
		bottleneck_output = self.bottleneck(downsample_feature_6)
		upsample_feature_1 = self.upsampler_1(bottleneck_output, downsample_feature_6)
		upsample_feature_2 = self.upsampler_2(upsample_feature_1, downsample_feature_5)
		upsample_feature_3 = self.upsampler_3(upsample_feature_2, downsample_feature_4)
		upsample_feature_4 = self.upsampler_4(upsample_feature_3, downsample_feature_3)
		upsample_feature_5 = self.upsampler_5(upsample_feature_4, downsample_feature_2)
		upsample_feature_6 = self.upsampler_6(upsample_feature_5, downsample_feature_1)
		output = torch.nn.functional.interpolate(upsample_feature_6, scale_factor = 2, mode = 'bilinear', align_corners = False)
		return bottleneck_output, upsample_feature_1, upsample_feature_2, upsample_feature_3, upsample_feature_4, upsample_feature_5, upsample_feature_6, output


class AADLayer(nn.Module):
	def __init__(self, input_channels : int, attr_channels : int, id_channels : int) -> None:
		super(AADLayer, self).__init__()
		self.input_channels = input_channels
		self.conv_beta = nn.Conv2d(attr_channels, input_channels, kernel_size = 1, stride = 1, padding = 0, bias = True)
		self.conv_gamma = nn.Conv2d(attr_channels, input_channels, kernel_size = 1, stride = 1, padding = 0, bias = True)
		self.fc_beta = nn.Linear(id_channels, input_channels)
		self.fc_gamma = nn.Linear(id_channels, input_channels)
		self.instance_norm = nn.InstanceNorm2d(input_channels, affine = False)
		self.conv_mask = nn.Conv2d(input_channels, 1, kernel_size = 1, stride = 1, padding = 0, bias = True)

	def forward(self, feature_map : Tensor, attr_embedding : Tensor, id_embedding : IDEmbedding) -> Tensor:
		feature_map = self.instance_norm(feature_map)
		attr_gamma = self.conv_gamma(attr_embedding)
		attr_beta = self.conv_beta(attr_embedding)
		attr_modulation = attr_gamma * feature_map + attr_beta
		id_gamma = self.fc_gamma(id_embedding).reshape(feature_map.shape[0], self.input_channels, 1, 1).expand_as(feature_map)
		id_beta = self.fc_beta(id_embedding).reshape(feature_map.shape[0], self.input_channels, 1, 1).expand_as(feature_map)
		id_modulation = id_gamma * feature_map + id_beta
		feature_mask = torch.sigmoid(self.conv_mask(feature_map))
		feature_blend = (1 - feature_mask) * attr_modulation + feature_mask * id_modulation
		return feature_blend


class AddBlocksSequential(nn.Sequential):
	def forward(self, *inputs : Tuple[Tensor, Tensor, IDEmbedding]) -> Tuple[Tuple[Tensor, Tensor, IDEmbedding], ...]:
		_, attr_embedding, id_embedding = inputs

		for index, module in enumerate(self._modules.values()):
			if index % 3 == 0 and index > 0:
				inputs = (inputs, attr_embedding, id_embedding) # type:ignore[assignment]
			if type(inputs) == tuple:
				inputs = module(*inputs)
			else:
				inputs = module(inputs)
		return inputs


class AADResBlock(nn.Module):
	def __init__(self, in_channels : int, out_channels : int, attr_channels : int, id_channels : int, num_blocks : int) -> None:
		super(AADResBlock, self).__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		primary_add_blocks = []

		for i in range(num_blocks):
			intermediate_channels = in_channels if i < (num_blocks - 1) else out_channels
			primary_add_blocks.extend(
				[
					AADLayer(in_channels, attr_channels, id_channels),
					nn.ReLU(inplace = True),
					nn.Conv2d(in_channels, intermediate_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)
				])
		self.primary_add_blocks = AddBlocksSequential(*primary_add_blocks)

		if in_channels != out_channels:
			auxiliary_add_blocks = \
				[
					AADLayer(in_channels, attr_channels, id_channels),
					nn.ReLU(inplace = True),
					nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)
				]
			self.auxiliary_add_blocks = AddBlocksSequential(*auxiliary_add_blocks)

	def forward(self, feature_map : Tensor, attr_embedding : Tensor, id_embedding : IDEmbedding) -> Tensor:
		primary_feature = self.primary_add_blocks(feature_map, attr_embedding, id_embedding)

		if self.in_channels != self.out_channels:
			feature_map = self.auxiliary_add_blocks(feature_map, attr_embedding, id_embedding)
		output_feature = primary_feature + feature_map
		return output_feature


class DownSample(nn.Module):
	def __init__(self, in_channels : int, out_channels : int) -> None:
		super(DownSample, self).__init__()
		self.conv = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 4, stride = 2, padding = 1, bias = False)
		self.batch_norm = nn.BatchNorm2d(out_channels)
		self.leaky_relu = nn.LeakyReLU(0.1, inplace = True)

	def forward(self, temp : Tensor) -> Tensor:
		temp = self.conv(temp)
		temp = self.batch_norm(temp)
		temp = self.leaky_relu(temp)
		return temp


class Upsample(nn.Module):
	def __init__(self, in_channels : int, out_channels : int) -> None:
		super(Upsample, self).__init__()
		self.deconv = nn.ConvTranspose2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 4, stride = 2, padding = 1, bias = False)
		self.batch_norm = nn.BatchNorm2d(out_channels)
		self.leaky_relu = nn.LeakyReLU(0.1, inplace = True)

	def forward(self, temp : Tensor, skip_tensor : Tensor) -> Tensor:
		temp = self.deconv(temp)
		temp = self.batch_norm(temp)
		temp = self.leaky_relu(temp)
		return torch.cat((temp, skip_tensor), dim = 1)


class PixelShuffleUpsample(nn.Module):
	def __init__(self, in_channels : int, out_channels : int) -> None:
		super(PixelShuffleUpsample, self).__init__()
		self.conv = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 3, stride = 1, padding = 1)
		self.pixel_shuffle = nn.PixelShuffle(upscale_factor = 2)

	def forward(self, temp : Tensor) -> Tensor:
		temp = self.conv(temp.view(temp.shape[0], -1, 1, 1))
		temp = self.pixel_shuffle(temp)
		return temp


def initialize_weight(module : nn.Module) -> None:
	if isinstance(module, nn.Linear):
		module.weight.data.normal_(0, 0.001)
		module.bias.data.zero_()

	if isinstance(module, nn.Conv2d):
		nn.init.xavier_normal_(module.weight.data)

	if isinstance(module, nn.ConvTranspose2d):
		nn.init.xavier_normal_(module.weight.data)


if __name__ == '__main__':
	model = AdaptiveEmbeddingIntegrationNetwork(512, 2)
	src = torch.randn(1, 512)
	trg = torch.randn(1, 3, 256, 256)
	out = model(trg, src)
	print(out[0].shape)
