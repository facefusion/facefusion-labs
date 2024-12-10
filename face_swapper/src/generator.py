import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveEmbeddingIntegrationNetwork(nn.Module):
	def __init__(self, id_channels=512, num_blocks=2):
		super(AdaptiveEmbeddingIntegrationNetwork, self).__init__()
		self.encoder = UNet()
		self.generator = AADGenerator(id_channels, num_blocks)

	def forward(self, target, source_embedding):
		target_attributes = self.get_attributes(target)
		swap = self.generator(target_attributes, source_embedding)
		return swap, target_attributes

	def get_attributes(self, target):
		return self.encoder(target)


class AADGenerator(nn.Module):
	def __init__(self, id_channels=512, num_blocks=2):
		super(AADGenerator, self).__init__()
		self.upsample = Upsample(id_channels, 1024 * 4)
		self.AADBlk1 = AAD_ResBlk(1024, 1024, 1024, id_channels, num_blocks)
		self.AADBlk2 = AAD_ResBlk(1024, 1024, 2048, id_channels, num_blocks)
		self.AADBlk3 = AAD_ResBlk(1024, 1024, 1024, id_channels, num_blocks)
		self.AADBlk4 = AAD_ResBlk(1024, 512, 512, id_channels, num_blocks)
		self.AADBlk5 = AAD_ResBlk(512, 256, 256, id_channels, num_blocks)
		self.AADBlk6 = AAD_ResBlk(256, 128, 128, id_channels, num_blocks)
		self.AADBlk7 = AAD_ResBlk(128, 64, 64, id_channels, num_blocks)
		self.AADBlk8 = AAD_ResBlk(64, 3, 64, id_channels, num_blocks)
		self.apply(initialize_weight)

	def forward(self, target_attributes, source_embedding):
		feature_map = self.upsample(source_embedding)
		feature_map_1 = F.interpolate(self.AADBlk1(feature_map, target_attributes[0], source_embedding), scale_factor=2, mode='bilinear', align_corners=False)
		feature_map_2 = F.interpolate(self.AADBlk2(feature_map_1, target_attributes[1], source_embedding), scale_factor=2, mode='bilinear', align_corners=False)
		feature_map_3 = F.interpolate(self.AADBlk3(feature_map_2, target_attributes[2], source_embedding), scale_factor=2, mode='bilinear', align_corners=False)
		feature_map_4 = F.interpolate(self.AADBlk4(feature_map_3, target_attributes[3], source_embedding), scale_factor=2, mode='bilinear', align_corners=False)
		feature_map_5 = F.interpolate(self.AADBlk5(feature_map_4, target_attributes[4], source_embedding), scale_factor=2, mode='bilinear', align_corners=False)
		feature_map_6 = F.interpolate(self.AADBlk6(feature_map_5, target_attributes[5], source_embedding), scale_factor=2, mode='bilinear', align_corners=False)
		feature_map_7 = F.interpolate(self.AADBlk7(feature_map_6, target_attributes[6], source_embedding), scale_factor=2, mode='bilinear', align_corners=False)
		output = self.AADBlk8(feature_map_7, target_attributes[7], source_embedding)
		return torch.tanh(output)


class UNet(nn.Module):
	def __init__(self):
		super(UNet, self).__init__()
		self.downsampler_1 = Conv4x4(3, 32)
		self.downsampler_2 = Conv4x4(32, 64)
		self.downsampler_3 = Conv4x4(64, 128)
		self.downsampler_4 = Conv4x4(128, 256)
		self.downsampler_5 = Conv4x4(256, 512)
		self.downsampler_6 = Conv4x4(512, 1024)

		self.bottleneck = Conv4x4(1024, 1024)

		self.upsampler_1 = DeConv4x4(1024, 1024)
		self.upsampler_2 = DeConv4x4(2048, 512)
		self.upsampler_3 = DeConv4x4(1024, 256)
		self.upsampler_4 = DeConv4x4(512, 128)
		self.upsampler_5 = DeConv4x4(256, 64)
		self.upsampler_6 = DeConv4x4(128, 32)
		self.apply(initialize_weight)

	def forward(self, input_tensor):
		downsample_feature_1 = self.downsampler_1(input_tensor)
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

		output = F.interpolate(upsample_feature_6, scale_factor=2, mode='bilinear', align_corners=False)

		return bottleneck_output, upsample_feature_1, upsample_feature_2, upsample_feature_3, upsample_feature_4, upsample_feature_5, upsample_feature_6, output


class AADLayer(nn.Module):
	def __init__(self, input_channels, attr_channels, id_channels):
		super(AADLayer, self).__init__()
		self.attr_channels = attr_channels
		self.id_channels = id_channels
		self.input_channels = input_channels

		self.conv_gamma = nn.Conv2d(attr_channels, input_channels, kernel_size=1, stride=1, padding=0, bias=True)
		self.conv_beta = nn.Conv2d(attr_channels, input_channels, kernel_size=1, stride=1, padding=0, bias=True)
		self.fc_gamma = nn.Linear(id_channels, input_channels)
		self.fc_beta = nn.Linear(id_channels, input_channels)
		self.instance_norm = nn.InstanceNorm2d(input_channels, affine=False)

		self.conv_mask = nn.Conv2d(input_channels, 1, kernel_size=1, stride=1, padding=0, bias=True)

	def forward(self, feature_map, attr_embedding, id_embedding):
		feature_map = self.instance_norm(feature_map)

		attr_gamma = self.conv_gamma(attr_embedding)
		attr_beta = self.conv_beta(attr_embedding)
		attr_modulation = attr_gamma * feature_map + attr_beta

		id_gamma = self.fc_gamma(id_embedding).reshape(feature_map.shape[0], self.input_channels, 1, 1).expand_as(
			feature_map)
		id_beta = self.fc_beta(id_embedding).reshape(feature_map.shape[0], self.input_channels, 1, 1).expand_as(
			feature_map)
		id_modulation = id_gamma * feature_map + id_beta

		feature_mask = torch.sigmoid(self.conv_mask(feature_map))
		feature_blend = (1 - feature_mask) * attr_modulation + feature_mask * id_modulation
		return feature_blend


class AddBlocksSequential(nn.Sequential):
	def forward(self, *inputs):
		h, attr_embedding, id_embedding = inputs

		for index, module in enumerate(self._modules.values()):
			if index % 3 == 0 and index > 0:
				inputs = (inputs, attr_embedding, id_embedding)
			if type(inputs) == tuple:
				inputs = module(*inputs)
			else:
				inputs = module(inputs)
		return inputs


class AAD_ResBlk(nn.Module):
	def __init__(self, in_channels, out_channels, attr_channels, id_channels, num_blocks):
		super(AAD_ResBlk, self).__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		primary_add_blocks = []

		for i in range(num_blocks):
			intermediate_channels = in_channels if i < (num_blocks - 1) else out_channels
			primary_add_blocks.extend(
				[
					AADLayer(in_channels, attr_channels, id_channels),
					nn.ReLU(inplace=True),
					nn.Conv2d(in_channels, intermediate_channels, kernel_size=3, stride=1, padding=1, bias=False)
				])
		self.primary_add_blocks = AddBlocksSequential(*primary_add_blocks)

		if in_channels != out_channels:
			auxiliary_add_blocks = \
				[
					AADLayer(in_channels, attr_channels, id_channels),
					nn.ReLU(inplace=True),
					nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
				]
			self.auxiliary_add_blocks = AddBlocksSequential(*auxiliary_add_blocks)

	def forward(self, feature_map, attr_embedding, id_embedding):
		primary_feature = self.primary_add_blocks(feature_map, attr_embedding, id_embedding)

		if self.in_channels != self.out_channels:
			feature_map = self.auxiliary_add_blocks(feature_map, attr_embedding, id_embedding)
		output_feature = primary_feature + feature_map
		return output_feature


class Conv4x4(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(Conv4x4, self).__init__()
		self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1,
							  bias=False)
		self.batch_norm = nn.BatchNorm2d(out_channels)
		self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)

	def forward(self, x):
		x = self.conv(x)
		x = self.batch_norm(x)
		x = self.leaky_relu(x)
		return x


class DeConv4x4(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(DeConv4x4, self).__init__()
		self.deconv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2,
										 padding=1, bias=False)
		self.batch_norm = nn.BatchNorm2d(out_channels)
		self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)

	def forward(self, x, skip):
		x = self.deconv(x)
		x = self.batch_norm(x)
		x = self.leaky_relu(x)
		return torch.cat((x, skip), dim=1)


class Upsample(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(Upsample, self).__init__()
		self.initial_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1,
									  padding=1)
		self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)

	def forward(self, x):
		x = self.initial_conv(x.view(x.shape[0], -1, 1, 1))
		x = self.pixel_shuffle(x)
		return x


def initialize_weight(module):
	if isinstance(module, nn.Linear):
		module.weight.data.normal_(0, 0.001)
		module.bias.data.zero_()

	if isinstance(module, nn.Conv2d):
		nn.init.xavier_normal_(module.weight.data)

	if isinstance(module, nn.ConvTranspose2d):
		nn.init.xavier_normal_(module.weight.data)
