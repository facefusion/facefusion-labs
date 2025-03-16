from configparser import ConfigParser
from typing import Tuple

import torch
from torch import Tensor, nn

from face_swapper.src.types import Feature


class UNet(nn.Module):
	def __init__(self, config_parser : ConfigParser) -> None:
		super().__init__()
		self.config_output_size = config_parser.getint('training.model.generator', 'output_size')
		self.down_samples = self.create_down_samples()
		self.up_samples = self.create_up_samples()

	def create_down_samples(self) -> nn.ModuleList:
		down_samples = nn.ModuleList(
		[
			DownSample(3, 32),
			DownSample(32, 64),
			DownSample(64, 128),
			DownSample(128, 256),
			DownSample(256, 512)
		])

		if self.config_output_size == 128:
			down_samples.extend(
			[
				DownSample(512, 512)
			])

		if self.config_output_size == 256:
			down_samples.extend(
			[
				DownSample(512, 1024),
				DownSample(1024, 1024)
			])

		if self.config_output_size == 512:
			down_samples.extend(
			[
				DownSample(512, 1024),
				DownSample(1024, 2048),
				DownSample(2048, 2048)
			])

		return down_samples

	def create_up_samples(self) -> nn.ModuleList:
		up_samples = nn.ModuleList()

		if self.config_output_size == 128:
			up_samples.extend(
			[
				UpSample(512, 512)
			])

		if self.config_output_size == 256:
			up_samples.extend(
			[
				UpSample(1024, 1024),
				UpSample(2048, 512)
			])

		if self.config_output_size == 512:
			up_samples.extend(
			[
				UpSample(2048, 2048),
				UpSample(4096, 1024),
				UpSample(2048, 512)
			])

		up_samples.extend(
		[
			UpSample(1024, 256),
			UpSample(512, 128),
			UpSample(256, 64),
			UpSample(128, 32)
		])

		return up_samples

	def forward(self, target_tensor : Tensor) -> Tuple[Feature, ...]:
		down_features = []
		up_features = []
		temp_feature = target_tensor

		for down_sample in self.down_samples:
			temp_feature = down_sample(temp_feature)
			down_features.append(temp_feature)

		bottleneck_feature = down_features[-1]
		temp_feature = bottleneck_feature

		for index, up_sample in enumerate(self.up_samples):
			skip_tensor = down_features[-(index + 2)]
			temp_feature = up_sample(temp_feature, skip_tensor)
			up_features.append(temp_feature)

		final_feature = nn.functional.interpolate(temp_feature, scale_factor = 2, mode = 'bilinear', align_corners = False)
		return bottleneck_feature, *up_features, final_feature


class UpSample(nn.Module):
	def __init__(self, input_channels : int, output_channels : int) -> None:
		super().__init__()
		self.sequences = self.create_sequences(input_channels, output_channels)

	@staticmethod
	def create_sequences(input_channels : int, output_channels : int) -> nn.Sequential:
		return nn.Sequential(
			nn.ConvTranspose2d(input_channels, output_channels, kernel_size = 4, stride = 2, padding = 1, bias = False),
			nn.BatchNorm2d(output_channels),
			nn.LeakyReLU(0.1, inplace = True)
		)

	def forward(self, input_tensor : Tensor, skip_tensor : Tensor) -> Tensor:
		output_tensor = self.sequences(input_tensor)
		output_tensor = torch.cat((output_tensor, skip_tensor), dim = 1)
		return output_tensor


class DownSample(nn.Module):
	def __init__(self, input_channels : int, output_channels : int) -> None:
		super().__init__()
		self.sequences = self.create_sequences(input_channels, output_channels)

	@staticmethod
	def create_sequences(input_channels : int, output_channels : int) -> nn.Sequential:
		return nn.Sequential(
			nn.Conv2d(input_channels, output_channels, kernel_size = 4, stride = 2, padding = 1, bias = False),
			nn.BatchNorm2d(output_channels),
			nn.LeakyReLU(0.1, inplace = True)
		)

	def forward(self, input_tensor : Tensor) -> Tensor:
		output_tensor = self.sequences(input_tensor)
		return output_tensor
