from typing import Tuple

import torch
from torch import Tensor, nn

class UNet(nn.Module):
	def __init__(self, output_size : int) -> None:
		super().__init__()
		self.output_size = output_size
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

		if self.output_size == 256:
			down_samples.extend(
			[
				DownSample(512, 1024),
				DownSample(1024, 1024)
			])

		if self.output_size == 512:
			down_samples.extend(
			[
				DownSample(512, 1024),
				DownSample(1024, 2048),
				DownSample(2048, 2048)
			])

		return down_samples

	def create_up_samples(self) -> nn.ModuleList:
		up_samples = nn.ModuleList()

		if self.output_size == 256:
			up_samples.extend(
			[
				UpSample(1024, 1024)
			])

		if self.output_size == 512:
			up_samples.extend(
			[
				UpSample(2048, 2048),
				UpSample(4096, 1024)
			])

		up_samples.extend(
		[
			UpSample(2048, 512),
			UpSample(1024, 256),
			UpSample(512, 128),
			UpSample(256, 64),
			UpSample(128, 32)
		])

		return up_samples

	def forward(self, target_tensor : Tensor) -> Tuple[Tensor, ...]:
		down_features = []
		up_features = []
		temp_tensor = target_tensor

		for down_sample in self.down_samples:
			temp_tensor = down_sample(temp_tensor)
			down_features.append(temp_tensor)

		bottleneck_tensor = down_features[-1]
		temp_tensor = bottleneck_tensor

		for index, up_sample in enumerate(self.up_samples):
			skip_tensor = down_features[-(index + 2)]
			temp_tensor = up_sample(temp_tensor, skip_tensor)
			up_features.append(temp_tensor)

		output_tensor = nn.functional.interpolate(temp_tensor, scale_factor = 2, mode = 'bilinear', align_corners = False)
		return bottleneck_tensor, *up_features, output_tensor


class UpSample(nn.Module):
	def __init__(self, input_channels : int, output_channels : int) -> None:
		super().__init__()
		self.conv_transpose = nn.ConvTranspose2d(in_channels = input_channels, out_channels = output_channels, kernel_size = 4, stride = 2, padding = 1, bias = False)
		self.batch_norm = nn.BatchNorm2d(output_channels)
		self.leaky_relu = nn.LeakyReLU(0.1, inplace = True)

	def forward(self, input_tensor : Tensor, skip_tensor : Tensor) -> Tensor:
		output_tensor = self.conv_transpose(input_tensor)
		output_tensor = self.batch_norm(output_tensor)
		output_tensor = self.leaky_relu(output_tensor)
		output_tensor = torch.cat((output_tensor, skip_tensor), dim = 1)
		return output_tensor


class DownSample(nn.Module):
	def __init__(self, input_channels : int, output_channels : int) -> None:
		super().__init__()
		self.conv = nn.Conv2d(in_channels = input_channels, out_channels = output_channels, kernel_size = 4, stride = 2, padding = 1, bias = False)
		self.batch_norm = nn.BatchNorm2d(output_channels)
		self.leaky_relu = nn.LeakyReLU(0.1, inplace = True)

	def forward(self, input_tensor : Tensor) -> Tensor:
		output_tensor = self.conv(input_tensor)
		output_tensor = self.batch_norm(output_tensor)
		output_tensor = self.leaky_relu(output_tensor)
		return output_tensor
