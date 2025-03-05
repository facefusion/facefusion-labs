from typing import Tuple

import torch
from torch import Tensor, nn
from torchvision import models
from torchvision.models import ResNet34_Weights


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
			DownSample(256, 512),
			DownSample(512, 1024),
			DownSample(1024, 1024)
		])

		if self.output_size in [ 384, 512, 768, 1024 ]:
			down_samples.append(DownSample(1024, 2048))
		if self.output_size in [ 512, 768, 1024 ]:
			down_samples.append(DownSample(2048, 4096))
		if self.output_size in [ 768, 1024 ]:
			down_samples.append(DownSample(4096, 8192))
		if self.output_size == 1024:
			down_samples.append(DownSample(8192, 16384))

		return down_samples

	def create_up_samples(self) -> nn.ModuleList:
		up_samples = nn.ModuleList(
		[
			UpSample(1024, 1024),
			UpSample(2048, 512),
			UpSample(1024, 256),
			UpSample(512, 128),
			UpSample(256, 64),
			UpSample(128, 32)
		])

		if self.output_size in [ 384, 512, 768, 1024 ]:
			up_samples.append(UpSample(32, 16))
		if self.output_size in [ 512, 768, 1024 ]:
			up_samples.append(UpSample(16, 8))
		if self.output_size in [ 768, 1024 ]:
			up_samples.append(UpSample(8, 4))
		if self.output_size == 1024:
			up_samples.append(UpSample(4, 2))

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


class UNetPro(UNet):
	def __init__(self, output_size : int) -> None:
		super().__init__(output_size)
		self.resnet = models.resnet34(weights = ResNet34_Weights.DEFAULT)
		self.down_samples = self.create_down_samples()
		self.up_samples = self.create_up_samples()

	def create_down_samples(self) -> nn.ModuleList:
		down_samples = nn.ModuleList(
		[
			nn.Sequential(
				self.resnet.conv1,
				self.resnet.bn1,
				self.resnet.relu,
				nn.Conv2d(64, 32, kernel_size = 1, bias = False),
				nn.BatchNorm2d(32),
				nn.LeakyReLU(0.1, inplace = True)
			),
			DownSample(32, 64),
			self.resnet.layer2,
			self.resnet.layer3,
			self.resnet.layer4,
			DownSample(512, 1024),
			DownSample(1024, 1024)
		])

		if self.output_size in [ 384, 512, 768, 1024 ]:
			down_samples.append(DownSample(1024, 2048))
		if self.output_size in [ 512, 768, 1024 ]:
			down_samples.append(DownSample(2048, 4096))
		if self.output_size in [ 768, 1024 ]:
			down_samples.append(DownSample(4096, 8192))
		if self.output_size == 1024:
			down_samples.append(DownSample(8192, 16384))

		return down_samples


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
