from typing import Tuple

import torch
from torch import Tensor, nn
from torchvision import models


class UNet(nn.Module):
	def __init__(self) -> None:
		super(UNet, self).__init__()
		self.down_samples = self.create_down_samples(self)
		self.up_samples = self.create_up_samples()

	@staticmethod
	def create_down_samples(self : nn.Module) -> nn.ModuleList:
		return nn.ModuleList(
		[
			DownSample(3, 32),
			DownSample(32, 64),
			DownSample(64, 128),
			DownSample(128, 256),
			DownSample(256, 512),
			DownSample(512, 1024),
			DownSample(1024, 1024)
		])

	@staticmethod
	def create_up_samples() -> nn.ModuleList:
		return nn.ModuleList(
		[
			UpSample(1024, 1024),
			UpSample(2048, 512),
			UpSample(1024, 256),
			UpSample(512, 128),
			UpSample(256, 64),
			UpSample(128, 32)
		])

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
	def __init__(self) -> None:
		super(UNet, self).__init__()
		self.resnet = models.resnet34(pretrained = True)
		self.down_samples = self.create_down_samples(self)
		self.up_samples = self.create_up_samples()

	@staticmethod
	def create_down_samples(self : nn.Module) -> nn.ModuleList:
		return nn.ModuleList(
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


class UpSample(nn.Module):
	def __init__(self, input_channels : int, output_channels : int) -> None:
		super(UpSample, self).__init__()
		self.conv_transpose = nn.ConvTranspose2d(in_channels = input_channels, out_channels = output_channels, kernel_size = 4, stride = 2, padding = 1, bias = False)
		self.batch_norm = nn.BatchNorm2d(output_channels)
		self.leaky_relu = nn.LeakyReLU(0.1, inplace = True)

	def forward(self, input_tensor : Tensor, skip_tensor : Tensor) -> Tensor:
		temp_tensor = self.conv_transpose(input_tensor)
		temp_tensor = self.batch_norm(temp_tensor)
		temp_tensor = self.leaky_relu(temp_tensor)
		temp_tensor = torch.cat((temp_tensor, skip_tensor), dim = 1)
		return temp_tensor


class DownSample(nn.Module):
	def __init__(self, input_channels : int, output_channels : int) -> None:
		super(DownSample, self).__init__()
		self.conv = nn.Conv2d(in_channels = input_channels, out_channels = output_channels, kernel_size = 4, stride = 2, padding = 1, bias = False)
		self.batch_norm = nn.BatchNorm2d(output_channels)
		self.leaky_relu = nn.LeakyReLU(0.1, inplace = True)

	def forward(self, input_tensor : Tensor) -> Tensor:
		temp_tensor = self.conv(input_tensor)
		temp_tensor = self.batch_norm(temp_tensor)
		temp_tensor = self.leaky_relu(temp_tensor)
		return temp_tensor
