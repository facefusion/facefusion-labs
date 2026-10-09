from configparser import ConfigParser

import torch
from torch import Tensor, nn

from ..types import Feature, Mask


class MaskNet(nn.Module):
	def __init__(self, config_parser : ConfigParser) -> None:
		super().__init__()
		self.config_input_channels = config_parser.getint('training.model.masker', 'input_channels')
		self.config_output_channels = config_parser.getint('training.model.masker', 'output_channels')
		self.config_num_filters = config_parser.getint('training.model.masker', 'num_filters')
		self.config_output_size = config_parser.getint('training.model.generator', 'output_size')
		self.scale_down_samples = self.create_scale_down_samples()
		self.base_down_samples = self.create_base_down_samples()
		self.base_up_samples = self.create_base_up_samples()
		self.scale_up_samples = self.create_scale_up_samples()
		self.bottleneck = BottleNeck(self.config_num_filters * 4)
		self.conv = nn.Conv2d(self.config_num_filters, self.config_output_channels, kernel_size = 1)
		self.sigmoid = nn.Sigmoid()

	def create_scale_down_samples(self) -> nn.ModuleList:
		scale_down_samples = nn.ModuleList(
		[
			DownSample(self.config_input_channels, self.config_num_filters)
		])

		if self.config_output_size == 512:
			scale_down_samples.extend(
			[
				DownSample(self.config_num_filters, self.config_num_filters)
			])

		if self.config_output_size == 1024:
			scale_down_samples.extend(
			[
				DownSample(self.config_num_filters, self.config_num_filters),
				DownSample(self.config_num_filters, self.config_num_filters)
			])

		return scale_down_samples

	def create_base_down_samples(self) -> nn.ModuleList:
		base_down_samples = nn.ModuleList(
		[
			DownSample(self.config_num_filters, self.config_num_filters * 2),
			DownSample(self.config_num_filters * 2, self.config_num_filters * 4)
		])

		return base_down_samples

	def create_base_up_samples(self) -> nn.ModuleList:
		base_up_samples = nn.ModuleList(
		[
			UpSample(self.config_num_filters * 4, self.config_num_filters * 2),
			UpSample(self.config_num_filters * 2, self.config_num_filters),
			UpSample(self.config_num_filters, self.config_num_filters)
		])

		return base_up_samples

	def create_scale_up_samples(self) -> nn.ModuleList:
		scale_up_samples = nn.ModuleList()

		if self.config_output_size == 512:
			scale_up_samples.extend(
			[
				UpSample(self.config_num_filters, self.config_num_filters)
			])

		if self.config_output_size == 1024:
			scale_up_samples.extend(
			[
				UpSample(self.config_num_filters, self.config_num_filters),
				UpSample(self.config_num_filters, self.config_num_filters)
			])

		return scale_up_samples

	def forward(self, input_tensor : Tensor, input_feature : Feature) -> Mask:
		output_mask = torch.cat([ input_tensor, input_feature ], dim = 1)

		for down_sample in self.scale_down_samples:
			output_mask = down_sample(output_mask)

		for down_sample in self.base_down_samples:
			output_mask = down_sample(output_mask)

		output_mask = self.bottleneck(output_mask)

		for up_sample in self.base_up_samples:
			output_mask = up_sample(output_mask)

		for up_sample in self.scale_up_samples:
			output_mask = up_sample(output_mask)

		output_mask = self.conv(output_mask)
		output_mask = self.sigmoid(output_mask)
		return output_mask


class BottleNeck(nn.Module):
	def __init__(self, num_filters : int):
		super().__init__()
		self.sequences = self.create_sequences(num_filters)
		self.relu = nn.ReLU()

	@staticmethod
	def create_sequences(num_filters : int) -> nn.Sequential:
		return nn.Sequential(
			nn.Conv2d(num_filters, num_filters, kernel_size = 3, padding = 1, bias = False),
			nn.BatchNorm2d(num_filters),
			nn.ReLU(),
			nn.Conv2d(num_filters, num_filters, kernel_size = 3, padding = 1, bias = False),
			nn.BatchNorm2d(num_filters),
			nn.ReLU()
		)

	def forward(self, input_tensor : Tensor) -> Tensor:
		output_tensor = self.sequences(input_tensor) + input_tensor
		output_tensor = self.relu(output_tensor)
		return output_tensor


class UpSample(nn.Module):
	def __init__(self, input_channels : int, output_channels : int) -> None:
		super().__init__()
		self.sequences = self.create_sequences(input_channels, output_channels)

	@staticmethod
	def create_sequences(input_channels : int, output_channels : int) -> nn.Sequential:
		return nn.Sequential(
			nn.ConvTranspose2d(input_channels, output_channels, kernel_size = 2, stride = 2),
			nn.ReLU()
		)

	def forward(self, input_tensor : Tensor) -> Tensor:
		output_tensor = self.sequences(input_tensor)
		return output_tensor


class DownSample(nn.Module):
	def __init__(self, input_channels : int, output_channels : int) -> None:
		super().__init__()
		self.sequences = self.create_sequences(input_channels, output_channels)

	@staticmethod
	def create_sequences(input_channels : int, output_channels : int) -> nn.Sequential:
		return nn.Sequential(
			nn.Conv2d(input_channels, output_channels, kernel_size = 3, padding = 1, bias = False),
			nn.BatchNorm2d(output_channels),
			nn.ReLU(),
			nn.MaxPool2d(2)
		)

	def forward(self, input_tensor : Tensor) -> Tensor:
		output_tensor = self.sequences(input_tensor)
		return output_tensor
