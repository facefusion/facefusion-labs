import torch
from torch import Tensor, nn


class MaskNet(nn.Module):
	def __init__(self, input_channels : int, output_channels : int, base_channels : int):
		super().__init__()
		self.down_samples = self.create_down_samples(input_channels, base_channels)
		self.up_samples = self.create_up_samples(base_channels)
		self.bottleneck = ResBlock(base_channels * 4)
		self.conv = nn.Conv2d(base_channels, output_channels, kernel_size = 1)
		self.sigmoid = nn.Sigmoid()

	def create_down_samples(self, input_channels : int, base_channels: int) -> nn.ModuleList:
		down_samples = nn.ModuleList(
			[
				DownSample(input_channels, base_channels),
				DownSample(base_channels, base_channels * 2),
				DownSample(base_channels * 2, base_channels * 4)
			])
		return down_samples

	def create_up_samples(self, base_channels : int) -> nn.ModuleList:
		down_samples = nn.ModuleList(
			[
				UpSample(base_channels * 4, base_channels * 2),
				UpSample(base_channels * 2, base_channels),
				UpSample(base_channels, base_channels)
			])
		return down_samples

	def forward(self, target_tensor : Tensor, target_attribute : Tensor) -> Tensor:
		output_tensor = torch.cat([ target_tensor, target_attribute ], dim=1)

		for down_sample in self.down_samples:
			output_tensor = down_sample(output_tensor)
		output_tensor = self.bottleneck(output_tensor)

		for up_sample in self.up_samples:
			output_tensor = up_sample(output_tensor)
		output_tensor = self.conv(output_tensor)
		output_tensor = self.activation(output_tensor)
		return output_tensor


class ResBlock(nn.Module):
	def __init__(self, channels: int):
		super().__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(channels),
			nn.ReLU(inplace=True)
		)
		self.relu = nn.ReLU(inplace=True)

	def forward(self, input_tensor: Tensor) -> Tensor:
		output_tensor = self.conv(input_tensor) + input_tensor
		output_tensor = self.relu(output_tensor)
		return output_tensor


class UpSample(nn.Module):
	def __init__(self, input_channels : int, output_channels : int) -> None:
		super().__init__()
		self.conv_transpose = nn.ConvTranspose2d(input_channels, output_channels, kernel_size = 2, stride = 2)
		self.relu = nn.ReLU(inplace=True)

	def forward(self, input_tensor : Tensor) -> Tensor:
		output_tensor = self.conv_transpose(input_tensor)
		output_tensor = self.relu(output_tensor)
		return output_tensor


class DownSample(nn.Module):
	def __init__(self, input_channels : int, output_channels : int) -> None:
		super().__init__()
		self.conv = nn.Conv2d(input_channels, output_channels, kernel_size = 3, padding = 1, bias = False)
		self.batch_norm = nn.BatchNorm2d(output_channels)
		self.relu = nn.ReLU(inplace = True)
		self.max_pool = nn.MaxPool2d(2)

	def forward(self, input_tensor : Tensor) -> Tensor:
		output_tensor = self.conv(input_tensor)
		output_tensor = self.batch_norm(output_tensor)
		output_tensor = self.relu(output_tensor)
		output_tensor = self.max_pool(output_tensor)
		return output_tensor
