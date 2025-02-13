import torch
from torch import Tensor, nn

from face_swapper.src.types import TargetAttributes


class UNet(nn.Module):
	def __init__(self) -> None:
		super(UNet, self).__init__()
		self.down = self.create_down()
		self.up = self.create_up()

	@staticmethod
	def create_down():
		return nn.ModuleList(
		[
			Down(3, 32),
			Down(32, 64),
			Down(64, 128),
			Down(128, 256),
			Down(256, 512),
			Down(512, 1024),
			Down(1024, 1024)
		])

	@staticmethod
	def create_up():
		return nn.ModuleList(
		[
			Up(1024, 1024),
			Up(2048, 512),
			Up(1024, 256),
			Up(512, 128),
			Up(256, 64),
			Up(128, 32)
		])

	def forward(self, target_tensor : Tensor) -> TargetAttributes:
		down_features = []
		up_features = []
		temp_tensor = target_tensor

		for down in self.down:
			temp_tensor = down(temp_tensor)
			down_features.append(temp_tensor)

		bottleneck_tensor = down_features[-1]
		temp_tensor = bottleneck_tensor

		for index, up in enumerate(self.up):
			down_index = -(index + 2)
			up_feature = up(temp_tensor, down_features[down_index])
			up_features.append(up_feature)

		output_tensor = nn.functional.interpolate(temp_tensor, scale_factor = 2, mode = 'bilinear', align_corners = False)
		return bottleneck_tensor, *up_features, output_tensor


class Up(nn.Module):
	def __init__(self, input_channels : int, output_channels : int) -> None:
		super(Up, self).__init__()
		self.conv_transpose = nn.ConvTranspose2d(in_channels = input_channels, out_channels = output_channels, kernel_size = 4, stride = 2, padding = 1, bias = False)
		self.batch_norm = nn.BatchNorm2d(output_channels)
		self.leaky_relu = nn.LeakyReLU(0.1, inplace = True)

	def forward(self, input_tensor : Tensor, skip_tensor : Tensor) -> Tensor:
		temp_tensor = self.conv_transpose(input_tensor)
		temp_tensor = self.batch_norm(temp_tensor)
		temp_tensor = self.leaky_relu(temp_tensor)
		temp_tensor = torch.cat((temp_tensor, skip_tensor), dim = 1)
		return temp_tensor


class Down(nn.Module):
	def __init__(self, input_channels : int, output_channels : int) -> None:
		super(Down, self).__init__()
		self.conv = nn.Conv2d(in_channels = input_channels, out_channels = output_channels, kernel_size = 4, stride = 2, padding = 1, bias = False)
		self.batch_norm = nn.BatchNorm2d(output_channels)
		self.leaky_relu = nn.LeakyReLU(0.1, inplace = True)

	def forward(self, input_tensor : Tensor) -> Tensor:
		temp_tensor = self.conv(input_tensor)
		temp_tensor = self.batch_norm(temp_tensor)
		temp_tensor = self.leaky_relu(temp_tensor)
		return temp_tensor
