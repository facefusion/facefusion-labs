import torch
from torch import Tensor, nn as nn

from face_swapper.src.types import TargetAttributes, VisionTensor


class Upsample(nn.Module):
	def __init__(self, input_channels : int, output_channels : int) -> None:
		super(Upsample, self).__init__()
		self.deconv = nn.ConvTranspose2d(in_channels = input_channels, out_channels = output_channels, kernel_size = 4, stride = 2, padding = 1, bias = False)
		self.batch_norm = nn.BatchNorm2d(output_channels)
		self.leaky_relu = nn.LeakyReLU(0.1, inplace = True)

	def forward(self, temp : Tensor, skip_tensor : Tensor) -> Tensor:
		temp = self.deconv(temp)
		temp = self.batch_norm(temp)
		temp = self.leaky_relu(temp)
		return torch.cat((temp, skip_tensor), dim = 1)


class DownSample(nn.Module):
	def __init__(self, input_channels : int, output_channels : int) -> None:
		super(DownSample, self).__init__()
		self.conv = nn.Conv2d(in_channels = input_channels, out_channels = output_channels, kernel_size = 4, stride = 2, padding = 1, bias = False)
		self.batch_norm = nn.BatchNorm2d(output_channels)
		self.leaky_relu = nn.LeakyReLU(0.1, inplace = True)

	def forward(self, temp : Tensor) -> Tensor:
		temp = self.conv(temp)
		temp = self.batch_norm(temp)
		temp = self.leaky_relu(temp)
		return temp


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

	def forward(self, target : VisionTensor) -> TargetAttributes:
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
