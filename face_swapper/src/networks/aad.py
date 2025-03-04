import torch
from torch import Tensor, nn

from ..types import Attributes, Embedding


class AAD(nn.Module):
	def __init__(self, identity_channels : int, output_channels : int, output_size : int, num_blocks : int) -> None:
		super().__init__()
		self.identity_channels = identity_channels
		self.output_channels = output_channels
		self.output_size = output_size
		self.num_blocks = num_blocks
		self.pixel_shuffle_up_sample = PixelShuffleUpSample(identity_channels, output_channels)
		self.layers = self.create_layers()

	def create_layers(self) -> nn.ModuleList:
		layers = nn.ModuleList(
		[
			AdaptiveFeatureModulation(1024, 1024, 1024, self.identity_channels, self.num_blocks),
			AdaptiveFeatureModulation(1024, 1024, 2048, self.identity_channels, self.num_blocks),
			AdaptiveFeatureModulation(1024, 1024, 1024, self.identity_channels, self.num_blocks),
			AdaptiveFeatureModulation(1024, 512, 512, self.identity_channels, self.num_blocks),
			AdaptiveFeatureModulation(512, 256, 256, self.identity_channels, self.num_blocks),
			AdaptiveFeatureModulation(256, 128, 128, self.identity_channels, self.num_blocks),
			AdaptiveFeatureModulation(128, 64, 64, self.identity_channels, self.num_blocks),
		])

		if self.output_size in [ 384, 512, 768, 1024 ]:
			layers.append(AdaptiveFeatureModulation(64, 32, 32, self.identity_channels, self.num_blocks))

		if self.output_size in [ 512, 768, 1024 ]:
			layers.append(AdaptiveFeatureModulation(32, 16, 16, self.identity_channels, self.num_blocks))

		if self.output_size in [ 768, 1024 ]:
			layers.append(AdaptiveFeatureModulation(16, 8, 8, self.identity_channels, self.num_blocks))

		if self.output_size == 1024:
			layers.append(AdaptiveFeatureModulation(8, 4, 4, self.identity_channels, self.num_blocks))

		if self.output_size == 256:
			layers.append(AdaptiveFeatureModulation(64, 3, 64, self.identity_channels, self.num_blocks))
		if self.output_size == 384:
			layers.append(AdaptiveFeatureModulation(32, 3, 32, self.identity_channels, self.num_blocks))
		if self.output_size == 512:
			layers.append(AdaptiveFeatureModulation(16, 3, 16, self.identity_channels, self.num_blocks))
		if self.output_size == 768:
			layers.append(AdaptiveFeatureModulation(8, 3, 8, self.identity_channels, self.num_blocks))
		if self.output_size == 1024:
			layers.append(AdaptiveFeatureModulation(4, 3, 4, self.identity_channels, self.num_blocks))

		return layers

	def forward(self, source_embedding : Embedding, target_attributes : Attributes) -> Tensor:
		temp_tensors = self.pixel_shuffle_up_sample(source_embedding)

		for index, layer in enumerate(self.layers[:-1]):
			temp_tensor = layer(temp_tensors, target_attributes[index], source_embedding)
			temp_tensors = nn.functional.interpolate(temp_tensor, scale_factor = 2, mode = 'bilinear', align_corners = False)

		temp_tensors = self.layers[-1](temp_tensors, target_attributes[-1], source_embedding)
		output_tensor = torch.tanh(temp_tensors)
		return output_tensor


class AdaptiveFeatureModulation(nn.Module):
	def __init__(self, input_channels : int, output_channels : int, attribute_channels : int, identity_channels : int, num_blocks : int) -> None:
		super().__init__()
		self.input_channels = input_channels
		self.output_channels = output_channels
		self.attribute_channels = attribute_channels
		self.identity_channels = identity_channels
		self.num_blocks = num_blocks
		self.primary_layers = self.create_primary_layers()
		self.shortcut_layers = self.create_shortcut_layers()

	def create_primary_layers(self) -> nn.ModuleList:
		primary_layers = nn.ModuleList()

		for index in range(self.num_blocks):
			primary_layers.extend(
			[
				FeatureModulation(self.input_channels, self.attribute_channels, self.identity_channels),
				nn.ReLU(inplace = True)
			])

			if index < self.num_blocks - 1:
				primary_layers.append(nn.Conv2d(self.input_channels, self.input_channels, kernel_size = 3, padding = 1, bias = False))
			else:
				primary_layers.append(nn.Conv2d(self.input_channels, self.output_channels, kernel_size = 3, padding = 1, bias = False))

		return primary_layers

	def create_shortcut_layers(self) -> nn.ModuleList:
		shortcut_layers = nn.ModuleList()

		if self.input_channels > self.output_channels:
			shortcut_layers.extend(
			[
				FeatureModulation(self.input_channels, self.attribute_channels, self.identity_channels),
				nn.ReLU(inplace = True),
				nn.Conv2d(self.input_channels, self.output_channels, kernel_size = 3, padding = 1, bias = False)
			])

		return shortcut_layers

	def forward(self, input_tensor : Tensor, attribute_embedding : Embedding, identity_embedding : Embedding) -> Tensor:
		primary_tensor = input_tensor

		for primary_layer in self.primary_layers:
			if isinstance(primary_layer, FeatureModulation):
				primary_tensor = primary_layer(primary_tensor, attribute_embedding, identity_embedding)
			else:
				primary_tensor = primary_layer(primary_tensor)

		if self.input_channels > self.output_channels:
			shortcut_tensor = input_tensor

			for shortcut_layer in self.shortcut_layers:
				if isinstance(shortcut_layer, FeatureModulation):
					shortcut_tensor = shortcut_layer(shortcut_tensor, attribute_embedding, identity_embedding)
				else:
					shortcut_tensor = shortcut_layer(shortcut_tensor)

			input_tensor = shortcut_tensor

		return primary_tensor + input_tensor


class FeatureModulation(nn.Module):
	def __init__(self, input_channels : int, attribute_channels : int, identity_channels : int) -> None:
		super().__init__()
		self.input_channels = input_channels
		self.conv1 = nn.Conv2d(attribute_channels, input_channels, kernel_size = 1)
		self.conv2 = nn.Conv2d(attribute_channels, input_channels, kernel_size = 1)
		self.conv3 = nn.Conv2d(input_channels, 1, kernel_size = 1)
		self.linear1 = nn.Linear(identity_channels, input_channels)
		self.linear2 = nn.Linear(identity_channels, input_channels)
		self.instance_norm = nn.InstanceNorm2d(input_channels)

	def forward(self, input_tensor : Tensor, attribute_embedding : Embedding, identity_embedding : Embedding) -> Tensor:
		temp_tensor = self.instance_norm(input_tensor)

		attribute_scale = self.conv1(attribute_embedding)
		attribute_shift = self.conv2(attribute_embedding)
		attribute_modulation = attribute_scale * temp_tensor + attribute_shift

		identity_scale = self.linear2(identity_embedding).reshape(temp_tensor.shape[0], self.input_channels, 1, 1).expand_as(temp_tensor)
		identity_shift = self.linear1(identity_embedding).reshape(temp_tensor.shape[0], self.input_channels, 1, 1).expand_as(temp_tensor)
		identity_modulation = identity_scale * temp_tensor + identity_shift

		temp_mask = torch.sigmoid(self.conv3(temp_tensor))
		output_tensor = (1 - temp_mask) * attribute_modulation + temp_mask * identity_modulation
		return output_tensor


class PixelShuffleUpSample(nn.Module):
	def __init__(self, input_channels : int, output_channels : int) -> None:
		super().__init__()
		self.conv = nn.Conv2d(input_channels, output_channels, kernel_size = 3, padding = 1)
		self.pixel_shuffle = nn.PixelShuffle(upscale_factor = 2)

	def forward(self, input_tensor : Tensor) -> Tensor:
		output_tensor = self.conv(input_tensor.view(input_tensor.shape[0], -1, 1, 1))
		output_tensor = self.pixel_shuffle(output_tensor)
		return output_tensor
