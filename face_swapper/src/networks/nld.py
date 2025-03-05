import math

from torch import Tensor, nn


class NLD(nn.Module):
	def __init__(self, input_channels : int, num_filters : int, num_layers : int, kernel_size : int) -> None:
		super().__init__()
		self.input_channels = input_channels
		self.num_filters = num_filters
		self.num_layers = num_layers
		self.kernel_size = kernel_size
		self.layers = self.create_layers()
		self.sequences = nn.Sequential(*self.layers)

	def create_layers(self) -> nn.ModuleList:
		padding = math.ceil((self.kernel_size - 1) / 2)
		current_filters = self.num_filters
		layers = nn.ModuleList(
		[
			nn.Conv2d(self.input_channels, current_filters, kernel_size = self.kernel_size, stride = 2, padding = padding),
			nn.LeakyReLU(0.2, True)
		])

		for _ in range(1, self.num_layers):
			previous_filters = current_filters
			current_filters = min(current_filters * 2, 512)
			layers +=\
			[
				nn.Conv2d(previous_filters, current_filters, kernel_size = self.kernel_size, stride = 2, padding = padding),
				nn.InstanceNorm2d(current_filters),
				nn.LeakyReLU(0.2, True)
			]

		previous_filters = current_filters
		current_filters = min(current_filters * 2, 512)
		layers +=\
		[
			nn.Conv2d(previous_filters, current_filters, kernel_size = self.kernel_size, padding = padding),
			nn.InstanceNorm2d(current_filters),
			nn.LeakyReLU(0.2, True),
			nn.Conv2d(current_filters, 1, kernel_size = self.kernel_size, padding = padding)
		]
		return layers

	def forward(self, input_tensor : Tensor) -> Tensor:
		return self.sequences(input_tensor)
