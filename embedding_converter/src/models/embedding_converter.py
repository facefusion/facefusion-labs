import torch
from torch import Tensor, nn


class EmbeddingConverter(nn.Module):
	def __init__(self) -> None:
		super(EmbeddingConverter, self).__init__()
		self.layers = self.create_layers()
		self.leaky_relu = nn.LeakyReLU()

	@staticmethod
	def create_layers() -> nn.ModuleList:
		return nn.ModuleList(
		[
			nn.Linear(512, 1024),
			nn.Linear(1024, 2048),
			nn.Linear(2048, 1024),
			nn.Linear(1024, 512)
		])

	def forward(self, input_tensor : Tensor) -> Tensor:
		output_tensor = input_tensor / torch.norm(input_tensor)

		for layer in self.layers[:-1]:
			output_tensor = self.leaky_relu(layer(output_tensor))

		output_tensor = self.layers[-1](output_tensor)
		return output_tensor
