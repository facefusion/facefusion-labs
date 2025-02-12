import torch
import torch.nn as nn

from ..types import VisionTensor


class EmbeddingConverter(nn.Module):
	def __init__(self) -> None:
		super(EmbeddingConverter, self).__init__()
		self.layers = self.create_layers()
		self.activation = nn.LeakyReLU()

	@staticmethod
	def create_layers() -> nn.ModuleList:
		layers = nn.ModuleList(
		[
			nn.Linear(512, 1024),
			nn.Linear(1024, 2048),
			nn.Linear(2048, 1024),
			nn.Linear(1024, 512)
		])
		return layers

	def forward(self, input_tensor : VisionTensor) -> VisionTensor:
		output_tensor = input_tensor / torch.norm(input_tensor)

		for layer in self.layers[:-1]:
			output_tensor = self.activation(layer(output_tensor))

		output_tensor = self.layers[-1](output_tensor)
		return output_tensor
