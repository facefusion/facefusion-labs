import torch
import torch.nn as nn
from torch import Tensor


class EmbeddingForge(nn.Module):
	def __init__(self) -> None:
		super(EmbeddingForge, self).__init__()
		self.fc1 = nn.Linear(512, 1024)
		self.fc2 = nn.Linear(1024, 2048)
		self.fc3 = nn.Linear(2048, 1024)
		self.fc4 = nn.Linear(1024, 512)
		self.activation = nn.LeakyReLU()

	def forward(self, inputs : Tensor) -> Tensor:
		norm_inputs = inputs / torch.norm(inputs)
		outputs = self.activation(self.fc1(norm_inputs))
		outputs = self.activation(self.fc2(outputs))
		outputs = self.activation(self.fc3(outputs))
		outputs = self.fc4(outputs)
		return outputs
