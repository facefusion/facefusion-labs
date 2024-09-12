import torch
import torch.nn as nn


class ArcFaceConverter(nn.Module):
	def __init__(self) -> None:
		super(ArcFaceConverter, self).__init__()
		input_dim = 512
		output_dim = 512
		self.fc1 = nn.Linear(input_dim, 1024)
		self.fc2 = nn.Linear(1024, 1024)
		self.fc3 = nn.Linear(1024, 1024)
		self.fc4 = nn.Linear(1024, output_dim)
		self.activation = nn.LeakyReLU()

	def forward(self, input_embedding: torch.Tensor) -> torch.Tensor:
		output_embedding = self.activation(self.fc1(input_embedding))
		output_embedding = self.activation(self.fc2(output_embedding))
		output_embedding = self.activation(self.fc3(output_embedding))
		output_embedding = self.fc4(output_embedding)
		return output_embedding
