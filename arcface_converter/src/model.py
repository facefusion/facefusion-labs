import torch.nn as nn
from torch import Tensor


class ArcFaceConverter(nn.Module):
	def __init__(self) -> None:
		super(ArcFaceConverter, self).__init__()
		self.fc1 = nn.Linear(512, 1024)
		self.fc2 = nn.Linear(1024, 1024)
		self.fc3 = nn.Linear(1024, 1024)
		self.fc4 = nn.Linear(1024, 512)
		self.activation = nn.LeakyReLU()

	def forward(self, input_embedding : Tensor) -> Tensor:
		output_embedding = self.activation(self.fc1(input_embedding))
		output_embedding = self.activation(self.fc2(output_embedding))
		output_embedding = self.activation(self.fc3(output_embedding))
		output_embedding = self.fc4(output_embedding)
		return output_embedding
