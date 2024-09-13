import torch
import torch.nn as nn
from torch import Tensor


class FaceLandmarker685(nn.Module):
	def __init__(self) -> None:
		super(FaceLandmarker685, self).__init__()
		self.fc1 = nn.Linear(2 * 5, 128)
		self.fc2 = nn.Linear(128, 256)
		self.fc3 = nn.Linear(256, 512)
		self.fc4 = nn.Linear(512, 2 * 68)
		self.activation = nn.ReLU()

	def forward(self, input_landmark : Tensor) -> Tensor:
		input_landmark = torch.flatten(input_landmark, start_dim = 1)
		output_landmark = self.activation(self.fc1(input_landmark))
		output_landmark = self.activation(self.fc2(output_landmark))
		output_landmark = self.activation(self.fc3(output_landmark))
		output_landmark = self.fc4(output_landmark)
		output_landmark = output_landmark.view(1, 68, 2)
		return output_landmark
