import torch
from torch import Tensor, nn

from ..types import Attribute


class MaskNet(nn.Module):
	def __init__(self) -> None:
		super().__init__()
		self.conv = nn.Conv2d(64, 1, 1)

	def forward(self, target_attribute : Attribute) -> Tensor:
		output_tensor = self.conv(target_attribute)
		output_tensor = torch.sigmoid(output_tensor)
		return output_tensor
