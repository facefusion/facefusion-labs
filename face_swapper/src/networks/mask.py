import torch
from torch import Tensor, nn

class MaskNet(nn.Module):
	def __init__(self) -> None:
		super().__init__()
		self.conv = nn.Conv2d(64, 1, 1)

	def forward(self, target_attribute : Tensor) -> Tensor:
		mask_tensor = self.conv(target_attribute)
		mask_tensor = torch.sigmoid(mask_tensor)
		return mask_tensor
