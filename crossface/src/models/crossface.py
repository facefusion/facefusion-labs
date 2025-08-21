from torch import Tensor, nn


class CrossFace(nn.Module):
	def __init__(self) -> None:
		super().__init__()
		self.sequence = self.create_sequence()
		self.linear = nn.Linear(512, 512)
		self.apply(init_weight)

	@staticmethod
	def create_sequence() -> nn.Sequential:
		return nn.Sequential(
			nn.Linear(512, 1024),
			nn.LayerNorm(1024),
			nn.GELU(),
			nn.Dropout(0.1),
			nn.Linear(1024, 2048),
			nn.LayerNorm(2048),
			nn.GELU(),
			nn.Dropout(0.1),
			nn.Linear(2048, 1024),
			nn.LayerNorm(1024),
			nn.GELU(),
			nn.Dropout(0.1),
			nn.Linear(1024, 512)
		)

	def forward(self, input_tensor : Tensor) -> Tensor:
		temp_tensor = nn.functional.normalize(input_tensor, p = 2, dim = -1)
		return self.sequence(temp_tensor) + 0.2 * self.linear(temp_tensor)


def init_weight(module : nn.Module) -> None:
	if isinstance(module, nn.Linear):
		nn.init.xavier_normal_(module.weight)
		nn.init.constant_(module.bias, 0.01)
