from torch import Tensor, nn


class CrossFace(nn.Module):
	def __init__(self) -> None:
		super().__init__()
		self.linear_layers = self.create_linear_layers()
		self.norm_layers = self.create_norm_layers()
		self.residual_layer = nn.Linear(512, 512)
		self.gelu = nn.GELU()
		self.dropout = nn.Dropout(0.1)
		self.apply(init_weight)

	@staticmethod
	def create_linear_layers() -> nn.ModuleList:
		return nn.ModuleList(
		[
			nn.Linear(512, 1024),
			nn.Linear(1024, 2048),
			nn.Linear(2048, 1024),
			nn.Linear(1024, 512)
		])

	@staticmethod
	def create_norm_layers() -> nn.ModuleList:
		return nn.ModuleList(
		[
			nn.LayerNorm(1024),
			nn.LayerNorm(2048),
			nn.LayerNorm(1024)
		])

	def forward(self, input_tensor : Tensor) -> Tensor:
		temp_tensor = nn.functional.normalize(input_tensor, p = 2, dim = -1)
		residual_tensor = self.residual_layer(temp_tensor)

		for index, layer in enumerate(self.linear_layers[:-1]):
			temp_tensor = layer(temp_tensor)
			temp_tensor = self.norm_layers[index](temp_tensor)
			temp_tensor = self.gelu(temp_tensor)
			temp_tensor = self.dropout(temp_tensor)

		temp_tensor = self.linear_layers[-1](temp_tensor)
		temp_tensor = temp_tensor + 0.2 * residual_tensor
		return temp_tensor


def init_weight(module : nn.Module) -> None:
	if isinstance(module, nn.Linear):
		nn.init.xavier_normal_(module.weight)
		nn.init.constant_(module.bias, 0.01)
