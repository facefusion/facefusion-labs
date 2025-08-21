from torch import Tensor, nn


class CrossFace(nn.Module):
	def __init__(self) -> None:
		super().__init__()
		self.linear_layers = self.create_linear_layers()
		self.norm_layers = self.create_norm_layers()
		self.skip_layer = nn.Linear(512, 512)
		self.gelu = nn.GELU()
		self.dropout = nn.Dropout(0.1)
		self._init_weights()

	def _init_weights(self):
		for layer in self.linear_layers:
			nn.init.xavier_uniform_(layer.weight)
			nn.init.constant_(layer.bias, 0.01)

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
		output_tensor = nn.functional.normalize(input_tensor, p = 2, dim = -1)
		identity_tensor = output_tensor

		for index, layer in enumerate(self.linear_layers[:-1]):
			output_tensor = layer(output_tensor)
			output_tensor = self.norm_layers[index](output_tensor)
			output_tensor = self.gelu(output_tensor)
			output_tensor = self.dropout(output_tensor)

		output_tensor = self.linear_layers[-1](output_tensor)
		output_tensor = output_tensor + 0.2 * self.skip_layer(identity_tensor)
		return output_tensor
