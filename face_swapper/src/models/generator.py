from configparser import ConfigParser
from typing import Tuple

from torch import Tensor, nn

from ..networks.aad import AAD
from ..networks.masknet import MaskNet
from ..networks.unet import UNet
from ..types import Attributes, Embedding


class Generator(nn.Module):
	def __init__(self, config_parser : ConfigParser) -> None:
		super().__init__()
		self.encoder = UNet(config_parser)
		self.generator = AAD(config_parser)
		self.masker = MaskNet(67, 1, 16)
		self.encoder.apply(init_weight)
		self.generator.apply(init_weight)
		self.masker.apply(init_weight)

	def forward(self, source_embedding : Embedding, target_tensor : Tensor) -> Tuple[Tensor, Tensor]:
		target_attributes = self.get_attributes(target_tensor)
		output_tensor = self.generator(source_embedding, target_attributes)
		mask_tensor = self.masker(target_tensor, target_attributes[-1])
		return output_tensor, mask_tensor

	def get_attributes(self, input_tensor : Tensor) -> Attributes:
		return self.encoder(input_tensor)


def init_weight(module : nn.Module) -> None:
	if isinstance(module, nn.Linear):
		module.weight.data.normal_(std = 0.001)
		module.bias.data.zero_()

	if isinstance(module, nn.Conv2d):
		nn.init.xavier_normal_(module.weight.data)

	if isinstance(module, nn.ConvTranspose2d):
		nn.init.xavier_normal_(module.weight.data)
