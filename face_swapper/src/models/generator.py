from configparser import ConfigParser
from typing import Tuple

from torch import Tensor, nn

from ..networks.aad import AAD
from ..networks.unet import UNet
from ..types import Embedding, Feature


class Generator(nn.Module):
	def __init__(self, config_parser : ConfigParser) -> None:
		super().__init__()
		self.encoder = UNet(config_parser)
		self.generator = AAD(config_parser)
		self.encoder.apply(init_weight)
		self.generator.apply(init_weight)

	def forward(self, source_embedding : Embedding, target_tensor : Tensor) -> Tuple[Tensor, Tuple[Feature, ...]]:
		target_features = self.encode_features(target_tensor)
		output_tensor = self.generator(source_embedding, target_features)
		return output_tensor, target_features

	def encode_features(self, input_tensor : Tensor) -> Tuple[Feature, ...]:
		return self.encoder(input_tensor)


def init_weight(module : nn.Module) -> None:
	if isinstance(module, nn.Linear):
		module.weight.data.normal_(std = 0.001)
		module.bias.data.zero_()

	if isinstance(module, nn.Conv2d):
		nn.init.xavier_normal_(module.weight.data)

	if isinstance(module, nn.ConvTranspose2d):
		nn.init.xavier_normal_(module.weight.data)
