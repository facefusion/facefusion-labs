import configparser
from typing import Tuple

from torch import nn

from ..networks.attribute_modulator import AADGenerator
from ..networks.unet import UNet
from ..types import Embedding, TargetAttributes, VisionTensor

CONFIG = configparser.ConfigParser()
CONFIG.read('config.ini')


class AdaptiveEmbeddingIntegrationNetwork(nn.Module):
	def __init__(self) -> None:
		super(AdaptiveEmbeddingIntegrationNetwork, self).__init__()
		id_channels = CONFIG.getint('training.model.generator', 'id_channels')
		num_blocks = CONFIG.getint('training.model.generator', 'num_blocks')

		self.encoder = UNet()
		self.generator = AADGenerator(id_channels, num_blocks)
		self.encoder.apply(init_weight)
		self.generator.apply(init_weight)

	def forward(self, target : VisionTensor, source_embedding : Embedding) -> Tuple[VisionTensor, TargetAttributes]:
		target_attributes = self.get_attributes(target)
		swap_tensor = self.generator(target_attributes, source_embedding)
		return swap_tensor, target_attributes

	def get_attributes(self, target : VisionTensor) -> TargetAttributes:
		return self.encoder(target)


def init_weight(module : nn.Module) -> None:
	if isinstance(module, nn.Linear):
		module.weight.data.normal_(std = 0.001)
		module.bias.data.zero_()

	if isinstance(module, nn.Conv2d):
		nn.init.xavier_normal_(module.weight.data)

	if isinstance(module, nn.ConvTranspose2d):
		nn.init.xavier_normal_(module.weight.data)
