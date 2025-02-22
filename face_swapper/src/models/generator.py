import configparser

from torch import Tensor, nn

from ..networks.attribute_modulator import AADGenerator
from ..networks.unet import UNet, UNetPro
from ..types import Attributes, Embedding

CONFIG = configparser.ConfigParser()
CONFIG.read('config.ini')


class Generator(nn.Module):
	def __init__(self) -> None:
		super(Generator, self).__init__()
		encoder_type = CONFIG.get('training.model.generator', 'encoder_type')
		id_channels = CONFIG.getint('training.model.generator', 'id_channels')
		num_blocks = CONFIG.getint('training.model.generator', 'num_blocks')

		if encoder_type == 'unet':
			self.encoder = UNet()
		if encoder_type == 'unet-pro':
			self.encoder = UNetPro()
		self.generator = AADGenerator(id_channels, num_blocks)
		self.encoder.apply(init_weight)
		self.generator.apply(init_weight)

	def forward(self, source_embedding : Embedding, target_tensor : Tensor) -> Tensor:
		target_attributes = self.get_attributes(target_tensor)
		output_tensor = self.generator(target_attributes, source_embedding)
		return output_tensor

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
