import os
from configparser import ConfigParser
from typing import Tuple

import torch
from torch import Tensor, nn

from .training import HyperSwapTrainer
from .types import Embedding, Mask, Module

CONFIG_PARSER = ConfigParser()
CONFIG_PARSER.read('config.ini')


class HalfPrecision(nn.Module):
	def __init__(self, model : Module) -> None:
		super().__init__()
		self.model = model.half()

	def forward(self, source_embedding : Embedding, target_tensor : Tensor) -> Tuple[Tensor, Mask]:
		source_embedding = source_embedding.half()
		target_tensor = target_tensor.half()
		output_tensor, output_mask = self.model(source_embedding, target_tensor)
		output_tensor = output_tensor.float()
		output_mask = output_mask.float()
		return output_tensor, output_mask


def export() -> None:
	config_directory_path = CONFIG_PARSER.get('exporting', 'directory_path')
	config_source_path = CONFIG_PARSER.get('exporting', 'source_path')
	config_target_path = CONFIG_PARSER.get('exporting', 'target_path')
	config_target_size = CONFIG_PARSER.getint('exporting', 'target_size')
	config_ir_version = CONFIG_PARSER.getint('exporting', 'ir_version')
	config_opset_version = CONFIG_PARSER.getint('exporting', 'opset_version')
	config_precision = CONFIG_PARSER.get('exporting', 'precision')

	os.makedirs(config_directory_path, exist_ok = True)
	model = HyperSwapTrainer.load_from_checkpoint(config_source_path, config_parser = CONFIG_PARSER, map_location = 'cpu').eval()

	if config_precision == 'half':
		model = HalfPrecision(model).eval()

	model.ir_version = torch.tensor(config_ir_version)
	source_tensor = torch.randn(1, 512)
	target_tensor = torch.randn(1, 3, config_target_size, config_target_size)
	torch.onnx.export(model, (source_tensor, target_tensor), config_target_path, input_names = [ 'source', 'target' ], output_names = [ 'output', 'mask' ], opset_version = config_opset_version)
