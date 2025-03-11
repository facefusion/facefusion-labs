import os
from configparser import ConfigParser


import torch

from .training import FaceSwapperTrainer

CONFIG_PARSER = ConfigParser()
CONFIG_PARSER.read('config.ini')


def export() -> None:
	config_directory_path = CONFIG_PARSER.get('exporting', 'directory_path')
	config_source_path = CONFIG_PARSER.get('exporting', 'source_path')
	config_target_path = CONFIG_PARSER.get('exporting', 'target_path')
	config_target_size = CONFIG_PARSER.getint('exporting', 'target_size')
	config_ir_version = CONFIG_PARSER.getint('exporting', 'ir_version')
	config_opset_version = CONFIG_PARSER.getint('exporting', 'opset_version')

	os.makedirs(config_directory_path, exist_ok = True)
	model = FaceSwapperTrainer.load_from_checkpoint(config_source_path, config_parser = CONFIG_PARSER, map_location = 'cpu').eval()
	model.ir_version = torch.tensor(config_ir_version)
	source_tensor = torch.randn(1, 512)
	target_tensor = torch.randn(1, 3, config_target_size, config_target_size)
	torch.onnx.export(model, (source_tensor, target_tensor), config_target_path, input_names = [ 'source', 'target' ], output_names = [ 'output', 'mask' ], opset_version = config_opset_version)
