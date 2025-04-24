import os
from configparser import ConfigParser

import torch

from .training import CrossFaceTrainer

CONFIG_PARSER = ConfigParser()
CONFIG_PARSER.read('config.ini')


def export() -> None:
	config_directory_path = CONFIG_PARSER.get('exporting', 'directory_path')
	config_source_path = CONFIG_PARSER.get('exporting', 'source_path')
	config_target_path = CONFIG_PARSER.get('exporting', 'target_path')
	config_ir_version = CONFIG_PARSER.getint('exporting', 'ir_version')
	config_opset_version = CONFIG_PARSER.getint('exporting', 'opset_version')

	os.makedirs(config_directory_path, exist_ok = True)
	model = CrossFaceTrainer.load_from_checkpoint(config_source_path, map_location = 'cpu').eval()
	model.ir_version = torch.tensor(config_ir_version)
	input_tensor = torch.randn(1, 512)
	torch.onnx.export(model, input_tensor, config_target_path, input_names = [ 'input' ], output_names = [ 'output' ], opset_version = config_opset_version)
