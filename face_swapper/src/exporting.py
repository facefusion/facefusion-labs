import os
from configparser import ConfigParser


import torch

from .training import FaceSwapperTrainer

CONFIG_PARSER = ConfigParser()
CONFIG_PARSER.read('config.ini')


def export() -> None:
	config =\
	{
		'directory_path': CONFIG_PARSER.get('exporting', 'directory_path'),
		'source_path': CONFIG_PARSER.get('exporting', 'source_path'),
		'target_path': CONFIG_PARSER.get('exporting', 'target_path'),
		'target_size': CONFIG_PARSER.getint('exporting', 'target_size'),
		'ir_version': CONFIG_PARSER.getint('exporting', 'ir_version'),
		'opset_version': CONFIG_PARSER.getint('exporting', 'opset_version')
	}

	os.makedirs(config.get('directory_path'), exist_ok = True)
	model = FaceSwapperTrainer.load_from_checkpoint(config.get('source_path'), map_location = 'cpu')
	model.eval()
	model.ir_version = torch.tensor(config.get('ir_version'))
	source_tensor = torch.randn(1, 512)
	target_tensor = torch.randn(1, 3, config.get('target_size'), config.get('target_size'))
	torch.onnx.export(model, (source_tensor, target_tensor), config.get('target_path'), input_names = [ 'source', 'target' ], output_names = [ 'output' ], opset_version = config.get('opset_version'))
