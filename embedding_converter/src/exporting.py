import configparser
from os import makedirs

import torch

from .training import EmbeddingConverterTrainer

CONFIG = configparser.ConfigParser()
CONFIG.read('config.ini')


def export() -> None:
	config =\
	{
		'directory_path': CONFIG.get('exporting', 'directory_path'),
		'source_path': CONFIG.get('exporting', 'source_path'),
		'target_path': CONFIG.get('exporting', 'target_path'),
		'ir_version': CONFIG.getint('exporting', 'ir_version'),
		'opset_version': CONFIG.getint('exporting', 'opset_version')
	}

	makedirs(config.get('directory_path'), exist_ok = True)
	model = EmbeddingConverterTrainer.load_from_checkpoint(config.get('source_path'), map_location = 'cpu')
	model.eval()
	model.ir_version = torch.tensor(config.get('ir_version'))
	input_tensor = torch.randn(1, 512)
	torch.onnx.export(model, input_tensor, config.get('target_path'), input_names = [ 'input' ], output_names = [ 'output' ], opset_version = config.get('opset_version'))
