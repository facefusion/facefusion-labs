import configparser
from os import makedirs

import torch

from .training import EmbeddingConverterTrainer

CONFIG_PARSER = configparser.ConfigParser()
CONFIG_PARSER.read('config.ini')


def export() -> None:
	config =\
	{
		'directory_path': CONFIG_PARSER.get('exporting', 'directory_path'),
		'source_path': CONFIG_PARSER.get('exporting', 'source_path'),
		'target_path': CONFIG_PARSER.get('exporting', 'target_path'),
		'ir_version': CONFIG_PARSER.getint('exporting', 'ir_version'),
		'opset_version': CONFIG_PARSER.getint('exporting', 'opset_version')
	}

	makedirs(config.get('directory_path'), exist_ok = True) # type:ignore[arg-type]
	model = EmbeddingConverterTrainer.load_from_checkpoint(config.get('source_path'), map_location = 'cpu')
	model.eval()
	model.ir_version = torch.tensor(config.get('ir_version'))
	input_tensor = torch.randn(1, 512)
	torch.onnx.export(model, input_tensor, config.get('target_path'), input_names = [ 'input' ], output_names = [ 'output' ], opset_version = config.get('opset_version'))
