import configparser
from os import makedirs

import torch

from .training import FaceSwapperTrainer

CONFIG = configparser.ConfigParser()
CONFIG.read('config.ini')


def export() -> None:
	directory_path = CONFIG.get('exporting', 'directory_path')
	source_path = CONFIG.get('exporting', 'source_path')
	target_path = CONFIG.get('exporting', 'target_path')
	target_size = CONFIG.getint('exporting', 'target_size')
	ir_version = CONFIG.getint('exporting', 'ir_version')
	opset_version = CONFIG.getint('exporting', 'opset_version')

	makedirs(directory_path, exist_ok = True)
	model = FaceSwapperTrainer.load_from_checkpoint(source_path, map_location = 'cpu')
	model.eval()
	model.ir_version = torch.tensor(ir_version)
	source_tensor = torch.randn(1, 512)
	target_tensor = torch.randn(1, 3, target_size, target_size)
	torch.onnx.export(model, (source_tensor, target_tensor), target_path, input_names = [ 'source', 'target' ], output_names = [ 'output' ], opset_version = opset_version)
