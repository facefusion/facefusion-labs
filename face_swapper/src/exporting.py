import configparser
from os import makedirs

import torch

from .generator import AdaptiveEmbeddingIntegrationNetwork

CONFIG = configparser.ConfigParser()
CONFIG.read('config.ini')


def export() -> None:
	directory_path = CONFIG.get('exporting', 'directory_path')
	source_path = CONFIG.get('exporting', 'source_path')
	target_path = CONFIG.get('exporting', 'target_path')
	opset_version = CONFIG.getint('exporting', 'opset_version')

	makedirs(directory_path, exist_ok = True)
	state_dict = torch.load(source_path, map_location = 'cpu')['state_dict']['generator']
	model = AdaptiveEmbeddingIntegrationNetwork(512, 2)
	model.load_state_dict(state_dict)
	model.eval()
	source_tensor = torch.randn(1, 512)
	target_tensor = torch.randn(1, 3, 256, 256)
	torch.onnx.export(model, (target_tensor, source_tensor), target_path, input_names = [ 'target', 'source' ], output_names = [ 'output' ], opset_version = opset_version)
