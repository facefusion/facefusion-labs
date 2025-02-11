import configparser
from os import makedirs

import torch

from .training import EmbeddingForgeTrainer

CONFIG = configparser.ConfigParser()
CONFIG.read('config.ini')


def export() -> None:
	directory_path = CONFIG.get('exporting', 'directory_path')
	source_path = CONFIG.get('exporting', 'source_path')
	target_path = CONFIG.get('exporting', 'target_path')
	opset_version = CONFIG.getint('exporting', 'opset_version')

	makedirs(directory_path, exist_ok = True)
	embedding_forge_trainer = EmbeddingForgeTrainer.load_from_checkpoint(source_path, map_location ='cpu')
	embedding_forge_trainer.eval()
	input_tensor = torch.randn(1, 512)
	torch.onnx.export(embedding_forge_trainer, input_tensor, target_path, input_names = [ 'input' ], output_names = [ 'output' ], opset_version = opset_version)
