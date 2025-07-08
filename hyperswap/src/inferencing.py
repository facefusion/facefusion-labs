import configparser

import torch
from torchvision import io

from .helper import calculate_face_embedding
from .training import HyperSwapTrainer

CONFIG_PARSER = configparser.ConfigParser()
CONFIG_PARSER.read('config.ini')


def infer() -> None:
	config_generator_path = CONFIG_PARSER.get('inferencing', 'generator_path')
	config_embedder_path = CONFIG_PARSER.get('inferencing', 'embedder_path')
	config_source_path = CONFIG_PARSER.get('inferencing', 'source_path')
	config_target_path = CONFIG_PARSER.get('inferencing', 'target_path')
	config_output_path = CONFIG_PARSER.get('inferencing', 'output_path')

	generator = HyperSwapTrainer.load_from_checkpoint(config_generator_path, config_parser = CONFIG_PARSER, map_location ='cpu').eval()
	embedder = torch.jit.load(config_embedder_path, map_location = 'cpu').eval()

	source_tensor = io.read_image(config_source_path)
	target_tensor = io.read_image(config_target_path)
	source_embedding = calculate_face_embedding(embedder, source_tensor, (0, 0, 0, 0))
	output_tensor, _ = generator(source_embedding, target_tensor)
	io.write_jpeg(output_tensor, config_output_path)
