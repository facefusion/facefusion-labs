import configparser

import torch
from torchvision import io

from .helper import calc_embedding
from .models.generator import Generator

CONFIG = configparser.ConfigParser()
CONFIG.read('config.ini')


def infer() -> None:
	generator_path = CONFIG.get('inferencing', 'generator_path')
	embedder_path = CONFIG.get('inferencing', 'embedder_path')
	source_path = CONFIG.get('inferencing', 'source_path')
	target_path = CONFIG.get('inferencing', 'target_path')
	output_path = CONFIG.get('inferencing', 'output_path')

	state_dict = torch.load(generator_path).get('state_dict').get('generator')
	generator = Generator()
	generator.load_state_dict(state_dict)
	generator.eval()
	embedder = torch.jit.load(embedder_path, map_location = 'cpu') # type:ignore[no-untyped-call]
	embedder.eval()

	source_tensor = io.read_image(source_path)
	target_tensor = io.read_image(target_path)
	source_embedding = calc_embedding(embedder, source_tensor, (0, 0, 0, 0))
	output_tensor = generator(source_embedding, target_tensor)[0]
	io.write_jpeg(output_tensor, output_path)
