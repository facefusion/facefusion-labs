import configparser

import torch
from torchvision import io

from .helper import calc_embedding
from .models.generator import Generator

CONFIG_PARSER = configparser.ConfigParser()
CONFIG_PARSER.read('config.ini')


def infer() -> None:
	config =\
	{
		'generator_path': CONFIG_PARSER.get('inferencing', 'generator_path'),
		'embedder_path': CONFIG_PARSER.get('inferencing', 'embedder_path'),
		'source_path': CONFIG_PARSER.get('inferencing', 'source_path'),
		'target_path': CONFIG_PARSER.get('inferencing', 'target_path'),
		'output_path': CONFIG_PARSER.get('inferencing', 'output_path')
	}

	state_dict = torch.load(config.get('generator_path')).get('state_dict').get('generator')
	generator = Generator(CONFIG_PARSER)
	generator.load_state_dict(state_dict)
	generator.eval()
	embedder = torch.jit.load(config.get('embedder_path'), map_location = 'cpu') # type:ignore[no-untyped-call]
	embedder.eval()

	source_tensor = io.read_image(config.get('source_path'))
	target_tensor = io.read_image(config.get('target_path'))
	source_embedding = calc_embedding(embedder, source_tensor, (0, 0, 0, 0))
	output_tensor = generator(source_embedding, target_tensor)[0]
	io.write_jpeg(output_tensor, config.get('output_path'))
