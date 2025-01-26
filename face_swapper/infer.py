import configparser

import cv2
import torch
from src.generator import AdaptiveEmbeddingIntegrationNetwork
from src.helper import infer, read_image

CONFIG = configparser.ConfigParser()
CONFIG.read('config.ini')


if __name__ == '__main__':
	generator_path = CONFIG.get('inference', 'generator_path')
	id_embedder_path = CONFIG.get('inference', 'id_embedder_path')
	source_path = CONFIG.get('inference', 'source_path')
	target_path = CONFIG.get('inference', 'target_path')
	output_path = CONFIG.get('inference', 'output_path')

	state_dict = torch.load(generator_path, map_location = 'cpu')['state_dict']['generator']
	generator = AdaptiveEmbeddingIntegrationNetwork(512, 2)
	generator.load_state_dict(state_dict)
	generator.eval()
	id_embedder = torch.jit.load(id_embedder_path, map_location = 'cpu') #type:ignore[no-untyped-call]
	id_embedder.eval()

	source_vision_frame = read_image(source_path)
	target_vision_frame = read_image(target_path)
	output_vision_frame = infer(generator, id_embedder, source_vision_frame, target_vision_frame)
	cv2.imwrite(output_path, output_vision_frame)
