import configparser

import cv2
import torch

from .generator import AdaptiveEmbeddingIntegrationNetwork
from .helper import calc_id_embedding, convert_to_vision_frame, convert_to_vision_tensor, read_image
from .types import Generator, IdEmbedder, VisionFrame

CONFIG = configparser.ConfigParser()
CONFIG.read('config.ini')


def run_swap(generator : Generator, id_embedder : IdEmbedder, source_vision_frame : VisionFrame, target_vision_frame : VisionFrame) -> VisionFrame:
	source_vision_tensor = convert_to_vision_tensor(source_vision_frame)
	target_vision_tensor = convert_to_vision_tensor(target_vision_frame)
	source_embedding = calc_id_embedding(id_embedder, source_vision_tensor, (0, 0, 0, 0))
	output_vision_tensor = generator(source_embedding, target_vision_tensor)[0]
	output_vision_frame = convert_to_vision_frame(output_vision_tensor)
	return output_vision_frame


def infer() -> None:
	generator_path = CONFIG.get('inferencing', 'generator_path')
	id_embedder_path = CONFIG.get('inferencing', 'id_embedder_path')
	source_path = CONFIG.get('inferencing', 'source_path')
	target_path = CONFIG.get('inferencing', 'target_path')
	output_path = CONFIG.get('inferencing', 'output_path')

	state_dict = torch.load(generator_path, map_location = 'cpu').get('state_dict').get('generator')
	generator = AdaptiveEmbeddingIntegrationNetwork(512, 2)
	generator.load_state_dict(state_dict)
	generator.eval()
	id_embedder = torch.jit.load(id_embedder_path, map_location = 'cpu')  # type:ignore[no-untyped-call]
	id_embedder.eval()

	source_vision_frame = read_image(source_path)
	target_vision_frame = read_image(target_path)
	output_vision_frame = run_swap(generator, id_embedder, source_vision_frame, target_vision_frame)
	cv2.imwrite(output_path, output_vision_frame)
