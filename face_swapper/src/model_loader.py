import configparser

import torch
import torch.nn as nn

from .discriminator import MultiscaleDiscriminator
from .generator import AdaptiveEmbeddingIntegrationNetwork

CONFIG = configparser.ConfigParser()
CONFIG.read('config.ini')


def load_generator() -> nn.Module:
	id_channels = CONFIG.getint('training.generator', 'id_channels')
	num_blocks = CONFIG.getint('training.generator', 'num_blocks')
	generator = AdaptiveEmbeddingIntegrationNetwork(id_channels, num_blocks)
	return generator


def load_discriminator() -> nn.Module:
	input_channels = CONFIG.getint('training.discriminator', 'input_channels')
	num_filters = CONFIG.getint('training.discriminator', 'num_filters')
	num_layers = CONFIG.getint('training.discriminator', 'num_layers')
	num_discriminators = CONFIG.getint('training.discriminator', 'num_discriminators')
	discriminator = MultiscaleDiscriminator(input_channels, num_filters, num_layers, num_discriminators)
	return discriminator


def load_arcface() -> nn.Module:
	model_path = CONFIG.get('auxiliary_models.paths', 'arcface_path')
	arcface = torch.load(model_path, map_location = 'cpu', weights_only = False)
	arcface.eval()
	return arcface


def load_landmarker() -> nn.Module:
	model_path = CONFIG.get('auxiliary_models.paths', 'landmarker_path')
	landmarker = torch.load(model_path, map_location = 'cpu', weights_only = False)
	landmarker.eval()
	return landmarker


def load_motion_extractor() -> nn.Module:
	from LivePortrait.src.modules.motion_extractor import MotionExtractor

	model_path = CONFIG.get('auxiliary_models.paths', 'motion_extractor_path')
	motion_extractor = MotionExtractor(num_kp = 21, backbone = 'convnextv2_tiny')
	motion_extractor.load_state_dict(torch.load(model_path, map_location = 'cpu', weights_only = True))
	motion_extractor.eval()
	return motion_extractor
