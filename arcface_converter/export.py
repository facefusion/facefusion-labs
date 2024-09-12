#!/usr/bin/env python3

import configparser

import torch

from arcface_converter.model import ArcFaceConverter

CONFIG = configparser.ConfigParser()
CONFIG.read('config.ini')


def export(output_path : str, export_path : str) -> None:
	model = ArcFaceConverter()
	model.load_state_dict(torch.load(output_path, map_location = 'cpu'))
	model.eval()
	input_embedding = torch.randn(1, 512)
	torch.onnx.export(model, input_embedding, export_path, input_names = [ 'input' ], output_names = [ 'output' ], opset_version = 15)


if __name__ == '__main__':
	export(CONFIG['outputs']['best_path'], CONFIG['exports']['best_path'])
	export(CONFIG['outputs']['final_path'], CONFIG['exports']['final_path'])
