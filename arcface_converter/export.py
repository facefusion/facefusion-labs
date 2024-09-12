#!/usr/bin/env python3

import configparser

import torch

from arcface_converter.model import ArcFaceConverter

CONFIG = configparser.ConfigParser()
CONFIG.read('config.ini')


def export() -> None:
	checkpoint_path = './checkpoints/checkpoint.pth'
	model = ArcFaceConverter()
	model.load_state_dict(torch.load(checkpoint_path, map_location = 'cpu'))
	model.eval()
	input_embedding = torch.randn(1, 512)
	torch.onnx.export(model, input_embedding, 'model.onnx', input_names = [ 'input' ], output_names = [ 'output' ], opset_version = 15)


if __name__ == '__main__':
	export()
