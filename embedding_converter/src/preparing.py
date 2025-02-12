import configparser
from os import makedirs
from os.path import isfile
from typing import List

import numpy
numpy.bool = numpy.bool_
from mxnet.io import ImageRecordIter
from onnxruntime import InferenceSession
from tqdm import tqdm

from .types import Embedding, EmbeddingDataset, VisionFrame

CONFIG = configparser.ConfigParser()
CONFIG.read('config.ini')


def prepare_crop_vision_frame(crop_vision_frame : VisionFrame) -> VisionFrame:
	crop_vision_frame = crop_vision_frame.astype(numpy.float32) / 255.0
	crop_vision_frame = (crop_vision_frame - 0.5) * 2
	return crop_vision_frame


def create_inference_session(model_path : str, execution_providers : List[str]) -> InferenceSession:
	inference_session = InferenceSession(model_path, providers = execution_providers)
	return inference_session


def forward(inference_session : InferenceSession, crop_vision_frame : VisionFrame) -> Embedding:
	embedding = inference_session.run(None,
	{
		'input': crop_vision_frame
	})[0]

	return embedding


def create_embedding_dataset(dataset_reader : ImageRecordIter, source_inference_session : InferenceSession, target_inference_session : InferenceSession) -> EmbeddingDataset:
	dataset_process_limit = CONFIG.getint('preparing.dataset', 'process_limit')
	embedding_pairs = []

	with tqdm(total = dataset_process_limit) as progress:
		for batch in dataset_reader:
			crop_vision_frame = batch.data[0].asnumpy()
			crop_vision_frame = prepare_crop_vision_frame(crop_vision_frame)
			source_embedding = forward(source_inference_session, crop_vision_frame)
			target_embedding = forward(target_inference_session, crop_vision_frame)
			embedding_pairs.append([ source_embedding, target_embedding ])
			progress.update()

			if progress.n == dataset_process_limit:
				return numpy.concatenate(embedding_pairs, axis = 1).T

	return numpy.concatenate(embedding_pairs, axis = 1).T


def prepare() -> None:
	dataset_path = CONFIG.get('preparing.dataset', 'dataset_path')
	dataset_crop_size = CONFIG.getint('preparing.dataset', 'crop_size')
	model_source_path = CONFIG.get('preparing.model', 'source_path')
	model_target_path = CONFIG.get('preparing.model', 'target_path')
	input_directory_path = CONFIG.get('preparing.input', 'directory_path')
	input_source_path = CONFIG.get('preparing.input', 'source_path')
	input_target_path = CONFIG.get('preparing.input', 'target_path')
	execution_providers = CONFIG.get('execution', 'providers').split(' ')

	makedirs(input_directory_path, exist_ok = True)
	if isfile(dataset_path) and isfile(model_source_path) and isfile(model_target_path):
		dataset_reader = ImageRecordIter(
			path_imgrec = dataset_path,
			data_shape = (3, dataset_crop_size, dataset_crop_size),
			batch_size = 1,
			shuffle = False
		)
		source_inference_session = create_inference_session(model_source_path, execution_providers)
		target_inference_session = create_inference_session(model_target_path, execution_providers)
		embedding_dataset = create_embedding_dataset(dataset_reader, source_inference_session, target_inference_session)
		numpy.save(input_source_path, embedding_dataset[..., 0].T)
		numpy.save(input_target_path, embedding_dataset[..., 1].T)
