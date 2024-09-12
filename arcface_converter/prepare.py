#!/usr/bin/env python3
import configparser
import os
from typing import Any, List, Tuple

import cv2
import facer
import numpy
import onnxruntime
import torch
from numpy.typing import NDArray
from onnxruntime import InferenceSession
from tqdm import tqdm

CONFIG = configparser.ConfigParser()
CONFIG.read('config.ini')

TEMPLATE_POINTS = numpy.array(
[
    [ 38.2946, 51.6963 ],
	[ 73.5318, 51.5014 ],
	[ 56.0252, 71.7366 ],
	[ 41.5493, 92.3655 ],
	[ 70.7299, 92.2041 ]
], numpy.float32)


def get_image_paths() -> List[str]:
	dataset_path = CONFIG['datasets']['dataset_path']
	image_names = os.listdir(dataset_path)
	image_paths = []

	for image_name in image_names:
		image_path = os.path.join(dataset_path, image_name)
		if os.path.isfile(image_path):
			image_paths.append(image_path)
	image_paths.sort()
	return image_paths


def warp_image(image : NDArray[Any], face_landmark_5 : NDArray[Any]) -> NDArray[Any]:
    matrix = cv2.estimateAffinePartial2D(face_landmark_5, TEMPLATE_POINTS, method = cv2.RANSAC, ransacReprojThreshold = 100)[0]
    crop_image = cv2.warpAffine(image, matrix, (112, 112), borderMode = cv2.BORDER_REPLICATE)
    return crop_image


def prepare_crop_image(image : NDArray[Any], face_landmarks_5 : NDArray[Any]) -> NDArray[Any]:
	crop_image = warp_image(image, face_landmarks_5)
	crop_image = crop_image.astype(numpy.float32) / 255
	crop_image = (crop_image - 0.5) * 2
	crop_image = crop_image.transpose(2, 0, 1)
	crop_image = crop_image[None]
	return crop_image


def get_arcface_sessions(device : str) -> Tuple[InferenceSession, InferenceSession]:
	source_path = CONFIG['models']['source_path']
	target_path = CONFIG['models']['target_path']
	provider = 'CUDAExecutionProvider' if device == 'cuda' else 'CPUExecutionProvider'
	source_session = onnxruntime.InferenceSession(source_path, providers = [ provider ])
	target_session = onnxruntime.InferenceSession(target_path, providers = [ provider ])
	return source_session, target_session


def forward_arcface_session(session : InferenceSession, crop_image : NDArray[Any]) -> NDArray[Any]:
	embedding = session.run(None,
	{
		'input': crop_image
	})[0]
	return embedding


def prepare(device : str) -> Tuple[NDArray[Any], NDArray[Any]]:
	face_detector = facer.face_detector('retinaface/mobilenet', device = device)
	image_paths = get_image_paths()
	source_session, target_session = get_arcface_sessions(device)
	source_embedding_list = []
	target_embedding_list = []

	for image_path in tqdm(image_paths):
		with torch.inference_mode():
			try:
				image = cv2.imread(image_path)[:, :, ::-1]
				image_torch = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device = device)
				face_landmarks_5_list = face_detector(image_torch)['points']
				for face_landmarks_5 in face_landmarks_5_list:
					face_landmarks_5 = face_landmarks_5.detach().cpu().numpy()
					crop_image = prepare_crop_image(image, face_landmarks_5)
					source_embedding = forward_arcface_session(source_session, crop_image)
					target_embedding = forward_arcface_session(target_session, crop_image)
					source_embedding_list.append(source_embedding)
					target_embedding_list.append(target_embedding)
			except:
				continue
	source_embeddings = numpy.concatenate(source_embedding_list, axis = 0)
	target_embeddings = numpy.concatenate(target_embedding_list, axis = 0)
	return source_embeddings, target_embeddings


if __name__ == '__main__':
	device = 'cuda'
	source_embeddings, target_embeddings = prepare(device)
	numpy.save(CONFIG['embeddings']['source_path'], source_embeddings)
	numpy.save(CONFIG['embeddings']['target_path'], target_embeddings)
