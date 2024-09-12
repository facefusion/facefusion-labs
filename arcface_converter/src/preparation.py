#!/usr/bin/env python3

import configparser
import os
from typing import List

import cv2
import facer
import numpy
import onnxruntime
import torch
from onnxruntime import InferenceSession
from tqdm import tqdm

from .typing import Embedding, FaceLandmark5, VisionFrame

CONFIG = configparser.ConfigParser()
CONFIG.read('config.ini')

WARP_TEMPLATE = numpy.array(
[
	[ 38.2946, 51.6963 ],
	[ 73.5318, 51.5014 ],
	[ 56.0252, 71.7366 ],
	[ 41.5493, 92.3655 ],
	[ 70.7299, 92.2041 ]
], numpy.float32)


def get_vision_frame_paths() -> List[str]:
	dataset_path = CONFIG['datasets']['dataset_path']
	vision_frame_names = os.listdir(dataset_path)
	vision_frame_paths = []

	for vision_frame_name in vision_frame_names:
		vision_frame_path = os.path.join(dataset_path, vision_frame_name)

		if os.path.isfile(vision_frame_path):
			vision_frame_paths.append(vision_frame_path)

	vision_frame_paths.sort()
	return vision_frame_paths


def warp_vision_frame(vision_frame : VisionFrame, face_landmark_5 : FaceLandmark5) -> VisionFrame:
	matrix = cv2.estimateAffinePartial2D(face_landmark_5, WARP_TEMPLATE, method = cv2.RANSAC, ransacReprojThreshold = 100)[0]
	crop_vision_frame = cv2.warpAffine(vision_frame, matrix, (112, 112), borderMode = cv2.BORDER_REPLICATE)
	return crop_vision_frame


def prepare_crop_vision_frame(vision_frame : VisionFrame, face_landmark_5 : FaceLandmark5) -> VisionFrame:
	crop_vision_frame = warp_vision_frame(vision_frame, face_landmark_5)
	crop_vision_frame = crop_vision_frame.astype(numpy.float32) / 255
	crop_vision_frame = (crop_vision_frame - 0.5) * 2
	crop_vision_frame = crop_vision_frame.transpose(2, 0, 1)
	crop_vision_frame = crop_vision_frame[None]
	return crop_vision_frame


def create_inference_session(model_path : str, execution_providers : List[str]) -> InferenceSession:
	inference_session = onnxruntime.InferenceSession(model_path, providers = execution_providers)
	return inference_session


def forward(inference_session : InferenceSession, crop_vision_frame : VisionFrame) -> Embedding:
	embedding = inference_session.run(None,
	{
		'input': crop_vision_frame
	})[0]

	return embedding


def prepare() -> None:
	face_detector = facer.face_detector('retinaface/mobilenet', device = CONFIG['execution']['device'])
	vision_frame_paths = get_vision_frame_paths()
	source_session = create_inference_session(CONFIG['models']['source_path'], [ CONFIG['execution']['providers'] ])
	target_session = create_inference_session(CONFIG['models']['target_path'], [ CONFIG['execution']['providers'] ])
	source_embedding_list = []
	target_embedding_list = []

	for vision_frame_path in tqdm(vision_frame_paths):
		with torch.inference_mode():
			try:
				vision_frame = cv2.imread(vision_frame_path)[:, :, ::-1]
				vision_frame_torch = torch.from_numpy(vision_frame).permute(2, 0, 1).unsqueeze(0).to(device = CONFIG['execution']['device'])
				face_landmarks_5 = face_detector(vision_frame_torch).get('points')

				for face_landmark_5 in face_landmarks_5:
					face_landmark_5 = face_landmark_5.detach().cpu().numpy()
					crop_vision_frame = prepare_crop_vision_frame(vision_frame, face_landmark_5)
					source_embedding = forward(source_session, crop_vision_frame)
					target_embedding = forward(target_session, crop_vision_frame)
					source_embedding_list.append(source_embedding)
					target_embedding_list.append(target_embedding)
			except Exception:
				continue

	source_embeddings = numpy.concatenate(source_embedding_list, axis = 0)
	target_embeddings = numpy.concatenate(target_embedding_list, axis = 0)
	numpy.save(CONFIG['embeddings']['source_path'], source_embeddings)
	numpy.save(CONFIG['embeddings']['target_path'], target_embeddings)
