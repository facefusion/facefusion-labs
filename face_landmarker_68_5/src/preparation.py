#!/usr/bin/env python3
import configparser
import os
from typing import List

import cv2
import facer
import numpy
import torch
from tqdm import tqdm

CONFIG = configparser.ConfigParser()
CONFIG.read('config.ini')


def get_vision_frame_paths() -> List[str]:
	dataset_path = CONFIG.get('datasets', 'dataset_path')
	vision_frame_names = os.listdir(dataset_path)
	vision_frame_paths = []

	for vision_frame_name in vision_frame_names:
		vision_frame_path = os.path.join(dataset_path, vision_frame_name)

		if os.path.isfile(vision_frame_path):
			vision_frame_paths.append(vision_frame_path)

	vision_frame_paths.sort()
	return vision_frame_paths


def prepare() -> None:
	device = CONFIG.get('execution', 'device')
	face_detector = facer.face_detector('retinaface/mobilenet', device = device)
	face_aligner = facer.face_aligner('farl/ibug300w/448', device = device)
	vision_frame_paths = get_vision_frame_paths()
	source_landmark_list = []
	target_landmark_list = []

	for vision_frame_path in tqdm(vision_frame_paths):
		with torch.inference_mode():
			try:
				vision_frame = cv2.imread(vision_frame_path)[:, :, ::-1]
				vision_frame_torch = torch.from_numpy(vision_frame.copy()).permute(2, 0, 1).unsqueeze(0).to(device = CONFIG.get('execution', 'device'))
				faces = face_detector(vision_frame_torch)
				faces = face_aligner(vision_frame_torch, faces)
				face_landmarks_5 = faces.get('points')
				face_landmarks_68 = faces.get('alignment')

				for face_landmark_5, face_landmark_68 in zip(face_landmarks_5, face_landmarks_68):
					face_landmark_5 = face_landmark_5.detach().cpu().numpy()
					face_landmark_68 = face_landmark_68.detach().cpu().numpy()
					source_landmark_list.append(face_landmark_5)
					target_landmark_list.append(face_landmark_68)
			except Exception:
				continue

	source_landmarks = numpy.concatenate(source_landmark_list, axis = 0)
	target_landmarks = numpy.concatenate(target_landmark_list, axis = 0)
	numpy.save(CONFIG.get('landmarks', 'source_path'), source_landmarks)
	numpy.save(CONFIG.get('landmarks', 'target_path'), target_landmarks)
