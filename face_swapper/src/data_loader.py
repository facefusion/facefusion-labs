import configparser
import glob
import os.path
import random

import cv2
import torch
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset

from .typing import Batch, VisionFrame

CONFIG = configparser.ConfigParser()
CONFIG.read('config.ini')


def read_image(image_path: str) -> VisionFrame:
	image = cv2.imread(image_path)[:, :, ::-1]
	return image


class DataLoaderVGG(TensorDataset):
	def __init__(self, dataset_path : str) -> None:
		self.same_person_probability = CONFIG.getfloat('preparing.dataloader', 'same_person_probability')
		self.image_paths = glob.glob('{}/*/*.*g'.format(dataset_path)) # @todo globs belong to the config
		self.folder_paths = glob.glob('{}/*'.format(dataset_path))
		self.image_path_set = {}
		self._current_index = 0

		for folder_path in self.folder_paths:
			image_paths = glob.glob('{}/*'.format(folder_path)) # @todo not sure about alls this globs being used here :-)
			self.image_path_set[folder_path] = image_paths
		self.dataset_total = len(self.image_paths)
		self.transforms = transforms.Compose(
		[
			transforms.ToPILImage(),
			transforms.Resize((256, 256), interpolation = transforms.InterpolationMode.BICUBIC),
			transforms.ColorJitter(brightness = 0.2, contrast = 0.2, saturation = 0.2, hue = 0.1),
			transforms.RandomAffine(4, translate = (0.01, 0.01), scale = (0.98, 1.02), shear = (1, 1), fill = 0),
			transforms.ToTensor(),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		])

	def __getitem__(self, item : int) -> Batch:
		source_image_path = self.image_paths[item]

		if random.random() > self.same_person_probability:
			return self.prepare_same_person(source_image_path)
		return self.prepare_different_person(source_image_path)

	def prepare_different_person(self, source_image_path : str) -> Batch:
		is_same_person = torch.tensor(0)
		target_image_path = random.choice(self.image_paths)
		source_vision_frame = read_image(source_image_path)
		target_vision_frame = read_image(target_image_path)
		source_tensor = self.transforms(source_vision_frame)
		target_tensor = self.transforms(target_vision_frame)
		return source_tensor, target_tensor, is_same_person

	def prepare_same_person(self, source_image_path : str) -> Batch:
		is_same_person = torch.tensor(1)
		target_image_path = random.choice(self.image_path_set.get(os.path.dirname(source_image_path)))
		source_vision_frame = read_image(source_image_path)
		target_vision_frame = read_image(target_image_path)
		source_tensor = self.transforms(source_vision_frame)
		target_tensor = self.transforms(target_vision_frame)
		return source_tensor, target_tensor, is_same_person

	def __len__(self) -> int:
		return self.dataset_total
