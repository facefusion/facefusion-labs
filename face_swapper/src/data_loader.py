import glob
import os.path
import random
from typing import Tuple

import torch
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset

from .helper import read_image
from .types import Batch, ImagePathList, ImagePathSet


class DataLoaderVGG(TensorDataset):
	def __init__(self, dataset_path : str, dataset_image_pattern : str, dataset_folder_pattern : str, same_person_probability : float) -> None:
		self.same_person_probability = same_person_probability
		self.directory_paths = glob.glob(dataset_folder_pattern.format(dataset_path))
		self.image_paths, self.image_path_set = self.prepare_image_paths(dataset_image_pattern)
		self.dataset_total = len(self.image_paths)
		self.transforms = self.compose_transforms()

	def prepare_image_paths(self, dataset_image_pattern : str) -> Tuple[ImagePathList, ImagePathSet]:
		image_paths = []
		image_path_set = {}

		for directory_path in self.directory_paths:
			image_paths.extend(glob.glob(dataset_image_pattern.format(directory_path)))
			image_path_set[directory_path] = image_paths
		return image_paths, image_path_set

	def compose_transforms(self) -> transforms:
		transform = transforms.Compose(
		[
			transforms.ToPILImage(),
			transforms.Resize((256, 256), interpolation = transforms.InterpolationMode.BICUBIC),
			transforms.ColorJitter(brightness = 0.2, contrast = 0.2, saturation = 0.2, hue = 0.1),
			transforms.RandomAffine(4, translate = (0.01, 0.01), scale = (0.98, 1.02), shear = (1, 1)),
			transforms.ToTensor(),
			transforms.Lambda(lambda temp_tensor : temp_tensor[[2, 1, 0], :, :]),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		])
		return transform

	def __getitem__(self, index : int) -> Batch:
		source_image_path = self.image_paths[index]

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
