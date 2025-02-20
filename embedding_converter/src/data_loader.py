import glob
import random

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from .helper import read_image
from .types import Batch, Paths


class DataLoaderRecognition(Dataset[torch.Tensor]):
	def __init__(self, dataset_file_pattern : str) -> None:
		self.image_paths = self.prepare_image_paths(dataset_file_pattern)
		self.dataset_total = len(self.image_paths)
		self.transforms = self.compose_transforms()

	def __getitem__(self, index : int) -> Batch:
		target_image_path = random.choice(self.image_paths)
		target_vision_frame = read_image(target_image_path)
		target_tensor = self.transforms(target_vision_frame)
		return target_tensor

	def __len__(self) -> int:
		return self.dataset_total

	def prepare_image_paths(self, dataset_file_pattern : str) -> Paths:
		return glob.glob(dataset_file_pattern)

	def compose_transforms(self) -> transforms:
		transform = transforms.Compose(
		[
			transforms.ToPILImage(),
			transforms.Resize((112, 112), interpolation = transforms.InterpolationMode.BICUBIC),
			transforms.ColorJitter(brightness = 0.2, contrast = 0.2, saturation = 0.2, hue = 0.1),
			transforms.ToTensor(),
			transforms.Lambda(lambda temp_tensor : temp_tensor[[2, 1, 0], :, :]),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		])
		return transform
