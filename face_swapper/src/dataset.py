import glob
import os
import random

import albumentations
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import io, transforms

from .helper import warp_tensor
from .types import Batch, BatchMode, WarpTemplate


class DynamicDataset(Dataset[Tensor]):
	def __init__(self, file_pattern : str, warp_template : WarpTemplate, batch_mode : BatchMode, batch_ratio : float) -> None:
		self.file_paths = glob.glob(file_pattern)
		self.warp_template = warp_template
		self.batch_mode = batch_mode
		self.batch_ratio = batch_ratio
		self.transforms = self.compose_transforms()

	def __getitem__(self, index : int) -> Batch:
		file_path = self.file_paths[index]

		if random.random() < self.batch_ratio:
			if self.batch_mode == 'equal':
				return self.prepare_equal_batch(file_path)
			if self.batch_mode == 'same':
				return self.prepare_same_batch(file_path)

		return self.prepare_different_batch(file_path)

	def __len__(self) -> int:
		return len(self.file_paths)

	def compose_transforms(self) -> transforms:
		augmentations = albumentations.Compose(
		[
			albumentations.RandomBrightnessContrast(p = 0.3),
			albumentations.OneOf(
			[
				albumentations.MotionBlur(p = 0.1),
				albumentations.MedianBlur(p = 0.1)
			], p = 0.3),
			albumentations.ColorJitter(p = 0.1),
		])
		return transforms.Compose(
		[
			transforms.Lambda(lambda temp : augmentations(image = temp.numpy().transpose(1, 2, 0)).get('image')),
			transforms.ToPILImage(),
			transforms.Resize(256),
			transforms.ToTensor(),
			transforms.Lambda(self.warp_tensor),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		])

	def warp_tensor(self, temp_tensor : Tensor) -> Tensor:
		return warp_tensor(temp_tensor.unsqueeze(0), self.warp_template).squeeze(0)

	def prepare_different_batch(self, source_path : str) -> Batch:
		target_path = random.choice(self.file_paths)
		source_tensor = io.read_image(source_path)
		source_tensor = self.transforms(source_tensor)
		target_tensor = io.read_image(target_path)
		target_tensor = self.transforms(target_tensor)
		return source_tensor, target_tensor

	def prepare_equal_batch(self, source_path : str) -> Batch:
		source_tensor = io.read_image(source_path)
		source_tensor = self.transforms(source_tensor)
		return source_tensor, source_tensor

	def prepare_same_batch(self, source_path : str) -> Batch:
		target_directory_path = os.path.dirname(source_path)
		target_file_name_and_extension = random.choice(os.listdir(target_directory_path))
		target_path = os.path.join(target_directory_path, target_file_name_and_extension)
		source_tensor = io.read_image(source_path)
		source_tensor = self.transforms(source_tensor)
		target_tensor = io.read_image(target_path)
		target_tensor = self.transforms(target_tensor)
		return source_tensor, target_tensor
