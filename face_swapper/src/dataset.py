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
	def __init__(self, file_pattern : str, warp_template : WarpTemplate, batch_mode : BatchMode, batch_ratio : float, resolution : int) -> None:
		self.file_paths = glob.glob(file_pattern)
		self.warp_template = warp_template
		self.batch_mode = batch_mode
		self.batch_ratio = batch_ratio
		self.resolution = resolution
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
		return transforms.Compose(
		[
			AugmentTransform(),
			transforms.ToPILImage(),
			transforms.Resize((self.resolution, self.resolution), interpolation = transforms.InterpolationMode.BICUBIC),
			transforms.ToTensor(),
			WarpTransform(self.warp_template),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		])

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


class AugmentTransform:
	def __init__(self) -> None:
		self.transforms = self.compose_transforms()

	def __call__(self, input_tensor : Tensor) -> Tensor:
		temp_tensor = input_tensor.numpy().transpose(1, 2, 0)
		return self.transforms(image = temp_tensor).get('image')

	@staticmethod
	def compose_transforms() -> albumentations.Compose:
		return albumentations.Compose(
		[
			albumentations.RandomBrightnessContrast(p = 0.3),
			albumentations.OneOf(
			[
				albumentations.MotionBlur(p = 0.1),
				albumentations.MedianBlur(p = 0.1)
			], p = 0.3),
			albumentations.ColorJitter(p = 0.1)
		])


class WarpTransform:
	def __init__(self, warp_template : WarpTemplate) -> None:
		self.warp_template = warp_template

	def __call__(self, input_tensor : Tensor) -> Tensor:
		temp_tensor = input_tensor.unsqueeze(0)
		return warp_tensor(temp_tensor, self.warp_template).squeeze(0)
