import glob
import os
import random
from configparser import ConfigParser
from typing import Tuple, cast

import albumentations
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import io, transforms

from .helper import warp_tensor
from .types import Batch, BatchMode, FileSet, WarpTemplate


class DynamicDataset(Dataset[Tensor]):
	def __init__(self, config_parser : ConfigParser) -> None:
		self.config_transform_size = config_parser.getint('training.dataset', 'transform_size')
		self.config_batch_mode = cast(BatchMode, config_parser.get('training.dataset', 'batch_mode'))
		self.config_batch_ratio = config_parser.getfloat('training.dataset', 'batch_ratio')
		self.config_parser = config_parser
		self.file_set = self.resolve_file_set()
		self.transforms = self.compose_transforms()

	def __getitem__(self, index : int) -> Batch:
		warp_template, file_paths = random.choice(list(self.file_set.items()))
		file_path = file_paths[index]

		if random.random() < self.config_batch_ratio:
			if self.config_batch_mode == 'equal':
				return self.prepare_equal_batch(file_path, warp_template)
			if self.config_batch_mode == 'same':
				return self.prepare_same_batch(file_path, warp_template)

		return self.prepare_different_batch(file_path, warp_template)

	def __len__(self) -> int:
		return len(self.file_set)

	def resolve_file_set(self) -> FileSet:
		file_set = {}

		for warp_template, file_pattern in self.config_parser.items('training.dataset.files'):
			warp_template = cast(WarpTemplate, warp_template)
			if file_pattern:
				file_set[warp_template] = glob.glob(file_pattern)

		return file_set

	def compose_transforms(self) -> transforms:
		return transforms.Compose(
		[
			AugmentTransform(),
			transforms.ToPILImage(),
			transforms.Resize((self.config_transform_size, self.config_transform_size), interpolation = transforms.InterpolationMode.BICUBIC),
			transforms.ToTensor(),
			WarpTransform(),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		])

	def prepare_different_batch(self, source_path : str, warp_template : WarpTemplate) -> Batch:
		_, file_paths = random.choice(list(self.file_set.items()))
		target_path = random.choice(file_paths)
		source_tensor = io.read_image(source_path)
		source_tensor = self.transforms(source_tensor, warp_template)
		target_tensor = io.read_image(target_path)
		target_tensor = self.transforms(target_tensor, warp_template)
		return source_tensor, target_tensor

	def prepare_equal_batch(self, source_path : str, warp_template : WarpTemplate) -> Batch:
		source_tensor = io.read_image(source_path)
		source_tensor = self.transforms(source_tensor, warp_template)
		return source_tensor, source_tensor

	def prepare_same_batch(self, source_path : str, warp_template : WarpTemplate) -> Batch:
		target_directory_path = os.path.dirname(source_path)
		target_file_name_and_extension = random.choice(os.listdir(target_directory_path))
		target_path = os.path.join(target_directory_path, target_file_name_and_extension)
		source_tensor = io.read_image(source_path)
		source_tensor = self.transforms(source_tensor, warp_template)
		target_tensor = io.read_image(target_path)
		target_tensor = self.transforms(target_tensor, warp_template)
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
	def __call__(self, input_tensor_and_warp_template : Tuple[Tensor, WarpTemplate]) -> Tensor:
		input_tensor, warp_template = input_tensor_and_warp_template
		temp_tensor = input_tensor.unsqueeze(0)
		output_tensor = warp_tensor(temp_tensor, warp_template).squeeze(0)
		return output_tensor
