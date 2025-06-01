import glob
import os
import random
from configparser import ConfigParser
from typing import List, cast

import albumentations
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import io, transforms

from .helper import convert_tensor
from .types import Batch, BatchMode, ConvertTemplate, FileSet, UsageMode


class FilePool:
	def __init__(self, config_parser : ConfigParser) -> None:
		self.config_parser = config_parser
		self.file_pool = self.create_file_pool()

	def create_file_pool(self) -> List[FileSet]:
		file_pool : List[FileSet] = []

		for config_section in self.config_parser.sections():

			if config_section.startswith('training.dataset'):
				usage_mode = cast(UsageMode, self.config_parser.get(config_section, 'usage_mode'))
				file_pattern = self.config_parser.get(config_section, 'file_pattern')
				file_paths = glob.glob(file_pattern)

				file_pool.append(
				{
					'dataset_name': config_section,
					'usage_mode': usage_mode,
					'file_paths': file_paths
				})

		return file_pool

	def find_by_dataset_name(self, dataset_name : str) -> List[str]:
		file_paths: List[str] = []

		for entry in self.file_pool:
			if entry.get('dataset_name') == dataset_name:
				file_paths.extend(entry.get('file_paths'))

		return file_paths

	def find_by_usage_mode(self, usage_mode: UsageMode) -> List[str]:
		file_paths : List[str] = []

		for entry in self.file_pool:
			if entry.get('usage_mode') == usage_mode:
				file_paths.extend(entry.get('file_paths'))

		return file_paths


class DynamicDataset(Dataset[Tensor]):
	def __init__(self, config_parser : ConfigParser, file_pool : FilePool) -> None:
		self.config_dataset_name = config_parser.get('training.dataset', 'dataset_name')
		self.config_file_pattern = config_parser.get('training.dataset', 'file_pattern')
		self.config_convert_template = cast(ConvertTemplate, config_parser.get('training.dataset', 'convert_template'))
		self.config_transform_size = config_parser.getint('training.dataset', 'transform_size')
		self.config_usage_mode = cast(BatchMode, config_parser.get('training.dataset', 'usage_mode'))
		self.config_batch_mode = cast(BatchMode, config_parser.get('training.dataset', 'batch_mode'))
		self.config_batch_ratio = config_parser.getfloat('training.dataset', 'batch_ratio')
		self.config_parser = config_parser
		self.file_pool = file_pool
		self.transforms = self.compose_transforms()

	def __getitem__(self, index : int) -> Batch:
		file_path = self.file_pool.find_by_dataset_name(self.config_dataset_name)[index]

		if random.random() < self.config_batch_ratio:
			if self.config_batch_mode == 'equal':
				return self.prepare_equal_batch(file_path)
			if self.config_batch_mode == 'same':
				return self.prepare_same_batch(file_path)

		if self.config_usage_mode == 'source':
			return self.prepare_source_batch(file_path)

		if self.config_usage_mode == 'target':
			return self.prepare_target_batch(file_path)

		return self.prepare_different_batch(file_path)

	def __len__(self) -> int:
		return len(self.file_pool.find_by_dataset_name(self.config_dataset_name))

	def compose_transforms(self) -> transforms:
		__transforms__ =\
		[
			AugmentTransform(),
			transforms.ToPILImage(),
			transforms.Resize((self.config_transform_size, self.config_transform_size), interpolation = transforms.InterpolationMode.BICUBIC),
			transforms.ToTensor()
		]

		if self.config_convert_template:
			__transforms__.append(ConvertTensorTransform(self.config_parser))

		__transforms__.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

		return transforms.Compose(__transforms__)

	def prepare_source_batch(self, source_path : str) -> Batch:
		target_path = random.choice(self.file_pool.find_by_usage_mode('both'))
		source_tensor = io.read_image(source_path)
		source_tensor = self.transforms(source_tensor)
		target_tensor = io.read_image(target_path)
		target_tensor = self.transforms(target_tensor)
		return source_tensor, target_tensor

	def prepare_target_batch(self, target_path : str) -> Batch:
		source_path = random.choice(self.file_pool.find_by_usage_mode('both'))
		source_tensor = io.read_image(source_path)
		source_tensor = self.transforms(source_tensor)
		target_tensor = io.read_image(target_path)
		target_tensor = self.transforms(target_tensor)
		return source_tensor, target_tensor

	def prepare_different_batch(self, source_path : str) -> Batch:
		target_path = random.choice(self.file_pool.find_by_dataset_name(self.config_dataset_name))
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
			albumentations.HorizontalFlip(),
			albumentations.OneOf(
			[
				albumentations.MotionBlur(p = 0.1),
				albumentations.ZoomBlur(max_factor = (1.0, 1.1), p = 0.1)
			], p = 0.2),
			albumentations.RandomBrightnessContrast(p = 0.7),
			albumentations.ColorJitter(p = 0.2),
			albumentations.RGBShift(p = 0.7),
			albumentations.Illumination(p = 0.2),
			albumentations.Affine(translate_percent = (-0.03, 0.03), scale = (0.98, 1.02), rotate = (-2, 2), border_mode = 1, p = 0.3)
		])


class ConvertTensorTransform:
	def __init__(self, config_parser : ConfigParser) -> None:
		self.config_convert_template = cast(ConvertTemplate, config_parser.get('training.dataset', 'convert_template'))

	def __call__(self, input_tensor : Tensor) -> Tensor:
		temp_tensor = input_tensor.unsqueeze(0)
		return convert_tensor(temp_tensor, self.config_convert_template).squeeze(0)
