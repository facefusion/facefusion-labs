import os
import random
from configparser import ConfigParser
from typing import cast

import torch
from kornia import augmentation
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import io, transforms

from .helper import convert_tensor, resolve_static_file_pattern
from .types import Batch, BatchMode, ConvertTemplate, UsageMode


class DynamicDataset(Dataset[Tensor]):
	def __init__(self, config_parser : ConfigParser) -> None:
		self.config_file_pattern = config_parser.get('training.dataset.current', 'file_pattern')
		self.config_convert_template = cast(ConvertTemplate, config_parser.get('training.dataset.current', 'convert_template'))
		self.config_transform_size = config_parser.getint('training.dataset.current', 'transform_size')
		self.config_usage_mode = cast(UsageMode, config_parser.get('training.dataset.current', 'usage_mode'))
		self.config_batch_mode = cast(BatchMode, config_parser.get('training.dataset.current', 'batch_mode'))
		self.config_batch_ratio = config_parser.getfloat('training.dataset.current', 'batch_ratio')
		self.config_parser = config_parser
		self.transforms = self.compose_transforms()

	def __getitem__(self, index : int) -> Batch:
		file_path = resolve_static_file_pattern(self.config_file_pattern)[index]

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
		return len(resolve_static_file_pattern(self.config_file_pattern))

	def prepare_equal_batch(self, source_path : str) -> Batch:
		return self.create_batch(source_path, source_path, self.config_convert_template, self.config_convert_template)

	def prepare_same_batch(self, source_path : str) -> Batch:
		target_directory_path = os.path.dirname(source_path)
		target_file_name_and_extension = random.choice(os.listdir(target_directory_path))
		target_path = os.path.join(target_directory_path, target_file_name_and_extension)
		return self.create_batch(source_path, target_path, self.config_convert_template, self.config_convert_template)

	def prepare_source_batch(self, source_path : str) -> Batch:
		config_parser = self.filter_config_by_usage_mode('both')
		config_section = random.choice(config_parser.sections())
		config_file_pattern = config_parser.get(config_section, 'file_pattern')
		config_convert_template = cast(ConvertTemplate, config_parser.get(config_section, 'convert_template'))
		target_path = random.choice(resolve_static_file_pattern(config_file_pattern))
		return self.create_batch(source_path, target_path, self.config_convert_template, config_convert_template)

	def prepare_target_batch(self, target_path : str) -> Batch:
		config_parser = self.filter_config_by_usage_mode('both')
		config_section = random.choice(config_parser.sections())
		config_file_pattern = config_parser.get(config_section, 'file_pattern')
		config_convert_template = cast(ConvertTemplate, config_parser.get(config_section, 'convert_template'))
		source_path = random.choice(resolve_static_file_pattern(config_file_pattern))
		return self.create_batch(source_path, target_path, config_convert_template, self.config_convert_template)

	def prepare_different_batch(self, source_path : str) -> Batch:
		target_path = random.choice(resolve_static_file_pattern(self.config_file_pattern))
		return self.create_batch(source_path, target_path, self.config_convert_template, self.config_convert_template)

	def compose_transforms(self) -> transforms:
		return transforms.Compose(
		[
			AugmentTransform(),
			transforms.Resize((self.config_transform_size, self.config_transform_size), interpolation = transforms.InterpolationMode.BICUBIC),
			transforms.ConvertImageDtype(torch.float32),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		])

	def filter_config_by_usage_mode(self, usage_mode : UsageMode) -> ConfigParser:
		config_parser = ConfigParser()

		for config_section in self.config_parser.sections():

			if config_section.startswith('training.dataset'):
				current_usage_mode = cast(UsageMode, self.config_parser.get(config_section, 'usage_mode'))
				if current_usage_mode == usage_mode:
					config_parser.add_section(config_section)

					for key, value in self.config_parser.items(config_section):
						config_parser.set(config_section, key, value)

		return config_parser

	def create_batch(self, source_path : str, target_path : str, source_convert_template : ConvertTemplate, target_convert_template : ConvertTemplate) -> Batch:
		source_tensor = io.read_image(source_path)
		source_tensor = self.transforms(source_tensor)
		source_tensor = self.conditional_convert_tensor(source_tensor, source_convert_template)
		target_tensor = io.read_image(target_path)
		target_tensor = self.transforms(target_tensor)
		target_tensor = self.conditional_convert_tensor(target_tensor, target_convert_template)
		return source_tensor, target_tensor

	@staticmethod
	def conditional_convert_tensor(input_tensor : Tensor, convert_template : ConvertTemplate) -> Tensor:
		if convert_template:
			temp_tensor = input_tensor.unsqueeze(0)
			return convert_tensor(temp_tensor, convert_template).squeeze(0)
		return input_tensor


class AugmentTransform:
	def __init__(self) -> None:
		self.transforms = self.compose_transforms()

	def __call__(self, input_tensor : Tensor) -> Tensor:
		temp_tensor = transforms.functional.convert_image_dtype(input_tensor, torch.float32).unsqueeze(0)
		temp_tensor = self.transforms(temp_tensor).squeeze(0).clamp(0, 1)
		return transforms.functional.convert_image_dtype(temp_tensor, torch.uint8)

	@staticmethod
	def compose_transforms() -> augmentation.AugmentationSequential:
		return augmentation.AugmentationSequential(
			augmentation.RandomHorizontalFlip(p = 0.5),
			augmentation.ImageSequential(
				augmentation.RandomMotionBlur(kernel_size = (3, 7), angle = (-90, 90), direction = (-1, 1), p = 0.1),
				augmentation.RandomGaussianBlur(kernel_size = (5, 5), sigma = (0.1, 2.0), p = 0.1),
				random_apply = 1
			),
			augmentation.ImageSequential(
				augmentation.RandomGamma(gamma = (0.8, 1.2), p = 0.2),
				augmentation.RandomBrightness(brightness = (0.8, 1.2), p = 0.2),
				augmentation.RandomContrast(contrast = (0.8, 1.2), p = 0.2),
				augmentation.RandomLinearIllumination(gain = (0.01, 0.2), p = 0.2),
				random_apply = 1
			),
			augmentation.ImageSequential(
				augmentation.ColorJitter(brightness = 0.2, contrast = 0.2, saturation = 0.2, hue = 0.1, p = 0.2),
				augmentation.RandomRGBShift(r_shift_limit = 0.08, g_shift_limit = 0.08, b_shift_limit = 0.08, p = 0.2),
				augmentation.RandomHue(hue = (-0.1, 0.1), p = 0.2),
				augmentation.RandomSaturation(saturation = (0.8, 1.2), p = 0.2),
				random_apply = 1
			),
			augmentation.RandomAffine(
				degrees = (-2, 2),
				translate = (0.05, 0.05),
				scale = (0.95, 1.05),
				padding_mode = 'border',
				p = 0.2
			)
		)
