import glob
import random

import cv2
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms

from .types import Batch


class DynamicDataset(Dataset[Tensor]):
	def __init__(self, file_pattern : str, batch_ratio : float) -> None:
		self.file_paths = glob.glob(file_pattern)
		self.transforms = self.compose_transforms()
		self.batch_ratio = batch_ratio

	def __getitem__(self, index : int) -> Batch:
		source_image_path = self.file_paths[index]

		if random.random() < self.batch_ratio:
			return self.prepare_equal_batch(source_image_path)

		return self.prepare_different_batch(source_image_path)

	def __len__(self) -> int:
		return len(self.file_paths)

	@staticmethod
	def compose_transforms() -> transforms:
		return transforms.Compose(
		[
			transforms.ToPILImage(),
			transforms.Resize((256, 256), interpolation = transforms.InterpolationMode.BICUBIC),
			transforms.ColorJitter(brightness = 0.2, contrast = 0.2, saturation = 0.2, hue = 0.1),
			transforms.RandomAffine(4, translate = (0.01, 0.01), scale = (0.98, 1.02), shear = (1, 1)),
			transforms.ToTensor(),
			transforms.Lambda(lambda temp_tensor : temp_tensor[[2, 1, 0], :, :]),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		])

	def prepare_different_batch(self, source_image_path : str) -> Batch:
		target_image_path = random.choice(self.file_paths)
		source_vision_frame = cv2.imread(source_image_path)
		target_vision_frame = cv2.imread(target_image_path)
		source_tensor = self.transforms(source_vision_frame)
		target_tensor = self.transforms(target_vision_frame)
		return source_tensor, target_tensor

	def prepare_equal_batch(self, source_image_path : str) -> Batch:
		source_vision_frame = cv2.imread(source_image_path)
		source_tensor = self.transforms(source_vision_frame)
		return source_tensor, source_tensor
