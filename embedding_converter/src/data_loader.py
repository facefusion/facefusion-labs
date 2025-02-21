import glob
import random

import cv2
import torch
import transforms from torchvision
from torch.utils.data import Dataset

from .types import Batch, Paths


class DataLoaderRecognition(Dataset[torch.Tensor]):
	def __init__(self, dataset_file_pattern : str) -> None:
		self.image_paths = glob.glob(dataset_file_pattern)
		self.transforms = self.compose_transforms()

	def __getitem__(self, index : int) -> Batch:
		image_path = random.choice(self.image_paths)
		vision_frame = cv2.imread(image_path)
		return self.transforms(vision_frame)

	def __len__(self) -> int:
		return len(self.image_paths)

	def compose_transforms(self) -> transforms:
		return transforms.Compose(
		[
			transforms.ToPILImage(),
			transforms.Resize((112, 112), interpolation = transforms.InterpolationMode.BICUBIC),
			transforms.ColorJitter(brightness = 0.2, contrast = 0.2, saturation = 0.2, hue = 0.1),
			transforms.ToTensor(),
			transforms.Lambda(lambda temp_tensor : temp_tensor[[2, 1, 0], :, :]),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		])
