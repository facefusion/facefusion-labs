import glob
import random

import cv2
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms

from .types import Batch


class DynamicDataset(Dataset[Tensor]):
	def __init__(self, file_pattern : str) -> None:
		self.file_paths = glob.glob(file_pattern)
		self.transforms = self.compose_transforms()

	def __getitem__(self, index : int) -> Batch:
		file_path = random.choice(self.file_paths)
		vision_frame = cv2.imread(file_path)
		return self.transforms(vision_frame)

	def __len__(self) -> int:
		return len(self.file_paths)

	@staticmethod
	def compose_transforms() -> transforms:
		return transforms.Compose(
		[
			transforms.ToPILImage(),
			transforms.Resize((112, 112), interpolation = transforms.InterpolationMode.BICUBIC),
			transforms.ColorJitter(brightness = 0.2, contrast = 0.2, saturation = 0.2, hue = 0.1),
			transforms.ToTensor(),
			transforms.Lambda(lambda temp_tensor: temp_tensor[[2, 1, 0], :, :]),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		])
