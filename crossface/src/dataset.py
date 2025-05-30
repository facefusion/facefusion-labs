import glob
from configparser import ConfigParser

from torch import Tensor
from torch.utils.data import Dataset
from torchvision import io, transforms

from .types import Batch


class StaticDataset(Dataset[Tensor]):
	def __init__(self, config_parser : ConfigParser) -> None:
		self.config_file_pattern = config_parser.get('training.dataset', 'file_pattern')
		self.file_paths = glob.glob(self.config_file_pattern)
		self.transforms = self.compose_transforms()

	def __getitem__(self, index : int) -> Batch:
		file_path = self.file_paths[index]
		temp_tensor = io.read_image(file_path)
		return self.transforms(temp_tensor)

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
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		])
