import configparser
import glob
import random

import cv2
import torchvision.transforms as transforms
import tqdm
from PIL import Image
from torch.utils.data import TensorDataset

from .augmentations import apply_random_motion_blur
from .sub_typing import Batch

CONFIG = configparser.ConfigParser()
CONFIG.read('config.ini')


def read_image(image_path: str) -> Image.Image:
	image = cv2.imread(image_path)[:, :, ::-1]
	pil_image = Image.fromarray(image)
	return pil_image


class DataLoaderVGG(TensorDataset):
	def __init__(self, dataset_path : str) -> None:
		self.same_person_probability = float(CONFIG.get('preparing.dataloader', 'same_person_probability'))
		self.image_paths = glob.glob('{}/*/*.*g'.format(dataset_path))
		self.folder_paths = glob.glob('{}/*'.format(dataset_path))
		self.image_path_dict = {}

		for folder_path in tqdm.tqdm(self.folder_paths):
			image_paths = glob.glob('{}/*'.format(folder_path))
			self.image_path_dict[folder_path] = image_paths
		self.dataset_total = len(self.image_paths)
		self.transforms_basic = transforms.Compose(
		[
			transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC),
			transforms.ToTensor(),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		])
		self.transforms_moderate = transforms.Compose(
		[
			transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC),
			transforms.ToTensor(),
			transforms.ColorJitter(brightness = 0.2, contrast = 0.2, saturation = 0.2, hue = 0.1),
			transforms.RandomAffine(4, translate = (0.01, 0.01), scale = (0.98, 1.02), shear = (1, 1), fill = 0),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		])
		self.transforms_complex = transforms.Compose(
		[
			transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC),
			transforms.ToTensor(),
			transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC),
			transforms.RandomHorizontalFlip(p = 0.5),
			transforms.RandomApply([ apply_random_motion_blur ], p = 0.3),
			transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation = 0.2, hue = 0.1),
			transforms.RandomAffine(8, translate = (0.02, 0.02), scale = (0.98, 1.02), shear = (1, 1), fill = 0),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		])

	def __getitem__(self, item : int) -> Batch:
		source_image_path = self.image_paths[item]
		source = read_image(source_image_path)

		if random.random() > self.same_person_probability:
			is_same_person = 0
			target_image_path = random.choice(self.image_paths)
			target = read_image(target_image_path)
			source_transform = self.transforms_moderate(source)
			target_transform = self.transforms_complex(target)
		else:
			is_same_person = 1
			source_folder_path = '/'.join(source_image_path.split('/')[:-1])
			target_image_path = random.choice(self.image_path_dict[source_folder_path])
			target = read_image(target_image_path)
			source_transform = self.transforms_basic(source)
			target_transform = self.transforms_basic(target)

		return source_transform, target_transform, is_same_person

	def __len__(self) -> int:
		return self.dataset_total
