#!/usr/bin/env python3

import configparser
from typing import Any, Tuple

import numpy
import pytorch_lightning
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.tuner.tuning import Tuner
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split

from .model import ArcFaceConverter
from .typing import Batch, Loader

CONFIG = configparser.ConfigParser()
CONFIG.read('config.ini')


class ArcFaceConverterTrainer(pytorch_lightning.LightningModule):
	def __init__(self) -> None:
		super(ArcFaceConverterTrainer, self).__init__()
		self.model = ArcFaceConverter()
		self.loss_fn = torch.nn.MSELoss()
		self.lr = 0.001

	def forward(self, source_embedding : Tensor) -> Tensor:
		return self.model(source_embedding)

	def training_step(self, batch : Batch, batch_index : int) -> Tensor:
		source, target = batch
		output = self(source)
		loss = self.loss_fn(output, target)
		self.log('train_loss', loss, prog_bar = True, logger = True)
		return loss

	def validation_step(self, batch : Batch, batch_index : int) -> Tensor:
		source, target = batch
		output = self(source)
		loss = self.loss_fn(output, target)
		self.log('val_loss', loss, prog_bar = True, logger = True)
		return loss

	def configure_optimizers(self) -> Any:
		optimizer = torch.optim.Adam(self.parameters(), lr = self.lr)
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

		return\
		{
			'optimizer': optimizer,
			'lr_scheduler':
			{
				'scheduler': scheduler,
				'monitor': 'train_loss',
				'interval': 'epoch',
				'frequency': 1
			}
		}


def create_loaders() -> Tuple[Loader, Loader]:
	loader_batch_size = CONFIG.getint('training.loader', 'batch_size')
	loader_num_workers = CONFIG.getint('training.loader', 'num_workers')

	training_dataset, validate_dataset = split_dataset()
	training_loader = DataLoader(training_dataset, batch_size = loader_batch_size, num_workers = loader_num_workers, shuffle = True, pin_memory = True)
	validation_loader = DataLoader(validate_dataset, batch_size = loader_batch_size, num_workers = loader_num_workers, shuffle = False, pin_memory = True)
	return training_loader, validation_loader


def split_dataset() -> Tuple[Dataset[Any], Dataset[Any]]:
	input_source_path = CONFIG.get('preparing.input', 'source_path')
	input_target_path = CONFIG.get('preparing.input', 'target_path')
	loader_split_ratio = CONFIG.getfloat('training.loader', 'split_ratio')

	source_input = torch.from_numpy(numpy.load(input_source_path)).float()
	target_input = torch.from_numpy(numpy.load(input_target_path)).float()
	dataset = TensorDataset(source_input, target_input)

	dataset_size = len(dataset)
	training_size = int(loader_split_ratio * len(dataset))
	validation_size = int(dataset_size - training_size)
	training_dataset, validate_dataset = random_split(dataset, [ training_size, validation_size ])
	return training_dataset, validate_dataset


def create_trainer() -> Trainer:
	trainer_max_epochs = CONFIG.getint('training.trainer', 'max_epochs')
	output_directory_path = CONFIG.get('training.output', 'directory_path')
	output_file_pattern = CONFIG.get('training.output', 'file_pattern')

	return Trainer(
		max_epochs = trainer_max_epochs,
		callbacks =
		[
			ModelCheckpoint(
				monitor = 'train_loss',
				dirpath = output_directory_path,
				filename = output_file_pattern,
				every_n_epochs = 10,
				save_top_k = 3,
				save_last = True
			)
		],
		enable_progress_bar = True,
		log_every_n_steps = 2
	)


def train() -> None:
	trainer = create_trainer()
	training_loader, validation_loader = create_loaders()
	model = ArcFaceConverterTrainer()
	tuner = Tuner(trainer)
	tuner.lr_find(model, training_loader, validation_loader)
	trainer.fit(model, training_loader, validation_loader)
