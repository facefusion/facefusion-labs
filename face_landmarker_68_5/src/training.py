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
from torch.utils.data import DataLoader, TensorDataset, random_split

from .model import FaceLandmarker685
from .typing import Batch, Loader

CONFIG = configparser.ConfigParser()
CONFIG.read('config.ini')


class FaceLandmarker685Trainer(pytorch_lightning.LightningModule):
	def __init__(self) -> None:
		super(FaceLandmarker685Trainer, self).__init__()
		self.model = FaceLandmarker685()
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
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience = 10, factor = 0.1, mode = 'min')

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


def create_loaders(split_ratio: float = 0.8) -> Tuple[Loader, Loader]:
	batch_size = CONFIG.getint('training', 'batch_size')
	source = torch.from_numpy(numpy.load(CONFIG.get('landmarks', 'source_path'))).float()
	target = torch.from_numpy(numpy.load(CONFIG.get('landmarks', 'target_path'))).float()
	dataset = TensorDataset(source, target)
	dataset_size = len(dataset)
	training_size = int(split_ratio * dataset_size)
	validation_size = int(dataset_size - training_size)
	training_dataset, validate_dataset = random_split(dataset, [training_size, validation_size])
	training_loader = DataLoader(training_dataset, batch_size = batch_size, shuffle = True, num_workers = 8, pin_memory = True)
	validation_loader = DataLoader(validate_dataset, batch_size = batch_size, shuffle = False, num_workers = 8, pin_memory = True)
	return training_loader, validation_loader


def create_trainer() -> Trainer:
	return Trainer(
		max_epochs = CONFIG.getint('training', 'max_epochs'),
		callbacks =
		[
			ModelCheckpoint(
				monitor = 'train_loss',
				dirpath = CONFIG.get('outputs', 'directory_path'),
				filename = CONFIG.get('outputs', 'file_name'),
				save_top_k = 3,
				mode = 'min',
				every_n_epochs = 10
			)
		],
		enable_progress_bar = True,
		log_every_n_steps = 2
	)


def train() -> None:
	trainer = create_trainer()
	training_loader, validation_loader = create_loaders()
	model = FaceLandmarker685Trainer()
	tuner = Tuner(trainer)
	tuner.lr_find(model, training_loader, validation_loader)
	trainer.fit(model, training_loader, validation_loader)
