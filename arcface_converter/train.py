#!/usr/bin/env python3

import configparser
from typing import Any, Tuple

import numpy
import pytorch_lightning
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.tuner.tuning import Tuner
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset, random_split

from arcface_converter.typing import Batch, Loader
from model import ArcFaceConverter

CONFIG = configparser.ConfigParser()
CONFIG.read('config.ini')


class ArcFaceConverterTrainer(pytorch_lightning.LightningModule):
	def __init__(self) -> None:
		super(ArcFaceConverterTrainer, self).__init__()
		self.model = ArcFaceConverter()
		self.loss_fn = torch.nn.MSELoss()
		self.lr = 0.001

	def forward(self, input_embedding : Tensor) -> Tensor:
		return self.model(input_embedding)

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


def create_data_loaders(batch_size : int, split_ratio : float = 0.8) -> Tuple[Loader, Loader]:
	source = torch.from_numpy(numpy.load(CONFIG['embeddings']['source_path'])).float()
	target = torch.from_numpy(numpy.load(CONFIG['embeddings']['target_path'])).float()
	dataset = TensorDataset(source, target)
	training_size = int(split_ratio * len(dataset))
	validation_size = len(dataset) - training_size
	training_dataset, validate_dataset = random_split(dataset, [ training_size, validation_size ])
	training_loader = DataLoader(training_dataset, batch_size = batch_size, shuffle = True, num_workers = 8, pin_memory = True)
	validation_loader = DataLoader(validate_dataset, batch_size = batch_size, shuffle = False, num_workers = 8, pin_memory = True)
	return training_loader, validation_loader


def train(trainer : Trainer, training_loader : Loader, validation_loader : Loader) -> None:
	model = ArcFaceConverterTrainer()
	tuner = Tuner(trainer)
	tuner.lr_find(model, training_loader, validation_loader)
	trainer.fit(model, training_loader, validation_loader)


def create_trainer() -> Trainer:
	checkpoint_callback = ModelCheckpoint(
		monitor = 'train_loss',
		dirpath = CONFIG['checkpoints']['directory_path'],
		filename = CONFIG['checkpoints']['file_name'],
		save_top_k = 3,
		mode = 'min',
		every_n_epochs = 10
	)
	trainer = Trainer(
		max_epochs = max_epochs,
		callbacks = [ checkpoint_callback ],
		enable_progress_bar = True,
		log_every_n_steps = 2
	)

	return trainer


if __name__ == '__main__':
	batch_size = 50000
	max_epochs = 5000
	trainer = create_trainer()
	training_loader, validation_loader = create_data_loaders(batch_size)

	logger = TensorBoardLogger('.logs', name = 'arcface_converter')
	train(trainer, training_loader, validation_loader)
