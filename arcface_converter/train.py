#!/usr/bin/env python3

import configparser
from typing import Any, Tuple

import numpy
import pytorch_lightning
import torch
from torch import Tensor

from arcface_converter.typing import DataLoaderSet
from model import ArcFaceConverter
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.tuner.tuning import Tuner
from torch.utils.data import DataLoader, TensorDataset, random_split


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

	def training_step(self, batch : Tuple[Tensor, Tensor], batch_index : int) -> Tensor:
		source, target = batch
		output = self(source)
		loss = self.loss_fn(output, target)
		self.log('train_loss', loss, prog_bar = True, logger = True)
		return loss

	def validation_step(self, batch : Tuple[Tensor, Tensor], batch_index : int) -> Tensor:
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


def create_data_loaders(batch_size : int, split_ratio : float = 0.8) -> Tuple[DataLoaderSet, DataLoaderSet]:
	source = torch.from_numpy(numpy.load(CONFIG['embeddings']['source_path'])).float()
	target = torch.from_numpy(numpy.load(CONFIG['embeddings']['target_path'])).float()
	dataset = TensorDataset(source, target)
	train_size = int(split_ratio * len(dataset))
	validate_size = len(dataset) - train_size
	train_dataset, validate_dataset = random_split(dataset, [ train_size, validate_size ])
	train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = 8, pin_memory = True)
	validate_loader = DataLoader(validate_dataset, batch_size = batch_size, shuffle = False, num_workers = 8, pin_memory = True)
	return train_loader, validate_loader


def train(trainer : Trainer, train_loader : DataLoaderSet, validate_loader : DataLoaderSet) -> None:
	model = ArcFaceConverterTrainer()
	tuner = Tuner(trainer)
	tuner.lr_find(model, train_loader, validate_loader)
	trainer.fit(model, train_loader, validate_loader)


if __name__ == '__main__':
	accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
	batch_size = 50000
	max_epochs = 5000
	train_loader, validate_loader = create_data_loaders(batch_size)

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
		accelerator = accelerator,
		callbacks = [ checkpoint_callback ],
		enable_progress_bar = True,
		log_every_n_steps = 2
	)
	logger = TensorBoardLogger('.logs', name = 'arcface_converter')
	train(trainer, train_loader, validate_loader)
