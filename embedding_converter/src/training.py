import configparser
import os
from typing import Any, Tuple

import lightning
import numpy
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.tuner import Tuner
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split

from .models.embedding_converter import EmbeddingConverter
from .types import Batch, Embedding

CONFIG = configparser.ConfigParser()
CONFIG.read('config.ini')


class EmbeddingConverterTrainer(lightning.LightningModule):
	def __init__(self) -> None:
		super(EmbeddingConverterTrainer, self).__init__()
		self.embedding_converter = EmbeddingConverter()
		self.mse_loss = nn.MSELoss()

	def forward(self, source_embedding : Embedding) -> Embedding:
		return self.embedding_converter(source_embedding)

	def training_step(self, batch : Batch, batch_index : int) -> Tensor:
		source_tensor, target = batch
		output_tensor = self(source_tensor)
		loss_training = self.mse_loss(output_tensor, target)
		self.log('loss_training', loss_training, prog_bar = True)
		return loss_training

	def validation_step(self, batch : Batch, batch_index : int) -> Tensor:
		source_tensor, target_tensor = batch
		output_tensor = self(source_tensor)
		validation = self.mse_loss(output_tensor, target_tensor)
		self.log('validation', validation, prog_bar = True)
		return validation

	def configure_optimizers(self) -> Any:
		learning_rate = CONFIG.getfloat('training.trainer', 'learning_rate')
		optimizer = torch.optim.Adam(self.parameters(), lr = learning_rate)
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


def create_loaders() -> Tuple[DataLoader, DataLoader]:
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

	source_tensor = torch.from_numpy(numpy.load(input_source_path)).float()
	target_tensor = torch.from_numpy(numpy.load(input_target_path)).float()
	dataset = TensorDataset(source_tensor, target_tensor)

	dataset_size = len(dataset)
	training_size = int(loader_split_ratio * len(dataset))
	validation_size = int(dataset_size - training_size)
	training_dataset, validate_dataset = random_split(dataset, [ training_size, validation_size ])
	return training_dataset, validate_dataset


def create_trainer() -> Trainer:
	trainer_max_epochs = CONFIG.getint('training.trainer', 'max_epochs')
	output_directory_path = CONFIG.get('training.output', 'directory_path')
	output_file_pattern = CONFIG.get('training.output', 'file_pattern')
	logger = TensorBoardLogger('.logs', name = 'embedding_converter')

	return Trainer(
		logger = logger,
		log_every_n_steps = 10,
		max_epochs = trainer_max_epochs,
		callbacks =
		[
			ModelCheckpoint(
				monitor = 'loss_training',
				dirpath = output_directory_path,
				filename = output_file_pattern,
				every_n_epochs = 10,
				save_top_k = 3,
				save_last = True
			)
		],
		val_check_interval = 1000
	)


def train() -> None:
	resume_file_path = CONFIG.get('training.output', 'resume_file_path')

	training_loader, validation_loader = create_loaders()
	embedding_converter_trainer = EmbeddingConverterTrainer()
	trainer = create_trainer()
	tuner = Tuner(trainer)
	tuner.lr_find(embedding_converter_trainer, training_loader, validation_loader)

	if os.path.exists(resume_file_path):
		trainer.fit(embedding_converter_trainer, training_loader, validation_loader, ckpt_path = resume_file_path)
	else:
		trainer.fit(embedding_converter_trainer, training_loader, validation_loader)
