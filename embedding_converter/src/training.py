import configparser
import os
from typing import Any, Tuple

import lightning
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.tuner import Tuner
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, random_split

from .data_loader import DataLoaderRecognition
from .models.embedding_converter import EmbeddingConverter
from .types import Batch, Embedding

CONFIG = configparser.ConfigParser()
CONFIG.read('config.ini')


class EmbeddingConverterTrainer(lightning.LightningModule):
	def __init__(self) -> None:
		super(EmbeddingConverterTrainer, self).__init__()
		source_path = CONFIG.get('training.model', 'source_path')
		target_path = CONFIG.get('training.model', 'target_path')
		self.lr = CONFIG.getfloat('training.trainer', 'learning_rate')
		self.embedding_converter = EmbeddingConverter()
		self.source_embedder = torch.jit.load(source_path) # type:ignore[no-untyped-call]
		self.target_embedder = torch.jit.load(target_path) # type:ignore[no-untyped-call]
		self.source_embedder.eval()
		self.target_embedder.eval()
		self.mse_loss = nn.MSELoss()

	def forward(self, source_embedding : Embedding) -> Embedding:
		return self.embedding_converter(source_embedding)

	def training_step(self, batch : Batch, batch_index : int) -> Tensor:
		with torch.no_grad():
			source_embedding = self.source_embedder(batch)
			target_embedding = self.target_embedder(batch)
		output_embedding = self(source_embedding)
		loss_training = self.mse_loss(output_embedding, target_embedding)
		self.log('loss_training', loss_training, prog_bar = True)
		return loss_training

	def validation_step(self, batch : Batch, batch_index : int) -> Tensor:
		with torch.no_grad():
			source_embedding = self.source_embedder(batch)
			target_embedding = self.target_embedder(batch)
		output_embedding = self(source_embedding)
		validation = self.mse_loss(output_embedding, target_embedding)
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
				'monitor': 'loss_training',
				'interval': 'epoch',
				'frequency': 1
			}
		}


def create_loaders(dataset : Dataset[Any]) -> Tuple[DataLoader[Any], DataLoader[Any]]:
	batch_size = CONFIG.getint('training.loader', 'batch_size')
	num_workers = CONFIG.getint('training.loader', 'num_workers')

	training_dataset, validate_dataset = split_dataset(dataset)
	training_loader = DataLoader(training_dataset, batch_size = batch_size, shuffle = True, num_workers = num_workers, drop_last = True, pin_memory = True, persistent_workers = True)
	validation_loader = DataLoader(validate_dataset, batch_size = batch_size, shuffle = False, num_workers = num_workers, drop_last = False, pin_memory = True, persistent_workers = True)
	return training_loader, validation_loader


def split_dataset(dataset : Dataset[Any]) -> Tuple[Dataset[Any], Dataset[Any]]:
	loader_split_ratio = CONFIG.getfloat('training.loader', 'split_ratio')
	training_size = int(loader_split_ratio * len(dataset)) # type:ignore[operator, arg-type]
	validation_size = len(dataset) - training_size # type:ignore[arg-type]
	training_dataset, validate_dataset = random_split(dataset, [training_size, validation_size])
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
		val_check_interval = 10
	)


def train() -> None:
	dataset_file_pattern = CONFIG.get('training.dataset', 'image_pattern')
	resume_file_path = CONFIG.get('training.output', 'resume_file_path')

	dataset = DataLoaderRecognition(dataset_file_pattern)
	training_loader, validation_loader = create_loaders(dataset)
	embedding_converter_trainer = EmbeddingConverterTrainer()
	trainer = create_trainer()
	tuner = Tuner(trainer)
	tuner.lr_find(embedding_converter_trainer, training_loader, validation_loader)

	if os.path.exists(resume_file_path):
		trainer.fit(embedding_converter_trainer, training_loader, validation_loader, ckpt_path = resume_file_path)
	else:
		trainer.fit(embedding_converter_trainer, training_loader, validation_loader)
