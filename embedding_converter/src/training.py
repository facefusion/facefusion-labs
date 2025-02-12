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

from .models.embedding_converter import EmbeddingConverter
from .types import Batch, Loader

CONFIG = configparser.ConfigParser()
CONFIG.read('config.ini')


class EmbeddingConverterTrainer(pytorch_lightning.LightningModule):
	def __init__(self) -> None:
		super(EmbeddingConverterTrainer, self).__init__()
		self.embedding_converter = EmbeddingConverter()
		self.mse_loss = torch.nn.MSELoss()

	def forward(self, source_embedding : Tensor) -> Tensor:
		return self.embedding_converter(source_embedding)

	def training_step(self, batch : Batch, batch_index : int) -> Tensor:
		source, target = batch
		output = self(source)
		loss_training = self.mse_loss(output, target)
		self.log('loss_training', loss_training, prog_bar = True)
		return loss_training

	def validation_step(self, batch : Batch, batch_index : int) -> Tensor:
		source, target = batch
		output = self(source)
		loss_validation = self.mse_loss(output, target)
		self.log('loss_validation', loss_validation, prog_bar = True)
		return loss_validation

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
	embedding_converter = EmbeddingConverterTrainer()
	tuner = Tuner(trainer)
	tuner.lr_find(embedding_converter, training_loader, validation_loader)
	trainer.fit(embedding_converter, training_loader, validation_loader)
