import os
from configparser import ConfigParser
from typing import Tuple

import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch import Tensor, nn
from torch.utils.data import Dataset, random_split
from torchdata.stateful_dataloader import StatefulDataLoader

from .dataset import StaticDataset
from .models.crossface import CrossFace
from .types import Batch, Embedding, OptimizerSet

CONFIG_PARSER = ConfigParser()
CONFIG_PARSER.read('config.ini')


class CrossFaceTrainer(LightningModule):
	def __init__(self, config_parser : ConfigParser) -> None:
		super().__init__()
		self.config_source_path = config_parser.get('training.model', 'source_path')
		self.config_target_path = config_parser.get('training.model', 'target_path')
		self.config_learning_rate = config_parser.getfloat('training.optimizer', 'learning_rate')
		self.crossface = CrossFace()
		self.source_embedder = torch.jit.load(self.config_source_path, map_location = 'cpu').eval()
		self.target_embedder = torch.jit.load(self.config_target_path, map_location = 'cpu').eval()
		self.mse_loss = nn.MSELoss()

	def forward(self, source_embedding : Embedding) -> Embedding:
		return self.crossface(source_embedding)

	def training_step(self, batch : Batch, batch_index : int) -> Tensor:
		with torch.no_grad():
			source_embedding = self.source_embedder(batch)
			target_embedding = self.target_embedder(batch)
		output_embedding = self(source_embedding)
		training_loss = self.mse_loss(output_embedding, target_embedding)
		self.log('training_loss', training_loss, prog_bar = True)
		return training_loss

	def validation_step(self, batch : Batch, batch_index : int) -> Tensor:
		with torch.no_grad():
			source_embedding = self.source_embedder(batch)
		output_embedding = self(source_embedding)
		validation_score = (nn.functional.cosine_similarity(source_embedding, output_embedding).mean() + 1) * 0.5
		self.log('validation_score', validation_score, sync_dist = True, prog_bar = True)
		return validation_score

	def configure_optimizers(self) -> OptimizerSet:
		optimizer = torch.optim.AdamW(self.parameters(), lr = self.config_learning_rate)
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
		optimizer_set =\
		{
			'optimizer': optimizer,
			'lr_scheduler':
			{
				'scheduler': scheduler,
				'monitor': 'training_loss',
				'interval': 'epoch',
				'frequency': 1
			}
		}

		return optimizer_set


def create_loaders(dataset : Dataset[Tensor]) -> Tuple[StatefulDataLoader[Tensor], StatefulDataLoader[Tensor]]:
	config_batch_size = CONFIG_PARSER.getint('training.loader', 'batch_size')
	config_num_workers = CONFIG_PARSER.getint('training.loader', 'num_workers')

	training_dataset, validate_dataset = split_dataset(dataset)
	training_loader = StatefulDataLoader(training_dataset, batch_size = config_batch_size, shuffle = True, num_workers = config_num_workers, drop_last = True, pin_memory = True, persistent_workers = True)
	validation_loader = StatefulDataLoader(validate_dataset, batch_size = config_batch_size, shuffle = False, num_workers = config_num_workers, pin_memory = True, persistent_workers = True)
	return training_loader, validation_loader


def split_dataset(dataset : Dataset[Tensor]) -> Tuple[Dataset[Tensor], Dataset[Tensor]]:
	config_split_ratio = CONFIG_PARSER.getfloat('training.loader', 'split_ratio')

	dataset_size = len(dataset) # type:ignore[arg-type]
	training_size = int(dataset_size * config_split_ratio)
	validation_size = int(dataset_size - training_size)
	training_dataset, validate_dataset = random_split(dataset, [ training_size, validation_size ])
	return training_dataset, validate_dataset


def create_trainer() -> Trainer:
	config_max_epochs = CONFIG_PARSER.getint('training.trainer', 'max_epochs')
	config_strategy = CONFIG_PARSER.get('training.trainer', 'strategy')
	config_precision = CONFIG_PARSER.get('training.trainer', 'precision')
	config_logger_path = CONFIG_PARSER.get('training.logger', 'logger_path')
	config_logger_name = CONFIG_PARSER.get('training.logger', 'logger_name')
	config_directory_path = CONFIG_PARSER.get('training.output', 'directory_path')
	config_file_pattern = CONFIG_PARSER.get('training.output', 'file_pattern')
	logger = TensorBoardLogger(config_logger_path, config_logger_name)

	return Trainer(
		logger = logger,
		log_every_n_steps = 10,
		max_epochs = config_max_epochs,
		strategy = config_strategy,
		precision = config_precision,
		callbacks =
		[
			ModelCheckpoint(
				monitor = 'training_loss',
				dirpath = config_directory_path,
				filename = config_file_pattern,
				every_n_epochs = 1,
				save_top_k = 5,
				save_last = True
			)
		]
	)


def train() -> None:
	config_resume_path = CONFIG_PARSER.get('training.output', 'resume_path')

	if torch.cuda.is_available():
		torch.set_float32_matmul_precision('high')

	dataset = StaticDataset(CONFIG_PARSER)
	training_loader, validation_loader = create_loaders(dataset)
	crossface_trainer = CrossFaceTrainer(CONFIG_PARSER)
	trainer = create_trainer()

	if os.path.exists(config_resume_path):
		trainer.fit(crossface_trainer, training_loader, validation_loader, ckpt_path = config_resume_path)
	else:
		trainer.fit(crossface_trainer, training_loader, validation_loader)
