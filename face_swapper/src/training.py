import configparser
import os
from typing import Tuple

import lightning
import torch
import torchvision
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Subset

from .data_loader import DataLoaderVGG
from .helper import calc_id_embedding
from .models.discriminator import Discriminator
from .models.generator import Generator
from .models.loss import FaceSwapperLoss
from .types import Batch, Embedding, VisionTensor

CONFIG = configparser.ConfigParser()
CONFIG.read('config.ini')


class FaceSwapperTrainer(lightning.LightningModule, FaceSwapperLoss):
	def __init__(self) -> None:
		super().__init__()
		FaceSwapperLoss.__init__(self)
		self.generator = Generator()
		self.discriminator = Discriminator()
		self.automatic_optimization = CONFIG.getboolean('training.trainer', 'automatic_optimization')

	def forward(self, target_tensor : VisionTensor, source_embedding : Embedding) -> Tensor:
		output_tensor = self.generator(source_embedding, target_tensor)
		return output_tensor

	def configure_optimizers(self) -> Tuple[Optimizer, Optimizer]:
		learning_rate = CONFIG.getfloat('training.trainer', 'learning_rate')
		generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr = learning_rate, betas = (0.0, 0.999), weight_decay = 1e-4)
		discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr = learning_rate, betas = (0.0, 0.999), weight_decay = 1e-4)
		return generator_optimizer, discriminator_optimizer

	def training_step(self, batch : Batch, batch_index : int) -> Tensor:
		source_tensor, target_tensor, is_same_person = batch
		generator_optimizer, discriminator_optimizer = self.optimizers() #type:ignore[attr-defined]
		source_embedding = calc_id_embedding(self.id_embedder, source_tensor, (0, 0, 0, 0))
		swap_tensor = self.generator(source_embedding, target_tensor)
		target_attributes = self.generator.get_attributes(target_tensor)
		swap_attributes = self.generator.get_attributes(swap_tensor)
		fake_discriminator_outputs = self.discriminator(swap_tensor)

		generator_losses = self.calc_generator_loss(swap_tensor, target_attributes, swap_attributes, fake_discriminator_outputs, batch)
		generator_optimizer.zero_grad()
		self.manual_backward(generator_losses.get('loss_generator'))
		generator_optimizer.step()

		real_discriminator_outputs = self.discriminator(source_tensor)
		fake_discriminator_outputs = self.discriminator(swap_tensor.detach())

		discriminator_losses = self.calc_discriminator_loss(real_discriminator_outputs, fake_discriminator_outputs)
		discriminator_optimizer.zero_grad()
		self.manual_backward(discriminator_losses.get('loss_discriminator'))
		discriminator_optimizer.step()

		if self.global_step % CONFIG.getint('training.trainer', 'preview_frequency') == 0:
			self.generate_preview(source_tensor, target_tensor, swap_tensor)

		self.log('loss_generator', generator_losses.get('loss_generator'), prog_bar = True)
		self.log('loss_discriminator', discriminator_losses.get('loss_discriminator'), prog_bar = True)
		self.log('loss_adversarial', generator_losses.get('loss_adversarial'))
		self.log('loss_attribute', generator_losses.get('loss_attribute'))
		self.log('loss_identity', generator_losses.get('loss_identity'))
		self.log('loss_reconstruction', generator_losses.get('loss_reconstruction'))
		return generator_losses.get('loss_generator')

	def validation_step(self, batch : Batch, batch_index : int) -> Tensor:
		source_tensor, target_tensor, _ = batch
		source_embedding = calc_id_embedding(self.id_embedder, source_tensor, (0, 0, 0, 0))
		output_tensor, target_attributes = self.generator(target_tensor, source_embedding)
		output_embedding = calc_id_embedding(self.id_embedder, output_tensor, (0, 0, 0, 0))
		validation = nn.functional.cosine_similarity(source_embedding, output_embedding).mean()
		self.log('validation', validation)
		return validation

	def generate_preview(self, source_tensor : VisionTensor, target_tensor : VisionTensor, output_tensor : VisionTensor) -> None:
		preview_limit = 8
		preview_items = []

		for source_tensor, target_tensor, output_tensor in zip(source_tensor[:preview_limit], target_tensor[:preview_limit], output_tensor[:preview_limit]):
			preview_items.append(torch.cat([ source_tensor, target_tensor, output_tensor] , dim = 2))

		preview_grid = torchvision.utils.make_grid(torch.cat(preview_items, dim = 1).unsqueeze(0), normalize = True, scale_each = True)
		self.logger.experiment.add_image('preview', preview_grid, self.global_step)


def create_trainer() -> Trainer:
	trainer_max_epochs = CONFIG.getint('training.trainer', 'max_epochs')
	output_directory_path = CONFIG.get('training.output', 'directory_path')
	output_file_pattern = CONFIG.get('training.output', 'file_pattern')
	trainer_precision = CONFIG.get('training.trainer', 'precision')
	logger = TensorBoardLogger('.logs', name = 'face_swapper')

	os.makedirs(output_directory_path, exist_ok = True)
	return Trainer(
		logger = logger,
		log_every_n_steps = 10,
		max_epochs = trainer_max_epochs,
		precision = trainer_precision,
		callbacks =
		[
			ModelCheckpoint(
				monitor = 'loss_generator',
				dirpath = output_directory_path,
				filename = output_file_pattern,
				every_n_train_steps = 1000,
				save_top_k = 3,
				save_last = True
			)
		],
		val_check_interval = 1000
	)


def train() -> None:
	dataset_path = CONFIG.get('preparing.dataset', 'dataset_path')
	dataset_image_pattern = CONFIG.get('preparing.dataset', 'image_pattern')
	dataset_directory_pattern = CONFIG.get('preparing.dataset', 'directory_pattern')
	same_person_probability = CONFIG.getfloat('preparing.dataset', 'same_person_probability')
	batch_size = CONFIG.getint('training.loader', 'batch_size')
	num_workers = CONFIG.getint('training.loader', 'num_workers')
	resume_file_path = CONFIG.get('training.output', 'resume_file_path')

	dataset = DataLoaderVGG(dataset_path, dataset_image_pattern, dataset_directory_pattern, same_person_probability)
	training_loader = DataLoader(dataset, batch_size = batch_size, shuffle = True, num_workers = num_workers, drop_last = True, pin_memory = True, persistent_workers = True)
	validation_loader = DataLoader(Subset(dataset, range(1000)), batch_size = batch_size, num_workers = num_workers, drop_last = True, pin_memory = True, persistent_workers = True)
	face_swapper_trainer = FaceSwapperTrainer()
	trainer = create_trainer()

	if os.path.isfile(resume_file_path):
		trainer.fit(face_swapper_trainer, training_loader, validation_loader, ckpt_path = resume_file_path)
	else:
		trainer.fit(face_swapper_trainer, training_loader, validation_loader)
