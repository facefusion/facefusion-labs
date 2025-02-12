import configparser
import os
from typing import Tuple

import pytorch_lightning
import torch
import torchvision
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.types import Optimizer
from torch import Tensor
from torch.utils.data import DataLoader

from .data_loader import DataLoaderVGG
from .helper import calc_id_embedding
from .models.discriminator import MultiscaleDiscriminator
from .models.generator import AdaptiveEmbeddingIntegrationNetwork
from .models.loss import FaceSwapperLoss
from .types import Batch, Embedding, TargetAttributes, VisionTensor

CONFIG = configparser.ConfigParser()
CONFIG.read('config.ini')


class FaceSwapperTrain(pytorch_lightning.LightningModule, FaceSwapperLoss):
	def __init__(self) -> None:
		super().__init__()
		self.generator = AdaptiveEmbeddingIntegrationNetwork()
		self.discriminator = MultiscaleDiscriminator()
		self.automatic_optimization = CONFIG.getboolean('training.trainer', 'automatic_optimization')

	def forward(self, target_tensor : VisionTensor, source_embedding : Embedding) -> Tuple[VisionTensor, TargetAttributes]:
		output = self.generator(target_tensor, source_embedding)
		return output

	def configure_optimizers(self) -> Tuple[Optimizer, Optimizer]:
		learning_rate = CONFIG.getfloat('training.trainer', 'learning_rate')
		generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr = learning_rate, betas = (0.0, 0.999), weight_decay = 1e-4)
		discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr = learning_rate, betas = (0.0, 0.999), weight_decay = 1e-4)
		return generator_optimizer, discriminator_optimizer

	def training_step(self, batch : Batch, batch_index : int) -> Tensor:
		source_tensor, target_tensor, is_same_person = batch
		generator_optimizer, discriminator_optimizer = self.optimizers() #type:ignore[attr-defined]
		source_embedding = calc_id_embedding(self.id_embedder, source_tensor, (0, 0, 0, 0))
		swap_tensor, target_attributes = self.generator(target_tensor, source_embedding)
		swap_attributes = self.generator.get_attributes(swap_tensor)
		real_discriminator_outputs = self.discriminator(source_tensor.detach())
		fake_discriminator_outputs = self.discriminator(swap_tensor.detach())

		generator_losses = self.calc_generator_loss(swap_tensor, target_attributes, swap_attributes, fake_discriminator_outputs, batch)
		generator_optimizer.zero_grad()
		self.manual_backward(generator_losses.get('loss_generator'))
		generator_optimizer.step()

		discriminator_losses = self.calc_discriminator_loss(real_discriminator_outputs, fake_discriminator_outputs)
		discriminator_optimizer.zero_grad()
		self.manual_backward(discriminator_losses.get('loss_discriminator'))
		discriminator_optimizer.step()

		if self.global_step % CONFIG.getint('training.output', 'preview_frequency') == 0:
			self.generate_preview(source_tensor, target_tensor, swap_tensor)

		self.log('loss_generator', generator_losses.get('loss_generator'), prog_bar = True)
		self.log('loss_discriminator', discriminator_losses.get('loss_discriminator'), prog_bar = True)
		self.log('loss_adversarial', generator_losses.get('loss_adversarial'))
		self.log('loss_attribute', generator_losses.get('loss_attribute'))
		self.log('loss_identity', generator_losses.get('loss_identity'))
		self.log('loss_reconstruction', generator_losses.get('loss_reconstruction'))
		return generator_losses.get('loss_generator')

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

	os.makedirs(output_directory_path, exist_ok = True)
	return Trainer(
		max_epochs = trainer_max_epochs,
		precision = trainer_precision,
		callbacks =
		[
			ModelCheckpoint(
				monitor = 'loss_generator',
				dirpath = output_directory_path,
				filename = output_file_pattern,
				every_n_train_steps = 1000,
				save_top_k = 5,
				save_last = True
			)
		],
		log_every_n_steps = 10
	)


def train() -> None:
	dataset_path = CONFIG.get('preparing.dataset', 'dataset_path')
	dataset_image_pattern = CONFIG.get('preparing.dataset', 'image_pattern')
	dataset_directory_pattern = CONFIG.get('preparing.dataset', 'directory_pattern')
	same_person_probability = CONFIG.getfloat('preparing.dataset', 'same_person_probability')
	batch_size = CONFIG.getint('training.loader', 'batch_size')
	num_workers = CONFIG.getint('training.loader', 'num_workers')
	output_file_path = CONFIG.get('training.output', 'file_path')

	dataset = DataLoaderVGG(dataset_path, dataset_image_pattern, dataset_directory_pattern, same_person_probability)
	data_loader = DataLoader(dataset, batch_size = batch_size, shuffle = True, num_workers = num_workers, drop_last = True, pin_memory = True, persistent_workers = True)
	face_swap_model = FaceSwapperTrain()
	trainer = create_trainer()
	trainer.fit(face_swap_model, data_loader, ckpt_path = output_file_path)
