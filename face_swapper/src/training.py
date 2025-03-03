import configparser
import os
import warnings
from typing import Tuple, cast

import lightning
import torch
import torchvision
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch import Tensor, nn
from torch.utils.data import Dataset, random_split
from torchdata.stateful_dataloader import StatefulDataLoader

from .dataset import DynamicDataset
from .helper import calc_embedding
from .models.discriminator import Discriminator
from .models.generator import Generator
from .models.loss import AdversarialLoss, AttributeLoss, DiscriminatorLoss, GazeLoss, IdentityLoss, PoseLoss, ReconstructionLoss
from .types import Batch, Embedding, OptimizerConfig, WarpTemplate

warnings.filterwarnings('ignore', category = UserWarning, module = 'torch')

CONFIG = configparser.ConfigParser()
CONFIG.read('config.ini')


class FaceSwapperTrainer(lightning.LightningModule):
	def __init__(self) -> None:
		super().__init__()
		embedder_path = CONFIG.get('training.model', 'embedder_path')
		gazer_path = CONFIG.get('training.model', 'gazer_path')
		motion_extractor_path = CONFIG.get('training.model', 'motion_extractor_path')

		self.embedder = torch.jit.load(embedder_path, map_location = 'cpu').eval()  # type:ignore[no-untyped-call]
		self.gazer = torch.jit.load(gazer_path, map_location = 'cpu').eval()  # type:ignore[no-untyped-call]
		self.motion_extractor = torch.jit.load(motion_extractor_path, map_location = 'cpu').eval()  # type:ignore[no-untyped-call]

		self.generator = Generator()
		self.discriminator = Discriminator()
		self.discriminator_loss = DiscriminatorLoss()
		self.adversarial_loss = AdversarialLoss()
		self.attribute_loss = AttributeLoss()
		self.reconstruction_loss = ReconstructionLoss(self.embedder)
		self.identity_loss = IdentityLoss(self.embedder)
		self.pose_loss = PoseLoss(self.motion_extractor)
		self.gaze_loss = GazeLoss(self.gazer)
		self.automatic_optimization = False

	def forward(self, source_embedding : Embedding, target_tensor : Tensor) -> Tensor:
		output_tensor = self.generator(source_embedding, target_tensor)
		return output_tensor

	def configure_optimizers(self) -> Tuple[OptimizerConfig, OptimizerConfig]:
		learning_rate = CONFIG.getfloat('training.trainer', 'learning_rate')
		generator_optimizer = torch.optim.AdamW(self.generator.parameters(), lr = learning_rate, betas = (0.0, 0.999), weight_decay = 1e-4)
		discriminator_optimizer = torch.optim.AdamW(self.discriminator.parameters(), lr = learning_rate, betas = (0.0, 0.999), weight_decay = 1e-4)
		generator_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(generator_optimizer, T_0 = 300, T_mult = 2)
		discriminator_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(discriminator_optimizer, T_0 = 300, T_mult = 2)

		generator_config =\
		{
			'optimizer': generator_optimizer,
			'lr_scheduler':
			{
				'scheduler': generator_scheduler,
				'interval': 'step'
			}
		}
		discriminator_config =\
		{
			'optimizer': discriminator_optimizer,
			'lr_scheduler':
			{
				'scheduler': discriminator_scheduler,
				'interval': 'step'
			}
		}
		return generator_config, discriminator_config

	def training_step(self, batch : Batch, batch_index : int) -> Tensor:
		preview_frequency = CONFIG.getfloat('training.trainer', 'preview_frequency')

		source_tensor, target_tensor = batch
		generator_optimizer, discriminator_optimizer = self.optimizers() #type:ignore[attr-defined]
		source_embedding = calc_embedding(self.embedder, source_tensor, (0, 0, 0, 0))
		target_attributes = self.generator.get_attributes(target_tensor)
		generator_output_tensor = self.generator(source_embedding, target_tensor)
		generator_output_attributes = self.generator.get_attributes(generator_output_tensor)
		discriminator_output_tensors = self.discriminator(generator_output_tensor)

		self.toggle_optimizer(generator_optimizer)
		adversarial_loss, weighted_adversarial_loss = self.adversarial_loss(discriminator_output_tensors)
		attribute_loss, weighted_attribute_loss = self.attribute_loss(target_attributes, generator_output_attributes)
		reconstruction_loss, weighted_reconstruction_loss = self.reconstruction_loss(source_tensor, target_tensor, generator_output_tensor)
		identity_loss, weighted_identity_loss = self.identity_loss(generator_output_tensor, source_tensor)
		pose_loss, weighted_pose_loss = self.pose_loss(target_tensor, generator_output_tensor)
		gaze_loss, weighted_gaze_loss = self.gaze_loss(target_tensor, generator_output_tensor)
		generator_loss = weighted_adversarial_loss + weighted_attribute_loss + weighted_reconstruction_loss + weighted_identity_loss + weighted_pose_loss + weighted_gaze_loss

		generator_optimizer.zero_grad()
		self.manual_backward(generator_loss)
		generator_optimizer.step()
		self.untoggle_optimizer(generator_optimizer)

		self.toggle_optimizer(discriminator_optimizer)
		discriminator_source_tensors = self.discriminator(source_tensor)
		discriminator_output_tensors = self.discriminator(generator_output_tensor.detach())
		discriminator_loss = self.discriminator_loss(discriminator_source_tensors, discriminator_output_tensors)

		discriminator_optimizer.zero_grad()
		self.manual_backward(discriminator_loss)
		discriminator_optimizer.step()
		self.untoggle_optimizer(discriminator_optimizer)

		if self.global_step % preview_frequency == 0:
			self.generate_preview(source_tensor, target_tensor, generator_output_tensor)

		self.log('generator_loss', generator_loss, prog_bar = True)
		self.log('discriminator_loss', discriminator_loss, prog_bar = True)
		self.log('adversarial_loss', adversarial_loss)
		self.log('attribute_loss', attribute_loss)
		self.log('reconstruction_loss', reconstruction_loss)
		self.log('identity_loss', identity_loss)
		self.log('pose_loss', pose_loss)
		self.log('gaze_loss', gaze_loss)
		return generator_loss

	def validation_step(self, batch : Batch, batch_index : int) -> Tensor:
		source_tensor, target_tensor = batch
		source_embedding = calc_embedding(self.embedder, source_tensor, (0, 0, 0, 0))
		output_tensor = self.generator(source_embedding, target_tensor)
		output_embedding = calc_embedding(self.embedder, output_tensor, (0, 0, 0, 0))
		validation_score = (nn.functional.cosine_similarity(source_embedding, output_embedding).mean() + 1) * 0.5
		self.log('validation_score', validation_score, prog_bar = True)
		return validation_score

	def generate_preview(self, source_tensor : Tensor, target_tensor : Tensor, output_tensor : Tensor) -> None:
		preview_limit = 8
		preview_cells = []

		for source_tensor, target_tensor, output_tensor in zip(source_tensor[:preview_limit], target_tensor[:preview_limit], output_tensor[:preview_limit]):
			preview_cell = torch.cat([ source_tensor, target_tensor, output_tensor] , dim = 2)
			preview_cells.append(preview_cell)

		preview_cells = torch.cat(preview_cells, dim = 1).unsqueeze(0)
		preview_grid = torchvision.utils.make_grid(preview_cells, normalize = True, scale_each = True)
		self.logger.experiment.add_image('preview', preview_grid, self.global_step) # type:ignore[attr-defined]


def create_loaders(dataset : Dataset[Tensor]) -> Tuple[StatefulDataLoader[Tensor], StatefulDataLoader[Tensor]]:
	batch_size = CONFIG.getint('training.loader', 'batch_size')
	num_workers = CONFIG.getint('training.loader', 'num_workers')

	training_dataset, validate_dataset = split_dataset(dataset)
	training_loader = StatefulDataLoader(training_dataset, batch_size = batch_size, shuffle = True, num_workers = num_workers, drop_last = True, pin_memory = True, persistent_workers = True)
	validation_loader = StatefulDataLoader(validate_dataset, batch_size = batch_size, shuffle = False, num_workers = num_workers, pin_memory = True, persistent_workers = True)
	return training_loader, validation_loader


def split_dataset(dataset : Dataset[Tensor]) -> Tuple[Dataset[Tensor], Dataset[Tensor]]:
	split_ratio = CONFIG.getfloat('training.loader', 'split_ratio')
	dataset_size = len(dataset) # type:ignore[arg-type]
	training_size = int(dataset_size * split_ratio)
	validation_size = int(dataset_size - training_size)
	training_dataset, validate_dataset = random_split(dataset, [ training_size, validation_size ])
	return training_dataset, validate_dataset


def create_trainer() -> Trainer:
	trainer_max_epochs = CONFIG.getint('training.trainer', 'max_epochs')
	output_directory_path = CONFIG.get('training.output', 'directory_path')
	output_file_pattern = CONFIG.get('training.output', 'file_pattern')
	trainer_precision = CONFIG.get('training.trainer', 'precision')
	logger = TensorBoardLogger('.logs', name = 'face_swapper')

	return Trainer(
		logger = logger,
		log_every_n_steps = 10,
		max_epochs = trainer_max_epochs,
		precision = trainer_precision, # type:ignore[arg-type]
		callbacks =
		[
			ModelCheckpoint(
				monitor = 'generator_loss',
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
	dataset_file_pattern = CONFIG.get('training.dataset', 'file_pattern')
	dataset_warp_template = cast(WarpTemplate, CONFIG.get('training.dataset', 'warp_template'))
	dataset_batch_ratio = CONFIG.getfloat('training.dataset', 'batch_ratio')
	output_resume_path = CONFIG.get('training.output', 'resume_path')

	if torch.cuda.is_available():
		torch.set_float32_matmul_precision('high')

	dataset = DynamicDataset(dataset_file_pattern, dataset_warp_template, dataset_batch_ratio)
	training_loader, validation_loader = create_loaders(dataset)
	face_swapper_trainer = FaceSwapperTrainer()
	trainer = create_trainer()

	if os.path.isfile(output_resume_path):
		trainer.fit(face_swapper_trainer, training_loader, validation_loader, ckpt_path = output_resume_path)
	else:
		trainer.fit(face_swapper_trainer, training_loader, validation_loader)
