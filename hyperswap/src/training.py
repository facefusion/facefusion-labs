import os
import shutil
import warnings
from configparser import ConfigParser
from copy import deepcopy
from pathlib import Path
from typing import List, Tuple, cast

import torch
import torchvision
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch import Tensor, nn
from torch.utils.data import ConcatDataset, Dataset, random_split
from torchdata.stateful_dataloader import StatefulDataLoader

from .dataset import DynamicDataset
from .helper import apply_noise, calculate_face_embedding, erode_mask, overlay_mask
from .models.discriminator import Discriminator
from .models.generator import Generator
from .models.loss import AdversarialLoss, CycleLoss, DiscriminatorLoss, FeatureLoss, GazeLoss, IdentityLoss, MaskLoss, ReconstructionLoss
from .types import Batch, Embedding, Mask, OptimizerSet, TrainerCompileMode, TrainerPrecision, TrainerStrategy

warnings.filterwarnings('ignore', category = UserWarning, module = 'torch')

CONFIG_PARSER = ConfigParser()
CONFIG_PARSER.read('config.ini')


class HyperSwapTrainer(LightningModule):
	def __init__(self, config_parser : ConfigParser) -> None:
		super().__init__()
		self.config_generator_embedder_path = config_parser.get('training.model', 'generator_embedder_path')
		self.config_loss_embedder_path = config_parser.get('training.model', 'loss_embedder_path')
		self.config_gazer_path = config_parser.get('training.model', 'gazer_path')
		self.config_face_masker_path = config_parser.get('training.model', 'face_masker_path')
		self.config_accumulate_size = config_parser.getfloat('training.trainer', 'accumulate_size')
		self.config_discriminator_ratio = config_parser.getfloat('training.trainer', 'discriminator_ratio')
		self.config_gradient_clip = config_parser.getfloat('training.trainer', 'gradient_clip')
		self.config_compile_mode = cast(TrainerCompileMode, config_parser.get('training.trainer', 'compile_mode'))
		self.config_preview_frequency = config_parser.getint('training.trainer', 'preview_frequency')
		self.config_mask_factor = config_parser.getfloat('training.modifier', 'mask_factor')
		self.config_noise_factor = config_parser.getfloat('training.modifier', 'noise_factor')
		self.config_generator_learning_rate = config_parser.getfloat('training.optimizer.generator', 'learning_rate')
		self.config_generator_momentum = config_parser.getfloat('training.optimizer.generator', 'momentum')
		self.config_generator_scheduler_factor = config_parser.getfloat('training.optimizer.generator', 'scheduler_factor')
		self.config_generator_scheduler_patience = config_parser.getint('training.optimizer.generator', 'scheduler_patience')
		self.config_discriminator_learning_rate = config_parser.getfloat('training.optimizer.discriminator', 'learning_rate')
		self.config_discriminator_momentum = config_parser.getfloat('training.optimizer.discriminator', 'momentum')
		self.config_discriminator_scheduler_factor = config_parser.getfloat('training.optimizer.discriminator', 'scheduler_factor')
		self.config_discriminator_scheduler_patience = config_parser.getint('training.optimizer.discriminator', 'scheduler_patience')
		self.generator_embedder = torch.jit.load(self.config_generator_embedder_path, map_location = 'cpu').eval()
		self.loss_embedder = torch.jit.load(self.config_loss_embedder_path, map_location = 'cpu').eval()
		self.gazer = torch.jit.load(self.config_gazer_path, map_location = 'cpu').eval()
		self.face_masker = torch.jit.load(self.config_face_masker_path, map_location ='cpu').eval()
		self.generator = Generator(config_parser)
		self.discriminator = Discriminator(config_parser)
		if self.config_compile_mode:
			self.generator = torch.compile(self.generator, mode = self.config_compile_mode)
			self.discriminator = torch.compile(self.discriminator, mode = self.config_compile_mode)
		self.discriminator_loss = DiscriminatorLoss()
		self.adversarial_loss = AdversarialLoss(config_parser)
		self.cycle_loss = CycleLoss(config_parser)
		self.feature_loss = FeatureLoss(config_parser)
		self.reconstruction_loss = ReconstructionLoss(config_parser, self.loss_embedder)
		self.identity_loss = IdentityLoss(config_parser, self.loss_embedder)
		self.gaze_loss = GazeLoss(config_parser, self.gazer)
		self.mask_loss = MaskLoss(config_parser, self.face_masker)
		self.automatic_optimization = False

	def forward(self, source_embedding : Embedding, target_tensor : Tensor) -> Tuple[Tensor, Mask]:
		with torch.no_grad():
			generator_target_features = self.generator.encode_features(target_tensor)
			output_tensor, output_mask = self.generator(source_embedding, target_tensor, generator_target_features)

		if self.config_mask_factor > 0:
			output_mask = erode_mask(output_mask, self.config_mask_factor)

		return output_tensor, output_mask

	def configure_optimizers(self) -> Tuple[OptimizerSet, OptimizerSet]:
		generator_optimizer = torch.optim.AdamW(self.generator.parameters(), lr = self.config_generator_learning_rate, betas = (self.config_generator_momentum, 0.999), weight_decay = 1e-4, eps = 1e-8)
		discriminator_optimizer = torch.optim.AdamW(self.discriminator.parameters(), lr = self.config_discriminator_learning_rate, betas = (self.config_discriminator_momentum, 0.999), weight_decay = 1e-4, eps = 1e-8)
		generator_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(generator_optimizer, mode = 'min', factor = self.config_generator_scheduler_factor, patience = self.config_generator_scheduler_patience, min_lr = 1e-8)
		discriminator_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(discriminator_optimizer, mode = 'min', factor = self.config_discriminator_scheduler_factor, patience = self.config_discriminator_scheduler_patience, min_lr = 1e-8)

		generator_config =\
		{
			'optimizer': generator_optimizer,
			'lr_scheduler':
			{
				'scheduler': generator_scheduler
			}
		}
		discriminator_config =\
		{
			'optimizer': discriminator_optimizer,
			'lr_scheduler':
			{
				'scheduler': discriminator_scheduler
			}
		}
		return generator_config, discriminator_config

	def training_step(self, batch : Batch, batch_index : int) -> Tensor:
		source_tensor, target_tensor = batch
		do_update = (batch_index + 1) % self.config_accumulate_size == 0
		generator_optimizer, discriminator_optimizer = self.optimizers() #type:ignore[attr-defined]
		generator_scheduler, discriminator_scheduler = self.lr_schedulers() #type:ignore[attr-defined]
		source_embedding = calculate_face_embedding(self.generator_embedder, source_tensor, (0, 0, 0, 0))
		target_embedding = calculate_face_embedding(self.generator_embedder, target_tensor, (0, 0, 0, 0))

		if self.config_noise_factor > 0:
			source_embedding = apply_noise(source_embedding, self.config_noise_factor)
			source_embedding = nn.functional.normalize(source_embedding, p = 2)

		generator_target_features = self.generator.encode_features(target_tensor)
		generator_output_tensor, generator_output_mask = self.generator(source_embedding, target_tensor, generator_target_features)
		generator_output_features = self.generator.encode_features(generator_output_tensor)
		cycle_output_tensor, cycle_output_mask = self.generator(target_embedding, generator_output_tensor, generator_output_features)
		cycle_output_features = self.generator.encode_features(cycle_output_tensor)
		discriminator_output_tensors = self.discriminator(generator_output_tensor)
		adversarial_loss, weighted_adversarial_loss = self.adversarial_loss(discriminator_output_tensors)
		cycle_loss, weighted_cycle_loss = self.cycle_loss(target_tensor, cycle_output_tensor, generator_target_features, cycle_output_features)
		feature_loss, weighted_feature_loss = self.feature_loss(generator_target_features, generator_output_features)
		reconstruction_loss, weighted_reconstruction_loss = self.reconstruction_loss(source_tensor, target_tensor, generator_output_tensor)
		identity_loss, weighted_identity_loss = self.identity_loss(generator_output_tensor, source_tensor)
		gaze_loss, weighted_gaze_loss = self.gaze_loss(target_tensor, generator_output_tensor)
		mask_loss, weighted_mask_loss = self.mask_loss(target_tensor, generator_output_mask)
		generator_loss = weighted_adversarial_loss + weighted_cycle_loss + weighted_feature_loss + weighted_reconstruction_loss + weighted_identity_loss + weighted_gaze_loss + weighted_mask_loss

		if torch.randn(1).item() < self.config_discriminator_ratio:
			discriminator_real_tensors = self.discriminator(source_tensor)
		else:
			discriminator_real_tensors = self.discriminator(target_tensor)
		discriminator_fake_tensors = self.discriminator(generator_output_tensor.detach())
		discriminator_loss = self.discriminator_loss(discriminator_real_tensors, discriminator_fake_tensors)

		self.toggle_optimizer(generator_optimizer)
		self.manual_backward(generator_loss)

		if do_update:
			if self.config_gradient_clip:
				self.clip_gradients(
					generator_optimizer,
					gradient_clip_val = self.config_gradient_clip,
					gradient_clip_algorithm = 'norm'
				)
			generator_optimizer.step()
			generator_optimizer.zero_grad()
		self.untoggle_optimizer(generator_optimizer)

		self.toggle_optimizer(discriminator_optimizer)
		self.manual_backward(discriminator_loss)

		if do_update:
			if self.config_gradient_clip:
				self.clip_gradients(
					discriminator_optimizer,
					gradient_clip_val = self.config_gradient_clip,
					gradient_clip_algorithm = 'norm'
				)
			discriminator_optimizer.step()
			discriminator_optimizer.zero_grad()
		self.untoggle_optimizer(discriminator_optimizer)

		if self.global_step % self.config_preview_frequency == 0:
			self.generate_preview(source_tensor, target_tensor, generator_output_tensor, generator_output_mask)

		self.log('generator_loss', generator_loss, prog_bar = True)
		self.log('discriminator_loss', discriminator_loss, prog_bar = True)
		self.log('adversarial_loss', adversarial_loss)
		self.log('cycle_loss', cycle_loss)
		self.log('feature_loss', feature_loss)
		self.log('reconstruction_loss', reconstruction_loss)
		self.log('identity_loss', identity_loss)
		self.log('gaze_loss', gaze_loss)
		self.log('mask_loss', mask_loss)

		if do_update:
			generator_scheduler.step(generator_loss)
			discriminator_scheduler.step(discriminator_loss)

		return generator_loss

	def validation_step(self, batch : Batch, batch_index : int) -> Tensor:
		source_tensor, target_tensor = batch
		source_embedding = calculate_face_embedding(self.generator_embedder, source_tensor, (0, 0, 0, 0))
		output_tensor, _ = self.forward(source_embedding, target_tensor)
		output_embedding = calculate_face_embedding(self.generator_embedder, output_tensor, (0, 0, 0, 0))
		validation_score = (nn.functional.cosine_similarity(source_embedding, output_embedding).mean() + 1) * 0.5
		self.log('validation_score', validation_score, sync_dist = True, prog_bar = True)
		return validation_score

	def generate_preview(self, source_tensor : Tensor, target_tensor : Tensor, output_tensor : Tensor, output_mask : Mask) -> None:
		preview_limit = 8
		preview_cells = []
		overlay_tensor = overlay_mask(output_tensor, output_mask)

		for source_tensor, target_tensor, output_tensor, overlay_tensor in zip(source_tensor[:preview_limit], target_tensor[:preview_limit], output_tensor[:preview_limit], overlay_tensor[:preview_limit]):
			preview_cell = torch.cat([ source_tensor, target_tensor, output_tensor, overlay_tensor ], dim = 2)
			preview_cells.append(preview_cell)

		preview_cells = torch.cat(preview_cells, dim = 1).unsqueeze(0)
		preview_grid = torchvision.utils.make_grid(preview_cells, normalize = True, scale_each = True)
		self.logger.experiment.add_image('preview', preview_grid, self.global_step) # type:ignore[attr-defined]


class ModelWithConfigCheckpoint(ModelCheckpoint):
	def _save_checkpoint(self, trainer : Trainer, checkpoint_path : str) -> None:
		super()._save_checkpoint(trainer, checkpoint_path)
		config_path = Path(checkpoint_path).with_suffix('.ini')
		shutil.copy2('config.ini', config_path)


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


def prepare_datasets(config_parser : ConfigParser) -> List[Dataset[Tensor]]:
	datasets = []

	for config_section in config_parser.sections():

		if config_section.startswith('training.dataset'):
			config_multiplier = config_parser.getint(config_section, 'multiplier')
			__config_parser__ = deepcopy(config_parser)
			__config_parser__.remove_section(config_section)
			__config_parser__.add_section('training.dataset.current')

			for key, value in config_parser.items(config_section):
				__config_parser__.set('training.dataset.current', key, value)

			dynamic_dataset = DynamicDataset(__config_parser__)
			datasets.extend([ dynamic_dataset ] * config_multiplier)

	return datasets


def create_trainer() -> Trainer:
	config_max_epochs = CONFIG_PARSER.getint('training.trainer', 'max_epochs')
	config_strategy = cast(TrainerStrategy, CONFIG_PARSER.get('training.trainer', 'strategy'))
	config_precision = cast(TrainerPrecision, CONFIG_PARSER.get('training.trainer', 'precision'))
	config_sync_batchnorm = CONFIG_PARSER.getboolean('training.trainer', 'sync_batchnorm')
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
		sync_batchnorm = config_sync_batchnorm,
		callbacks =
		[
			ModelWithConfigCheckpoint(
				monitor = 'generator_loss',
				dirpath = config_directory_path,
				filename = config_file_pattern,
				every_n_train_steps = 1000,
				save_top_k = 5,
				save_last = True
			)
		],
		val_check_interval = 1000
	)


def train() -> None:
	config_resume_path = CONFIG_PARSER.get('training.output', 'resume_path')

	if torch.cuda.is_available():
		torch.set_float32_matmul_precision('high')

	dataset = ConcatDataset(prepare_datasets(CONFIG_PARSER))
	training_loader, validation_loader = create_loaders(dataset)
	hyperswap_trainer = HyperSwapTrainer(CONFIG_PARSER)
	trainer = create_trainer()

	if os.path.isfile(config_resume_path):
		trainer.fit(hyperswap_trainer, training_loader, validation_loader, ckpt_path = config_resume_path)
	else:
		trainer.fit(hyperswap_trainer, training_loader, validation_loader)
