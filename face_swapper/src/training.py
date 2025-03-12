import os
import warnings
from configparser import ConfigParser
from typing import Tuple

import torch
import torchvision
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch import Tensor, nn
from torch.utils.data import Dataset, random_split
from torchdata.stateful_dataloader import StatefulDataLoader

from .dataset import DynamicDataset
from .helper import calc_embedding, overlay_mask
from .models.discriminator import Discriminator
from .models.generator import Generator
from .models.loss import AdversarialLoss, AttributeLoss, DiscriminatorLoss, GazeLoss, IdentityLoss, MaskLoss, MotionLoss, ReconstructionLoss
from .networks.masknet import MaskNet
from .types import Batch, Embedding, OptimizerSet

warnings.filterwarnings('ignore', category = UserWarning, module = 'torch')

CONFIG_PARSER = ConfigParser()
CONFIG_PARSER.read('config.ini')


class FaceSwapperTrainer(LightningModule):
	def __init__(self, config_parser : ConfigParser) -> None:
		super().__init__()
		self.config_embedder_path = config_parser.get('training.model', 'embedder_path')
		self.config_gazer_path = config_parser.get('training.model', 'gazer_path')
		self.config_motion_extractor_path = config_parser.get('training.model', 'motion_extractor_path')
		self.config_face_parser_path = config_parser.get('training.model', 'face_parser_path')
		self.config_accumulate_size = config_parser.getfloat('training.trainer', 'accumulate_size')
		self.config_learning_rate = config_parser.getfloat('training.trainer', 'learning_rate')
		self.config_preview_frequency = config_parser.getint('training.trainer', 'preview_frequency')
		self.embedder = torch.jit.load(self.config_embedder_path, map_location = 'cpu').eval()
		self.gazer = torch.jit.load(self.config_gazer_path, map_location = 'cpu').eval()
		self.motion_extractor = torch.jit.load(self.config_motion_extractor_path, map_location = 'cpu').eval()
		self.face_parser = torch.jit.load(self.config_face_parser_path, map_location ='cpu').eval()
		self.generator = Generator(config_parser)
		self.discriminator = Discriminator(config_parser)
		self.masker = MaskNet(config_parser)
		self.discriminator_loss = DiscriminatorLoss()
		self.adversarial_loss = AdversarialLoss(config_parser)
		self.attribute_loss = AttributeLoss(config_parser)
		self.reconstruction_loss = ReconstructionLoss(config_parser, self.embedder)
		self.identity_loss = IdentityLoss(config_parser, self.embedder)
		self.motion_loss = MotionLoss(config_parser, self.motion_extractor)
		self.gaze_loss = GazeLoss(config_parser, self.gazer)
		self.mask_loss = MaskLoss(config_parser, self.face_parser)
		self.automatic_optimization = False

	def forward(self, source_embedding : Embedding, target_tensor : Tensor) -> Tuple[Tensor, Tensor]:
		with torch.no_grad():
			output_tensor, target_attributes = self.generator(source_embedding, target_tensor)
			target_attribute = target_attributes[-1]
			mask_tensor = self.masker(target_tensor, target_attribute)

		return output_tensor, mask_tensor

	def configure_optimizers(self) -> Tuple[OptimizerSet, OptimizerSet, OptimizerSet]:
		generator_optimizer = torch.optim.AdamW(self.generator.parameters(), lr = self.config_learning_rate, betas = (0.0, 0.999), weight_decay = 1e-4)
		discriminator_optimizer = torch.optim.AdamW(self.discriminator.parameters(), lr = self.config_learning_rate, betas = (0.0, 0.999), weight_decay = 1e-4)
		masker_optimizer = torch.optim.AdamW(self.masker.parameters(), lr = self.config_learning_rate, betas = (0.0, 0.999), weight_decay = 1e-4)
		generator_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(generator_optimizer, T_0 = 300, T_mult = 2)
		discriminator_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(discriminator_optimizer, T_0 = 300, T_mult = 2)
		masker_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(masker_optimizer, T_0 = 300, T_mult = 2)

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
		masker_config =\
		{
			'optimizer': masker_optimizer,
			'lr_scheduler':
			{
				'scheduler': masker_scheduler,
				'interval': 'step'
			}
		}
		return generator_config, discriminator_config, masker_config

	def training_step(self, batch : Batch, batch_index : int) -> Tensor:
		source_tensor, target_tensor = batch
		do_update = (batch_index + 1) % self.config_accumulate_size == 0
		generator_optimizer, discriminator_optimizer, masker_optimizer = self.optimizers() #type:ignore[attr-defined]

		source_embedding = calc_embedding(self.embedder, source_tensor, (0, 0, 0, 0))
		generator_output_tensor, generator_output_attributes = self.generator(source_embedding, target_tensor)
		discriminator_output_tensors = self.discriminator(generator_output_tensor)
		adversarial_loss, weighted_adversarial_loss = self.adversarial_loss(discriminator_output_tensors)
		attribute_loss, weighted_attribute_loss = self.attribute_loss(generator_output_attributes, generator_output_attributes)
		reconstruction_loss, weighted_reconstruction_loss = self.reconstruction_loss(source_tensor, target_tensor, generator_output_tensor)
		identity_loss, weighted_identity_loss = self.identity_loss(generator_output_tensor, source_tensor)
		pose_loss, weighted_pose_loss, expression_loss, weighted_expression_loss = self.motion_loss(target_tensor, generator_output_tensor)
		gaze_loss, weighted_gaze_loss = self.gaze_loss(target_tensor, generator_output_tensor)
		generator_loss = weighted_adversarial_loss + weighted_attribute_loss + weighted_reconstruction_loss + weighted_identity_loss + weighted_pose_loss + weighted_gaze_loss + weighted_expression_loss

		discriminator_source_tensors = self.discriminator(source_tensor)
		discriminator_output_tensors = self.discriminator(generator_output_tensor.detach())
		discriminator_loss = self.discriminator_loss(discriminator_source_tensors, discriminator_output_tensors)

		generator_output_attribute = generator_output_attributes[-1]
		mask_tensor = self.masker(generator_output_tensor.detach(), generator_output_attribute.detach())
		mask_loss = self.mask_loss(target_tensor, mask_tensor)

		self.toggle_optimizer(generator_optimizer)
		self.manual_backward(generator_loss)
		if do_update:
			generator_optimizer.step()
			generator_optimizer.zero_grad()
		self.untoggle_optimizer(generator_optimizer)

		self.toggle_optimizer(discriminator_optimizer)
		self.manual_backward(discriminator_loss)
		if do_update:
			discriminator_optimizer.step()
			discriminator_optimizer.zero_grad()
		self.untoggle_optimizer(discriminator_optimizer)

		self.toggle_optimizer(masker_optimizer)
		self.manual_backward(mask_loss)
		if do_update:
			masker_optimizer.step()
			masker_optimizer.zero_grad()
		self.untoggle_optimizer(masker_optimizer)

		if self.global_step % self.config_preview_frequency == 0:
			self.generate_preview(source_tensor, target_tensor, generator_output_tensor, mask_tensor)

		self.log('generator_loss', generator_loss, prog_bar = True)
		self.log('discriminator_loss', discriminator_loss, prog_bar = True)
		self.log('adversarial_loss', adversarial_loss)
		self.log('attribute_loss', attribute_loss)
		self.log('reconstruction_loss', reconstruction_loss)
		self.log('identity_loss', identity_loss)
		self.log('pose_loss', pose_loss)
		self.log('gaze_loss', gaze_loss)
		self.log('mask_loss', mask_loss)
		return generator_loss

	def validation_step(self, batch : Batch, batch_index : int) -> Tensor:
		source_tensor, target_tensor = batch
		source_embedding = calc_embedding(self.embedder, source_tensor, (0, 0, 0, 0))
		output_tensor, _ = self.generator(source_embedding, target_tensor)
		output_embedding = calc_embedding(self.embedder, output_tensor, (0, 0, 0, 0))
		validation_score = (nn.functional.cosine_similarity(source_embedding, output_embedding).mean() + 1) * 0.5
		self.log('validation_score', validation_score, prog_bar = True)
		return validation_score

	def generate_preview(self, source_tensor : Tensor, target_tensor : Tensor, output_tensor : Tensor, mask_tensor : Tensor) -> None:
		preview_limit = 8
		preview_cells = []
		overlay_tensor = overlay_mask(output_tensor, mask_tensor)

		for source_tensor, target_tensor, output_tensor, overlay_tensor in zip(source_tensor[:preview_limit], target_tensor[:preview_limit], output_tensor[:preview_limit], overlay_tensor[:preview_limit]):
			preview_cell = torch.cat([ source_tensor, target_tensor, output_tensor, overlay_tensor ], dim = 2)
			preview_cells.append(preview_cell)

		preview_cells = torch.cat(preview_cells, dim = 1).unsqueeze(0)
		preview_grid = torchvision.utils.make_grid(preview_cells, normalize = True, scale_each = True)
		self.logger.experiment.add_image('preview', preview_grid, self.global_step) # type:ignore[attr-defined]


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
	config_logger_path = CONFIG_PARSER.get('training.trainer', 'logger_path')
	config_logger_name = CONFIG_PARSER.get('training.trainer', 'logger_name')
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
				monitor = 'generator_loss',
				dirpath = config_directory_path,
				filename = config_file_pattern,
				every_n_train_steps = 1000,
				save_top_k = 3,
				save_last = True
			)
		],
		val_check_interval = 1000
	)


def train() -> None:
	config_resume_path = CONFIG_PARSER.get('training.output', 'resume_path')

	if torch.cuda.is_available():
		torch.set_float32_matmul_precision('high')

	dataset = DynamicDataset(CONFIG_PARSER)
	training_loader, validation_loader = create_loaders(dataset)
	face_swapper_trainer = FaceSwapperTrainer(CONFIG_PARSER)
	trainer = create_trainer()

	if os.path.isfile(config_resume_path):
		trainer.fit(face_swapper_trainer, training_loader, validation_loader, ckpt_path = config_resume_path)
	else:
		trainer.fit(face_swapper_trainer, training_loader, validation_loader)
