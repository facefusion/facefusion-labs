import configparser
import os
from typing import Tuple

import pytorch_lightning
import torch
import torchvision
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.types import Optimizer
from pytorch_msssim import ssim
from torch import Tensor
from torch.utils.data import DataLoader

from .data_loader import DataLoaderVGG
from .discriminator import MultiscaleDiscriminator
from .generator import AdaptiveEmbeddingIntegrationNetwork
from .helper import calc_id_embedding, hinge_fake_loss, hinge_real_loss
from .typing import Batch, DiscriminatorLossSet, DiscriminatorOutputs, FaceLandmark203, GeneratorLossSet, LossTensor, SourceEmbedding, SwapAttributes, TargetAttributes, VisionTensor

CONFIG = configparser.ConfigParser()
CONFIG.read('config.ini')


class FaceSwapperLoss:
	def __init__(self) -> None:
		id_embedder_path = CONFIG.get('training.model', 'id_embedder_path')
		landmarker_path = CONFIG.get('training.model', 'landmarker_path')
		motion_extractor_path = CONFIG.get('training.model', 'motion_extractor_path')
		self.batch_size = CONFIG.getint('training.loader', 'batch_size')
		self.mse_loss = torch.nn.MSELoss()
		self.id_embedder = torch.jit.load(id_embedder_path, map_location = 'cpu')  # type:ignore[no-untyped-call]
		self.landmarker = torch.jit.load(landmarker_path, map_location = 'cpu')  # type:ignore[no-untyped-call]
		self.motion_extractor = torch.jit.load(motion_extractor_path, map_location = 'cpu')  # type:ignore[no-untyped-call]
		self.id_embedder.eval()
		self.landmarker.eval()
		self.motion_extractor.eval()

	def calc_generator_loss(self, swap_tensor : VisionTensor, target_attributes : TargetAttributes, swap_attributes : SwapAttributes, discriminator_outputs : DiscriminatorOutputs, batch : Batch) -> GeneratorLossSet:
		source_tensor, target_tensor, is_same_person = batch
		weight_adversarial = CONFIG.getfloat('training.losses', 'weight_adversarial')
		weight_id = CONFIG.getfloat('training.losses', 'weight_id')
		weight_attribute = CONFIG.getfloat('training.losses', 'weight_attribute')
		weight_reconstruction = CONFIG.getfloat('training.losses', 'weight_reconstruction')
		weight_pose = CONFIG.getfloat('training.losses', 'weight_pose')
		weight_gaze = CONFIG.getfloat('training.losses', 'weight_gaze')
		generator_loss_set = {}

		generator_loss_set['loss_adversarial'] = self.calc_adversarial_loss(discriminator_outputs)
		generator_loss_set['loss_id'] = self.calc_id_loss(source_tensor, swap_tensor)
		generator_loss_set['loss_attribute'] = self.calc_attribute_loss(target_attributes, swap_attributes)
		generator_loss_set['loss_reconstruction'] = self.calc_reconstruction_loss(swap_tensor, target_tensor, is_same_person)

		if weight_pose > 0:
			generator_loss_set['loss_pose'] = self.calc_pose_loss(swap_tensor, target_tensor)
		else:
			generator_loss_set['loss_pose'] = torch.tensor(0).to(swap_tensor.device).to(swap_tensor.dtype)

		if weight_gaze > 0:
			generator_loss_set['loss_gaze'] = self.calc_gaze_loss(swap_tensor, target_tensor)
		else:
			generator_loss_set['loss_gaze'] = torch.tensor(0).to(swap_tensor.device).to(swap_tensor.dtype)

		generator_loss_set['loss_generator'] = generator_loss_set.get('loss_adversarial') * weight_adversarial
		generator_loss_set['loss_generator'] += generator_loss_set.get('loss_id') * weight_id
		generator_loss_set['loss_generator'] += generator_loss_set.get('loss_attribute') * weight_attribute
		generator_loss_set['loss_generator'] += generator_loss_set.get('loss_reconstruction') * weight_reconstruction
		generator_loss_set['loss_generator'] += generator_loss_set.get('loss_pose') * weight_pose
		generator_loss_set['loss_generator'] += generator_loss_set.get('loss_gaze') * weight_gaze
		return generator_loss_set

	def calc_discriminator_loss(self, real_discriminator_outputs : DiscriminatorOutputs, fake_discriminator_outputs : DiscriminatorOutputs) -> DiscriminatorLossSet:
		discriminator_loss_set = {}
		loss_fake = torch.Tensor(0)

		for fake_discriminator_output in fake_discriminator_outputs:
			loss_fake += hinge_fake_loss(fake_discriminator_output[0]).mean()

		loss_true = torch.Tensor(0)

		for true_discriminator_output in real_discriminator_outputs:
			loss_true += hinge_real_loss(true_discriminator_output[0]).mean()

		discriminator_loss_set['loss_discriminator'] = (loss_true.mean() + loss_fake.mean()) * 0.5
		return discriminator_loss_set

	def calc_adversarial_loss(self, discriminator_outputs : DiscriminatorOutputs) -> LossTensor:
		loss_adversarial = torch.Tensor(0)

		for discriminator_output in discriminator_outputs:
			loss_adversarial += hinge_real_loss(discriminator_output[0])

		loss_adversarial = torch.mean(loss_adversarial)
		return loss_adversarial

	def calc_attribute_loss(self, target_attributes : TargetAttributes, swap_attributes : SwapAttributes) -> LossTensor:
		loss_attribute = torch.Tensor(0)

		for swap_attribute, target_attribute in zip(swap_attributes, target_attributes):
			loss_attribute += torch.mean(torch.pow(swap_attribute - target_attribute, 2).reshape(self.batch_size, -1), dim = 1).mean()

		loss_attribute *= 0.5
		return loss_attribute

	def calc_reconstruction_loss(self, swap_tensor : VisionTensor, target_tensor : VisionTensor, is_same_person : Tensor) -> LossTensor:
		loss_reconstruction = torch.pow(swap_tensor - target_tensor, 2).reshape(self.batch_size, -1)
		loss_reconstruction = torch.mean(loss_reconstruction, dim = 1) * 0.5
		loss_reconstruction = torch.sum(loss_reconstruction * is_same_person) / (is_same_person.sum() + 1e-4)
		loss_ssim = 1 - ssim(swap_tensor, target_tensor, data_range = float(torch.max(swap_tensor) - torch.min(swap_tensor))).mean()
		loss_reconstruction = (loss_reconstruction + loss_ssim) * 0.5
		return loss_reconstruction

	def calc_id_loss(self, source_tensor : VisionTensor, swap_tensor : VisionTensor) -> LossTensor:
		swap_embedding = calc_id_embedding(self.id_embedder, swap_tensor, (30, 0, 10, 10))
		source_embedding = calc_id_embedding(self.id_embedder, source_tensor, (30, 0, 10, 10))
		loss_id = (1 - torch.cosine_similarity(source_embedding, swap_embedding)).mean()
		return loss_id

	def calc_pose_loss(self, swap_tensor : VisionTensor, target_tensor : VisionTensor) -> LossTensor:
		swap_motion_features = self.get_pose_features(swap_tensor)
		target_motion_features = self.get_pose_features(target_tensor)
		loss_pose = torch.tensor(0).to(swap_tensor.device).to(swap_tensor.dtype)

		for swap_motion_feature, target_motion_feature in zip(swap_motion_features, target_motion_features):
			loss_pose += self.mse_loss(swap_motion_feature, target_motion_feature)

		return loss_pose

	def calc_gaze_loss(self, swap_tensor : VisionTensor, target_tensor : VisionTensor) -> LossTensor:
		swap_landmark = self.get_face_landmarks(swap_tensor)
		target_landmark = self.get_face_landmarks(target_tensor)
		left_gaze_loss = self.mse_loss(swap_landmark[:, 198], target_landmark[:, 198])
		right_gaze_loss = self.mse_loss(swap_landmark[:, 197], target_landmark[:, 197])
		gaze_loss = left_gaze_loss + right_gaze_loss
		return gaze_loss

	def get_face_landmarks(self, vision_tensor : VisionTensor) -> FaceLandmark203:
		vision_tensor_norm = (vision_tensor + 1) * 0.5
		vision_tensor_norm = torch.nn.functional.interpolate(vision_tensor_norm, size = (224, 224), mode = 'bilinear')
		landmarks = self.landmarker(vision_tensor_norm)[2].view(-1, 203, 2)
		return landmarks

	def get_pose_features(self, vision_tensor : Tensor) -> Tuple[Tensor, Tensor, Tensor]:
		vision_tensor_norm = (vision_tensor + 1) * 0.5
		pitch, yaw, roll, translation, expression, scale, _ = self.motion_extractor(vision_tensor_norm)
		rotation = torch.cat([ pitch, yaw, roll ], dim = 1)
		return translation, scale, rotation


class FaceSwapperTrain(pytorch_lightning.LightningModule, FaceSwapperLoss):
	def __init__(self) -> None:
		super().__init__()
		id_channels = CONFIG.getint('training.model.generator', 'id_channels')
		num_blocks = CONFIG.getint('training.model.generator', 'num_blocks')
		input_channels = CONFIG.getint('training.model.discriminator', 'input_channels')
		num_filters = CONFIG.getint('training.model.discriminator', 'num_filters')
		num_layers = CONFIG.getint('training.model.discriminator', 'num_layers')
		num_discriminators = CONFIG.getint('training.model.discriminator', 'num_discriminators')
		kernel_size = CONFIG.getint('training.model.discriminator', 'kernel_size')
		self.generator = AdaptiveEmbeddingIntegrationNetwork(id_channels, num_blocks)
		self.discriminator = MultiscaleDiscriminator(input_channels, num_filters, num_layers, num_discriminators, kernel_size)
		self.automatic_optimization = CONFIG.getboolean('training.trainer', 'automatic_optimization')

	def forward(self, target_tensor : VisionTensor, source_embedding : SourceEmbedding) -> Tuple[VisionTensor, TargetAttributes]:
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

		self.log('l_G', generator_losses.get('loss_generator'), prog_bar = True)
		self.log('l_D', discriminator_losses.get('loss_discriminator'), prog_bar = True)
		self.log('l_ADV', generator_losses.get('loss_adversarial'), prog_bar = True)
		self.log('l_ATTR', generator_losses.get('loss_attribute'), prog_bar = True)
		self.log('l_ID', generator_losses.get('loss_id'), prog_bar = True)
		self.log('l_REC', generator_losses.get('loss_reconstruction'), prog_bar = True)
		return generator_losses.get('loss_generator')

	def generate_preview(self, source_tensor : VisionTensor, target_tensor : VisionTensor, swap_tensor : VisionTensor) -> None:
		max_preview = 8
		source_tensors = source_tensor[:max_preview]
		target_tensors = target_tensor[:max_preview]
		swap_tensors = swap_tensor[:max_preview]
		rows = [ torch.cat([ source_tensor, target_tensor, swap_tensor ], dim = 2) for source_tensor, target_tensor, swap_tensor in zip(source_tensors, target_tensors, swap_tensors) ]
		grid = torchvision.utils.make_grid(torch.cat(rows, dim = 1).unsqueeze(0), nrow = 1, normalize = True, scale_each = True)
		self.logger.experiment.add_image("Generator Preview", grid, self.global_step)


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
				monitor = 'l_G',
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
	batch_size = CONFIG.getint('training.loader', 'batch_size')
	num_workers = CONFIG.getint('training.loader', 'num_workers')
	checkpoint_path = CONFIG.get('training.output', 'checkpoint_path')
	dataset_path = CONFIG.get('preparing.dataset', 'dataset_path')
	dataset_image_pattern = CONFIG.get('preparing.dataset', 'image_pattern')
	dataset_directory_pattern = CONFIG.get('preparing.dataset', 'directory_pattern')
	same_person_probability = CONFIG.getfloat('preparing.dataset', 'same_person_probability')
	dataset = DataLoaderVGG(dataset_path, dataset_image_pattern, dataset_directory_pattern, same_person_probability)

	data_loader = DataLoader(dataset, batch_size = batch_size, shuffle = True, num_workers = num_workers, drop_last = True, pin_memory = True, persistent_workers = True)
	face_swap_model = FaceSwapperTrain()
	trainer = create_trainer()
	trainer.fit(face_swap_model, data_loader, ckpt_path = checkpoint_path)
