import configparser
import os
from typing import Tuple

import pytorch_lightning
import torch
import torchvision
from LivePortrait.src.modules.motion_extractor import MotionExtractor
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.types import Optimizer
from pytorch_msssim import ssim
from torch import Tensor
from torch.utils.data import DataLoader

from .data_loader import DataLoaderVGG
from .discriminator import MultiscaleDiscriminator
from .generator import AdaptiveEmbeddingIntegrationNetwork
from .helper import hinge_fake_loss, hinge_real_loss
from .typing import Batch, DiscriminatorLossSet, DiscriminatorOutputs, FaceLandmark203, GeneratorLossSet, IdEmbedding, LossTensor, Padding, SourceEmbedding, TargetAttributes, VisionTensor

CONFIG = configparser.ConfigParser()
CONFIG.read('config.ini')


class FaceSwapper(pytorch_lightning.LightningModule):
	def __init__(self) -> None:
		super().__init__()
		id_channels = CONFIG.getint('training.generator', 'id_channels')
		num_blocks = CONFIG.getint('training.generator', 'num_blocks')
		input_channels = CONFIG.getint('training.discriminator', 'input_channels')
		num_filters = CONFIG.getint('training.discriminator', 'num_filters')
		num_layers = CONFIG.getint('training.discriminator', 'num_layers')
		num_discriminators = CONFIG.getint('training.discriminator', 'num_discriminators')
		arcface_path = CONFIG.get('auxiliary_models.paths', 'arcface_path')
		landmarker_path = CONFIG.get('auxiliary_models.paths', 'landmarker_path')
		motion_extractor_path = CONFIG.get('auxiliary_models.paths', 'motion_extractor_path')

		self.generator = AdaptiveEmbeddingIntegrationNetwork(id_channels, num_blocks)
		self.discriminator = MultiscaleDiscriminator(input_channels, num_filters, num_layers, num_discriminators)
		self.arcface = torch.load(arcface_path, map_location = 'cpu', weights_only = False)
		self.landmarker = torch.load(landmarker_path, map_location = 'cpu', weights_only = False)
		self.motion_extractor = MotionExtractor(num_kp = 21, backbone = 'convnextv2_tiny')
		self.motion_extractor.load_state_dict(torch.load(motion_extractor_path, map_location = 'cpu', weights_only = True))
		self.arcface.eval()
		self.landmarker.eval()
		self.motion_extractor.eval()
		self.automatic_optimization = False
		self.mse_loss = torch.nn.MSELoss()
		self.batch_size = CONFIG.getint('training.loader', 'batch_size')

	def forward(self, target_tensor : VisionTensor, source_embedding : SourceEmbedding) -> Tuple[VisionTensor, TargetAttributes]:
		output = self.generator(target_tensor, source_embedding)
		return output

	def configure_optimizers(self) -> Tuple[Optimizer, Optimizer]:
		generator_learning_rate = CONFIG.getfloat('training.generator', 'learning_rate')
		discriminator_learning_rate = CONFIG.getfloat('training.discriminator', 'learning_rate')
		generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr = generator_learning_rate, betas = (0.0, 0.999), weight_decay = 1e-4)
		discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr = discriminator_learning_rate, betas = (0.0, 0.999), weight_decay = 1e-4)
		return generator_optimizer, discriminator_optimizer

	def training_step(self, batch : Batch, batch_index : int) -> Tensor:
		source_tensor, target_tensor, is_same_person = batch
		generator_optimizer, discriminator_optimizer = self.optimizers() #type:ignore[attr-defined]
		source_embedding = self.get_id_embedding(source_tensor, (0, 0, 0, 0))
		swap_tensor, target_attributes = self.generator(target_tensor, source_embedding)
		discriminator_outputs = self.discriminator(swap_tensor)

		generator_losses = self.calc_generator_loss(swap_tensor, target_attributes, discriminator_outputs, batch)
		generator_optimizer.zero_grad()
		self.manual_backward(generator_losses.get('loss_generator'))
		generator_optimizer.step()

		discriminator_losses = self.calc_discriminator_loss(swap_tensor, source_tensor)
		discriminator_optimizer.zero_grad()
		self.manual_backward(discriminator_losses.get('loss_discriminator'))
		discriminator_optimizer.step()

		if self.global_step % CONFIG.getint('training.output', 'preview_frequency') == 0:
			self.log_generator_preview(source_tensor, target_tensor, swap_tensor)

		self.log('l_G', generator_losses.get('loss_generator'), prog_bar = True)
		self.log('l_D', discriminator_losses.get('loss_discriminator'), prog_bar = True)
		self.log('l_ADV', generator_losses.get('loss_adversarial'), prog_bar = True)
		self.log('l_ATTR', generator_losses.get('loss_attribute'), prog_bar = True)
		self.log('l_ID', generator_losses.get('loss_id'), prog_bar=True)
		self.log('l_REC', generator_losses.get('loss_reconstruction'), prog_bar = True)
		return generator_losses.get('loss_generator')

	def calc_adversarial_loss(self, discriminator_outputs : DiscriminatorOutputs) -> LossTensor:
		loss_adversarial = torch.Tensor(0)

		for discriminator_output in discriminator_outputs:
			loss_adversarial += hinge_real_loss(discriminator_output[0]).mean(dim = [ 1, 2, 3 ])
		loss_adversarial = torch.mean(loss_adversarial)
		return loss_adversarial

	def calc_attribute_loss(self, swap_tensor : VisionTensor, target_attributes : TargetAttributes) -> LossTensor:
		loss_attribute = torch.Tensor(0)
		swap_attributes = self.generator.get_attributes(swap_tensor)

		for swap_attribute, target_attribute in zip(swap_attributes, target_attributes):
			loss_attribute += torch.mean(torch.pow(swap_attribute - target_attribute, 2).reshape(self.batch_size, -1), dim = 1).mean()
		loss_attribute *= 0.5
		return loss_attribute

	def calc_reconstruction_loss(self, swap_tensor : VisionTensor, target_tensor : VisionTensor, is_same_person : Tensor) -> LossTensor:
		loss_reconstruction = torch.sum(0.5 * torch.mean(torch.pow(swap_tensor - target_tensor, 2).reshape(self.batch_size, -1), dim = 1) * is_same_person) / (is_same_person.sum() + 1e-4)
		loss_ssim = 1 - ssim(swap_tensor, target_tensor, data_range = float(torch.max(swap_tensor) - torch.min(swap_tensor))).mean()
		loss_reconstruction = (loss_reconstruction + loss_ssim) * 0.5
		return loss_reconstruction

	def calc_id_loss(self, source_tensor : VisionTensor, swap_tensor : VisionTensor) -> LossTensor:
		swap_embedding = self.get_id_embedding(swap_tensor, (30, 0, 10, 10))
		source_embedding = self.get_id_embedding(source_tensor, (30, 0, 10, 10))
		loss_id = (1 - torch.cosine_similarity(source_embedding, swap_embedding, dim = 1)).mean()
		return loss_id

	def calc_tsr_loss(self, swap_tensor : VisionTensor, target_tensor : VisionTensor) -> LossTensor:
		swap_motion_features = self.get_pose_features(swap_tensor)
		target_motion_features = self.get_pose_features(target_tensor)
		loss_tsr = torch.tensor(0).to(swap_tensor.device).to(swap_tensor.dtype)

		for swap_motion_feature, target_motion_feature in zip(swap_motion_features, target_motion_features):
			loss_tsr += self.mse_loss(swap_motion_feature, target_motion_feature)
		return loss_tsr

	def calc_gaze_loss(self, swap_tensor : VisionTensor, target_tensor : VisionTensor) -> LossTensor:
		swap_landmark = self.get_face_landmarks(swap_tensor)
		target_landmark = self.get_face_landmarks(target_tensor)
		left_gaze_loss = self.mse_loss(swap_landmark[:, 198], target_landmark[:, 198])
		right_gaze_loss = self.mse_loss(swap_landmark[:, 197], target_landmark[:, 197])
		gaze_loss = left_gaze_loss + right_gaze_loss
		return gaze_loss

	def calc_generator_loss(self, swap_tensor : VisionTensor, target_attributes : TargetAttributes, discriminator_outputs : DiscriminatorOutputs, batch : Batch) -> GeneratorLossSet:
		source_tensor, target_tensor, is_same_person = batch
		weight_adversarial = CONFIG.getfloat('training.losses', 'weight_adversarial')
		weight_id = CONFIG.getfloat('training.losses', 'weight_id')
		weight_attribute = CONFIG.getfloat('training.losses', 'weight_attribute')
		weight_reconstruction = CONFIG.getfloat('training.losses', 'weight_reconstruction')
		weight_tsr = CONFIG.getfloat('training.losses', 'weight_tsr')
		weight_gaze = CONFIG.getfloat('training.losses', 'weight_gaze')

		generator_loss_set = {}
		generator_loss_set['loss_adversarial'] = self.calc_adversarial_loss(discriminator_outputs)
		generator_loss_set['loss_id'] = self.calc_id_loss(source_tensor, swap_tensor)
		generator_loss_set['loss_attribute'] = self.calc_attribute_loss(swap_tensor, target_attributes)
		generator_loss_set['loss_reconstruction'] = self.calc_reconstruction_loss(swap_tensor, target_tensor, is_same_person)

		if weight_tsr > 0:
			generator_loss_set['loss_tsr'] = self.calc_tsr_loss(swap_tensor, target_tensor)
		else:
			generator_loss_set['loss_tsr'] = torch.tensor(0).to(swap_tensor.device).to(swap_tensor.dtype)

		if weight_gaze > 0:
			generator_loss_set['loss_gaze'] = self.calc_gaze_loss(swap_tensor, target_tensor)
		else:
			generator_loss_set['loss_gaze'] = torch.tensor(0).to(swap_tensor.device).to(swap_tensor.dtype)
		generator_loss_set['loss_generator'] = generator_loss_set.get('loss_adversarial') * weight_adversarial
		generator_loss_set['loss_generator'] += generator_loss_set.get('loss_id') * weight_id
		generator_loss_set['loss_generator'] += generator_loss_set.get('loss_attribute') * weight_attribute
		generator_loss_set['loss_generator'] += generator_loss_set.get('loss_reconstruction') * weight_reconstruction
		generator_loss_set['loss_generator'] += generator_loss_set.get('loss_tsr') * weight_tsr
		generator_loss_set['loss_generator'] += generator_loss_set.get('loss_gaze') * weight_gaze
		return generator_loss_set

	def calc_discriminator_loss(self, swap_tensor : VisionTensor, source_tensor : VisionTensor) -> DiscriminatorLossSet:
		discriminator_loss_set = {}
		fake_discriminator_outputs = self.discriminator(swap_tensor.detach())
		loss_fake = torch.Tensor(0)

		for fake_discriminator_output in fake_discriminator_outputs:
			loss_fake += torch.mean(hinge_fake_loss(fake_discriminator_output[0]).mean(dim = [ 1, 2, 3 ]))
		true_discriminator_outputs = self.discriminator(source_tensor)
		loss_true = torch.Tensor(0)

		for true_discriminator_output in true_discriminator_outputs:
			loss_true += torch.mean(hinge_real_loss(true_discriminator_output[0]).mean(dim = [ 1, 2, 3 ]))
		discriminator_loss_set['loss_discriminator'] = (loss_true.mean() + loss_fake.mean()) * 0.5
		return discriminator_loss_set

	def get_id_embedding(self, vision_tensor : VisionTensor, padding : Padding) -> IdEmbedding:
		crop_vision_tensor = torch.nn.functional.interpolate(vision_tensor, size = (112, 112), mode = 'area')
		crop_vision_tensor = crop_vision_tensor[:, :, 0:112, 8:128]
		crop_vision_tensor[:, :, :padding[0], :] = 0
		crop_vision_tensor[:, :, 112 - padding[1]:, :] = 0
		crop_vision_tensor[:, :, :, :padding[2]] = 0
		crop_vision_tensor[:, :, :, 112 - padding[3]:] = 0
		embedding = self.arcface(crop_vision_tensor)
		embedding = torch.nn.functional.normalize(embedding, p = 2, dim = 1)
		return embedding

	def get_face_landmarks(self, vision_tensor : VisionTensor) -> FaceLandmark203:
		vision_tensor_norm = (vision_tensor + 1) * 0.5
		vision_tensor_norm = torch.nn.functional.interpolate(vision_tensor_norm, size = (224, 224), mode = 'bilinear')
		landmarks = self.landmarker(vision_tensor_norm)[2].view(-1, 203, 2)
		return landmarks

	def get_pose_features(self, vision_tensor : Tensor) -> Tuple[Tensor, Tensor, Tensor]:
		vision_tensor_norm = (vision_tensor + 1) * 0.5
		motion_dict = self.motion_extractor(vision_tensor_norm)
		translation = motion_dict.get('t')
		scale = motion_dict.get('scale')
		rotation = torch.cat([ motion_dict.get('pitch'), motion_dict.get('yaw'), motion_dict.get('roll') ], dim = 1)
		return translation, scale, rotation

	def log_generator_preview(self, source_tensor : VisionTensor, target_tensor : VisionTensor, swap_tensor : VisionTensor) -> None:
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
	os.makedirs(output_directory_path, exist_ok = True)

	return Trainer(
		max_epochs = trainer_max_epochs,
		precision = '16-mixed',
		callbacks =
		[
			ModelCheckpoint(
				monitor = 'l_G',
				dirpath = output_directory_path,
				filename = output_file_pattern,
				every_n_train_steps = 1000,
				save_top_k = 5,
				mode = 'min',
				save_last = True
			)
		],
		log_every_n_steps = 10,
		accumulate_grad_batches = 1,
	)


def train() -> None:
	batch_size = CONFIG.getint('training.loader', 'batch_size')
	num_workers = CONFIG.getint('training.loader', 'num_workers')
	checkpoint_path = CONFIG.get('training.output', 'checkpoint_path')
	dataset = DataLoaderVGG(CONFIG.get('preparing.dataset', 'dataset_path'))

	data_loader = DataLoader(dataset, batch_size = batch_size, shuffle = True, num_workers = num_workers, drop_last = True, pin_memory = True, persistent_workers = True)
	face_swap_model = FaceSwapper()
	trainer = create_trainer()
	trainer.fit(face_swap_model, data_loader, ckpt_path = checkpoint_path)
