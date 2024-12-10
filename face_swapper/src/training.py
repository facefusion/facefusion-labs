import configparser
import random
import numpy

from typing import Tuple
import os
import cv2
import torchvision

import pytorch_lightning
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from pytorch_lightning.utilities.types import OptimizerLRScheduler
import torch

from .discriminator import MultiscaleDiscriminator
from .generator import AdaptiveEmbeddingIntegrationNetwork
from .data_loader import DataLoaderVGG, read_image

from .typing import Tensor, LossDict, TargetAttributes, DiscriminatorOutputs, Batch
from .helper import hinge_loss, calc_distance_ratio, L2_loss, randomize_expression
from pytorch_msssim import ssim

CONFIG = configparser.ConfigParser()
CONFIG.read('config.ini')


def load_models():
	id_channels = CONFIG.getint('training.generator', 'id_channels')
	num_blocks = CONFIG.getint('training.generator', 'num_blocks')
	generator = AdaptiveEmbeddingIntegrationNetwork(id_channels, num_blocks)

	input_channels = CONFIG.getint('training.discriminator', 'input_channels')
	num_filters = CONFIG.getint('training.discriminator', 'num_filters')
	num_layers = CONFIG.getint('training.discriminator', 'num_layers')
	num_discriminators = CONFIG.getint('training.discriminator', 'num_discriminators')
	discriminator = MultiscaleDiscriminator(input_channels, num_filters, num_layers, num_discriminators)

	model_path = CONFIG.get('auxiliary_models.paths', 'arcface_path')
	arcface = torch.load(model_path, map_location = 'cpu', weights_only = False)
	arcface.eval()

	if CONFIG.getfloat('training.losses', 'weight_eye_gaze') > 0 or CONFIG.getfloat('training.losses', 'weight_eye_open') > 0 or CONFIG.getfloat('training.losses', 'weight_lip_open') > 0:
		model_path = CONFIG.get('auxiliary_models.paths', 'landmarker_path')
		landmarker = torch.load(model_path, map_location = 'cpu', weights_only = False)
		landmarker.eval()
	else:
		landmarker = None

	if CONFIG.getfloat('training.losses', 'weight_tsr') > 0 or CONFIG.getboolean('preparing.augmentation', 'expression'):
		from LivePortrait.src.modules.motion_extractor import MotionExtractor

		model_path = CONFIG.get('auxiliary_models.paths', 'motion_extractor_path')
		motion_extractor = MotionExtractor(num_kp = 21, backbone = 'convnextv2_tiny')
		motion_extractor.load_state_dict(torch.load(model_path, map_location = 'cpu', weights_only = True))
		motion_extractor.eval()
	else:
		motion_extractor = None

	if CONFIG.getboolean('preparing.augmentation', 'expression'):
		from LivePortrait.src.modules.appearance_feature_extractor import AppearanceFeatureExtractor
		from LivePortrait.src.modules.warping_network import WarpingNetwork
		from LivePortrait.src.modules.spade_generator import SPADEDecoder

		feature_extractor_path = CONFIG.get('auxiliary_models.paths', 'feature_extractor_path')
		feature_extractor = AppearanceFeatureExtractor(3, 64, 2, 512, 32, 16, 6)
		feature_extractor.load_state_dict(torch.load(feature_extractor_path, map_location = 'cpu', weights_only = True))
		feature_extractor.eval()

		warping_network_path = CONFIG.get('auxiliary_models.paths', 'warping_network_path')
		dense_motion_params = {
			'block_expansion': 32,
			'max_features': 1024,
			'num_blocks': 5,
			'reshape_depth': 16,
			'compress': 4
		}
		warping_network = WarpingNetwork(num_kp = 21, block_expansion = 64, max_features = 512, num_down_blocks = 2, reshape_channel = 32, estimate_occlusion_map = True, dense_motion_params = dense_motion_params)
		warping_network.load_state_dict(torch.load(warping_network_path, map_location='cpu', weights_only=True))
		warping_network.eval()

		spade_generator_path = CONFIG.get('auxiliary_models.paths', 'spade_generator_path')
		spade_generator = SPADEDecoder(upscale = 2, block_expansion = 64, max_features = 512, num_down_blocks = 2)
		spade_generator.load_state_dict(torch.load(spade_generator_path, map_location = 'cpu', weights_only = True))
		spade_generator.eval()
	else:
		feature_extractor = None
		warping_network = None
		spade_generator = None
	return generator, discriminator, arcface, landmarker, motion_extractor, feature_extractor, warping_network, spade_generator


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
				# every_n_epochs = 1,
				every_n_train_steps = 1000,
				save_top_k = 5,
				mode = 'min',
				save_last = True
			)
		],
		log_every_n_steps = 10,
		accumulate_grad_batches = 1,
	)


def train():
	batch_size = CONFIG.getint('training.loader', 'batch_size')
	num_workers = CONFIG.getint('training.loader', 'num_workers')
	checkpoint_path = CONFIG.get('training.output', 'checkpoint_path')
	dataset = DataLoaderVGG(CONFIG.get('preparing.dataset', 'dataset_path'))

	if not (checkpoint_path and os.path.exists(checkpoint_path)):
		checkpoint_path = None
	data_loader = DataLoader(dataset, batch_size = batch_size, shuffle = True, num_workers = num_workers, drop_last = True, pin_memory = True, persistent_workers = True)
	face_swap_model = FaceSwapper(*load_models())
	trainer = create_trainer()
	trainer.fit(face_swap_model, data_loader, ckpt_path = checkpoint_path)


class FaceSwapper(pytorch_lightning.LightningModule):
	def __init__(self, generator, discriminator, arcface, landmarker, motion_extractor, feature_extractor, warping_network, spade_generator) -> None:
		super().__init__()
		self.generator = generator
		self.discriminator = discriminator
		self.arcface = arcface
		self.landmarker = landmarker
		self.motion_extractor = motion_extractor
		self.feature_extractor = feature_extractor
		self.warping_network = warping_network
		self.spade_generator = spade_generator

		self.loss_adversarial_accumulated = 20
		self.automatic_optimization = False
		self.batch_size = CONFIG.getint('training.loader', 'batch_size')


	def forward(self, target_tensor : Tensor, source_embedding : Tensor) -> Tensor:
		output = self.generator(target_tensor, source_embedding)
		return output


	def state_dict(self, *args, **kwargs):
		return {
			"generator": self.generator.state_dict(),
			"discriminator": self.discriminator.state_dict(),
		}

	def load_state_dict(self, state_dict, strict: bool = True):
		if "generator" in state_dict:
			self.generator.load_state_dict(state_dict["generator"], strict = strict)
		if "discriminator" in state_dict:
			self.discriminator.load_state_dict(state_dict["discriminator"], strict = strict)


	def configure_optimizers(self) -> OptimizerLRScheduler:
		generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr = CONFIG.getfloat('training.generator', 'learning_rate'), betas = (0.0, 0.999), weight_decay = 1e-4)
		discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr = CONFIG.getfloat('training.discriminator', 'learning_rate'), betas = (0.0, 0.999), weight_decay = 1e-4)
		generator_scheduler = torch.optim.lr_scheduler.StepLR(generator_optimizer, step_size = CONFIG.getint('training.schedulers', 'step'), gamma = CONFIG.getfloat('training.schedulers', 'gamma'))
		discriminator_scheduler = torch.optim.lr_scheduler.StepLR(discriminator_optimizer, step_size = CONFIG.getint('training.schedulers', 'step'), gamma = CONFIG.getfloat('training.schedulers', 'gamma'))
		return (
			{
				"optimizer": generator_optimizer,
				"lr_scheduler": generator_scheduler
			},
			{
				"optimizer": discriminator_optimizer,
				"lr_scheduler": discriminator_scheduler
			})


	def training_step(self, batch : Batch, batch_index : int) -> Tensor:
		source_tensor, target_tensor, is_same_person = batch
		generator_optimizer, discriminator_optimizer = self.optimizers()
		source_embedding = self.get_arcface_embedding(source_tensor, (0, 0, 0, 0))

		if random.random() > 0.5 and CONFIG.getboolean('preparing.augmentation', 'expression'):
			target_tensor = randomize_expression(target_tensor, self.feature_extractor, self.motion_extractor, self.warping_network, self.spade_generator)

		swap_tensor, target_attributes = self(target_tensor, source_embedding)
		discriminator_outputs = self.discriminator(swap_tensor)

		generator_losses = self.calc_generator_loss(swap_tensor, target_attributes, discriminator_outputs, batch)
		generator_optimizer.zero_grad()
		self.manual_backward(generator_losses.get('loss_generator'))
		generator_optimizer.step()

		discriminator_losses = self.calc_discriminator_loss(swap_tensor, source_tensor)
		discriminator_optimizer.zero_grad()
		self.manual_backward(discriminator_losses.get('loss_discriminator'))

		if not CONFIG.getboolean('training.discriminator', 'disable') or self.loss_adversarial_accumulated < 0.4:
			discriminator_optimizer.step()

		if self.global_step % CONFIG.getint('training.output', 'preview_frequency') == 0:
			self.log_generator_preview(source_tensor, target_tensor, swap_tensor)

		if self.global_step % CONFIG.getint('training.output', 'validation_frequency') == 0:
			self.log_validation_preview()
		self.log('l_G', generator_losses.get('loss_generator'), prog_bar = True)
		self.log('l_D', discriminator_losses.get('loss_discriminator'), prog_bar = True)
		self.log('l_ADV_A', self.loss_adversarial_accumulated, prog_bar = True)
		self.log('l_ADV', generator_losses.get('loss_adversarial'), prog_bar = False)
		self.log('l_id', generator_losses.get('loss_identity'), prog_bar = True)
		self.log('l_attr', generator_losses.get('loss_attribute'), prog_bar = True)
		self.log('l_rec', generator_losses.get('loss_reconstruction'), prog_bar = True)
		return generator_losses.get('loss_generator')


	def calc_generator_loss(self, swap_tensor : Tensor, target_attributes : TargetAttributes, discriminator_outputs : DiscriminatorOutputs, batch : Batch) -> LossDict:
		source_tensor, target_tensor, is_same_person = batch
		generator_losses = {}
		# adversarial loss
		loss_adversarial = 0

		for discriminator_output in discriminator_outputs:
			loss_adversarial += hinge_loss(discriminator_output[0], True).mean(dim = [ 1, 2, 3 ])
		loss_adversarial = torch.mean(loss_adversarial)
		generator_losses['loss_adversarial'] = loss_adversarial
		generator_losses['loss_generator'] = loss_adversarial * CONFIG.getfloat('training.losses', 'weight_adversarial')
		self.loss_adversarial_accumulated = self.loss_adversarial_accumulated * 0.98 + loss_adversarial.item() * 0.02

		# identity loss
		swap_embedding = self.get_arcface_embedding(swap_tensor, (30, 0, 10, 10))
		source_embedding = self.get_arcface_embedding(source_tensor, (30, 0, 10, 10))
		loss_identity = (1 - torch.cosine_similarity(source_embedding, swap_embedding, dim = 1)).mean()
		generator_losses['loss_identity'] = loss_identity
		generator_losses['loss_generator'] += loss_identity * CONFIG.getfloat('training.losses', 'weight_identity')

		# attribute loss
		loss_attribute = 0
		swap_attributes = self.generator.get_attributes(swap_tensor)

		for swap_attribute, target_attribute in zip(swap_attributes, target_attributes):
			loss_attribute += torch.mean(torch.pow(swap_attribute - target_attribute, 2).reshape(self.batch_size, -1), dim = 1).mean()
		loss_attribute *= 0.5
		generator_losses['loss_attribute'] = loss_attribute
		generator_losses['loss_generator'] += loss_attribute * CONFIG.getfloat('training.losses', 'weight_attribute')

		# reconstruction loss
		loss_reconstruction = torch.sum(0.5 * torch.mean(torch.pow(swap_tensor - target_tensor, 2).reshape(self.batch_size, -1), dim = 1) * is_same_person) / (is_same_person.sum() + 1e-4)
		loss_ssim = 1 - ssim(swap_tensor, target_tensor, data_range = float(torch.max(swap_tensor) - torch.min(swap_tensor))).mean()
		loss_reconstruction = loss_reconstruction * 0.3 + loss_ssim * 0.7
		generator_losses['loss_reconstruction'] = loss_reconstruction
		generator_losses['loss_generator'] += CONFIG.getfloat('training.losses', 'weight_reconstruction')

		if CONFIG.getfloat('training.losses', 'weight_tsr') > 0:
			# tsr loss
			swap_motion_features = self.get_motion_features(swap_tensor)
			target_motion_features = self.get_motion_features(target_tensor)
			loss_tsr = torch.tensor(0).to(swap_tensor.device).to(swap_tensor.dtype)

			for swap_motion_feature, target_motion_feature in zip(swap_motion_features, target_motion_features):
				loss_tsr += L2_loss(swap_motion_feature, target_motion_feature)
			generator_losses['loss_tsr'] = loss_tsr
			generator_losses['loss_generator'] += loss_tsr * CONFIG.getfloat('training.losses', 'weight_tsr')


		if CONFIG.getfloat('training.losses', 'weight_eye_gaze') > 0 or CONFIG.getfloat('training.losses', 'weight_eye_open') > 0 or CONFIG.getfloat('training.losses', 'weight_lip_open') > 0:
			swap_landmark_features = self.get_landmark_features(swap_tensor)
			target_landmark_features = self.get_landmark_features(target_tensor)

			# eye gaze loss
			loss_left_eye_gaze = L2_loss(swap_landmark_features[3], target_landmark_features[3])
			loss_right_eye_gaze = L2_loss(swap_landmark_features[4], target_landmark_features[4])
			loss_eye_gaze = loss_left_eye_gaze + loss_right_eye_gaze
			generator_losses['loss_eye_gaze'] = loss_eye_gaze
			generator_losses['loss_generator'] += loss_eye_gaze * CONFIG.getfloat('training.losses', 'weight_eye_gaze')

			# eye open loss
			loss_left_eye_open = L2_loss(swap_landmark_features[0], target_landmark_features[0])
			loss_right_eye_open = L2_loss(swap_landmark_features[1], target_landmark_features[1])
			loss_eye_open = loss_left_eye_open + loss_right_eye_open
			generator_losses['loss_eye_open'] = loss_eye_open * CONFIG.getfloat('training.losses', 'weight_eye_open')
			generator_losses['loss_generator'] += loss_eye_open

			# lip open loss
			loss_lip_open = L2_loss(swap_landmark_features[2], target_landmark_features[2])
			generator_losses['loss_lip_open'] = loss_lip_open * CONFIG.getfloat('training.losses', 'weight_lip_open')
			generator_losses['loss_generator'] += loss_lip_open
		return generator_losses


	def calc_discriminator_loss(self, swap_tensor : Tensor, source_tensor : Tensor) -> LossDict:
		discriminator_losses = {}
		fake_discriminator_outputs = self.discriminator(swap_tensor.detach())
		loss_fake = 0

		for fake_discriminator_output in fake_discriminator_outputs:
			loss_fake += torch.mean(hinge_loss(fake_discriminator_output[0], False).mean(dim=[1, 2, 3]))
		true_discriminator_outputs = self.discriminator(source_tensor)
		loss_true = 0

		for true_discriminator_output in true_discriminator_outputs:
			loss_true += torch.mean(hinge_loss(true_discriminator_output[0], True).mean(dim=[1, 2, 3]))
		discriminator_losses['loss_discriminator'] = (loss_true.mean() + loss_fake.mean()) * 0.5
		return discriminator_losses


	def get_arcface_embedding(self, vision_tensor : Tensor, padding : Tuple[int, int, int, int]) -> Tensor:
		_, _, height, width = vision_tensor.shape
		crop_height = int(height * 0.0586)
		crop_width = int(width * 0.0586)
		crop_vision_tensor = vision_tensor[:, :, crop_height : height - crop_height, crop_width : width - crop_width]
		crop_vision_tensor = torch.nn.functional.interpolate(crop_vision_tensor, size = (112, 112), mode = 'bilinear')
		crop_vision_tensor[:, :, :padding[0], :] = 0
		crop_vision_tensor[:, :, -padding[1]:, :] = 0
		crop_vision_tensor[:, :, :, :padding[2]] = 0
		crop_vision_tensor[:, :, :, -padding[3]:] = 0
		embedding = self.arcface(crop_vision_tensor)
		embedding = torch.nn.functional.normalize(embedding, p = 2, dim = 1)
		return embedding


	def get_landmark_features(self, vision_tensor : Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
		vision_tensor_norm = (vision_tensor + 1) * 0.5
		vision_tensor_norm = torch.nn.functional.interpolate(vision_tensor_norm, size = (224, 224), mode = 'bilinear')
		landmarks = self.landmarker(vision_tensor_norm)[2]
		landmarks = landmarks.view(-1, 203, 2) * 256
		left_eye_open_ratio = calc_distance_ratio(landmarks, (6, 18, 0, 12))
		right_eye_open_ratio = calc_distance_ratio(landmarks, (30, 42, 24, 36))
		lip_open_ratio = calc_distance_ratio(landmarks, (90, 102, 48, 66))
		left_eye_gaze = landmarks[:, 198]
		right_eye_gaze = landmarks[:, 197]
		return left_eye_open_ratio, right_eye_open_ratio, lip_open_ratio, left_eye_gaze, right_eye_gaze


	def get_motion_features(self, vision_tensor : Tensor) -> Tuple[Tensor, Tensor, Tensor]:
		vision_tensor_norm = (vision_tensor + 1) * 0.5
		motion_dict = self.motion_extractor(vision_tensor_norm)
		translation = motion_dict.get('t')
		scale = motion_dict.get('scale')
		rotation = torch.cat([ motion_dict.get('pitch'), motion_dict.get('yaw'), motion_dict.get('roll') ], dim = 1)
		return translation, scale, rotation


	def log_generator_preview(self, source_tensor, target_tensor, swap_tensor):
		max_preview = 8
		source_tensor = source_tensor[:max_preview]
		target_tensor = target_tensor[:max_preview]
		swap_tensor = swap_tensor[:max_preview]
		rows = [torch.cat([src, tgt, swp], dim=2) for src, tgt, swp in zip(source_tensor, target_tensor, swap_tensor)]
		grid = torchvision.utils.make_grid(torch.cat(rows, dim=1).unsqueeze(0), nrow=1, normalize=True, scale_each=True)
		os.makedirs("previews", exist_ok=True)
		torchvision.utils.save_image(grid, f"previews/step_{self.global_step}.jpg")
		self.logger.experiment.add_image("Generator Preview", grid, self.global_step)

	def log_validation_preview(self):
		validation_source_path = CONFIG.get('training.validation', 'sources')
		validation_target_path = CONFIG.get('training.validation', 'targets')
		sources = [read_image(os.path.join(validation_source_path, f)) for f in os.listdir(validation_source_path) if f.lower().endswith('.jpg') or f.lower().endswith('.png')]
		targets = [read_image(os.path.join(validation_target_path, f)) for f in os.listdir(validation_target_path) if f.lower().endswith('.jpg') or f.lower().endswith('.png')]
		transforms = torchvision.transforms.Compose(
			[
				torchvision.transforms.Resize((256, 256), interpolation = torchvision.transforms.InterpolationMode.BICUBIC),
				torchvision.transforms.ToTensor(),
				torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
			])
		to_numpy = lambda x: (x.cpu().detach().numpy()[0].transpose(1, 2, 0).clip(-1, 1)[:,:,::-1] + 1) * 127.5
		self.generator.eval()
		results = []

		for source, target in zip(sources, targets):
			source_tensor = transforms(source).unsqueeze(0).to(self.device).half()
			target_tensor = transforms(target).unsqueeze(0).to(self.device).half()
			source_embedding = self.get_arcface_embedding(source_tensor, (0, 0, 0, 0))

			with torch.no_grad():
				output, _ = self.generator(target_tensor, source_embedding)
				results.append(numpy.hstack([to_numpy(source_tensor), to_numpy(target_tensor), to_numpy(output)]))
		preview = numpy.vstack(results)
		os.makedirs("validation_previews", exist_ok=True)
		cv2.imwrite(f"validation_previews/step_{self.global_step}.jpg", preview)
		self.generator.train()
