import configparser
import os

import cv2
import numpy
import pytorch_lightning
import torch
import torchvision
from LivePortrait.src.modules.motion_extractor import MotionExtractor
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.types import Optimizer
from pytorch_msssim import ssim
from torch.utils.data import DataLoader

from .data_loader import DataLoaderVGG, read_image
from .discriminator import MultiscaleDiscriminator
from .generator import AdaptiveEmbeddingIntegrationNetwork
from .helper import L2_loss, hinge_loss
from .typing import Batch, DiscriminatorOutputs, IDEmbedding, LossDict, TargetAttributes, Tensor, Tuple

CONFIG = configparser.ConfigParser()
CONFIG.read('config.ini')


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

	if not (checkpoint_path and os.path.exists(checkpoint_path)):
		checkpoint_path = None
	data_loader = DataLoader(dataset, batch_size = batch_size, shuffle = True, num_workers = num_workers, drop_last = True, pin_memory = True, persistent_workers = True)
	face_swap_model = FaceSwapper()
	trainer = create_trainer()
	trainer.fit(face_swap_model, data_loader, ckpt_path = checkpoint_path)


class FaceSwapper(pytorch_lightning.LightningModule):
	def __init__(self) -> None:
		super().__init__()
		self.generator = AdaptiveEmbeddingIntegrationNetwork(CONFIG.getint('training.generator', 'id_channels'), CONFIG.getint('training.generator', 'num_blocks'))
		self.discriminator = MultiscaleDiscriminator(CONFIG.getint('training.discriminator', 'input_channels'), CONFIG.getint('training.discriminator', 'num_filters'), CONFIG.getint('training.discriminator', 'num_layers'), CONFIG.getint('training.discriminator', 'num_discriminators'))
		self.arcface = torch.load(CONFIG.get('auxiliary_models.paths', 'arcface_path'), map_location = 'cpu', weights_only = False)
		self.landmarker = torch.load(CONFIG.get('auxiliary_models.paths', 'landmarker_path'), map_location = 'cpu', weights_only = False)
		self.motion_extractor = MotionExtractor(num_kp = 21, backbone = 'convnextv2_tiny')
		self.motion_extractor.load_state_dict(torch.load(CONFIG.get('auxiliary_models.paths', 'motion_extractor_path'), map_location = 'cpu', weights_only = True))
		self.arcface.eval()
		self.landmarker.eval()
		self.motion_extractor.eval()
		self.automatic_optimization = False
		self.batch_size = CONFIG.getint('training.loader', 'batch_size')

	def forward(self, target_tensor : Tensor, source_embedding : IDEmbedding) -> Tuple[Tensor, TargetAttributes]:
		output = self.generator(target_tensor, source_embedding)
		return output

	def configure_optimizers(self) -> Tuple[Optimizer, Optimizer]:
		generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr = CONFIG.getfloat('training.generator', 'learning_rate'), betas = (0.0, 0.999), weight_decay = 1e-4)
		discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr = CONFIG.getfloat('training.discriminator', 'learning_rate'), betas = (0.0, 0.999), weight_decay = 1e-4)
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

		if not CONFIG.getboolean('training.discriminator', 'disable'):
			discriminator_optimizer.step()

		if self.global_step % CONFIG.getint('training.output', 'preview_frequency') == 0:
			self.log_generator_preview(source_tensor, target_tensor, swap_tensor)

		if self.global_step % CONFIG.getint('training.output', 'validation_frequency') == 0:
			self.log_validation_preview()
		self.log('l_G', generator_losses.get('loss_generator'), prog_bar = True)
		self.log('l_D', discriminator_losses.get('loss_discriminator'), prog_bar = True)
		self.log('l_ADV', generator_losses.get('loss_adversarial'), prog_bar = False)
		self.log('l_ATTR', generator_losses.get('loss_attribute'), prog_bar = True)
		self.log('l_ID', generator_losses.get('loss_identity'), prog_bar=True)
		self.log('l_REC', generator_losses.get('loss_reconstruction'), prog_bar = True)
		return generator_losses.get('loss_generator')

	def calc_generator_loss(self, swap_tensor : Tensor, target_attributes : TargetAttributes, discriminator_outputs : DiscriminatorOutputs, batch : Batch) -> LossDict:
		source_tensor, target_tensor, is_same_person = batch
		generator_losses = {}
		# adversarial loss
		loss_adversarial = torch.Tensor(0)

		for discriminator_output in discriminator_outputs:
			loss_adversarial += hinge_loss(discriminator_output[0], True).mean(dim = [ 1, 2, 3 ])
		loss_adversarial = torch.mean(loss_adversarial)
		generator_losses['loss_adversarial'] = loss_adversarial
		generator_losses['loss_generator'] = loss_adversarial * CONFIG.getfloat('training.losses', 'weight_adversarial')

		# identity loss
		swap_embedding = self.get_id_embedding(swap_tensor, (30, 0, 10, 10))
		source_embedding = self.get_id_embedding(source_tensor, (30, 0, 10, 10))
		loss_identity = (1 - torch.cosine_similarity(source_embedding, swap_embedding, dim = 1)).mean()
		generator_losses['loss_identity'] = loss_identity
		generator_losses['loss_generator'] += loss_identity * CONFIG.getfloat('training.losses', 'weight_identity')

		# attribute loss
		loss_attribute = torch.Tensor(0)
		swap_attributes = self.generator.get_attributes(swap_tensor)

		for swap_attribute, target_attribute in zip(swap_attributes, target_attributes):
			loss_attribute += torch.mean(torch.pow(swap_attribute - target_attribute, 2).reshape(self.batch_size, -1), dim = 1).mean()
		loss_attribute *= 0.5
		generator_losses['loss_attribute'] = loss_attribute
		generator_losses['loss_generator'] += loss_attribute * CONFIG.getfloat('training.losses', 'weight_attribute')

		# reconstruction loss
		loss_reconstruction = torch.sum(0.5 * torch.mean(torch.pow(swap_tensor - target_tensor, 2).reshape(self.batch_size, -1), dim = 1) * is_same_person) / (is_same_person.sum() + 1e-4)
		loss_ssim = 1 - ssim(swap_tensor, target_tensor, data_range = float(torch.max(swap_tensor) - torch.min(swap_tensor))).mean()
		loss_reconstruction = (loss_reconstruction + loss_ssim) * 0.5
		generator_losses['loss_reconstruction'] = loss_reconstruction
		generator_losses['loss_generator'] += loss_reconstruction * CONFIG.getfloat('training.losses', 'weight_reconstruction')

		if CONFIG.getfloat('training.losses', 'weight_tsr') > 0:
			# tsr loss
			swap_motion_features = self.get_motion_features(swap_tensor)
			target_motion_features = self.get_motion_features(target_tensor)
			loss_tsr = torch.tensor(0).to(swap_tensor.device).to(swap_tensor.dtype)

			for swap_motion_feature, target_motion_feature in zip(swap_motion_features, target_motion_features):
				loss_tsr += L2_loss(swap_motion_feature, target_motion_feature)
			generator_losses['loss_tsr'] = loss_tsr
			generator_losses['loss_generator'] += loss_tsr * CONFIG.getfloat('training.losses', 'weight_tsr')

		if CONFIG.getfloat('training.losses', 'weight_eye_gaze') > 0:
			swap_landmark_features = self.get_landmark_features(swap_tensor)
			target_landmark_features = self.get_landmark_features(target_tensor)
			loss_left_eye_gaze = L2_loss(swap_landmark_features[0], target_landmark_features[1])
			loss_right_eye_gaze = L2_loss(swap_landmark_features[0], target_landmark_features[1])
			loss_eye_gaze = loss_left_eye_gaze + loss_right_eye_gaze
			generator_losses['loss_eye_gaze'] = loss_eye_gaze
			generator_losses['loss_generator'] += loss_eye_gaze * CONFIG.getfloat('training.losses', 'weight_eye_gaze')
		return generator_losses

	def calc_discriminator_loss(self, swap_tensor : Tensor, source_tensor : Tensor) -> LossDict:
		discriminator_losses = {}
		fake_discriminator_outputs = self.discriminator(swap_tensor.detach())
		loss_fake = torch.Tensor(0)

		for fake_discriminator_output in fake_discriminator_outputs:
			loss_fake += torch.mean(hinge_loss(fake_discriminator_output[0], False).mean(dim=[1, 2, 3]))
		true_discriminator_outputs = self.discriminator(source_tensor)
		loss_true = torch.Tensor(0)

		for true_discriminator_output in true_discriminator_outputs:
			loss_true += torch.mean(hinge_loss(true_discriminator_output[0], True).mean(dim=[1, 2, 3]))
		discriminator_losses['loss_discriminator'] = (loss_true.mean() + loss_fake.mean()) * 0.5
		return discriminator_losses

	def get_id_embedding(self, vision_tensor : Tensor, padding : Tuple[int, int, int, int]) -> Tensor:
		_, _, height, width = vision_tensor.shape
		crop_height = int(height * 0.0586)
		crop_width = int(width * 0.0586)
		crop_vision_tensor = vision_tensor[:, :, crop_height : height - crop_height, crop_width : width - crop_width]
		crop_vision_tensor = torch.nn.functional.interpolate(crop_vision_tensor, size = (112, 112), mode = 'bilinear')
		crop_vision_tensor[:, :, :padding[0], :] = 0
		crop_vision_tensor[:, :, 112 - padding[1]:, :] = 0
		crop_vision_tensor[:, :, :, :padding[2]] = 0
		crop_vision_tensor[:, :, :, 112 - padding[3]:] = 0
		embedding = self.arcface(crop_vision_tensor)
		embedding = torch.nn.functional.normalize(embedding, p = 2, dim = 1)
		return embedding

	def get_landmark_features(self, vision_tensor : Tensor) -> Tuple[Tensor, Tensor]:
		vision_tensor_norm = (vision_tensor + 1) * 0.5
		vision_tensor_norm = torch.nn.functional.interpolate(vision_tensor_norm, size = (224, 224), mode = 'bilinear')
		landmarks = self.landmarker(vision_tensor_norm)[2]
		landmarks = landmarks.view(-1, 203, 2) * 256
		return landmarks[:, 198], landmarks[:, 197]

	def get_motion_features(self, vision_tensor : Tensor) -> Tuple[Tensor, Tensor, Tensor]:
		vision_tensor_norm = (vision_tensor + 1) * 0.5
		motion_dict = self.motion_extractor(vision_tensor_norm)
		translation = motion_dict.get('t')
		scale = motion_dict.get('scale')
		rotation = torch.cat([ motion_dict.get('pitch'), motion_dict.get('yaw'), motion_dict.get('roll') ], dim = 1)
		return translation, scale, rotation

	def log_generator_preview(self, source_tensor : Tensor, target_tensor : Tensor, swap_tensor : Tensor) -> None:
		max_preview = 8
		source_tensor = source_tensor[:max_preview]
		target_tensor = target_tensor[:max_preview]
		swap_tensor = swap_tensor[:max_preview]
		rows = [torch.cat([src, tgt, swp], dim = 2) for src, tgt, swp in zip(source_tensor, target_tensor, swap_tensor)]
		grid = torchvision.utils.make_grid(torch.cat(rows, dim = 1).unsqueeze(0), nrow = 1, normalize = True, scale_each = True)
		self.logger.experiment.add_image("Generator Preview", grid, self.global_step)

	def log_validation_preview(self) -> None:
		read_images = lambda path :  [read_image(os.path.join(path, f)) for f in sorted(os.listdir(path)) if f.lower().endswith('.jpg') or f.lower().endswith('.png')]
		to_numpy = lambda x: (x.cpu().detach().numpy()[0].transpose(1, 2, 0).clip(-1, 1)[:, :, ::-1] + 1) * 127.5
		transforms = torchvision.transforms.Compose(
			[
				torchvision.transforms.Resize((256, 256), interpolation = torchvision.transforms.InterpolationMode.BICUBIC),
				torchvision.transforms.ToTensor(),
				torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
			])
		sources = read_images(CONFIG.get('training.validation', 'sources'))
		targets_front = read_images(CONFIG.get('training.validation', 'targets_front'))
		targets_side = read_images(CONFIG.get('training.validation', 'targets_side'))
		targets_makeup = read_images(CONFIG.get('training.validation', 'targets_makeup'))
		targets_occlusion = read_images(CONFIG.get('training.validation', 'targets_occlusion'))

		self.generator.eval()

		results_source = []
		results_front = []
		results_side = []
		results_makeup = []
		results_occlusion = []

		for source, target_front, target_side, target_makeup, target_occlusion in zip(sources, targets_front, targets_side, targets_makeup, targets_occlusion):
			source_tensor = transforms(source).unsqueeze(0).to(self.device).half()
			source_embedding = self.get_id_embedding(source_tensor, (0, 0, 0, 0))
			target_front_tensor = transforms(target_front).unsqueeze(0).to(self.device).half()
			target_side_tensor = transforms(target_side).unsqueeze(0).to(self.device).half()
			target_makeup_tensor = transforms(target_makeup).unsqueeze(0).to(self.device).half()
			target_occlusion_tensor = transforms(target_occlusion).unsqueeze(0).to(self.device).half()

			with torch.no_grad():
				output_front, _ = self.generator(target_front_tensor, source_embedding)
				output_side, _ = self.generator(target_side_tensor, source_embedding)
				output_makeup, _ = self.generator(target_makeup_tensor, source_embedding)
				output_occlusion, _ = self.generator(target_occlusion_tensor, source_embedding)

			results_source.append(to_numpy(source_tensor))
			results_front.append(numpy.hstack([to_numpy(target_front_tensor), to_numpy(output_front)]))
			results_side.append(numpy.hstack([to_numpy(target_side_tensor), to_numpy(output_side)]))
			results_makeup.append(numpy.hstack([to_numpy(target_makeup_tensor), to_numpy(output_makeup)]))
			results_occlusion.append(numpy.hstack([to_numpy(target_occlusion_tensor), to_numpy(output_occlusion)]))

		sources_vertical = numpy.vstack(results_source)
		results_front_vertical = numpy.vstack(results_front)
		results_side_vertical = numpy.vstack(results_side)
		results_makeup_vertical = numpy.vstack(results_makeup)
		results_occlusion_vertical = numpy.vstack(results_occlusion)
		pad = numpy.zeros((sources_vertical.shape[0], 10, 3), dtype = sources_vertical.dtype)
		preview = numpy.hstack([sources_vertical, pad, results_front_vertical, pad, results_side_vertical, pad, results_makeup_vertical, pad, results_occlusion_vertical])

		os.makedirs("validation_previews", exist_ok=True)
		cv2.imwrite(f"validation_previews/step_{self.global_step}.jpg", preview)
		self.generator.train()
