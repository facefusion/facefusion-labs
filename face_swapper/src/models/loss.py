import configparser
from typing import Tuple

import torch
from pytorch_msssim import ssim
from torch import Tensor

from face_swapper.src.helper import calc_id_embedding, hinge_fake_loss, hinge_real_loss
from face_swapper.src.types import Batch, DiscriminatorLossSet, DiscriminatorOutputs, FaceLandmark203, GeneratorLossSet, LossTensor, SwapAttributes, TargetAttributes, VisionTensor

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
		generator_loss_set =\
		{
			'loss_adversarial': self.calc_adversarial_loss(discriminator_outputs),
			'loss_id': self.calc_id_loss(source_tensor, swap_tensor),
			'loss_attribute': self.calc_attribute_loss(target_attributes, swap_attributes),
			'loss_reconstruction': self.calc_reconstruction_loss(swap_tensor, target_tensor, is_same_person)
		}

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
