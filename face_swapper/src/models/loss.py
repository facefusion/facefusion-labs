import configparser
from typing import List, Tuple

import torch
from pytorch_msssim import ssim
from sqlalchemy.dialects.mssql.information_schema import identity_columns
from torch import Tensor, nn

from ..helper import calc_embedding, hinge_fake_loss, hinge_real_loss
from ..types import Attributes, Batch, DiscriminatorLossSet, DiscriminatorOutputs, FaceLandmark203, GeneratorLossSet, LossTensor, SwapAttributes, TargetAttributes, VisionTensor

CONFIG = configparser.ConfigParser()
CONFIG.read('config.ini')


class FaceSwapperLoss:
	def __init__(self) -> None:
		embedder_path = CONFIG.get('training.model', 'embedder_path')
		landmarker_path = CONFIG.get('training.model', 'landmarker_path')
		motion_extractor_path = CONFIG.get('training.model', 'motion_extractor_path')
		self.batch_size = CONFIG.getint('training.loader', 'batch_size')
		self.mse_loss = nn.MSELoss()
		self.embedder = torch.jit.load(embedder_path, map_location = 'cpu') # type:ignore[no-untyped-call]
		self.landmarker = torch.jit.load(landmarker_path, map_location = 'cpu') # type:ignore[no-untyped-call]
		self.motion_extractor = torch.jit.load(motion_extractor_path, map_location = 'cpu') # type:ignore[no-untyped-call]
		self.embedder.eval()
		self.landmarker.eval()
		self.motion_extractor.eval()

	def calc_generator_loss(self, swap_tensor : VisionTensor, target_attributes : TargetAttributes, swap_attributes : SwapAttributes, discriminator_outputs : DiscriminatorOutputs, batch : Batch) -> GeneratorLossSet:
		weight_adversarial = CONFIG.getfloat('training.losses', 'weight_adversarial')
		weight_identity = CONFIG.getfloat('training.losses', 'weight_identity')
		weight_attribute = CONFIG.getfloat('training.losses', 'weight_attribute')
		weight_reconstruction = CONFIG.getfloat('training.losses', 'weight_reconstruction')
		weight_pose = CONFIG.getfloat('training.losses', 'weight_pose')
		weight_gaze = CONFIG.getfloat('training.losses', 'weight_gaze')
		source_tensor, target_tensor = batch
		is_same_person = torch.tensor(0) if torch.equal(source_tensor, target_tensor) else torch.tensor(1)
		generator_loss_set =\
		{
			'loss_adversarial': self.calc_adversarial_loss(discriminator_outputs),
			'loss_identity': self.calc_identity_loss(source_tensor, swap_tensor),
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
		generator_loss_set['loss_generator'] += generator_loss_set.get('loss_identity') * weight_identity
		generator_loss_set['loss_generator'] += generator_loss_set.get('loss_attribute') * weight_attribute
		generator_loss_set['loss_generator'] += generator_loss_set.get('loss_reconstruction') * weight_reconstruction
		generator_loss_set['loss_generator'] += generator_loss_set.get('loss_pose') * weight_pose
		generator_loss_set['loss_generator'] += generator_loss_set.get('loss_gaze') * weight_gaze
		return generator_loss_set

	def calc_discriminator_loss(self, real_discriminator_outputs : DiscriminatorOutputs, fake_discriminator_outputs : DiscriminatorOutputs) -> DiscriminatorLossSet:
		discriminator_loss_set = {}
		loss_fakes = []

		for fake_discriminator_output in fake_discriminator_outputs:
			loss_fakes.append(hinge_fake_loss(fake_discriminator_output[0]))

		loss_trues = []

		for true_discriminator_output in real_discriminator_outputs:
			loss_trues.append(hinge_real_loss(true_discriminator_output[0]))

		loss_fake = torch.stack(loss_fakes).mean()
		loss_true = torch.stack(loss_trues).mean()
		discriminator_loss_set['loss_discriminator'] = (loss_true + loss_fake) * 0.5
		return discriminator_loss_set

	def calc_adversarial_loss(self, discriminator_outputs : DiscriminatorOutputs) -> LossTensor:
		loss_adversarials = []

		for discriminator_output in discriminator_outputs:
			loss_adversarials.append(hinge_real_loss(discriminator_output[0]).mean())

		loss_adversarial = torch.stack(loss_adversarials).mean()
		return loss_adversarial

	def calc_attribute_loss(self, target_attributes : TargetAttributes, swap_attributes : SwapAttributes) -> LossTensor:
		loss_attributes = []

		for swap_attribute, target_attribute in zip(swap_attributes, target_attributes):
			loss_attributes.append(torch.mean(torch.pow(swap_attribute - target_attribute, 2).reshape(self.batch_size, -1), dim = 1).mean())

		loss_attribute = torch.stack(loss_attributes).mean() * 0.5
		return loss_attribute

	def calc_reconstruction_loss(self, swap_tensor : VisionTensor, target_tensor : VisionTensor, is_same_person : Tensor) -> LossTensor:
		loss_reconstruction = torch.pow(swap_tensor - target_tensor, 2).reshape(self.batch_size, -1)
		loss_reconstruction = torch.mean(loss_reconstruction, dim = 1) * 0.5
		loss_reconstruction = torch.sum(loss_reconstruction * is_same_person) / (is_same_person.sum() + 1e-4)
		loss_ssim = 1 - ssim(swap_tensor, target_tensor, data_range = float(torch.max(swap_tensor) - torch.min(swap_tensor))).mean()
		loss_reconstruction = (loss_reconstruction + loss_ssim) * 0.5
		return loss_reconstruction

	def calc_identity_loss(self, source_tensor : VisionTensor, swap_tensor : VisionTensor) -> LossTensor:
		swap_embedding = calc_embedding(self.embedder, swap_tensor, (30, 0, 10, 10))
		source_embedding = calc_embedding(self.embedder, source_tensor, (30, 0, 10, 10))
		loss_identity = (1 - torch.cosine_similarity(source_embedding, swap_embedding)).mean()
		return loss_identity

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
		vision_tensor_norm = nn.functional.interpolate(vision_tensor_norm, size = (224, 224), mode = 'bilinear')
		landmarks = self.landmarker(vision_tensor_norm)[2].view(-1, 203, 2)
		return landmarks

	def get_pose_features(self, vision_tensor : Tensor) -> Tuple[Tensor, Tensor, Tensor]:
		vision_tensor_norm = (vision_tensor + 1) * 0.5
		pitch, yaw, roll, translation, expression, scale, _ = self.motion_extractor(vision_tensor_norm)
		rotation = torch.cat([ pitch, yaw, roll ], dim = 1)
		return translation, scale, rotation


class AdversarialLoss(torch.nn.Module):
	def __init__(self) -> None:
		super(AdversarialLoss, self).__init__()

	def calc(self, discriminator_output_tensors : List[Tensor]) -> Tuple[Tensor, Tensor]:
		adversarial_weight = CONFIG.getfloat('training.losses', 'adversarial_weight')
		temp_tensors = []

		for discriminator_output_tensor in discriminator_output_tensors:
			temp_tensor = torch.relu(1 - discriminator_output_tensor[0]).mean(dim = [ 1, 2, 3 ]).mean()
			temp_tensors.append(temp_tensor)

		adversarial_loss = torch.stack(temp_tensors).mean()
		weighted_adversarial_loss = adversarial_loss * adversarial_weight
		return adversarial_loss, weighted_adversarial_loss


class AttributeLoss(torch.nn.Module):
	def __init__(self) -> None:
		super(AttributeLoss, self).__init__()

	def calc(self, target_attributes : Attributes, output_attributes : Attributes) -> Tuple[Tensor, Tensor]:
		batch_size = CONFIG.getint('training.loader', 'batch_size')
		attribute_weight = CONFIG.getfloat('training.losses', 'attribute_weight')
		temp_tensors = []

		for target_attribute, output_attribute in zip(target_attributes, output_attributes):
			temp_tensor = torch.mean(torch.pow(output_attribute - target_attribute, 2).reshape(batch_size, -1), dim = 1).mean()
			temp_tensors.append(temp_tensor)

		attribute_loss = torch.stack(temp_tensors).mean() * 0.5
		weighted_attribute_loss = attribute_loss * attribute_weight
		return attribute_loss, weighted_attribute_loss


class ReconstructionLoss(torch.nn.Module):
	def __init__(self) -> None:
		super(ReconstructionLoss, self).__init__()

	def calc(self, source_tensor : Tensor, target_tensor : Tensor, output_tensor : Tensor) -> Tuple[Tensor, Tensor]:
		batch_size = CONFIG.getint('training.loader', 'batch_size')
		reconstruction_weight = CONFIG.getfloat('training.losses', 'reconstruction_weight')
		reconstruction_loss = torch.pow(output_tensor - target_tensor, 2).reshape(batch_size, -1)
		reconstruction_loss = torch.mean(reconstruction_loss, dim = 1) * 0.5

		if torch.equal(source_tensor, target_tensor):
			reconstruction_loss = torch.sum(reconstruction_loss * torch.tensor(0)) / (torch.tensor(0).sum() + 1e-4)
		else:
			reconstruction_loss = torch.sum(reconstruction_loss * torch.tensor(1)) / (torch.tensor(1).sum() + 1e-4)

		data_range = float(torch.max(output_tensor) - torch.min(output_tensor))
		similarity = 1 - ssim(output_tensor, target_tensor, data_range = data_range).mean()

		reconstruction_loss = (reconstruction_loss + similarity) * 0.5
		weighted_reconstruction_loss = reconstruction_loss * reconstruction_weight
		return reconstruction_loss, weighted_reconstruction_loss


class IdentityLoss(torch.nn.Module):
	def __init__(self) -> None:
		super(IdentityLoss, self).__init__()
		embedder_path = CONFIG.get('training.model', 'embedder_path')
		self.embedder = torch.jit.load(embedder_path, map_location = 'cpu') # type:ignore[no-untyped-call]
		self.embedder.eval()

	def calc(self, source_tensor : Tensor, output_tensor : Tensor) -> Tuple[Tensor, Tensor]:
		identity_weight = CONFIG.getfloat('training.losses', 'identity_weight')
		output_embedding = calc_embedding(self.embedder, output_tensor, (30, 0, 10, 10))
		source_embedding = calc_embedding(self.embedder, source_tensor, (30, 0, 10, 10))
		identity_loss = (1 - torch.cosine_similarity(source_embedding, output_embedding)).mean()
		weighted_identity_loss = identity_loss * identity_weight
		return identity_loss, weighted_identity_loss
