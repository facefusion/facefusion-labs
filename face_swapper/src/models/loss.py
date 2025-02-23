import configparser
from typing import List, Tuple

import torch
from pytorch_msssim import ssim
from torch import Tensor, nn

from ..helper import calc_embedding
from ..types import Attributes, FaceLandmark203

CONFIG = configparser.ConfigParser()
CONFIG.read('config.ini')


class DiscriminatorLoss(nn.Module):
	def __init__(self) -> None:
		super().__init__()

	def calc(self, discriminator_source_tensors : List[Tensor], discriminator_output_tensors : List[Tensor]) -> Tensor:
		temp1_tensors = []
		temp2_tensors = []

		for discriminator_output_tensor in discriminator_output_tensors:
			temp1_tensor = torch.relu(discriminator_output_tensor[0] + 1).mean(dim = [ 1, 2, 3 ])
			temp1_tensors.append(temp1_tensor)

		for discriminator_source_tensor in discriminator_source_tensors:
			temp2_tensor = torch.relu(1 - discriminator_source_tensor[0]).mean(dim = [ 1, 2, 3 ])
			temp2_tensors.append(temp2_tensor)

		discriminator1_loss = torch.stack(temp1_tensors).mean()
		discriminator2_loss = torch.stack(temp2_tensors).mean()
		discriminator_loss = (discriminator1_loss + discriminator2_loss) * 0.5
		return discriminator_loss


class AdversarialLoss(nn.Module):
	def __init__(self) -> None:
		super().__init__()

	def calc(self, discriminator_output_tensors : List[Tensor]) -> Tuple[Tensor, Tensor]:
		adversarial_weight = CONFIG.getfloat('training.losses', 'adversarial_weight')
		temp_tensors = []

		for discriminator_output_tensor in discriminator_output_tensors:
			temp_tensor = torch.relu(1 - discriminator_output_tensor[0]).mean(dim = [ 1, 2, 3 ]).mean()
			temp_tensors.append(temp_tensor)

		adversarial_loss = torch.stack(temp_tensors).mean()
		weighted_adversarial_loss = adversarial_loss * adversarial_weight
		return adversarial_loss, weighted_adversarial_loss


class AttributeLoss(nn.Module):
	def __init__(self) -> None:
		super().__init__()

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


class ReconstructionLoss(nn.Module):
	def __init__(self) -> None:
		super().__init__()
		self.mse_loss = nn.MSELoss()

	def calc(self, source_tensor : Tensor, target_tensor : Tensor, output_tensor : Tensor) -> Tuple[Tensor, Tensor]:
		reconstruction_weight = CONFIG.getfloat('training.losses', 'reconstruction_weight')
		temp_tensors = []

		for _source_tensor, _target_tensor in zip(source_tensor, target_tensor):
			temp_tensor = self.mse_loss(_source_tensor, _target_tensor)

			if torch.equal(_source_tensor, _target_tensor):
				temp_tensors.append(temp_tensor)
			else:
				temp_tensors.append(temp_tensor * 0)
		reconstruction_loss = torch.stack(temp_tensors).mean() * 0.5
		data_range = float(torch.max(output_tensor) - torch.min(output_tensor))
		similarity = 1 - ssim(output_tensor, target_tensor, data_range = data_range).mean()
		reconstruction_loss = (reconstruction_loss + similarity) * 0.5
		weighted_reconstruction_loss = reconstruction_loss * reconstruction_weight
		return reconstruction_loss, weighted_reconstruction_loss


class IdentityLoss(nn.Module):
	def __init__(self) -> None:
		super().__init__()
		embedder_path = CONFIG.get('training.model', 'embedder_path')
		self.embedder = torch.jit.load(embedder_path, map_location = 'cpu') # type:ignore[no-untyped-call]

	def calc(self, source_tensor : Tensor, output_tensor : Tensor) -> Tuple[Tensor, Tensor]:
		identity_weight = CONFIG.getfloat('training.losses', 'identity_weight')
		output_embedding = calc_embedding(self.embedder, output_tensor, (30, 0, 10, 10))
		source_embedding = calc_embedding(self.embedder, source_tensor, (30, 0, 10, 10))
		identity_loss = (1 - torch.cosine_similarity(source_embedding, output_embedding)).mean()
		weighted_identity_loss = identity_loss * identity_weight
		return identity_loss, weighted_identity_loss


class PoseLoss(nn.Module):
	def __init__(self) -> None:
		super().__init__()
		motion_extractor_path = CONFIG.get('training.model', 'motion_extractor_path')
		self.motion_extractor = torch.jit.load(motion_extractor_path, map_location = 'cpu') # type:ignore[no-untyped-call]
		self.mse_loss = nn.MSELoss()

	def calc(self, target_tensor : Tensor, output_tensor : Tensor, ) -> Tuple[Tensor, Tensor]:
		pose_weight = CONFIG.getfloat('training.losses', 'pose_weight')
		output_motion_features = self.get_motion_features(output_tensor)
		target_motion_features = self.get_motion_features(target_tensor)
		temp_tensors = []

		for target_motion_feature, output_motion_feature in zip(target_motion_features, output_motion_features):
			temp_tensor = self.mse_loss(target_motion_feature, output_motion_feature)
			temp_tensors.append(temp_tensor)

		pose_loss = torch.stack(temp_tensors).mean()
		weighted_pose_loss = pose_loss * pose_weight
		return pose_loss, weighted_pose_loss

	def get_motion_features(self, input_tensor : Tensor) -> Tuple[Tensor, Tensor, Tensor]:
		input_tensor = (input_tensor + 1) * 0.5
		pitch, yaw, roll, translation, expression, scale, _ = self.motion_extractor(input_tensor)
		rotation = torch.cat([ pitch, yaw, roll ], dim = 1)
		return translation, scale, rotation


class GazeLoss(nn.Module):
	def __init__(self) -> None:
		super().__init__()
		landmarker_path = CONFIG.get('training.model', 'landmarker_path')
		self.landmarker = torch.jit.load(landmarker_path, map_location = 'cpu') # type:ignore[no-untyped-call]
		self.mse_loss = nn.MSELoss()

	def calc(self, target_tensor : Tensor, output_tensor : Tensor, ) -> Tuple[Tensor, Tensor]:
		gaze_weight = CONFIG.getfloat('training.losses', 'gaze_weight')
		output_face_landmark = self.detect_face_landmark(output_tensor)
		target_face_landmark = self.detect_face_landmark(target_tensor)

		left_gaze_loss = self.mse_loss(output_face_landmark[:, 198], target_face_landmark[:, 198])
		right_gaze_loss = self.mse_loss(output_face_landmark[:, 197], target_face_landmark[:, 197])

		gaze_loss = left_gaze_loss + right_gaze_loss
		weighted_gaze_loss = gaze_loss * gaze_weight
		return gaze_loss, weighted_gaze_loss

	def detect_face_landmark(self, input_tensor : Tensor) -> FaceLandmark203:
		input_tensor = (input_tensor + 1) * 0.5
		input_tensor = nn.functional.interpolate(input_tensor, size = (224, 224), mode = 'bilinear')
		face_landmarks_203 = self.landmarker(input_tensor)[2].view(-1, 203, 2)
		return face_landmarks_203
