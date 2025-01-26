import cv2
import numpy
import torch

from .typing import IdEmbedding, Padding, Tensor, VisionFrame, VisionTensor


def read_image(image_path : str) -> VisionFrame:
	image = cv2.imread(image_path)
	return image


def convert_to_vision_tensor(vision_frame : VisionFrame) -> VisionTensor:
	vision_tensor = torch.from_numpy(vision_frame[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32))
	vision_tensor = vision_tensor / 255
	vision_tensor = (vision_tensor - 0.5) * 2
	vision_tensor = vision_tensor.unsqueeze(0)
	return vision_tensor


def convert_to_vision_frame(vision_tensor : VisionTensor) -> VisionFrame:
	vision_frame = vision_tensor.detach().cpu().numpy()[0]
	vision_frame = vision_frame.transpose(1, 2, 0)
	vision_frame = (vision_frame + 1) * 127.5
	vision_frame = vision_frame.clip(0, 255).astype(numpy.uint8)
	vision_frame = vision_frame[:, :, ::-1]
	return vision_frame


def hinge_real_loss(tensor : Tensor) -> Tensor:
	return torch.relu(1 - tensor)


def hinge_fake_loss(tensor : Tensor) -> Tensor:
	return torch.relu(tensor + 1)


def calc_id_embedding(id_embedder : torch.nn.Module, vision_tensor : VisionTensor, padding : Padding) -> IdEmbedding:
	crop_vision_tensor = vision_tensor[:, :, 15 : 241, 15 : 241]
	crop_vision_tensor = torch.nn.functional.interpolate(crop_vision_tensor, size = (112, 112), mode = 'area')
	crop_vision_tensor[:, :, :padding[0], :] = 0
	crop_vision_tensor[:, :, 112 - padding[1]:, :] = 0
	crop_vision_tensor[:, :, :, :padding[2]] = 0
	crop_vision_tensor[:, :, :, 112 - padding[3]:] = 0
	source_embedding = id_embedder(crop_vision_tensor)
	source_embedding = torch.nn.functional.normalize(source_embedding, p = 2, dim = 1)
	return source_embedding


def infer(generator : torch.nn.Module, id_embedder : torch.nn.Module, source_vision_frame : VisionFrame, target_vision_frame : VisionFrame) -> VisionFrame:
	source_vision_tensor = convert_to_vision_tensor(source_vision_frame)
	target_vision_tensor = convert_to_vision_tensor(target_vision_frame)
	source_embedding = calc_id_embedding(id_embedder, source_vision_tensor, (0, 0, 0, 0))
	output_vision_tensor = generator(source_embedding, target_vision_tensor)[0]
	output_vision_frame = convert_to_vision_frame(output_vision_tensor)
	return output_vision_frame
