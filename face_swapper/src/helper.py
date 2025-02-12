import platform

import cv2
import numpy
import torch

from .types import Embedder, Embedding, Padding, Tensor, VisionFrame, VisionTensor


def is_windows() -> bool:
	return platform.system().lower() == 'windows'


def read_image(image_path : str) -> VisionFrame:
	if is_windows():
		image_buffer = numpy.fromfile(image_path, dtype = numpy.uint8)
		return cv2.imdecode(image_buffer, cv2.IMREAD_COLOR)
	return cv2.imread(image_path)


def convert_to_vision_tensor(vision_frame : VisionFrame) -> VisionTensor:
	vision_tensor = torch.from_numpy(vision_frame[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32))
	vision_tensor = vision_tensor / 255.0
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
	real_loss = torch.relu(1 - tensor)
	real_loss = real_loss.mean(dim = [ 1, 2, 3 ])
	return real_loss


def hinge_fake_loss(tensor : Tensor) -> Tensor:
	fake_loss = torch.relu(tensor + 1)
	fake_loss = fake_loss.mean(dim = [ 1, 2, 3 ])
	return fake_loss


def calc_id_embedding(id_embedder : Embedder, vision_tensor : VisionTensor, padding : Padding) -> Embedding:
	crop_vision_tensor = vision_tensor[:, :, 15 : 241, 15 : 241]
	crop_vision_tensor = torch.nn.functional.interpolate(crop_vision_tensor, size = (112, 112), mode = 'area')
	crop_vision_tensor[:, :, :padding[0], :] = 0
	crop_vision_tensor[:, :, 112 - padding[1]:, :] = 0
	crop_vision_tensor[:, :, :, :padding[2]] = 0
	crop_vision_tensor[:, :, :, 112 - padding[3]:] = 0
	source_embedding = id_embedder(crop_vision_tensor)
	source_embedding = torch.nn.functional.normalize(source_embedding, p = 2)
	return source_embedding
