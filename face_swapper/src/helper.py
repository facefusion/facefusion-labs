import numpy
import torch
from torch import Tensor, nn

from .types import EmbedderModule, Embedding, Padding, VisionFrame, VisionTensor


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


def hinge_real_loss(input_tensor : Tensor) -> Tensor:
	real_loss = torch.relu(1 - input_tensor)
	real_loss = real_loss.mean(dim = [ 1, 2, 3 ])
	return real_loss


def hinge_fake_loss(input_tensor : Tensor) -> Tensor:
	fake_loss = torch.relu(input_tensor + 1)
	fake_loss = fake_loss.mean(dim = [ 1, 2, 3 ])
	return fake_loss


def calc_embedding(embedder : EmbedderModule, input_tensor : Tensor, padding : Padding) -> Embedding:
	crop_tensor = input_tensor[:, :, 15: 241, 15: 241]
	crop_tensor = nn.functional.interpolate(crop_tensor, size = (112, 112), mode = 'area')
	crop_tensor[:, :, :padding[0], :] = 0
	crop_tensor[:, :, 112 - padding[1]:, :] = 0
	crop_tensor[:, :, :, :padding[2]] = 0
	crop_tensor[:, :, :, 112 - padding[3]:] = 0
	embedding = embedder(crop_tensor)
	embedding = nn.functional.normalize(embedding, p = 2)
	return embedding
