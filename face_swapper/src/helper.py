from torch import Tensor, nn

from .types import EmbedderModule, Embedding, Padding


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
