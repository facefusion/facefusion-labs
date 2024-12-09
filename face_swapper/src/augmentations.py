import torch
from torch import Tensor


def apply_random_motion_blur(tensor_image : Tensor) -> Tensor:
	kernel_size = 9
	kernel = torch.zeros((kernel_size, kernel_size), dtype=torch.float32)
	random_angle = torch.empty(1).uniform_(-2 * torch.pi, 2 * torch.pi)
	dx = torch.cos(random_angle)
	dy = torch.sin(random_angle)
	center = kernel_size // 2

	for i in range(kernel_size):
		x = int(center + (i - center) * dx)
		y = int(center + (i - center) * dy)
		if 0 <= x < kernel_size and 0 <= y < kernel_size:
			kernel[y, x] = 1
	kernel /= kernel.sum()
	kernel = kernel.unsqueeze(0).unsqueeze(0)
	blurred_channels = []

	for channel in tensor_image:
		channel = channel.unsqueeze(0).unsqueeze(0)
		channel = torch.nn.functional.conv2d(channel, kernel, padding=kernel_size // 2)
		channel = channel.squeeze(0).squeeze(0)
		blurred_channels.append(channel)
	return torch.stack(blurred_channels)
