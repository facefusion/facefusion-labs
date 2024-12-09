from typing import List

import numpy
import torch.nn as nn

from .sub_typing import Tensor


class NLayerDiscriminator(nn.Module):
    def __init__(self, input_channels : int, num_filters : int, num_layers : int) -> None:
        super(NLayerDiscriminator, self).__init__()
        self.num_layers = num_layers
        kernel_size = 4
        padding_size = int(numpy.ceil((kernel_size - 1.0) / 2))
        model_layers = [
			[
				nn.Conv2d(input_channels, num_filters, kernel_size = kernel_size, stride = 2, padding = padding_size),
			 	nn.LeakyReLU(0.2, True)
			]]
        current_filters = num_filters

        for layer_index in range(1, num_layers):
            previous_filters = current_filters
            current_filters = min(current_filters * 2, 512)
            model_layers += [
				[
                	nn.Conv2d(previous_filters, current_filters, kernel_size = kernel_size, stride = 2, padding = padding_size),
                	nn.InstanceNorm2d(current_filters), nn.LeakyReLU(0.2, True)
        		]]
        previous_filters = current_filters
        current_filters = min(current_filters * 2, 512)
        model_layers += [
			[
            	nn.Conv2d(previous_filters, current_filters, kernel_size = kernel_size, stride = 1, padding = padding_size),
            	nn.InstanceNorm2d(current_filters),
            	nn.LeakyReLU(0.2, True)
        	]]
        model_layers += [
			[
				nn.Conv2d(current_filters, 1, kernel_size = kernel_size, stride = 1, padding = padding_size)
			]]
        combined_layers = []

        for layer in model_layers:
            combined_layers += layer
        self.model = nn.Sequential(*combined_layers)

    def forward(self, input_tensor : Tensor) -> Tensor:
        return self.model(input_tensor)


# input_channels=3, num_filters=64, num_layers=5, num_discriminators=3
class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_channels : int, num_filters : int, num_layers : int, num_discriminators : int):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_discriminators = num_discriminators
        self.num_layers = num_layers

        for discriminator_index in range(num_discriminators):
            single_discriminator = NLayerDiscriminator(input_channels, num_filters, num_layers)
            setattr(self, 'discriminator_layer_{}'.format(discriminator_index), single_discriminator.model)
        self.downsample = nn.AvgPool2d(kernel_size = 3, stride = 2, padding = [ 1, 1 ], count_include_pad = False) # type:ignore[arg-type]

    def single_discriminator_forward(self, model_layers : nn.Sequential, input_tensor : Tensor) -> List[Tensor]:

        if self.return_intermediate_features:
            feature_maps = [ input_tensor ]

            for layer in model_layers:
                feature_maps.append(layer(feature_maps[-1]))
            return feature_maps[1:]
        else:
            return [ model_layers(input_tensor) ]

    def forward(self, input_tensor : Tensor) -> List[Tensor]:
        discriminator_outputs = []
        downsampled_input = input_tensor

        for discriminator_index in range(self.num_discriminators):
            model_layers = getattr(self, 'discriminator_layer_{}'.format(self.num_discriminators - 1 - discriminator_index))
            discriminator_outputs.append(self.single_discriminator_forward(model_layers, downsampled_input))

            if discriminator_index != (self.num_discriminators - 1):
                downsampled_input = self.downsample(downsampled_input)
        return discriminator_outputs
