#!/usr/bin/env python3
from pytorch_lightning.loggers import TensorBoardLogger

from src.training import create_loaders, create_trainer, train

if __name__ == '__main__':
	trainer = create_trainer()
	training_loader, validation_loader = create_loaders()
	logger = TensorBoardLogger('.logs', name = 'arcface_converter')
	train(trainer, training_loader, validation_loader)
