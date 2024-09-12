#!/usr/bin/env python3
import numpy
import configparser
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.tuner.tuning import Tuner
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, random_split, TensorDataset
from model import ArcFaceConverter

CONFIG = configparser.ConfigParser()
CONFIG.read('config.ini')


class ArcFaceConverterTrainer(pl.LightningModule):
    def __init__(self):
        super(ArcFaceConverterTrainer, self).__init__()
        self.model = ArcFaceConverter()
        self.loss_fn = torch.nn.MSELoss()
        self.lr = 0.001

    def forward(self, input_embedding: torch.Tensor) -> torch.Tensor:
        return self.model(input_embedding)

    def training_step(self, batch, batch_idx):
        source, target = batch
        output = self(source)
        loss = self.loss_fn(output, target)
        self.log('train_loss', loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        source, target = batch
        output = self(source)
        loss = self.loss_fn(output, target)
        self.log('val_loss', loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.1, mode='min')
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'train_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }


def get_data_loader(batch_size, split_ratio = 0.8):
	source = torch.from_numpy(numpy.load(CONFIG['embeddings']['source_path'])).float()
	target = torch.from_numpy(numpy.load(CONFIG['embeddings']['target_path'])).float()
	dataset = TensorDataset(source, target)
	train_size = int(split_ratio * len(dataset))
	val_size = len(dataset) - train_size
	train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
	val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
	return train_loader, val_loader


def train(trainer, train_loader, val_loader) -> None:
	model = ArcFaceConverterTrainer()
	tuner = Tuner(trainer)
	tuner.lr_find(model, train_loader, val_loader)
	trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
	accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
	device = 1 if torch.cuda.is_available() else None
	batch_size = 50000
	train_loader, val_loader = get_data_loader(batch_size)

	checkpoint_callback = ModelCheckpoint(
		monitor="train_loss",
		dirpath=CONFIG['checkpoints']['save_path'],
		filename=CONFIG['checkpoints']['save_name'] + '-{epoch:02d}-{val_loss:.4f}',
		save_top_k=3,
		mode="min",
		every_n_epochs=10
	)

	trainer = pl.Trainer(
		max_epochs=1000,
		devices=device,
		accelerator=accelerator,
		callbacks=[checkpoint_callback],
		enable_progress_bar=True,
		log_every_n_steps=2
	)
	logger = TensorBoardLogger('logs/', name='arcface_converter')
	train(trainer, train_loader, val_loader)
