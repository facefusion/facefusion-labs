import pytorch_lightning


class ArcFaceConverter(pytorch_lightning.LightningModule):
	def __init__(self) -> None:
		super(ArcFaceConverter, self).__init__()

