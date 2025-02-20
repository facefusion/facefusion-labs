import cv2

from .types import VisionFrame


def read_image(image_path : str) -> VisionFrame:
	return cv2.imread(image_path)
