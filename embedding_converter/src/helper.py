import platform

import cv2
import numpy

from .types import VisionFrame


def is_windows() -> bool:
	return platform.system().lower() == 'windows'


def read_image(image_path : str) -> VisionFrame:
	if is_windows():
		image_buffer = numpy.fromfile(image_path, dtype = numpy.uint8)
		return cv2.imdecode(image_buffer, cv2.IMREAD_COLOR)
	return cv2.imread(image_path)
