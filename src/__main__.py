import argparse
import numpy as np
import cv2
import os

from src.utils import Parser
from src.core import Data, Color, Shape

if __name__ == "__main__":
    parser = Parser()

    image_data = Data(os.path.join("./data", parser.filename))
    
    color_detector = Color(image_data.image)
    image_color = color_detector.find_color()

    shape_detector = Shape(image_data.image)
    image_shape = shape_detector.find_shape()

    # Show the output image
    cv2.imshow('Final Result', image_data.image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()