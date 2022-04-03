import cv2

class Data:
    def __init__(self, filename) -> None:
        self.filename = filename
        self.image = cv2.imread(filename)
