import imutils
import cv2

class Shape:
    def __init__(self, image) -> None:
        self.image = image

    def find_shape(self):
        resized = imutils.resize(self.image, width=300)
        ratio = self.image.shape[0] / float(resized.shape[0])
        # convert the resized image to grayscale, blur it slightly, and threshold it
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
        # find contours in the thresholded image and initialize the shape detector
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # cnts = imutils.grab_contours(cnts)
        # sd = ShapeDetector()

        return "circle"