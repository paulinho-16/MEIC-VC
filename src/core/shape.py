import imutils
import cv2

class ShapeDetector:
    def __init__(self, red, blue, image) -> None:
        self.red = red
        self.blue = blue
        self.image = image

    def detect(self, contour):
            # initialize the shape name and approximate the contour
            shape = "unidentified"
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.001 * peri, True)
            # if the shape is a triangle, it will have 3 vertices
            if len(approx) == 3:
                shape = "triangle"
            # if the shape has 4 vertices, it is either a square or a rectangle
            elif len(approx) == 4:
                # compute the bounding box of the contour and use the bounding box to compute the aspect ratio
                (x, y, w, h) = cv2.boundingRect(approx)
                ar = w / float(h)
                # a square will have an aspect ratio that is approximately equal to one, otherwise, the shape is a rectangle
                shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
            # if the shape is a pentagon, it will have 5 vertices
            elif len(approx) == 5:
                shape = "pentagon"
            # otherwise, we assume the shape is a circle
            else:
                shape = "circle"
            # return the name of the shape
            return shape

    def find_shape(self):
        self.image = self.red
        
        resized = imutils.resize(self.image, width=300)
        ratio = self.image.shape[0] / float(resized.shape[0])

        # convert the resized image to grayscale, blur it slightly, and threshold it
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # blurred = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)[1]

        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        # imgLCwithCLAHE = clahe.apply(gray)

        # imgLCwithCLAHE = cv2.cvtColor(imgLCwithCLAHE, cv2.COLOR_GRAY2BGR)

        # self.image = imgLCwithCLAHE

        # cv2.imshow('Antes', self.image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # find contours in the thresholded image and initialize the shape detector
        cnts = cv2.findContours(blurred.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        # loop over the contours
        for c in cnts:
            AREA = 2000
            # AREA = img.shape[0]*img.shape[1]/20
            if cv2.contourArea(c) < AREA:
                continue
            # compute the center of the contour, then detect the name of the shape using only the contour
            M = cv2.moments(c)
            # print(M)
            cX = int((M["m10"] / (M["m00"] + 1e-7)) * ratio)
            cY = int((M["m01"] / (M["m00"] + 1e-7)) * ratio)
            shape = self.detect(c)
            # multiply the contour (x, y)-coordinates by the resize ratio, then draw the contours and the name of the shape on the image
            c = c.astype("float")
            c *= ratio
            c = c.astype("int")
            cv2.drawContours(self.image, [c], -1, (0, 255, 0), 2)
            cv2.putText(self.image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow('Result', self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return "circle"