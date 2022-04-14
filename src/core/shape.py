import imutils
import cv2
import numpy as np

class ShapeDetector:
    def __init__(self, red, blue, image) -> None:
        self.red = red
        self.blue = blue
        self.image = image

    def detect(self, contour):
        # initialize the shape name and approximate the contour
        shape = "unidentified"
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.01 * peri, True)
        # if the shape is a triangle, it will have 3 vertices
        if len(approx) == 3:
            shape = "triangle"
        # if the shape has 4 vertices, it is either a square or a rectangle
        elif len(approx) == 4:
            # compute the bounding box of the contour and use the bounding box to compute the aspect ratio
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
            # a square will have an aspect ratio that is approximately equal to one, otherwise, the shape is a rectangle
            if ar >= 0.95 and ar <= 1.05:
                shape = "square"

        # if the shape is a pentagon, it will have 5 vertices
        elif len(approx) == 5:
            shape = "pentagon"
        elif len(approx) == 8:
            shape = "stop"
        # otherwise, we assume the shape is a circle
        # else:
        #     shape = "circle"
            
        # return the name of the shape
        return shape

    def find_shape(self):
        red_contours = self.find_image_shape(self.red, "red")
        blue_contours = self.find_image_shape(self.blue, "blue")

        return red_contours + blue_contours

    def find_image_shape(self, image, colour):
        # resized = imutils.resize(image, width=300)
        # image = imutils.resize(image, width=300)
        #self.red = imutils.resize(self.red, width=300)
        #self.blue = imutils.resize(self.blue, width=300)
        # ratio = image.shape[0] / float(resized.shape[0])

        # convert the resized image to grayscale, blur it slightly, and threshold it
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # TODO: era resized
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # blurred = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)[1]

        final_contours = []

        # find contours in the thresholded image and initialize the shape detector
        cnts = cv2.findContours(blurred.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        # Draw circles
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT_ALT, 2, 30, param1=250, param2=0.8, minRadius=1) # param1 estava a 100, mudamos por causa do road53.png
        if circles is None: circles = [[]] 
        circles = np.uint16(np.around(circles))

        processed_centers = {}

        for i in circles[0,:]:
            if (i[0],i[1]) in processed_centers.keys():
                if processed_centers[(i[0],i[1])] < i[2]:
                    processed_centers[(i[0],i[1])] = i[2]
                continue
            else:
                processed_centers[(i[0], i[1])] = i[2]

        # loop over the contours
        for c in cnts:
            AREA = 2000
            # AREA = img.shape[0]*img.shape[1]/20
            if cv2.contourArea(c) < AREA:
                continue
            # compute the center of the contour, then detect the name of the shape using only the contour
            M = cv2.moments(c)
            cX = int((M["m10"] / (M["m00"] + 1e-7))) #* ratio)
            cY = int((M["m01"] / (M["m00"] + 1e-7))) #* ratio)
            shape = self.detect(c)

            if shape == "unidentified":
                continue
            
            # multiply the contour (x, y)-coordinates by the resize ratio, then draw the contours and the name of the shape on the image
            c = c.astype("float")
            # c *= ratio
            c = c.astype("int")

            (x,y), contour_radius = cv2.minEnclosingCircle(c)
            
            # if not contains_circle:
            if not (shape == "stop" and colour == "blue"): # TODO: If the building is red still detects a stop sign - road78.png
                final_contours.append((shape, c, colour))
                cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
                classification = f"{colour} {shape}" if shape != "stop" else "stop sign"
                cv2.putText(image, classification, (cX - 35, cY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        for (i[0],i[1]) in list(processed_centers.keys()):
            circle = {'center': (i[0],i[1]), 'radius': (processed_centers[i[0],i[1]])}
            final_contours.append(("circle", circle, colour))
            # draw the outer circle
            cv2.circle(image,(i[0],i[1]),processed_centers[i[0],i[1]],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(image,(i[0],i[1]),2,(0,0,255),3)
            # write the shape
            cv2.putText(image, f"{colour} circle", (i[0] - 35, i[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow('Result', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return final_contours
