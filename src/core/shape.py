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
            
            # TODO: specific case of road102.png (blue sky)
            # img_h, img_w, _ = self.image.shape
            # if img_w != w or img_h != h: # if the rectangle consists of the whole image
            #     shape = "rectangle"

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
        red_detection = self.find_image_shape(self.red)
        blue_detection = self.find_image_shape(self.blue)
        final_detection = cv2.bitwise_or(red_detection, blue_detection)

        return final_detection

    def find_image_shape(self, image):        
        # resized = imutils.resize(image, width=300)
        # image = imutils.resize(image, width=300)
        #self.red = imutils.resize(self.red, width=300)
        #self.blue = imutils.resize(self.blue, width=300)
        # ratio = image.shape[0] / float(resized.shape[0])

        # convert the resized image to grayscale, blur it slightly, and threshold it
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # TODO: era resized
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # blurred = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)[1]

        #cv2.imshow('no imgLCwithCLAHE', gray)

        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        # imgLCwithCLAHE = clahe.apply(gray)

        #cv2.imshow('imgLCwithCLAHE', imgLCwithCLAHE)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        # imgLCwithCLAHE = cv2.cvtColor(imgLCwithCLAHE, cv2.COLOR_GRAY2BGR)

        # image = imgLCwithCLAHE

        # cv2.imshow('Antes', image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # num_labels, labels_im = cv2.connectedComponents(gray)

        # def imshow_components(labels):
        #     # Map component labels to hue val
        #     label_hue = np.uint8(179*labels/np.max(labels))
        #     blank_ch = 255*np.ones_like(label_hue)
        #     labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

        #     # cvt to BGR for display
        #     labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

        #     # set bg label to black
        #     labeled_img[label_hue==0] = 0

        #     cv2.imshow('labeled.png', labeled_img)
        #     cv2.waitKey()
        #     cv2.destroyAllWindows()

        # imshow_components(labels_im)

        # find contours in the thresholded image and initialize the shape detector
        cnts = cv2.findContours(blurred.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        # Draw circles
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT_ALT, 2, 30, param1=100, param2=0.8, minRadius=1)
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

        # cv2.imshow('detected circles', image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # loop over the contours
        for c in cnts:
            AREA = 2000
            # AREA = img.shape[0]*img.shape[1]/20
            if cv2.contourArea(c) < AREA:
                continue
            # compute the center of the contour, then detect the name of the shape using only the contour
            M = cv2.moments(c)
            # print(M)
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
            # min_enclosing_circle_center_x, min_enclosing_circle_center_y = (int(x),int(y))
            # min_enclosing_circle_radius = int(contour_radius)

            # for (c1, c2) in processed_centers.keys():
                # if (c1 - min_enclosing_circle_center_x)**2 + (c2 - min_enclosing_circle_center_y)**2 < min_enclosing_circle_radius**2: # check if center of circle is inside the contour

            contains_circle = False
            for (c1, c2) in list(processed_centers.keys()):
                circle_radius = processed_centers[(c1, c2)]
                dist = cv2.pointPolygonTest(c, (c1,c2), True)
                if dist > 0: # center of a circle is inside the contour
                    del processed_centers[(c1, c2)]
                    if circle_radius >= 2/3 * contour_radius:  # the radius of the contour and the circle are approximately the same
                        contains_circle = True
            
            if not contains_circle:
                cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
                cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        for (i[0],i[1]) in list(processed_centers.keys()):
            # draw the outer circle
            cv2.circle(image,(i[0],i[1]),processed_centers[i[0],i[1]],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(image,(i[0],i[1]),2,(0,0,255),3)
            # write the shape
            cv2.putText(image, "circle", (i[0],i[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow('Result', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return image
