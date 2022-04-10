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

        #cv2.imshow('no imgLCwithCLAHE', gray)

        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        # imgLCwithCLAHE = clahe.apply(gray)

        #cv2.imshow('imgLCwithCLAHE', imgLCwithCLAHE)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        # imgLCwithCLAHE = cv2.cvtColor(imgLCwithCLAHE, cv2.COLOR_GRAY2BGR)

        # self.image = imgLCwithCLAHE

        # cv2.imshow('Antes', self.image)
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

    def connected_component_label(self, path):
        # Getting the input image
        img = cv2.imread(path, 0)
        # Converting those pixels with values 1-127 to 0 and others to 1
        img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]
        # Applying cv2.connectedComponents() 
        num_labels, labels = cv2.connectedComponents(img)
        
        # Map component labels to hue val, 0-179 is the hue range in OpenCV
        label_hue = np.uint8(179*labels/np.max(labels))
        blank_ch = 255*np.ones_like(label_hue)
        labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

        # Converting cvt to BGR
        labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

        # set bg label to black
        labeled_img[label_hue==0] = 0
        
        
        # Showing Original Image
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title("Orginal Image")
        plt.show()
        
        #Showing Image after Component Labeling
        plt.imshow(cv2.cvtColor(labeled_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title("Image after Component Labeling")
        plt.show()

