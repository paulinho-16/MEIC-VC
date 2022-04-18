import argparse
import numpy as np
import cv2
import os
import imutils

from src.utils import Parser
from src.core import Data, ColorDetector, ShapeDetector

def print_contour(t, c, colour):
    print(f'TYPE {t}, colour {colour}')

if __name__ == "__main__":
    parser = Parser()

    image_data = Data(os.path.join("./data", parser.filename))
    
    color_detector = ColorDetector(image_data.image)
    image_color, red_result, blue_result, color_result = color_detector.find_color()

    shape_detector = ShapeDetector(red_result, blue_result, color_result)
    contours = shape_detector.find_shape()

    processed_contours = [] # list of tuples (t, c, colour)
    
    contains_contour = False

    print(f'LEN CONTOURS:::: {len(contours)}')

    for t, c, colour in contours:
        print_contour(t, c, colour)

    print('---------------------')

    for t, c, colour in contours:
        # print('AQUII')
        # print_contour(t, c, colour)
        if t == "circle":
            center = (c['center'][0], c['center'][1])
        else:
            M = cv2.moments(c)
            cX = int((M["m10"] / (M["m00"] + 1e-7))) #* ratio)
            cY = int((M["m01"] / (M["m00"] + 1e-7))) #* ratio)
            center = (cX, cY)

        contour_radius = c['radius'] if t == "circle" else cv2.minEnclosingCircle(c)[1]
        if not processed_contours:
            processed_contours.append((t, c, colour))
            continue

        invalid_contours = []
        append_contour = False
        processed_verified = 0
        
        print(f'LEN PROCESSED:::: {len(processed_contours)}')

        for (tp, cp, colourp) in processed_contours:
            print('PROCESSED')
            print_contour(tp, cp, colourp)

            circle_inside = False
            dist = 0
            processed_contour_radius = cp['radius'] if tp == "circle" else cv2.minEnclosingCircle(cp)[1]

            if tp == "circle":
                print(f'processed_contour_radius: {processed_contour_radius}')
                print(f'center do processed: {int(cp["center"][0]), int(cp["center"][1])}')
                print(f'center do contorno: {int(center[0]), int(center[1])}')
                print(f'quadrado 1: {(int(cp["center"][0]) - int(center[0]))**2}')
                print(f'quadrado 2: {(int(cp["center"][1]) - int(center[1]))**2}')
                print(f'quadrado 3: {int(processed_contour_radius)**2}')
                if (int(cp['center'][0]) - int(center[0]))**2 + (int(cp['center'][1]) - int(center[1]))**2 < processed_contour_radius**2: # check if center of circle is inside the contour
                    circle_inside = True
            else:
                dist = cv2.pointPolygonTest(cp, center, True)
            if dist > 0 or circle_inside: # center of the contour is inside a processed contour
                print(f'PRIMEIRA CONDIÇAO {(contour_radius >= 2/3 * processed_contour_radius and ((colourp == "red" and t == "stop") or (colour == "red" and tp != "stop")))}')
                print(f'SEGUNDA CONDIÇAO {(contour_radius >= processed_contour_radius and colour != "blue")}')
                if (contour_radius >= 2/3 * processed_contour_radius and colour == "red" and not (t == "stop" or tp == "stop")) or (contour_radius >= processed_contour_radius and colour != "blue"): # the radius of the contour is bigger than the radius of the processed contour
                    invalid_contours.append((tp, cp, colourp))
                    append_contour = True
            else:
                processed_verified += 1


        for i in invalid_contours:
            processed_contours.remove(i)
        if append_contour or (processed_verified == len(processed_contours)):
            processed_contours.append((t, c, colour))

    stop_count = 0
    for t, c, colour in processed_contours:
        if t == "circle":
            cv2.circle(image_data.image,(c['center']),c['radius'],(0,255,0),2) # draw the outer circle
            cv2.circle(image_data.image,(c['center']),2,(0,0,255),3) # draw the center of the circle
            cv2.putText(image_data.image, f"{colour} circle", (c['center'][0] - 35, c['center'][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2) # write the shape
        else:
            if t == "stop": stop_count += 1
            M = cv2.moments(c)
            cX = int((M["m10"] / (M["m00"] + 1e-7))) #* ratio)
            cY = int((M["m01"] / (M["m00"] + 1e-7))) #* ratio)
            cv2.drawContours(image_data.image, [c], -1, (0, 255, 0), 2)
            classification = f"{colour} {t}" if t != "stop" else "stop sign"
            cv2.putText(image_data.image, classification, (cX - 35, cY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    if stop_count == 0:
        # OpenCV opens images as BRG 
        # but we want it as RGB We'll 
        # also need a grayscale version
        img_gray = cv2.cvtColor(image_data.image, cv2.COLOR_BGR2GRAY)
        
        # Use minSize because for not 
        # bothering with extra-small 
        # dots that would look like STOP signs
        stop_data = cv2.CascadeClassifier(os.path.join("./src/core", 'stop_data.xml'))
        
        found = stop_data.detectMultiScale(img_gray, minSize =(20, 20), minNeighbors=8)
        
        # Don't do anything if there's 
        # no sign
        amount_found = len(found)
        
        if amount_found != 0:
            
            # There may be more than one
            # sign in the image
            for (x, y, width, height) in found:
                
                # We draw a green rectangle around
                # every recognized sign
                cv2.rectangle(image_data.image, (x, y), 
                            (x + height, y + width), 
                            (0, 255, 0), 5)
                classification = "stop sign"
                cv2.putText(image_data.image, classification, (x + int(height/2), y + int(width/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
        # Creates the environment of 
        # the picture and shows it
        cv2.imshow('STOPS', image_data.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # final_result = cv2.bitwise_or(image_data.image, image_shape)

    # Show the output image
    cv2.imshow('Final Result', image_data.image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()