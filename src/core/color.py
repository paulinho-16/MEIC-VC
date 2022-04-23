import cv2
import numpy as np

DEBUG = False

class ColorDetector:
    def __init__(self, image) -> None:
        self.image = image
        pass

    def find_color(self):
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

        # split HSV
        h,s,v = cv2.split(hsv)

        saturation = hsv[:, :, 1].mean()
        print(f' SATTT MEDIA: {saturation}')

        # Increasing Contrast with CLAHE in saturation and value
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        swithCLAHE = clahe.apply(s)
        vwithCLAHE = clahe.apply(v)

        hsv = cv2.merge([h, swithCLAHE, vwithCLAHE])

        # Generate lower mask (0-10) and upper mask (170-180) of red
        if saturation < 61:
            red_mask1 = cv2.inRange(hsv, (0,90,50), (10,255,255)) # deteta tudo no road56.png
            red_mask2 = cv2.inRange(hsv, (170,90,50), (180,255,255)) # deteta tudo no road56.png
        else:
            red_mask1 = cv2.inRange(hsv, (0,65,30), (20,255,255)) # deteta tudo no road57.png
            red_mask2 = cv2.inRange(hsv, (160,65,30), (180,255,255)) # deteta tudo no road57.png
            red_mask1 = cv2.inRange(hsv, (0,30,30), (20,255,255)) # deteta tudo no road66.png
            red_mask2 = cv2.inRange(hsv, (160,30,30), (180,255,255)) # deteta tudo no road66.png

        # Merge the mask and crop the red regions
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        # Generate mask (100-140) of blue
        if saturation < 40:
            blue_mask = cv2.inRange(hsv, (100,130,50), (140,255,255))
        elif saturation < 120:
            blue_mask = cv2.inRange(hsv, (100,150,50), (140,255,255))
        else:
            blue_mask = cv2.inRange(hsv, (100,250,50), (140,255,255))

        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
        # red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_DILATE, np.ones((3,3),np.uint8))

        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
        # blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_DILATE, np.ones((3,3),np.uint8))

        red = cv2.bitwise_and(self.image, self.image, mask=red_mask)
        blue = cv2.bitwise_and(self.image, self.image, mask=blue_mask)

        # Show red tracing
        if DEBUG:
            cv2.imshow('ANTES DO SEGUNDO THRESHOLD', blue)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        red_hslimage  = cv2.cvtColor(red, cv2.COLOR_BGR2HLS) # TODO: tirar isto
        red_Lchannel = red_hslimage[:,:,1]
        red_lvalue =cv2.mean(red_Lchannel)[0]
        print(f' VALORRRR DO RED: {red_lvalue}')

        blue_hslimage  = cv2.cvtColor(blue, cv2.COLOR_BGR2HLS)
        blue_Lchannel = blue_hslimage[:,:,1]
        blue_lvalue =cv2.mean(blue_Lchannel)[0]
        print(f' VALORRRR DO BLUE: {blue_lvalue}')

        red_hsv = cv2.cvtColor(red, cv2.COLOR_BGR2HSV)
        average_hsv_1 = cv2.mean(red_hsv, red_mask1)[:3]
        average_hsv_2 = cv2.mean(red_hsv, red_mask2)[:3]
        print(f' AVERAGGEEEEEE 11111111: {average_hsv_1}')
        print(f' AVERAGGEEEEEE 22222222: {average_hsv_2}')
        average_hsv_red = cv2.mean(red_hsv,  red_mask)[:3]
        print(f' AVERAGGEEEEEE: {average_hsv_red}')

        blue_hsv = cv2.cvtColor(blue, cv2.COLOR_BGR2HSV)
        average_hsv_blue = cv2.mean(blue_hsv,  blue_mask)[:3]

        min_value_saturation_red = red_hsv[np.where(red_hsv[:,:,1]>0)][:,1].min() if red_hsv.any() else 100
        min_value_saturation_blue = blue_hsv[np.where(blue_hsv[:,:,1]>0)][:,1].min() if blue_hsv.any() else 100

        print(f' MINIMUMMM RED: {min_value_saturation_red}')
        red_threshold = (average_hsv_red[1] + min_value_saturation_red) / 2
        print(f' THRESHOLD RED A PARTIR DE: {red_threshold}')

        print(f' MINIMUMMM BLUE: {min_value_saturation_blue}')
        blue_threshold = (average_hsv_blue[1] + min_value_saturation_blue) / 2
        print(f' THRESHOLD BLUE A PARTIR DE: {blue_threshold}')

        # Show red tracing
        if DEBUG:
            show_hsv = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            cv2.imshow('EFEITO DO CLAHE', show_hsv)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        max_red1 = 10 if average_hsv_1[0] <= 15 else 20
        min_red2 = 170 if average_hsv_1[0] > 175 else 160 # TODO: devia ser average_hsv_2, mudar para verificar se não estraga

        red_mask1 = cv2.inRange(hsv, (0, red_threshold, 50), (max_red1, 255, 255))
        red_mask2 = cv2.inRange(hsv, (min_red2, red_threshold, 50), (180, 255, 255))
        
        blue_mask = cv2.inRange(hsv, (90, blue_threshold, 50), (130, 255, 255))

        # Merge the mask and crop the red regions
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))

        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))

        red = cv2.bitwise_and(self.image, self.image, mask=red_mask)
        blue = cv2.bitwise_and(self.image, self.image, mask=blue_mask)

        # Show red tracing
        if DEBUG:
            cv2.imshow('Red Color', red)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Show blue tracing
        if DEBUG:
            cv2.imshow('Blue Color', blue)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        mask = cv2.bitwise_or(red, blue)

        result = cv2.bitwise_and(self.image, mask)

        # Show blue and red tracing
        if DEBUG:
            cv2.imshow('Red Color Detection', red)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return ("gray", red, blue, result)