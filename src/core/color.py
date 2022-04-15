import cv2
import numpy as np

class ColorDetector:
    def __init__(self, image) -> None:
        self.image = image
        pass

    def find_color(self):
        hslimage  = cv2.cvtColor(self.image, cv2.COLOR_BGR2HLS)
        Lchannel = hslimage[:,:,1]
        lvalue =cv2.mean(Lchannel)[0]
        print(f' VALORRRR: {lvalue}')

        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

        # split HSV
        h,s,v = cv2.split(hsv)

        if lvalue < 125:
            # Increasing Contrast with CLAHE in saturation and value
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            swithCLAHE = clahe.apply(s)
            vwithCLAHE = clahe.apply(v)

            hsv = cv2.merge([h, swithCLAHE, vwithCLAHE])
            # Generate lower mask (0-10) and upper mask (170-180) of red
            red_mask1 = cv2.inRange(hsv, (0,100,30), (20,255,255))
            red_mask2 = cv2.inRange(hsv, (160,100,30), (180,255,255))
            # AQUIIIIIIIIIII APAGAR EM BAIXO
            red_mask1 = cv2.inRange(hsv, (0,70,30), (10,255,255))
            red_mask2 = cv2.inRange(hsv, (170,70,30), (180,255,255))
        elif lvalue >= 125 and lvalue <= 170: # era 130
            # Increasing Contrast with CLAHE in saturation and value
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            swithCLAHE = clahe.apply(s)
            vwithCLAHE = clahe.apply(v)

            hsv = cv2.merge([h, swithCLAHE, vwithCLAHE])
            # Generate lower mask (0-10) and upper mask (170-180) of red
            red_mask1 = cv2.inRange(hsv, (0,100,50), (20,255,255))
            red_mask2 = cv2.inRange(hsv, (160,100,50), (180,255,255))
        else:
            # Increasing Contrast with CLAHE in saturation and value
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            swithCLAHE = clahe.apply(s)
            vwithCLAHE = clahe.apply(v)

            hsv = cv2.merge([h, swithCLAHE, vwithCLAHE])
            red_mask1 = cv2.inRange(hsv, (0,104,100), (20,255,255)) # melhores com clahe de 8
            red_mask2 = cv2.inRange(hsv, (160,104,100), (180,255,255))

        # red_mask1 = cv2.inRange(hsv, (0,50,20), (10,255,255))
        # red_mask2 = cv2.inRange(hsv, (170,50,20), (180,255,255))

        # Merge the mask and crop the red regions
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        red_show = cv2.bitwise_and(hsv, hsv, mask=red_mask)
        red_show = cv2.cvtColor(red_show, cv2.COLOR_HSV2BGR)

        # red_show = cv2.fastNlMeansDenoisingColored(red_show,None,10,10,7,21)

        # Show red tracing
        # cv2.imshow('Red', red_show)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # Generate mask (90-130) of blue
        blue_mask = cv2.inRange(hsv, (90,30,50), (130,255,255)) # valores originais
        # blue_mask = cv2.inRange(hsv, (80,30,20), (150,255,255))

        blue_show = cv2.bitwise_and(hsv, hsv, mask=blue_mask)
        blue_show = cv2.cvtColor(blue_show, cv2.COLOR_HSV2BGR)

        # Show blue tracing
        # cv2.imshow('Blue', blue_show)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
        # red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_DILATE, np.ones((3,3),np.uint8))

        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
        # blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_GRADIENT, np.ones((3,3),np.uint8))

        red = cv2.bitwise_and(self.image, self.image, mask=red_mask)
        blue = cv2.bitwise_and(self.image, self.image, mask=blue_mask)

        mask = cv2.bitwise_or(red, blue)

        print(mask)
        # TODO: Perceber porquê de só dar sem o segundo image
        result = cv2.bitwise_and(self.image, mask)

        # Show blue and red tracing
        # cv2.imshow('Result', result)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return ("gray", red, blue, result)