import cv2

class Color:
    def __init__(self, image) -> None:
        self.image = image
        pass

    def find_color(self):
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

        # Generate lower mask (0-10) and upper mask (170-180) of red
        red_mask1 = cv2.inRange(hsv, (0,50,20), (10,255,255))
        red_mask2 = cv2.inRange(hsv, (170,50,20), (180,255,255))

        # Merge the mask and crop the red regions
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        red = cv2.bitwise_and(hsv, hsv, mask=red_mask)
        red = cv2.cvtColor(red, cv2.COLOR_HSV2BGR)

        cv2.imshow('Red', red)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Generate mask (90-130) of blue
        blue_mask = cv2.inRange(hsv, (90,30,50), (130,255,255))

        blue = cv2.bitwise_and(hsv, hsv, mask=blue_mask)
        blue = cv2.cvtColor(blue, cv2.COLOR_HSV2BGR)

        cv2.imshow('Blue', blue)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        red = cv2.bitwise_and(self.image, self.image, mask=red_mask)
        blue = cv2.bitwise_and(self.image, self.image, mask=blue_mask)

        mask = cv2.bitwise_or(red, blue)

        print(mask)
        # TODO: Perceber porquê de só dar sem o segundo image
        result = cv2.bitwise_and(self.image, mask)

        cv2.imshow('Result', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return "gray"