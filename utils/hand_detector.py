'''
detecting hand
'''
# from cvzone.HandTrackingModule import HandDetector
import cv2
import numpy as np
import os

class hand_detector():
    def __init__(self, con=0.8):
        self.detector = HandDetector(detectionCon=con, maxHands=2)
        self.hand1 = []
        self.hand2 = []

    def detect(self, img):
        hands = self.detector.findHands(img, draw=False)
        output = []

        if hands:
            self.hand1 = hands[0]
            bbox1 = self.hand1["bbox"] #x,y,w,h
            center1 = self.hand1["center"] # cx, cy
            handType1 = self.hand1["type"] #"Left", "Right"
            output = [[bbox1, center1, handType1]]


            if len(hands) == 2:
                self.hand2 = hands[1]
                bbox2 = self.hand2["bbox"]
                center2 = self.hand2["center"]
                handType2 = self.hand2["type"]
                output.append([bbox2, center2, handType2])

        return output



if __name__ == "__main__":
    file_dir = 'D:/datasets/RHD_v1-1/RHD_published_v2/training/color'
    filenames = os.listdir(file_dir)
    hd = hand_detector()

    for i in range(len(filenames)):
        path = os.path.join(file_dir, filenames[i])
        img = cv2.imread(path)
        li = hd.detect(img)
        print(np.array(li).shape)

