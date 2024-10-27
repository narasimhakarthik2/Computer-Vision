import math
import time

import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
folder = "Data/3"
counter = 0
offset = 20

while True:
    success, image = cap.read()
    hands, image = detector.findHands(image)
    if hands:
        hand = hands[0]
        x,y,w,h = hand['bbox']

        imgWhite = np.ones((300,300,3),np.uint8)*255

        imgCrop = image[y-offset : y+h+offset, x-offset : x+w+offset]

        aspectRatio = h/w
        if aspectRatio > 1:
            Constant = 300/h
            width_cal = math.ceil(Constant*w)
            imgResize = cv2.resize(imgCrop,(width_cal, 300))
            width_gap = math.ceil((300-width_cal)/2)
            imgWhite[:, width_gap:width_cal+width_gap] = imgResize
        else:
            Constant = 300 / w
            height_cal = math.ceil(Constant * h)
            imgResize = cv2.resize(imgCrop, (height_cal, 300))
            height_gap = math.ceil((300 - height_cal) / 2)
            imgWhite[:, height_gap:height_cal + height_gap] = imgResize

        cv2.imshow("img white", imgWhite)

    cv2.imshow("Image", image)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgWhite)
        print(counter)
