import cv2
import cvzone
import pickle
import numpy as np
from networkx.algorithms.bipartite import color

width, height= 105, 43

with open('CarParkPos', 'rb') as f:
    posList = pickle.load(f)

#video feed
cap = cv2.VideoCapture('Dataset/carPark.mp4')

def checkParkingSpace(imgPro):

    freeSpaceCounter = 0

    for pos in posList:
        x,y = pos
        imgCrop = imgPro[y:y+height, x:x+width]
        # cv2.imshow(str(x*y), imgCrop)
        count = cv2.countNonZero(imgCrop)

        if count < 850:
            color = (0,255,0)
            freeSpaceCounter += 1
        else:
            color = (0,0,255)

        cv2.rectangle(img, pos, (pos[0]+width,pos[1]+height), color, 2)
        cvzone.putTextRect(img, str(count), (x,y+height-3),scale=1,thickness=2, offset=0, colorR=color)


    cvzone.putTextRect(img, f'Free: {freeSpaceCounter}/{len(posList)}', (100,50), scale=3, thickness=3, offset=10, colorR=(0,200,0))


while True:
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    success, img = cap.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (3,3),1)
    imgThreshold = cv2.adaptiveThreshold(imgBlur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25,16)

    imgMedian = cv2.medianBlur(imgThreshold, 5)
    kernal = np.ones((3,3),np.uint8)
    imgDilate = cv2.dilate(imgMedian, kernal, iterations=1)

    checkParkingSpace(imgDilate)

    cv2.imshow("Image", img)
    # cv2.imshow("ImageBlur", imgBlur)
    # cv2.imshow("ImageThreshold", imgThreshold)
    # cv2.imshow("imgDilate", imgDilate)


    cv2.waitKey(10)