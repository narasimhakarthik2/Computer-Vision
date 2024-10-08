import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
import math
import torch
from sort import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]


cap = cv2.VideoCapture("Videos/cars.mp4")
mask = cv2.imread("Images/mask.png")

tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

model = YOLO('yolo11n.pt')

while True:
    success, img = cap.read()
    imgRegion = cv2.bitwise_and(img, mask)

    results = model(imgRegion, stream=True)

    detections = np.empty((0,5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            #Bounding Box
            x1,y1,x2,y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255), 3)

            w, h = x2-x1, y2-y1
            bbox = int(x1), int(y1), int(w), int(h)

            #Confidence
            conf = math.ceil(box.conf[0]*100)/100

            #Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            if currentClass == "car" and conf > 0.3:
                cvzone.cornerRect(img, bbox, l=9, rt=5)
                # cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(40, y1)), scale=1, thickness=1, offset=3)
                currentArray = np.array([x1,y1,x2,y2,conf])
                detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)

    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),
                           scale=2, thickness=3, offset=10)

    tracker.update(detections)
    cv2.imshow("Image", img)
    cv2.waitKey(1)