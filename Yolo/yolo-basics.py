import cv2
from ultralytics import YOLO
import cvzone
import math
import torch

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
              "teddy bear", "hair drier", "toothbrush"
              ]

#YOLO with images
# model = YOLO('yolo11n.pt')
# results = model('images/2.jpg', show=True)
# cv2.waitKey(0)

#YOLO with webcam
# cap = cv2.VideoCapture(0)
# cap.set(3, 1280)
# cap.set(4, 720)

cap = cv2.VideoCapture("Videos/bikes.mp4")


model = YOLO('yolo11n.pt')

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            #Bounding Box
            x1,y1,x2,y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255), 3)

            w, h = x2-x1, y2-y1
            bbox = int(x1), int(y1), int(w), int(h)
            cvzone.cornerRect(img, bbox)

            #Confidence
            conf = math.ceil(box.conf[0]*100)/100

            #Class Name
            cls = int(box.cls[0])

            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(40, y1)), scale=1, thickness=1)


    cv2.imshow("Image", img)
    cv2.waitKey(1)