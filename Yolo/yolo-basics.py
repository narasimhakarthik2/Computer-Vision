import cv2
from PIL.ImageOps import scale
from ultralytics import YOLO
import cvzone
import math

classNames = ['Person']


#YOLO with images
# model = YOLO('yolo11n.pt')
# results = model('images/2.jpg', show=True)
# cv2.waitKey(0)

#YOLO with webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

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

            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0,x1) ,max(40,y1)), scale=0.7, thickness=1)


    cv2.imshow("Image", img)
    cv2.waitKey(1)