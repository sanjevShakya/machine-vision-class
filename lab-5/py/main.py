import cv2
import numpy as np
import os
import glob

cap = cv2.VideoCapture()
cap.open(0, cv2.CAP_ANY)
if not cap.isOpened():
    print("ERROR! Unable to open camera\n")
    exit()
print("Start grabbing")
print("Press s to save images and q to terminate")
frameAdd = 0
while True:
    _, frame = cap.read()
    if frame is None:
        print("ERROR! blank frame grabbed\n")
        exit()
    cv2.imshow("Live", frame)
    iKey = cv2.waitKey(5)
    if iKey == ord('s') or iKey == ord('S'):
        cv2.imwrite("./images/frame" + str(frameAdd) + ".jpg", frame)
        frameAdd += 1
        print("Frame: ", frameAdd, " has been saved.")
    elif iKey == ord('q') or iKey == ord('Q'):
        break
