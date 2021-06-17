import numpy as np
import cv2 as cv
import argparse

parser = argparse.ArgumentParser(description="Sample Demo Lucas-Kanade")
parser.add_argument('--video', type=str, help="path to video file")
parser.add_argument('--useCamera', type=bool,
                    help="check this flag to capture from inbuilt camera")

args = parser.parse_args()
cap = None

if(args.useCamera):
    cap = cv.VideoCapture(0)
else:
    cap = cv.VideoCapture(args.video)

feature_params = dict(maxCorners=100,
                      qualityLevel=0.01,
                      minDistance=7,
                      blockSize=7)

lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(
    cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

color = np.random.randint(0, 255, (100, 3))

ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

mask = np.zeros_like(old_frame)

while(1):
    ret, frame = cap.read()
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    p1, st, err = cv.calcOpticalFlowPyrLK(
        old_gray, frame_gray, p0, None, **lk_params)
    if p1 is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]
    else:
        old_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        continue;

    for i,(new,old) in enumerate(zip(good_new, good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        # mask = cv.line(mask, (int(a),int(b)),(int(c),int(d)), color[i].tolist(), 2)
        frame = cv.circle(frame,(int(a),int(b)),5,color[i].tolist(),-1)
    img = frame

    cv.imshow('frame',img)
    k = cv.waitKey(30) & 0xff
    print('key', k)
    while(k != 32):
        break;
    if k == 27:
        break
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)