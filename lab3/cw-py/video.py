
import cv2
import numpy as np

VIDEO_FILE = "robot.mp4"
HOMOGRAPHY_FILE = "robot-homography.yml"

def displayFrame(matFrameDisplay, iFrame, cFrames, pHomographyData):
    for i in range(pHomographyData.cPoints):
        cv2.circle(matFrameDisplay, pHomographyData.aPoints[i], 10, (255, 0, 0), 2, cv.LINE_8, 0)

    cv2.imshow(VIDEO_FILE, matFrameDisplay)
    ss = "Frame " + str(iFrame) + "/" + str(cFrames)
    ss += ": hit <space> for next frame or 'q' to quit";
    # cv2.displayOverlay(VIDEO_FILE, ss, 0);  # for linux + qt
    cv2.putText(matFrameDisplay, ss, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3);

def main():
    global matFinal, matResult, matPauseScreen
    key = -1;

    videoCapture = cv2.VideoCapture(VIDEO_FILE)
    if not videoCapture.isOpened():
        print("ERROR! Unable to open input video file ", VIDEO_FILE)
        return -1

    width  = videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
    height = videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`

    cFrames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)

    cv2.namedWindow(VIDEO_FILE, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)

    homographyData = readHomography(HOMOGRAPHY_FILE)
    if homographyData is None:
        print("ERROR! Unable to read homography data file ", HOMOGRAPHY_FILE)
        return -1

    iFrame = 0
    # Capture loop 
    while (key < 0):
        # Get the next frame
        _, matFrameCapture = videoCapture.read()
        if matFrameCapture is None:
            # End of video file
            break

        matFrameCapture = videoCapture.read(matFrameCapture)
        if matFrameCapture is None:
            # End of video file
            break

        displayFrame(matFrameCapture, iFrame, cFrames, homographyData)

        iKey = -1
        while iKey != ord(' '):
            iKey = cv2.waitKey(10)
            if (cv2.getWindowProperty(VIDEO_FILE, cv2.WND_PROP_VISIBLE) == 0):
                return 0
            if (iKey == ord('q') or iKey == ord('Q')):
                return 0

        iFrame += 1

        return

main()