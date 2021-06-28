import cv2
import numpy as np
import cv2
import numpy as np
from typing import List  # use it for :List[...]


class Homograpy:
    matH = np.zeros((3, 3))
    widthOut: int
    heightOut: int
    cPoints: int
    aPoints: list = []

    def __init__(self, homography_file=None):
        self.cPoints = 0
        if homography_file is not None:
            self.read(homography_file)

    def read(self, homography_file):
        fileStorage = cv2.FileStorage(homography_file, cv2.FILE_STORAGE_READ)
        if not fileStorage.isOpened():
            return False

        self.cPoints = 0
        for i in range(points.size()):
            points = fileStorage.getNode("aPoints" + str(i))
            self.aPoints.append(points.mat())
            self.cPoints += 1
        self.matH = fileStorage.getNode("matH").mat()
        self.widthOut = int(fileStorage.getNode("widthOut").real())
        self.heightOut = int(fileStorage.getNode("heightOut").real())
        fileStorage.release()
        return True

    def write(self, homography_file):
        fileStorage = cv2.FileStorage(homography_file, cv2.FILE_STORAGE_WRITE)
        if not fileStorage.isOpened():
            return False

        for i in range(4):
            fileStorage.write("aPoints" + str(i), self.aPoints[i])

        fileStorage.write("matH", self.matH)
        fileStorage.write("widthOut", self.widthOut)
        fileStorage.write("heightOut", self.heightOut)
        fileStorage.release()
        return True


VIDEO_FILE = "robot.mp4"
HOMOGRAPHY_FILE = "robot-homography.yml"

matResult = None
matFinal = None
matPauseScreen = None

point = (-1, -1)
pts = []
var = 0
drag = 0

# Mouse handler function has 5 parameters input (no matter what)


def mouseHandler(event, x, y, flags, param):
    global point, pts, var, drag, matFinal, matResult

    if (var >= 4):                           # if homography points are more than 4 points, do nothing
        return
    if (event == cv2.EVENT_LBUTTONDOWN):     # When Press mouse left down
        drag = 1                             # Set it that the mouse is in pressing down mode
        matResult = matFinal.copy()          # copy final image to draw image
        # memorize current mouse position to point var
        point = (x, y)
        if (var >= 1):                       # if the point has been added more than 1 points, draw a line
            # draw a green line with thickness 2
            cv2.line(matResult, pts[var - 1], point, (0, 255, 0, 255), 2)
        cv2.circle(matResult, point, 2, (0, 255, 0), -1, 8,
                   0)             # draw a current green point
        cv2.imshow("Source", matResult)      # show the current drawing
    if (event == cv2.EVENT_LBUTTONUP and drag):  # When Press mouse left up
        drag = 0                             # no more mouse drag
        pts.append(point)                    # add the current point to pts
        var += 1                             # increase point number
        matFinal = matResult.copy()          # copy the current drawing image to final image
        # if the homograpy points are done
        if (var >= 4):
            # draw the last line
            cv2.line(matFinal, pts[0], pts[3], (0, 255, 0, 255), 2)
            # draw polygon from points
            cv2.fillConvexPoly(matFinal, np.array(
                pts, 'int32'), (0, 120, 0, 20))
        cv2.imshow("Source", matFinal)
    if (drag):                                    # if the mouse is dragging
        matResult = matFinal.copy()               # copy final images to draw image
        # memorize current mouse position to point var
        point = (x, y)
        if (var >= 1):                            # if the point has been added more than 1 points, draw a line
            # draw a green line with thickness 2
            cv2.line(matResult, pts[var - 1], point, (0, 255, 0, 255), 2)
        cv2.circle(matResult, point, 2, (0, 255, 0), -1, 8,
                   0)         # draw a current green point
        cv2.imshow("Source", matResult)           # show the current drawing


def main():
    global matFinal, matResult, matPauseScreen
    key = -1

    videoCapture = cv2.VideoCapture(VIDEO_FILE)
    if not videoCapture.isOpened():
        print("ERROR! Unable to open input video file ", VIDEO_FILE)
        return -1

    width = videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
    height = videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`

    # Capture loop
    while (key < 0):
        # Get the next frame
        _, matFrameCapture = videoCapture.read()
        if matFrameCapture is None:
            # End of video file
            break

        ratio = 640.0 / width
        dim = (int(width * ratio), int(height * ratio))
        matFrameDisplay = cv2.resize(matFrameDisplay, dim)

        cv2.imshow(VIDEO_FILE, matFrameDisplay)
        key = cv2.waitKey(30)

        if (key >= 0):
            cv2.destroyWindow(VIDEO_FILE)
            matPauseScreen = matFrameCapture
            matFinal = matPauseScreen.copy()
            cv2.namedWindow("Source", cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback("Source", mouseHandler)
            cv2.imshow("Source", matPauseScreen)
            cv2.waitKey(0)
            cv2.destroyWindow("Source")

            if (len(pts) < 4):
                return

            src = np.array(pts).astype(np.float32)
            reals = np.array([(800, 800),
                              (1000, 800),
                              (1000, 1000),
                              (800, 1000)], np.float32)

            homography_matrix = cv2.getPerspectiveTransform(src, reals)
            print("Estimated Homography Matrix is:")
            print(homography_matrix)

            h, w, ch = matPauseScreen.shape
            matResult = cv2.warpPerspective(
                matPauseScreen, homography_matrix, (w, h), cv2.INTER_LINEAR)
            matPauseScreen = cv2.resize(matPauseScreen, dim)
            cv2.imshow("Source", matPauseScreen)
            matResult = cv2.resize(matResult, dim)
            cv2.imshow("Result", matResult)

            cv2.waitKey(0)
        homographyData = Homograpy()
        homographyData.cPoints = 0
        for i in range(4):
            homographyData.aPoints.append(src[i])
            homographyData.cPoints += 1
        homographyData.matH = homography_matrix
        homographyData.widthOut = width
        homographyData.heightOut = height
        homographyData.write(HOMOGRAPHY_FILE)

        ...

        # Read H from file

        homographyData = Homograpy(HOMOGRAPHY_FILE)
        if homographyData.cPoints == 0:
            print("ERROR! Unable to read homography data file ", HOMOGRAPHY_FILE)
            return -1


main()
