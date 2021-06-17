import cv2
import numpy as np
import sys
import argparse


parser = argparse.ArgumentParser(description="Sample Demo Lucas-Kanade")
parser.add_argument('--video', type=str, help="path to video file")
parser.add_argument('--useCamera', type=bool,
                    help="check this flag to capture from inbuilt camera")

args = parser.parse_args()
VIDEO_FILE = args.video
ROTATE = False


class HomographyFileUtil:
    output_file = "homography.yaml"
    file_storage = None

    def __init__(self):
        return None

    def open_read_mode(self):
        self.file_storage = cv2.FileStorage(
            self.output_file, cv2.FILE_STORAGE_READ)

    def open_write_mode(self):

        self.file_storage = cv2.FileStorage(
            self.output_file, cv2.FILE_STORAGE_WRITE)

    def write_matrix(self, row, key='R_MAT'):
        self.file_storage.write(key, np.array(row))

    def read_matrix(self, key='R_MAT'):
        return self.file_storage.getNode(key).mat()

    def close_file(self):
        self.file_storage.release()
        return True


class HomographyDTO:
    def __init__(self, point=(-1, -1), matFinal=None, matResult=None, var=0, drag=0, pts=[]):
        self.point = point
        self.pts = pts
        self.var = var
        self.drag = drag
        self.matFinal = matFinal
        self.matResult = matResult


class Homography:
    def __init__(self, src, dst):
        self.src = src
        self.dst = dst
        self.compute_coefficents()
        self.compute_homography_matrix()

    def get_homography_matrix(self):
        return self.H

    def compute_coefficents(self):
        # print(X[:,0])
        coeffs = []
        X = self.src
        Xprime = self.dst
        for i in range(X.shape[0]):
            x = X[i]
            x_prime = Xprime[i]
            upper_coeff = np.array(
                [-x[0], -x[1], -1, 0, 0, 0, x_prime[0] * x[0], x_prime[0] * x[1], x_prime[0]])
            lower_coeff = np.array(
                [0, 0, 0, -x[0], -x[1], -1, x_prime[1] * x[0], x_prime[1] * x[1], x_prime[1]])
            coeffs.append(upper_coeff)
            coeffs.append(lower_coeff)
        self.coeffs = np.array(coeffs)

    def compute_homography_matrix(self):
        U, s, VT = np.linalg.svd(self.coeffs)
        self.H = VT[-1].reshape(3, 3)

    def wrap_perspective(self, points):
        wrapped_pts = self.H @ points.T
        return wrapped_pts / wrapped_pts[-1:, :]  # convert to homogeneous pts


def mouseHandler(event, x, y, flags, homographyDto: HomographyDTO):
    # call global variable to use in this function

    # if homography homographyDto.points are more than 4 homographyDto.points, do nothing
    if (homographyDto.var >= 4):
        return
    if (event == cv2.EVENT_LBUTTONDOWN):     # When Press mouse left down
        # Set it that the mouse is in pressing down mode
        homographyDto.drag = 1
        # copy final image to draw image
        homographyDto.matResult = homographyDto.matFinal.copy()
        # memorize current mouse position to homographyDto.point homographyDto.var
        homographyDto.point = (x, y)
        # if the homographyDto.point has been added more than 1 homographyDto.points, draw a line
        if (homographyDto.var >= 1):
            # draw a green line with thickness 2
            cv2.line(homographyDto.matResult, homographyDto.pts[homographyDto.var - 1],
                     homographyDto.point, (0, 255, 0, 255), 2)
        cv2.circle(homographyDto.matResult, homographyDto.point, 2, (0, 255, 0), -1, 8,
                   0)             # draw a current green homographyDto.point
        # show the current drawing
        cv2.imshow("Source", homographyDto.matResult)
    if (event == cv2.EVENT_LBUTTONUP and homographyDto.drag):  # When Press mouse left up
        homographyDto.drag = 0                             # no more mouse drag
        # add the current homographyDto.point to homographyDto.pts
        homographyDto.pts.append(homographyDto.point)
        # increase homographyDto.point number
        homographyDto.var += 1
        # copy the current drawing image to final image
        homographyDto.matFinal = homographyDto.matResult.copy()
        # if the homograpy homographyDto.points are done
        if (homographyDto.var >= 4):
            # draw the last line
            cv2.line(homographyDto.matFinal,
                     homographyDto.pts[0], homographyDto.pts[3], (0, 255, 0, 255), 2)
            # draw polygon from homographyDto.points
            cv2.fillConvexPoly(homographyDto.matFinal, np.array(
                homographyDto.pts, 'int32'), (0, 120, 0, 20))
        cv2.imshow("Source", homographyDto.matFinal)
    if homographyDto.drag:                                    # if the mouse is dragging
        # copy final images to draw image
        homographyDto.matResult = homographyDto.matFinal.copy()
        # memorize current mouse position to homographyDto.point homographyDto.var
        homographyDto.point = (x, y)
        # if the homographyDto.point has been added more than 1 homographyDto.points, draw a line
        if (homographyDto.var >= 1):
            # draw a green line with thickness 2
            cv2.line(homographyDto.matResult, homographyDto.pts[homographyDto.var - 1],
                     homographyDto.point, (0, 255, 0, 255), 2)
        cv2.circle(homographyDto.matResult, homographyDto.point, 2, (0, 255, 0), -1, 8,
                   0)         # draw a current green point
        # show the current drawing
        cv2.imshow("Source", homographyDto.matResult)


if __name__ == '__main__':

    key = -1
    homographyDto = HomographyDTO()
    h_fileUtil = HomographyFileUtil()

    # Open input video file
    videoCapture = cv2.VideoCapture(VIDEO_FILE)
    if not videoCapture.isOpened():
        print('Error: Unable to open input video file', VIDEO_FILE)
        sys.exit('Unable to open input video file')

    width = videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
    height = videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
    window_name = 'window_1'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Capture loop
    # Capture loop
    while (key < 0):        # play video until press any key
        # Get the next frame
        _, matFrameCapture = videoCapture.read()
        if matFrameCapture is None:   # no more frame capture from the video
            # End of video file
            break

        # Rotate if needed, some video has output like top go down, so we need to rotate it
        if ROTATE:
            # rotate 180 degree and put the image to matFrameDisplay
            _, matFrameDisplay = cv2.rotate(matFrameCapture, cv2.ROTATE_180)
        else:
            matFrameDisplay = matFrameCapture

        ratio = 640.0 / width
        dim = (int(width * ratio), int(height * ratio))
        # resize image to 480 * 640 for showing
        matFrameDisplay = cv2.resize(matFrameDisplay, dim)

        # Show the image in window named "robot.mp4"
        cv2.imshow(VIDEO_FILE, matFrameDisplay)
        key = cv2.waitKey(30)

        # --------------------- [STEP 2: pause the screen and show an image] ---------------------
        if (key >= 0):
            matPauseScreen = matFrameCapture     # transfer the current image to process
            homographyDto.matFinal = matPauseScreen.copy()     # copy image to final image

    # --------------------- [STEP 3: use mouse handler to select 4 points] ---------------------
    if (matFrameCapture is not None):
        # reset number of saving points
        homographyDto.var = 0
        # reset all points
        homographyDto.pts.clear()
        # create a windown named source
        cv2.namedWindow("Source", cv2.WINDOW_AUTOSIZE)
        # set mouse event handler "mouseHandler" at Window "Source"
        cv2.setMouseCallback("Source", mouseHandler, homographyDto)
        cv2.imshow("Source", matPauseScreen)                # Show the image
        # wait until press anykey
        cv2.waitKey(0)
        # destroy the window
        cv2.destroyWindow("Source")
    else:
        print("No pause before end of video finish. Exiting.")

    if (len(homographyDto.pts) == 4):
        src = np.array(homographyDto.pts).astype(np.float32)

        reals = np.array([(800, 800),
                          (1000, 800),
                          (1000, 1000),
                          (800, 1000)], np.float32)

        homography = Homography(src, reals)

        manual_homography_matrix = homography.get_homography_matrix()

        homography_matrix = cv2.getPerspectiveTransform(src, reals)
        print("Estimated Homography Matrix is:")
        print(homography_matrix)
        print('Manual Homography Matrix is:')
        print(manual_homography_matrix)

        # perspective transform operation using transform matrix

        h, w, ch = matPauseScreen.shape
        matResult = cv2.warpPerspective(
            matPauseScreen, manual_homography_matrix, (w, h), cv2.INTER_LINEAR)
        matPauseScreen = cv2.resize(matPauseScreen, dim)

        cv2.imshow("Source", matPauseScreen)
        matResult = cv2.resize(matResult, dim)
        cv2.imshow("Result", matResult)

        h_fileUtil.open_write_mode()
        h_fileUtil.write_matrix(
            manual_homography_matrix, 'manual_homography_matrix')
        h_fileUtil.write_matrix(homography_matrix, 'homography_matrix')
        h_fileUtil.close_file()

        cv2.waitKey(0)
