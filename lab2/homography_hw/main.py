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


class CameraMatrixData:
    mtx = None
    dist = None
    rvecs = None
    tvecs = None

    def __init__(self):
        pass

    def write(self, fs, name):
        fs.startWriteStruct(name, cv2.FileNode_MAP | cv2.FileNode_FLOW)
        fs.write('mtx', self.mtx)
        fs.write('dist', self.dist)
        fs.write('rvecs', np.array(self.rvecs))
        fs.write('tvecs', np.array(self.tvecs))
        fs.endWriteStruct()

    def read(self, node):
        if (not node.empty()):
            self.mtx = node.getNode('mtx').mat()
            self.dist = node.getNode('dist').mat()
            self.rvecs = node.getNode('rvecs').mat()
            self.tvecs = node.getNode('tvecs').mat()
        else:
            self.mtx = None
            self.dist = None
            self.rvecs = None
            self.tvecs = None
        return self


class HomographyFileUtil:
    output_file = "homography.yaml"
    file_storage = None

    def __init__(self, output_file="homography.yaml"):
        self.output_file = output_file
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


def get_optical_flow_points(p0, p1, old_frame, new_frame, lk_params):
    frame_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)

    if p1 is None:
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            old_gray, frame_gray, p0, None, **lk_params)

    if p1 is not None:
        good_new_point = p1[st == 1]
        good_old_point = p0[st == 1]
        return (good_old_point, good_new_point)
    else:
        old_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **lk_params)
        return get_optical_flow_points(p0, p1, old_frame, new_frame, lk_params)


def main():
    key = -1

    videoCapture = cv2.VideoCapture(VIDEO_FILE)
    homographyFileUtil = HomographyFileUtil()
    homographyFileUtil.open_read_mode()
    manualHomographyMatrix = homographyFileUtil.read_matrix(
        'manual_homography_matrix')
    homographyMatrix = homographyFileUtil.read_matrix('homography_matrix')
    print('>>>>>', manualHomographyMatrix)
    if not videoCapture.isOpened():
        print('Error: Unable to open input video file', VIDEO_FILE)
        sys.exit('Unable to open input video file')

    original_window = 'original_window'
    perspective_window = 'perspective_window'
    original_window_undistorted = 'original_window_undistorted'
    perspective_window_undistored = 'perspective_window_undistored'

    cv2.namedWindow(original_window, cv2.WINDOW_NORMAL)
    cv2.namedWindow(perspective_window, cv2.WINDOW_NORMAL)
    cv2.namedWindow(original_window_undistorted, cv2.WINDOW_NORMAL)
    cv2.namedWindow(perspective_window_undistored, cv2.WINDOW_NORMAL)

    width = videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
    height = videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`

    ratio = 640.0 / width
    dim = (int(width * ratio), int(height * ratio))

    calibrateFile = HomographyFileUtil("calibrate_robot.yaml")
    calibrateFile.open_read_mode()
    rvecs = calibrateFile.read_matrix('rvecs')
    tvecs = calibrateFile.read_matrix('tvecs')
    dist = calibrateFile.read_matrix('dist')
    mtx = calibrateFile.read_matrix('mtx')
    calibrateFile.close_file()

    print('cmatrixData', rvecs)
    # cmatrixData.write(fileStorage, 'CameraMatrixData')

    while(key < 0):
        _, matFrameCapture = videoCapture.read()
        if matFrameCapture is None:
            # End of video
            break

        matFrameDisplay = matFrameCapture

        h, w, ch = matFrameDisplay.shape
        matHomographyDisplay = cv2.warpPerspective(
            matFrameDisplay, manualHomographyMatrix, (w, h), cv2.INTER_LINEAR)

        matFrameDisplay = cv2.resize(matFrameDisplay, dim)
        matHomographyDisplay = cv2.resize(matHomographyDisplay, dim)
        cv2.imshow(original_window, matFrameDisplay)
        cv2.imshow(perspective_window, matHomographyDisplay)

        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            mtx, dist, (w, h), 1, (w, h))
        x, y, w, h = roi
        matFrameRes = cv2.undistort(matFrameDisplay, mtx, dist)
        matHomographyRes = cv2.undistort(matHomographyDisplay, mtx, dist)
        # matFrameRes = matFrameRes[y:y+h, x:x+w]
        # matHomographyRes = matHomographyRes[y:y+h, x:x+w]
        cv2.imshow(original_window_undistorted, matFrameRes)
        cv2.imshow(perspective_window_undistored, matHomographyRes)
        key = cv2.waitKey(30)

        if(key > 0):
            print('break from here', key)
            break


if __name__ == '__main__':
    main()
