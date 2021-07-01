#!/usr/bin/env python

import cv2
import numpy as np
import glob


class CameraMatrixData:
    mtx = None
    dist = None
    rvecs = None
    tvecs = None

    def __init__(self, mtx, dist, rvecs, tvecs):
        self.mtx = mtx;
        self.dist = dist;
        self.rvecs = rvecs;
        self.tvecs = tvecs;

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

# Defining the dimensions of checkerboard
CHECKERBOARD = (6, 9)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Creating vector to store vectors of 3D points for each checkerboard image
objpoints = []
# Creating vector to store vectors of 2D points for each checkerboard image
imgpoints = []


# Defining the world coordinates for 3D points
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0],
                          0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None

# Extracting path of individual image stored in a given directory
images = glob.glob('./robot-images/*.jpg')
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    # If desired number of corners are found in the image then ret = true
    ret, corners = cv2.findChessboardCorners(
        gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

    """
    If desired number of corner are detected,
    we refine the pixel coordinates and display 
    them on the images of checker board
    """
    if ret == True:
        objpoints.append(objp)
        # refining pixel coordinates for given 2d points.
        corners2 = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria)

        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)

    cv2.imshow('img', img)
    cv2.waitKey(0)

cv2.destroyAllWindows()

h, w = img.shape[:2]

"""
Performing camera calibration by 
passing the value of known 3D points (objpoints)
and corresponding pixel coordinates of the 
detected corners (imgpoints)
"""
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None)

print("Camera matrix : \n")
print(mtx)
print("dist : \n")
print(dist)
print("rvecs : \n")
print(rvecs)
print("tvecs : \n")
print(tvecs)

fileStorage = cv2.FileStorage('calibrate_robot.yaml', cv2.FileStorage_WRITE)
cmatrixData = CameraMatrixData(mtx, dist, rvecs, tvecs)
cmatrixData.write(fileStorage, 'CameraMatrixData')


# Show images of undistorted
# for fname in images:
#     img = cv2.imread(fname)
#     res = cv2.undistort(img, mtx, dist)
#     cv2.imshow('img', img)
#     cv2.imshow('res', res)
#     cv2.waitKey(0)

# for fname in images:
#     img = cv2.imread(fname)
#     h,  w = img.shape[:2]
#     newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
#         mtx, dist, (w, h), 1, (w, h))

#     res = cv2.undistort(img, mtx, dist)

#     # crop the image
#     x, y, w, h = roi
#     res = res[y:y+h, x:x+w]

#     cv2.imshow('distorted_image', img)
#     cv2.imshow('undistorted_image', res)
#     cv2.waitKey(0)


mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(
        objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error
print("total error: {}".format(mean_error/len(objpoints)))
