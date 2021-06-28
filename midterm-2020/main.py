# First read the yaml file and store the camera extrinsic parameters
# Load the images
# Undistort the image
import numpy as np
from scipy import linalg
import cv2 as cv
import sys


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


class CameraData:
    image_width = 0
    image_height = 0
    board_width = None
    square_size = None
    camera_matrix = None
    distortion_coefficients = None
    rot = None
    trans = None

    def __init__(self, fs):
        self.fs = fs
        pass

    def read(self):
        self.image_width = self.fs.getNode('image_width').real()
        self.image_height = self.fs.getNode('image_height').real()
        self.board_width = self.fs.getNode('board_width').real()
        self.square_size = self.fs.getNode('square_size').real()
        self.camera_matrix = self.fs.getNode('camera_matrix').mat()
        self.distortion_coefficients = self.fs.getNode(
            'distortion_coefficients').mat()
        self.rot = self.fs.getNode('rot').mat()
        self.trans = self.fs.getNode('trans').mat()
        return self


def help(filename):
    print(
        '''
        {0} shows the usage of the OpenCV serialization functionality. \n\n
        usage:\n
            python3 {0} outputfile.yml.gz\n\n
        The output file may be either in XML, YAML or JSON. You can even compress it\n
        by specifying this in its extension like xml.gz yaml.gz etc... With\n
        FileStorage you can serialize objects in OpenCV.\n\n
        For example: - create a class and have it serialized\n
                     - use it to read and write matrices.\n
        '''.format(filename)
    )


def main(argv):
    if len(argv) != 2:
        help(argv[0])
        exit(1)

    filename = argv[1]
    fs = cv.FileStorage()
    fs.open(filename, cv.FileStorage_READ)
    camera_data = CameraData(fs).read()
    mtx = camera_data.camera_matrix
    dist = camera_data.distortion_coefficients
    img1 = cv.imread('images/examImg1.png')
    h, w = img1.shape[:2]
    newcammtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    undistorted_img = cv.undistort(img1, mtx, dist)
    x, y, w, h = roi
    # cv.imshow('distorted_image', img1)
    # cv.imshow('undistorted_image', undistorted_img)
    # cv.waitKey(0)

    # Question 2
    rot_mat, _ = cv.Rodrigues(camera_data.rot)
    projection_mat = np.hstack((rot_mat, camera_data.trans))
    projection_mat = np.vstack((projection_mat, [[0, 0, 0, 1]]))
    print(projection_mat)
    tcr = projection_mat
    point = np.array([0, 0, 0, 1]).T
    result = np.dot(tcr, point)
    print('x = H x :', result)
    print('Transformation from robot origin to world-coordinate')
    print('----------------------------------------------------')
    print('START Question 3 -------')
    tcr_inv = np.linalg.pinv(tcr)
    Pr = np.array([[763.037448, 948.626962, -70.027011, -249.063249], [-27.153472,
                  413.013589, -834.460406, 146.980511], [-0.047150, 0.996552, -0.068276, -0.347539]]).T
    Pr_dash = np.array([[529.243367, 1096.649189, -65.460341, -541.617311], [-114.715323,
                       397.195667, -834.696173, 54.425526], [-0.270799, 0.960176, -0.068768, -0.625205]]).T
    n_Pr = np.dot(Pr.T, tcr_inv)
    n_Pr_dash = np.dot(Pr_dash.T, tcr_inv)
    print('Normalized Pr', n_Pr)
    print('Normalized Pr_dash', n_Pr_dash)
    print('----------------------------------------------------')
    print('START Question 4 --------')
    # k1, r1 = linalg.rq(n_Pr)
    plane = np.array([[0, 0, 1, 0]]).T
    input_points = np.array([[1, 1, 0, 1], [1, 2, 0, 1], [2, 1, 0, 1], [3, 1, 0, 1]]).T
    plane_prime = np.dot(plane.T, input_points);
    print(plane_prime)
    ps_prime = np.dot(Pr.T, input_points);
    ps_prime = ps_prime / ps_prime[-1]
    print(ps_prime)
    # print('Input points:\n', input_points)
    # print('Output points:\n', points_prime)
    homography = Homography(ps_prime, input_points)
    h_mat = homography.get_homography_matrix()
    print('Homography mat: \n')
    print(h_mat)


if __name__ == '__main__':
    main(sys.argv)
