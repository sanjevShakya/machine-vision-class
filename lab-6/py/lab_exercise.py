import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
from utils import Points, HomographyFileUtil


FRAME_1 = '../Frame1.png'
FRAME_2 = '../Frame2.png'

akaze_thresh: float = 3e-4
ransac_thresh: float = 2.5  # RANSAC inlier threshold
nn_match_ratio: float = 0.8  # Nearest-neighbour matching ratio
bb_min_inliers: int = 100  # Minimal number of inliers to draw bounding box
stats_update_period: int = 10  # On-screen statistics are updated every 10 frames


def fetch_camera_calibration():
    calibrateFile = HomographyFileUtil("../calibrate_robot.yaml")
    calibrateFile.open_read_mode()
    rvecs = calibrateFile.read_matrix('rvecs')
    tvecs = calibrateFile.read_matrix('tvecs')
    dist = calibrateFile.read_matrix('dist')
    mtx = calibrateFile.read_matrix('mtx')
    calibrateFile.close_file()
    return (mtx, dist, rvecs, tvecs)


def detectComputeKeyPoints(frame, detector, camera_matrix=None, dist_coeffs=None, undistort=False):
    copied_frame = frame.copy()
    kp_detect = detector.detect(frame, None)
    kps, desc = detector.compute(frame, kp_detect)
    kp_len = len(kps)
    if undistort:
        points = []
        points = Points(kps)
        xy_undistoted = cv2.undistortPoints(points, camera_matrix, dist_coeffs)
        # TODO change xy_undistorted points to keypoints

        # print('xy_points', xy_undistoted.shape);
        # kps = xy_undistoted
        # for i in range(kp_len):
        #     x = xy_undistoted[i][0]
        #     y = xy_undistoted[i][1]
        #     # points.append()
        # print('kps', xy_undistoted)
    res = cv2.drawKeypoints(copied_frame, kps, None, color=(
        255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return (res, desc, kps)


def main():
    mtx, dist, rvec, tvecs = fetch_camera_calibration()

    frame1 = cv2.imread(FRAME_1)
    frame2 = cv2.imread(FRAME_2)

    akaze_window = 'akaze_window'
    orb_window = 'orb_window'
    epipole_frame2 = 'epipole_frame2'
    epipole_frame1 = 'epipole_frame1'

    # orb_window = 'orb_window'
    matcher = cv2.DescriptorMatcher_create("BruteForce-Hamming")

    cv2.namedWindow(akaze_window, flags=cv2.WINDOW_GUI_EXPANDED)
    cv2.namedWindow(epipole_frame1, flags=cv2.WINDOW_GUI_EXPANDED)
    cv2.namedWindow(epipole_frame2, flags=cv2.WINDOW_GUI_EXPANDED)

    height, width, _ = frame1.shape
    scaleFactor = 0.4
    resized_height = int(scaleFactor * height)
    resized_width = int(scaleFactor * width)
    cv2.resizeWindow(akaze_window, (resized_width, resized_height))
    cv2.resizeWindow(epipole_frame1, (resized_width, resized_height))
    cv2.resizeWindow(epipole_frame2, (resized_width, resized_height))

    # cv2.namedWindow(orb_window)

    akaze = cv2.AKAZE_create()
    akaze.setThreshold(akaze_thresh)
    orb = cv2.ORB_create()

    res_akaze_frame1, desc_akaze_frame1, key_points_frame1 = detectComputeKeyPoints(
        frame1, akaze, mtx, dist, undistort=True)
    res_akaze_frame2, desc_akaze_frame2, key_points_frame2 = detectComputeKeyPoints(
        frame2, akaze, mtx, dist, undistort=True)
    # res_orb = detectComputeKeyPoints(frame2, orb)

    matches = matcher.knnMatch(desc_akaze_frame1, desc_akaze_frame2, k=2)
    matched1 = []
    matched2 = []
    matched1_kps = []
    matched2_kps = []
    good = []

    for i, (m, n) in enumerate(matches):
        if m.distance < 0.8 * n.distance:
            good.append(m)
            matched1_kps.append(
                key_points_frame1[matches[i][0].queryIdx])
            matched2_kps.append(key_points_frame2[matches[i][0].trainIdx])

    matched1 = np.float32(
        [key_points_frame1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    matched2 = np.float32(
        [key_points_frame2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    res = cv2.drawMatches(frame1, key_points_frame1, frame2, key_points_frame2,
                          good, None, matchColor=(255, 0, 0), singlePointColor=(255, 0, 0))

    # Now use RANSAC
    homography = None
    start_time = time.time()
    if(len(matched1) >= 4):
        homography, inlier_mask = cv2.findHomography(
            matched1, matched2, method=cv2.RANSAC, ransacReprojThreshold=1)
        dt = time.time() - start_time
        # fps = 1. / dt

    if (len(matched1) < 4 or homography is None):
        res = cv2.hconcat([frame1, frame2])
        return sys.exit(0)

    inliers_frame1 = []
    inliers_frame2 = []
    inliers_frame1_kps = []
    inliers_frame2_kps = []
    for i in range(len(good)):
        if (inlier_mask[i] > 0):
            inliers_frame1.append(matched1[i])
            inliers_frame2.append(matched2[i])
            inliers_frame1_kps.append(key_points_frame1[i])
            inliers_frame2_kps.append(key_points_frame2[i])
    inlier_matches = [cv2.DMatch(
        _imgIdx=0, _queryIdx=idx, _trainIdx=idx, _distance=0) for idx in range(len(inliers_frame1))]
    inliers_frame1 = np.array(inliers_frame1, dtype=np.float32)
    inliers_frame2 = np.array(inliers_frame2, dtype=np.float32)

    res = cv2.drawMatches(frame1, inliers_frame1_kps, frame2, inliers_frame2_kps,
                          inlier_matches, None, matchColor=(255, 0, 0), singlePointColor=(255, 0, 0))

    essential_mtx, mask = cv2.findEssentialMat(
        matched1, matched2, mtx, method=cv2.RANSAC, threshold=1)

    matchesMask = []
    j = 0
    for i in range(len(good)):
        if matches[i]:
            if inlier_mask[j] > 0:
                matchesMask.append((1, 0))
            else:
                matchesMask.append((0, 0))
            j += 1
        else:
            matchesMask.append((0, 0))
    cv2.imshow(akaze_window, res)
    print(inliers_frame1[0].shape)

    def epipolar_line(x, x_prime, F):
        x = np.array(x).T
        x = np.vstack((x, [1]))
        x_prime = np.array(x_prime).T
        x_prime = np.vstack((x_prime, [1]))
        frame2_l = np.dot(F, x)
        frame1_l = np.dot(F.T, x_prime)

        frame1_l = frame1_l / frame1_l[2]
        frame2_l = frame2_l / frame2_l[2]

        return (frame1_l, frame2_l)

    def line_x(homogeneous_coordinate, x):
        a = homogeneous_coordinate[0]
        b = homogeneous_coordinate[1]
        c = homogeneous_coordinate[2]
        return (-(a * x + c)) / b

    def line_y(homogeneous_coordinate, y):
        a = homogeneous_coordinate[0]
        b = homogeneous_coordinate[1]
        c = homogeneous_coordinate[2]
        return (-(b * y + c)) / a

    def draw_line_frame(window_name, frame, line):
        x1 = 0
        y1 = int(line_x(line, x1)[0])
        y2 = height
        x2 = int(line_y(line, y2)[0])
        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imshow(window_name, frame)

    F, mask = cv2.findFundamentalMat(
        matched1, matched2, cv2.FM_RANSAC)

    epipolar_lines = []
    for i in range(len(inliers_frame1)):
        x = inliers_frame1[i]
        xp = inliers_frame2[i]
        frame1_l, frame2_l = epipolar_line(x, xp, F)
        draw_line_frame(epipole_frame1, frame1,
                        frame1_l)

        draw_line_frame(epipole_frame2, frame2,
                        frame2_l)

    print('inlier frame 1 shape', inliers_frame1.shape)
    print('inlier frame 1 shape', inliers_frame2.shape)
    newPts1, newPts2 = cv2.correctMatches(F, matched1.reshape(
        1, len(matched1), 2), matched2.reshape(1, len(matched2), 2))
    print(newPts1.shape)
    print(newPts2.shape)

    points, R_est, t_est, mask_pose = cv2.recoverPose(
        essential_mtx, inliers_frame1, inliers_frame2)

    R1, R2, t = cv2.decomposeEssentialMat(essential_mtx)
    P1 = np.hstack((R1, t))
    P2 = np.hstack((R2, t))

    pp_1 = []
    pp_2 = []
    indices = []
    print('inliers_frame1 0', inliers_frame1[0])
    for i in range(len(inliers_frame1)):
        p1 = inliers_frame1[i][0]
        p2 = inliers_frame2[i][0]
        pp_1.append([p1[0], p1[1], 1.0])
        pp_2.append([p2[0], p2[1], 1.0])
    pp_1 = np.array(pp_1).T
    pp_2 = np.array(pp_2).T
    pts1 = np.dot(np.linalg.pinv(P1), pp_1)
    pts2 = np.dot(np.linalg.pinv(P2), pp_2)

    three_d_points = cv2.triangulatePoints(P1, P2, pts1[:2], pts2[:2])
    three_d_points /= three_d_points[3]
    print(three_d_points)
    print(three_d_points.shape)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    x = three_d_points[0]
    print(x)
    ax.scatter(three_d_points[0], three_d_points[1],
               three_d_points[2], marker='o')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    cv2.waitKey(0)
    plt.savefig('3d_reconstruction.png')
    plt.show()


if __name__ == '__main__':
    main()
