import cv2 
import numpy as np
from stats import Stats
import time
from utils import drawBoundingBox, Points

class Tracker:
    nn_match_ratio: float = 0.8
    bb_min_inliers: int = 100  # Minimal number of inliers to draw bounding box
    ransac_thresh: float = 2.5  # RANSAC inlier threshold

    def __init__(self, detector, matcher, camera_matrix, dist_coeffs):
        self.detector = detector
        self.matcher = matcher
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs

    def setFirstFrame(self, frame, bb, title: str):
        iSize = len(bb)
        stat = Stats()
        ptContain = np.zeros((iSize, 2))
        i = 0
        for b in bb:
            #ptMask[i] = (b[0], b[1])
            ptContain[i, 0] = b[0]
            ptContain[i, 1] = b[1]
            i += 1

        self.first_frame = frame.copy()
        matMask = np.zeros(frame.shape, dtype=np.uint8)
        cv2.fillPoly(matMask, np.int32([ptContain]), (255, 0, 0))

        # cannot use in ORB
        # self.first_kp, self.first_desc = self.detector.detectAndCompute(self.first_frame, matMask)

        # find the keypoints with ORB
        kp = self.detector.detect(self.first_frame, None)

        # compute the descriptors with ORB
        self.first_kp, self.first_desc = self.detector.compute(
            self.first_frame, kp)

        # print(self.first_kp[0].pt[0])
        # print(self.first_kp[0].pt[1])
        # print(self.first_kp[0].angle)
        # print(self.first_kp[0].size)
        # change here ??
        points = Points(self.first_kp)
        xy_undistorted = cv2.undistortPoints(
            points, self.camera_matrix, self.dist_coeffs)

        res = cv2.drawKeypoints(self.first_frame, xy_undistorted, None, color=(
            255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        stat.keypoints = len(self.first_kp)
        drawBoundingBox(self.first_frame, bb)

        cv2.imshow("key points of {0}".format(title), res)
        cv2.waitKey(0)
        cv2.destroyWindow("key points of {0}".format(title))

        cv2.putText(self.first_frame, title, (0, 60),
                    cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 0), 4)
        self.object_bb = bb
        return stat

    def process(self, frame):
        stat = Stats()
        start_time = time.time()
        kp, desc = self.detector.detectAndCompute(frame, None)
        stat.keypoints = len(kp)
        matches = self.matcher.knnMatch(self.first_desc, desc, k=2)

        matched1 = []
        matched2 = []
        matched1_keypoints = []
        matched2_keypoints = []
        good = []

        for i, (m, n) in enumerate(matches):
            if m.distance < self.nn_match_ratio * n.distance:
                good.append(m)
                matched1_keypoints.append(
                    self.first_kp[matches[i][0].queryIdx])
                matched2_keypoints.append(kp[matches[i][0].trainIdx])

        matched1 = np.float32(
            [self.first_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        matched2 = np.float32(
            [kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        stat.matches = len(matched1)
        homography = None
        if (len(matched1) >= 4):
            homography, inlier_mask = cv2.findHomography(
                matched1, matched2, cv2.RANSAC, self.ransac_thresh)
        dt = time.time() - start_time
        stat.fps = 1. / dt
        if (len(matched1) < 4 or homography is None):
            res = cv2.hconcat([self.first_frame, frame])
            stat.inliers = 0
            stat.ratio = 0
            return res, stat
        inliers1 = []
        inliers2 = []
        inliers1_keypoints = []
        inliers2_keypoints = []
        for i in range(len(good)):
            if (inlier_mask[i] > 0):
                new_i = len(inliers1)
                inliers1.append(matched1[i])
                inliers2.append(matched2[i])
                inliers1_keypoints.append(matched1_keypoints[i])
                inliers2_keypoints.append(matched2_keypoints[i])
        inlier_matches = [cv2.DMatch(
            _imgIdx=0, _queryIdx=idx, _trainIdx=idx, _distance=0) for idx in range(len(inliers1))]
        inliers1 = np.array(inliers1, dtype=np.float32)
        inliers2 = np.array(inliers2, dtype=np.float32)

        stat.inliers = len(inliers1)
        stat.ratio = stat.inliers * 1.0 / stat.matches
        bb = np.array([self.object_bb], dtype=np.float32)
        new_bb = cv2.perspectiveTransform(bb, homography)
        frame_with_bb = frame.copy()
        if (stat.inliers >= self.bb_min_inliers):
            drawBoundingBox(frame_with_bb, new_bb[0])

        res = cv2.drawMatches(self.first_frame, inliers1_keypoints, frame_with_bb, inliers2_keypoints,
                              inlier_matches, None, matchColor=(255, 0, 0), singlePointColor=(255, 0, 0))
        return res, stat

    def getDetector(self):
        return self.detector
