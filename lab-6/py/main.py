import cv2
import numpy as np
import time
from stats import Stats
from utils import drawBoundingBox, drawStatistics, printStatistics, Points, HomographyFileUtil

# AKAZE detection threshold set to locate about 1000 keypoints
akaze_thresh: float = 3e-4
ransac_thresh: float = 2.5  # RANSAC inlier threshold
nn_match_ratio: float = 0.8  # Nearest-neighbour matching ratio
bb_min_inliers: int = 100  # Minimal number of inliers to draw bounding box
stats_update_period: int = 10  # On-screen statistics are updated every 10 frames


# def detectDrawKeyPoints(frame, detector):
#     keyPoints = detector.detect(frame, None)
#     kps, dps = detector.compute(frame, keyPoints)
#     cv2.drawKeypoints(frame, kps, None, color=(255, 0, 0),
#                       flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


class Tracker:
    def __init__(self, detector, matcher):
        self.detector = detector
        self.matcher = matcher

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
        res = cv2.drawKeypoints(self.first_frame, self.first_kp, None, color=(
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
            if m.distance < nn_match_ratio * n.distance:
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
                matched1, matched2, cv2.RANSAC, ransac_thresh)
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
        if (stat.inliers >= bb_min_inliers):
            drawBoundingBox(frame_with_bb, new_bb[0])

        res = cv2.drawMatches(self.first_frame, inliers1_keypoints, frame_with_bb, inliers2_keypoints,
                              inlier_matches, None, matchColor=(255, 0, 0), singlePointColor=(255, 0, 0))
        return res, stat

    def getDetector(self):
        return self.detector


def main():
    video_name = "../robot.mp4"
    calibrateFile = HomographyFileUtil("../calibrate_robot.yaml")
    calibrateFile.open_read_mode()
    rvecs = calibrateFile.read_matrix('rvecs')
    tvecs = calibrateFile.read_matrix('tvecs')
    dist = calibrateFile.read_matrix('dist')
    mtx = calibrateFile.read_matrix('mtx')
    calibrateFile.close_file()

    video_in = cv2.VideoCapture()
    video_in.open(video_name)
    if (not video_in.isOpened()):
        print("Couldn't open ", video_name)
        return -1

    akaze_stats = Stats()
    orb_stats = Stats()

    akaze = cv2.AKAZE_create()
    akaze.setThreshold(akaze_thresh)

    orb = cv2.ORB_create()

    matcher = cv2.DescriptorMatcher_create("BruteForce-Hamming")

    akaze_tracker = Tracker(akaze, matcher)
    orb_tracker = Tracker(orb, matcher)

    cv2.namedWindow(video_name, cv2.WINDOW_NORMAL)
    print("\nPress any key to stop the video and select a bounding box")

    key = -1

    while(key < 1):
        _, frame = video_in.read()
        w, h, ch = frame.shape
        cv2.resizeWindow(video_name, (h, w))
        cv2.imshow(video_name, frame)
        # detectDrawKeyPoints(frame, akaze)
        key = cv2.waitKey(1)

    print("Select a ROI and then press SPACE or ENTER button!")
    print("Cancel the selection process by pressing c button!")
    uBox = cv2.selectROI(video_name, frame)
    bb = []
    bb.append((uBox[0], uBox[1]))
    bb.append((uBox[0] + uBox[2], uBox[0]))
    bb.append((uBox[0] + uBox[2], uBox[0] + uBox[3]))
    bb.append((uBox[0], uBox[0] + uBox[3]))

    stat_a = akaze_tracker.setFirstFrame(frame, bb, "AKAZE",)
    stat_o = orb_tracker.setFirstFrame(frame, bb, "ORB")

    akaze_draw_stats = stat_a.copy()
    orb_draw_stats = stat_o.copy()

    i = 0
    video_in.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while True:
        i += 1
        update_stats = (i % stats_update_period == 0)
        _, frame = video_in.read()
        if frame is None:
            # End of video
            break
        akaze_res, stat = akaze_tracker.process(frame)
        akaze_stats + stat
        if (update_stats):
            akaze_draw_stats = stat
        orb.setMaxFeatures(stat.keypoints)
        orb_res, stat = orb_tracker.process(frame)
        orb_stats + stat
        if (update_stats):
            orb_draw_stats = stat
        drawStatistics(akaze_res, akaze_draw_stats)
        drawStatistics(orb_res, orb_draw_stats)
        res_frame = cv2.vconcat([akaze_res, orb_res])
        # cv2.imshow(video_name, akaze_res)
        cv2.imshow(video_name, res_frame)
        if (cv2.waitKey(1) == 27):  # quit on ESC button
            break

    akaze_stats / (i - 1)
    orb_stats / (i - 1)
    printStatistics("AKAZE", akaze_stats)
    printStatistics("ORB", orb_stats)
    return 0


if __name__ == '__main__':
    main()
