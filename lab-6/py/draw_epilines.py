import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def drawlines(img1, img2, lines, pts1, pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r, c, _ = img1.shape
    # img1 = cv.cvtColor(img1, cv.COLOR_GRAY2BGR)
    # img2 = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        img1 = cv.line(img1, (x0, y0), (x1, y1), color, 1)
        # img1 = cv.circle(img1, tuple(pt1), 5, color, -1)
        # img2 = cv.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2


img1 = cv.imread('../Frame1.png')  # queryimage # left image
img2 = cv.imread('../Frame2.png')  # trainimage # right image
sift = cv.AKAZE_create()
# find the keypoints and descriptors with SIFT
kp1 = sift.detect(img1, None)
kp1, des1 = sift.compute(img1, kp1)
kp2 = sift.detect(img2, None)
kp2, des2 = sift.compute(img2, kp2)

# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
# matcher = cv.FlannBasedMatcher(index_params, search_params)
matcher = cv.DescriptorMatcher_create("BruteForce-Hamming")
matches = matcher.knnMatch(des1, des2, k=2)
pts1 = []
pts2 = []
good = []
# ratio test as per Lowe's paper
for i, (m, n) in enumerate(matches):

    if m.distance < 0.8*n.distance:
        good.append(m)
        pts1.append(kp1[matches[i][0].queryIdx])
        pts2.append(kp2[matches[i][0].trainIdx])

pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

F, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_LMEDS)
# We select only inlier points
pts1 = pts1[mask.ravel() == 1]
pts2 = pts2[mask.ravel() == 1]

# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
lines1 = lines1.reshape(-1, 3)
img5, img6 = drawlines(img1, img2, lines1, pts1, pts2)
# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
lines2 = lines2.reshape(-1, 3)
img3, img4 = drawlines(img2, img1, lines2, pts2, pts1)
img3 = cv.cvtColor(img3, cv.COLOR_BGR2RGB)
img4 = cv.cvtColor(img4, cv.COLOR_BGR2RGB)

plt.subplot(121), plt.imshow(img3)
plt.subplot(122), plt.imshow(img4)
plt.show()
essential_mtx, mask = cv.findEssentialMat(
    pts1, pts2, method=cv.RANSAC, threshold=1)
R1, R2, t = cv.decomposeEssentialMat(essential_mtx)
P1 = np.hstack((R1, t))
P2 = np.hstack((R2, t))
pp_1 = []
pp_2 = []
for i in range(len(pts1)):
    p1 = pts1[i][0]
    p2 = pts2[i][0]
    pp_1.append([p1[0], p1[1], 1.0])
    pp_2.append([p2[0], p2[1], 1.0])
pp_1 = np.array(pp_1).T
pp_2 = np.array(pp_2).T
pts1 = np.dot(np.linalg.pinv(P1), pp_1)
pts2 = np.dot(np.linalg.pinv(P2), pp_2)
three_d_points = cv.triangulatePoints(P1, P2, pts1[:2], pts2[:2])
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
ax.set_xlim((-2, 0))
plt.savefig('3d_reconstruction2.png')
plt.show()