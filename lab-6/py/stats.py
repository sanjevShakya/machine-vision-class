import numpy as np

class Stats:
    """
    Statistic class

    Attributes
    ----------
    matches=0 (int):
        total number of matching

    inliers=0 (int):
        number of inliner matching

    ratio=0. (float):
        Nearest-neighbour matching ratio

    keypoints=0 (int):
        Wall

    fps=0. (float):
        frame per 1 sec
    
    Methods
    -------
    add(Stats) - overload + function:
        plus the information into this class

    divide(Stats) - overload + function:
        divide the information into this class
    """
    matches:int
    inliers:int
    ratio:float
    keypoints:int
    fps:float

    def __init__(self, matches = 0, inliers = 0, ratio = 0., keypoints = 0, fps = 0.):
        self.matches = matches
        self.inliers = inliers
        self.ratio = ratio
        self.keypoints = keypoints
        self.fps = fps

    def __add__(self, op:"Stats") -> "Stats":
        self.matches += op.matches
        self.inliers += op.inliers
        self.ratio += op.ratio
        self.keypoints += op.keypoints
        self.fps += op.fps
        return self

    def __truediv__(self, num:int) -> "Stats":
        self.matches //= num
        self.inliers //= num
        self.ratio /= num
        self.keypoints //= num
        self.fps /= num
        return self

    def __str__(self) -> str:
        return "matches({0}) inliner({1}) ratio({2:.2f}) keypoints({3}) fps({4:.2f})".format(self.matches, self.inliers, self.ratio, self.keypoints, self.fps)

    __repr__ = __str__

    def to_strings(self):
        """
        Convert to string set of matches, inliners, ratio, and fps
        """
        str1 = "Matches: {0}".format(self.matches)
        str2 = "Inliers: {0}".format(self.inliers)
        str3 = "Inlier ratio: {0:.2f}".format(self.ratio)
        str4 = "Keypoints: {0}".format(self.keypoints)
        str5 = "FPS: {0:.2f}".format(self.fps)
        return str1, str2, str3, str4, str5

    def copy(self):
        return Stats(self.matches, self.inliers, self.ratio, self.keypoints, self.fps)