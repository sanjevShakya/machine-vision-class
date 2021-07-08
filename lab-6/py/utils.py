from stats import Stats
import cv2
import numpy as np
from typing import List  # use it for :List[...]


def drawBoundingBox(image, bb):
    """
    Draw the bounding box from the points set

    Parameters
    ----------
        image (array):
            image which you want to draw
        bb (List):
            points array set
    """
    color = (0, 0, 255)
    for i in range(len(bb) - 1):
        b1 = (int(bb[i][0]), int(bb[i][1]))
        b2 = (int(bb[i + 1][0]), int(bb[i + 1][1]))
        cv2.line(image, b1, b2, color, 2)
    b1 = (int(bb[len(bb) - 1][0]), int(bb[len(bb) - 1][1]))
    b2 = (int(bb[0][0]), int(bb[0][1]))
    cv2.line(image, b1, b2, color, 2)


def drawStatistics(image, stat: Stats):
    """
    Draw the statistic to images

    Parameters
    ----------
        image (array):
            image which you want to draw
        stat (Stats):
            statistic values
    """
    font = cv2.FONT_HERSHEY_PLAIN

    str1, str2, str3, str4, str5 = stat.to_strings()

    shape = image.shape

    cv2.putText(image, str1, (0, shape[0] - 120), font, 2, (0, 0, 255), 3)
    cv2.putText(image, str2, (0, shape[0] - 90), font, 2, (0, 0, 255), 3)
    cv2.putText(image, str3, (0, shape[0] - 60), font, 2, (0, 0, 255), 3)
    cv2.putText(image, str5, (0, shape[0] - 30), font, 2, (0, 0, 255), 3)


def printStatistics(name: str, stat: Stats):
    """
    Print the statistic

    Parameters
    ----------
        name (str):
            image which you want to draw
        stat (Stats):
            statistic values
    """
    print(name)
    print("----------")
    str1, str2, str3, str4, str5 = stat.to_strings()
    print(str1)
    print(str2)
    print(str3)
    print(str4)
    print(str5)
    print()


def Points(keypoints):
    return  np.float32([keypoints[idx].pt for idx, m in enumerate(keypoints)]).reshape(-1, 1, 2)


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
