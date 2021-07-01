import cv2
import math
import sys
import numpy as np

IMAGE_FILE_1 = "video1_frame.png"
IMAGE_FILE_2 = "video2_frame.png"
VIDEO_FILE = "roboto2.mp4"
ONNX_NETWORK_DEFINITION = "fcn_resnet18_floor.onnx"
IMAGENET_MEANS = (123.67, 116.28, 103.53)
IMAGENET_SCALE_FACTOR = 1.0 / 255.0 / (0.224 + 0.229 + 0.225 / 3)
WINDOW_NAME = 'getmented_frame'
WINDOW_FLAGS = cv2.WINDOW_GUI_EXPANDED

aStringClasses = [
    "nonfloor", "floor"
]

aColorClasses = [
    (0, 0, 0), (255, 255, 0)
]

# Read CNN definition
net = cv2.dnn.readNetFromONNX(ONNX_NETWORK_DEFINITION)
nClasses = len(aColorClasses)


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


def segmentCapturedFrame(matFrame):
    tensorFrame = cv2.dnn.blobFromImage(
        matFrame, IMAGENET_SCALE_FACTOR, (320, 320), IMAGENET_MEANS, True, False)
    # print(tensorFrame.shape)

    net.setInput(tensorFrame)
    matScore = net.forward()
    # Colorize the image and display
    matFrameCopied = matFrame.copy()
    # scores = 1.0 / (1.0 + np.exp(-matScore.squeeze(0)))
    classes = matScore.squeeze(0).argmax(0) + 1
    # classes = matScore.squeeze(0)[1:, :, :].argmax(0) + 1
    color_mask = np.zeros((3, 320, 320))
    for iClass in range(nClasses):
        mask = classes == iClass
        colors = np.array((aColorClasses[iClass][0] * np.ones((320, 320)),
                          aColorClasses[iClass][1] * np.ones((320, 320)),
                          aColorClasses[iClass][2] * np.ones((320, 320))))
        color_mask += mask * colors
    color_mask = cv2.resize(color_mask.transpose(
        1, 2, 0), (matFrame.shape[1], matFrame.shape[0]), cv2.INTER_NEAREST)
    colorizedFrame = 0.75 * matFrameCopied + 0.5 * color_mask

    freq = cv2.getTickFrequency() / 1000
    t, layersTimes = net.getPerfProfile()  # / freq;
    layersTimes /= freq
    label = "Inference time: " + str(layersTimes[0]) + " ms"
    cv2.putText(colorizedFrame, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0))
    return colorizedFrame

# Read input image
# matFrame = cv2.imread(IMAGE_FILE_2)
# if (matFrame is None):
#     print("Cannot open image file ", IMAGE_FILE_2)
#     sys.exit()


def main():
    key = -1
    iFrame = 0
    window1 = 'original_frame'
    window2 = 'segmented_video'
    window3 = 'original_frame_homography'
    window4 = 'original_frame_segmented'
    videoCapture = cv2.VideoCapture('robot3.mp4')
    homographyFileUtil = HomographyFileUtil()
    homographyFileUtil.open_read_mode()
    manualHomographyMatrix = homographyFileUtil.read_matrix(
        'manual_homography_matrix')
    if not videoCapture.isOpened():
        print("ERROR! Unable to open input video file ", VIDEO_FILE)
        return -1

    width = videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
    height = videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
    scaleFactor = 0.4
    resized_width = int(width * scaleFactor)
    resized_height = int(height * scaleFactor)
    cv2.namedWindow(window1, cv2.WINDOW_GUI_EXPANDED)
    cv2.namedWindow(window2, cv2.WINDOW_GUI_EXPANDED)
    cv2.namedWindow(window3, cv2.WINDOW_GUI_EXPANDED)
    cv2.namedWindow(window4, cv2.WINDOW_GUI_EXPANDED)
    cv2.resizeWindow(window1, (resized_width,resized_height))
    cv2.resizeWindow(window2, (resized_width, resized_height))
    cv2.resizeWindow(window3, (resized_width, resized_height))
    cv2.resizeWindow(window4, (resized_width, resized_height))
    iFrame = 0

    # while (True):
    #     _, matFrameCapture = videoCapture.read()
    #     if(matFrameCapture is None):
    #         break
    #     cv2.imshow(window1, matFrameCapture)
    #     colorizedFrame = segmentCapturedFrame(matFrameCapture)
    #     cv2.imshow(window2, colorizedFrame)

    #     key = -1
    #     while key != ord(' '):
    #         key = cv2.waitKey(10)
    #         if (cv2.getWindowProperty(VIDEO_FILE, cv2.WND_PROP_VISIBLE) == 0):
    #             return 0
    #         if (key == ord('q') or key == ord('Q')):
    #             return 0

    #     iFrame += 1

    while(videoCapture.isOpened()):

        # videoCaptureture frame-by-frame
        ret, frame = videoCapture.read()
        if ret == True:
            h, w, ch = frame.shape
            # Display the resulting frame
            cv2.imshow(window1, frame)
            colorizedFrame = segmentCapturedFrame(frame)
            cv2.imshow(window2, colorizedFrame)
            # Press Q on keyboard to  exit
            frameHomography = cv2.warpPerspective(
                frame, manualHomographyMatrix, (w, h), cv2.INTER_LINEAR)
            segmentedHomgraphy = cv2.warpPerspective(
                colorizedFrame, manualHomographyMatrix, (w, h), cv2.INTER_LINEAR)
            cv2.imshow(window3, frameHomography)
            cv2.imshow(window4, segmentedHomgraphy)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break
    videoCapture.release()
    # matFrame = cv2.imread(IMAGE_FILE_2)

    # if (matFrame is None):
    #     print("Cannot open image file ", IMAGE_FILE_2)
    #     sys.exit()
    # colorizedFrame = segmentCapturedFrame(matFrame)
    # cv2.imwrite('mask2.png', colorizedFrame)
    # cv2.waitKey(0)


main()
