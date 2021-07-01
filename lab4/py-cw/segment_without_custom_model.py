import cv2
import math
import sys
import numpy as np

IMAGE_FILE_1 = "video1_frame.png"
IMAGE_FILE_2 = "video2_frame.png"
VIDEO_FILE = "robot3.mp4"
ONNX_NETWORK_DEFINITION = "fcn_resnet18.onnx"
IMAGENET_MEANS = (123.67, 116.28, 103.53)
IMAGENET_SCALE_FACTOR = 1.0 / 255.0 / (0.224 + 0.229 + 0.225 / 3)
WINDOW_NAME = 'getmented_frame'
WINDOW_FLAGS = cv2.WINDOW_GUI_EXPANDED

aStringClasses = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

aColorClasses = [
    (0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 255,
                                          120), (0, 0, 255), (255, 0, 255), (70, 70, 70),
    (102, 102, 156), (190, 153, 153), (180, 165,
                                       180), (150, 100, 100), (153, 153, 153),
    (250, 170, 30), (220, 220, 0), (107, 142, 35), (192,
                                                    128, 128), (70, 130, 180), (220, 20, 60),
    (0, 0, 142), (0, 0, 70), (119, 11, 32)
]


# Read CNN definition
net = cv2.dnn.readNetFromONNX(ONNX_NETWORK_DEFINITION)
nClasses = len(aColorClasses)


# def displayFrame(matFrameDisplay, iFrame, cFrames, pHomographyData):
#     for i in range(pHomographyData.cPoints):
#         cv2.circle(matFrameDisplay,
#                    pHomographyData.aPoints[i], 10, (255, 0, 0), 2, cv2.LINE_8, 0)

#     cv2.imshow(VIDEO_FILE, matFrameDisplay)
#     ss = "Frame " + str(iFrame) + "/" + str(cFrames)
#     ss += ": hit <space> for next frame or 'q' to quit"
#     # cv2.displayOverlay(VIDEO_FILE, ss, 0);  # for linux + qt
#     cv2.putText(matFrameDisplay, ss, (10, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)


def segmentCapturedFrame(matFrame):
    tensorFrame = cv2.dnn.blobFromImage(
        matFrame, IMAGENET_SCALE_FACTOR, (320, 320), IMAGENET_MEANS, True, False)
    # print(tensorFrame.shape)

    net.setInput(tensorFrame)
    matScore = net.forward()

    # Colorize the image and display
    matColored = matFrame.copy()
    scores = 1.0 / (1.0 + np.exp(-matScore.squeeze(0)))
    # Temporarily ignore the "background" class as it seems we get that
    classes = matScore.squeeze(0)[1:, :, :].argmax(0) + 1
    color_mask = np.zeros((3, 10, 10))
    for iClass in range(nClasses):
        mask = classes == iClass
        colors = np.array((aColorClasses[iClass][0] * np.ones((10, 10)),
                          aColorClasses[iClass][1] * np.ones((10, 10)),
                          aColorClasses[iClass][2] * np.ones((10, 10))))
        color_mask += mask * colors
    color_mask = cv2.resize(color_mask.transpose(
        1, 2, 0), (matFrame.shape[1], matFrame.shape[0]), cv2.INTER_NEAREST)
    colorizedFrame = 0.5 * matFrame + 0.5 * color_mask
    # print(np.ones((3,10,10)) * np.array(aColorClasses[iClass].reshape(3, 1, 1))
    # print(np.expand_dims(aColorClasses[iClass], 1) @ np.expand_dims(mask,0))
    # print('arg max classes', classes)
    # matColored = np.array(colorizedFrame, dtype=np.uint8) # colorizeSegmentation(matFrame, matScore, aColorClasses, aStringClasses, nClasses)

    # Add timing information
    freq = cv2.getTickFrequency() / 1000
    t, layersTimes = net.getPerfProfile()  # / freq;
    layersTimes /= freq
    # print('Inference time', layersTimes)
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
    videoCapture = cv2.VideoCapture('robot3.mp4')

    if not videoCapture.isOpened():
        print("ERROR! Unable to open input video file ", VIDEO_FILE)
        return -1

    width = videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
    height = videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
    cv2.namedWindow(window1, cv2.WINDOW_NORMAL |
                    cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
    cv2.namedWindow(window2, cv2.WINDOW_NORMAL |
                    cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)

    while(videoCapture.isOpened()):

        # videoCaptureture frame-by-frame
        ret, frame = videoCapture.read()
        if ret == True:

            # Display the resulting frame
            cv2.imshow(window1, frame)
            colorizedFrame = segmentCapturedFrame(frame)
            cv2.imshow(window2, colorizedFrame)
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break
    videoCapture.release()
    cv2.destoryAllWindows()


main()
