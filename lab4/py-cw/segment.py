import cv2
import math
import sys
import numpy as np

IMAGE_FILE_1 = "video1_frame.png"
IMAGE_FILE_2 = "video2_frame.png"
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


def colorizeSegmentation():
    return True


nClasses = len(aColorClasses)

# Read CNN definition
net = cv2.dnn.readNetFromONNX(ONNX_NETWORK_DEFINITION)

# Read input image
matFrame = cv2.imread(IMAGE_FILE_2)
if (matFrame is None):
    print("Cannot open image file ", IMAGE_FILE_2)
    sys.exit()

# prepare blob for input image
# Query: mean value are applied before or after scaling to 0-1?
# from dnn.cpp lines 370-371 first the mean is subtracted then we divide by std deviation
# We know ImageNet does [(R, G, B)/255-(0.485, 0.456, 0.406)] / (0.229, 0.224, 0.225)
# ImageNet:   o = (color / 255 - mean) / std
# OpenCV:     o = (color - mean) * scalefactor
# mean must be (0.485, 0.456, 0.406) * 255
# scaleFactor must be 1.0/255.0/std where std is average of 0.229, 0.224, 0.225
# swapRB = True because the means are for RGB
# crop = False to resize the image without any crop ( distorting the aspect ratio but processing the whole image)

tensorFrame = cv2.dnn.blobFromImage(
    matFrame, IMAGENET_SCALE_FACTOR, (320, 320), IMAGENET_MEANS, True, False)
# print(tensorFrame.shape)

net.setInput(tensorFrame)
matScore = net.forward()
print('matscore shape', matScore.shape)

# Colorize the image and display
matColored = matFrame.copy()
scores = 1.0 / (1.0 + np.exp(-matScore.squeeze(0)))
# Temporarily ignore the "background" class as it seems we get that
classes = matScore.squeeze(0)[1:, :, :].argmax(0) + 1
print('classes shape', classes.shape)
color_mask = np.zeros((3, 10, 10))
for iClass in range(nClasses):
    mask = classes == iClass
    print(mask.shape)
    colors = np.array((aColorClasses[iClass][0] * np.ones((10, 10)),
                      aColorClasses[iClass][1] * np.ones((10, 10)),
                      aColorClasses[iClass][2] * np.ones((10, 10))))
    color_mask += mask * colors
color_mask = cv2.resize(color_mask.transpose(1,2,0),(matFrame.shape[1], matFrame.shape[0]), cv2.INTER_NEAREST)
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
cv2.putText(matColored, label, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0))

# Display
cv2.namedWindow(WINDOW_NAME, WINDOW_FLAGS)
cv2.imwrite('mask.png', colorizedFrame)
# cv2.imshow(WINDOW_NAME, matFrame)
cv2.waitKey(0)
