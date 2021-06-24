#include <iostream>
#include <opencv2/opencv.hpp>
#include "HomographyData.cpp"

using namespace cv;
using namespace std;

#define VIDEO_FILE "robot.mp4"
#define HOMOGRAPHY_FILE "robot-homography.yml"

Mat matPauseScreen, matResult, matFinal;
Point point;
vector<Point> pts;
int var = 0;
int drag = 0;

// Create mouse handler function
void mouseHandler(int event, int x, int y, int, void *)
{
    if (var >= 4)
        return;
    if (event == EVENT_LBUTTONDOWN) // Left button down
    {
        drag = 1;
        matResult = matFinal.clone();
        point = Point(x, y);
        if (var >= 1)
        {
            line(matResult, pts[var - 1], point, Scalar(0, 255, 0, 255), 2);
        }
        circle(matResult, point, 2, Scalar(0, 255, 0), -1, 8, 0);
        imshow("Source", matResult);
    }
    if (event == EVENT_LBUTTONUP && drag) // When Press mouse left up
    {
        drag = 0;
        var++;
        pts.push_back(point);
        matFinal = matResult.clone();
        if (var >= 4)
        {
            line(matFinal, pts[0], pts[3], Scalar(0, 255, 0, 255), 2);
            fillPoly(matFinal, pts, Scalar(0, 120, 0, 20), 8, 0);

            setMouseCallback("Source", NULL, NULL);
        }
        imshow("Source", matFinal);
    }
    if (drag)
    {
        matResult = matFinal.clone();
        point = Point(x, y);
        if (var >= 1)
            line(matResult, pts[var - 1], point, Scalar(0, 255, 0, 255), 2);
        circle(matResult, point, 2, Scalar(0, 255, 0), -1, 8, 0);
        imshow("Source", matResult);
    }
}

int main()
{
    Mat matFrameCapture;
    Mat matFrameDisplay;
    int key = -1;

    VideoCapture videoCapture(VIDEO_FILE);
    if (!videoCapture.isOpened())
    {
        cerr << "ERROR! Unable to open input video file " << VIDEO_FILE << endl;
        return -1;
    }

    while (key < 0) // play video until press any key
    {
        // Get the next frame
        videoCapture.read(matFrameCapture);
        if (matFrameCapture.empty())
        {
            // End of video file
            break;
        }
        float ratio = 640.0 / matFrameCapture.cols;
        resize(matFrameCapture, matFrameDisplay, cv::Size(), ratio, ratio, INTER_LINEAR);

        imshow(VIDEO_FILE, matFrameDisplay);
        key = waitKey(30);

        if (key >= 0)
        {
            destroyWindow(VIDEO_FILE);
            matPauseScreen = matFrameCapture;
            matFinal = matPauseScreen.clone();

            namedWindow("Source", cv::WINDOW_GUI_EXPANDED);
            setMouseCallback("Source", mouseHandler, NULL);
            imshow("Source", matPauseScreen);
            waitKey(0);
            destroyWindow("Source");

            Point2f src[4];
            for (int i = 0; i < 4; i++)
            {
                src[i].x = pts[i].x * 1.0;
                src[i].y = pts[i].y * 1.0;
            }
            Point2f reals[4];
            reals[0] = Point2f(800.0, 800.0);
            reals[1] = Point2f(1000.0, 800.0);
            reals[2] = Point2f(1000.0, 1000.0);
            reals[3] = Point2f(800.0, 1000.0);

            Mat homography_matrix = getPerspectiveTransform(src, reals);
            std::cout << "Estimated Homography Matrix is:" << std::endl;
            std::cout << homography_matrix << std::endl;

            // perspective transform operation using transform matrix
            cv::warpPerspective(matPauseScreen, matResult, homography_matrix, matPauseScreen.size(), cv::INTER_LINEAR);
            imshow("Source", matPauseScreen);
            imshow("Result", matResult);

            // Write H to file

            HomographyData homographyData;
            for (int i = 0; i < 4; i++)
            {
                homographyData.aPoints[i] = src[i];
                homographyData.cPoints++;
            }
            homographyData.matH = homography_matrix;
            homographyData.widthOut = matPauseScreen.cols;
            homographyData.heightOut = matPauseScreen.rows;
            if (!homographyData.write(HOMOGRAPHY_FILE))
            {
                cerr << "ERROR! Unable to write homography data file " << HOMOGRAPHY_FILE << endl;
                return -1;
            }

            // HomographyData homographyData(HOMOGRAPHY_FILE);
            // if (!homographyData.cPoints == 0)
            // {
            //     cerr << "ERROR! Unable to read homography data file " << HOMOGRAPHY_FILE << endl;
            //     return -1;
            // }

            waitKey(0);
        }
    }

    return 0;
}