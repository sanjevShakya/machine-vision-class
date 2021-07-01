
#include <opencv2/opencv.hpp>
#include <iostream>
#include "HomographyData.cpp"

using namespace cv;
using namespace std;

#define VIDEO_FILE "robot3.mp4"
#define HOMOGRAPHY_FILE "robot-homography.yml"

#define H_BIN 64
#define S_BIN 32
#define V_BIN 16
#define H_RS_BIT 6
#define S_RS_BIT 5
#define V_RS_BIT 4
#define GRAY_PIXEL 200

void displayFrame(cv::Mat &matFrameDisplay, int iFrame, int cFrames, HomographyData homographyData)
{
    for (int i = 0; i < homographyData.cPoints; i++)
    {
        cv::circle(matFrameDisplay, homographyData.aPoints[i], 10, cv::Scalar(255, 0, 0), 2, cv::LINE_8, 0);
    }
    imshow(VIDEO_FILE, matFrameDisplay);
    stringstream ss;
    ss << "Frame " << iFrame << "/" << cFrames;
    ss << ": hit <space> for next frame or 'q' to quit";
    // cv::displayOverlay(VIDEO_FILE, ss.str(), 0);  // for linux + qt
    putText(matFrameDisplay, ss.str(), Point(10, 30), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 255), 3);
}

void segmentFrame(cv::Mat &matFrame, double aProbFloorHS[H_BIN][S_BIN][V_BIN], double aProbNonFloorHS[H_BIN][S_BIN][V_BIN])
{
    cv::Mat matFrameHSV;
    cvtColor(matFrame, matFrameHSV, COLOR_BGR2HSV);

    for (int i = 0; i < matFrame.rows; i++)
    {
        for (int j = 0; j < matFrame.cols; j++)
        {
            Vec3b hsv = matFrameHSV.at<Vec3b>(i, j);
            double probHSVFloor = aProbFloorHS[hsv[0] >> H_RS_BIT][hsv[1] >> S_RS_BIT][hsv[2] >> V_RS_BIT];
            double probHSVgivenNonFloor = aProbNonFloorHS[hsv[0] >> H_RS_BIT][hsv[1] >> S_RS_BIT][hsv[2] >> V_RS_BIT];

            if (probHSVFloor < probHSVgivenNonFloor)
            {
                // cout << "here inside" << endl;
                Vec3b &BGR = matFrame.at<Vec3b>(i, j);
                BGR[0] = (int)(0.5 * BGR[0] + 0.5 * 0);
                BGR[1] = (int)(0.5 * BGR[1] + 0.5 * 0);
                BGR[2] = (int)(0.5 * BGR[2] + 0.5 * 255);
            }
        }
    }
}

// Read mask files and images and use them to update a floor and non-floor model
void getHistogram(double aProbFloorHS[H_BIN][S_BIN][V_BIN], double aProbNonFloorHS[H_BIN][S_BIN][V_BIN])
{
    std::vector<std::string> frames{"frame-001.png", "frame-002.png", "frame-003.png", "frame-004.png", "frame-005.png", "frame-006.png", "frame-007.png", "frame-008.png", "frame-009.png", "frame-010.png", "frame-011.png", "frame-012.png", "frame-013.png", "frame-014.png", "frame-015.png", "frame-016.png", "frame-017.png", "frame-018.png", "frame-019.png", "frame-020.png", "frame-022.png", "frame-023.png", "frame-024.png", "frame-025.png", "frame-026.png", "frame-027.png", "frame-028.png", "frame-029.png"};
    std::vector<std::string> mask_frames{"frame-001-mask.png", "frame-002-mask.png", "frame-003-mask.png", "frame-004-mask.png", "frame-005-mask.png", "frame-006-mask.png", "frame-007-mask.png", "frame-008-mask.png", "frame-009-mask.png", "frame-010-mask.png", "frame-011-mask.png", "frame-012-mask.png", "frame-013-mask.png", "frame-014-mask.png", "frame-015-mask.png", "frame-016-mask.png", "frame-017-mask.png", "frame-018-mask.png", "frame-019-mask.png", "frame-020-mask.png", "frame-022-mask.png", "frame-023-mask.png", "frame-024-mask.png", "frame-025-mask.png", "frame-026-mask.png", "frame-027-mask.png", "frame-028-mask.png", "frame-029-mask.png"};

    int nImages = frames.size();
    cout << "N images" << nImages << endl;
    for (int iImage = 0; iImage < nImages; iImage++)
    {
        cv::Mat matFrame = cv::imread(frames[iImage]);
        cv::Mat matMask = cv::imread(mask_frames[iImage]);
        cv::Mat matFrameHSV;
        cvtColor(matFrame, matFrameHSV, COLOR_BGR2HSV);
        for (int iRow = 0; iRow < matFrame.rows; iRow++)
        {
            for (int iCol = 0; iCol < matFrame.cols; iCol++)
            {
                Vec3b HSV = matFrameHSV.at<Vec3b>(iRow, iCol);
                Vec3b MaskRGB = matMask.at<Vec3b>(iRow, iCol);
                int Hindex = HSV[0] >> H_RS_BIT;
                int Sindex = HSV[1] >> S_RS_BIT;
                int Vindex = HSV[2] >> V_RS_BIT;
                int probValue = 1;

                // if(iRow < matFrame.rows / 1.5) {
                //     probValue = 0;
                // } else {
                //     probValue = 1;
                // }

                if (MaskRGB[0] > GRAY_PIXEL)
                {
                    aProbFloorHS[Hindex][Sindex][Vindex] += probValue;
                }
                else
                {
                    aProbNonFloorHS[Hindex][Sindex][Vindex] += probValue;
                }
            }
        }
        int scaleH = matFrame.rows >> 4;
        int scaleW = matFrame.cols >> 4;
        double area = scaleH * scaleW * 1.0;
        for (int iH = 0; iH < H_BIN; iH++)
        {
            for (int iS = 0; iS < S_BIN; iS++)
            {
                for (int iV = 0; iV < V_BIN; iV++)
                {
                    aProbFloorHS[iH][iS][iV] = aProbFloorHS[iH][iS][iV] / area;
                    aProbNonFloorHS[iH][iS][iV] = 1 - aProbFloorHS[iH][iS][iV];
                }
            }
        }
    }
}

int main()
{
    cv::Mat matFrameCapture;
    cv::Mat matFrameDisplay;
    cv::Mat matHomographyDisplay;
    int cFrames;
    double aProbFloorHS[H_BIN][S_BIN][V_BIN] = {0};
    double aProbNonFloorHS[H_BIN][S_BIN][V_BIN] = {0};
    cout << "Address at main" << aProbFloorHS << endl;

    getHistogram(aProbNonFloorHS, aProbNonFloorHS);

    HomographyData homographyData;

    cv::VideoCapture videoCapture(VIDEO_FILE);
    if (!videoCapture.isOpened())
    {
        cerr << "ERROR! Unable to open input video file " << VIDEO_FILE << endl;
        return -1;
    }
    cFrames = (int)videoCapture.get(cv::CAP_PROP_FRAME_COUNT);

    // Create a named window that will be used later to display each frame
    cv::namedWindow(VIDEO_FILE, (unsigned int)cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO | cv::WINDOW_GUI_EXPANDED);
    cv::namedWindow("homography_result", (unsigned int)cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO | cv::WINDOW_GUI_EXPANDED);
    // Read homography from file
    if (!homographyData.read(HOMOGRAPHY_FILE))
    {
        cerr << "ERROR! Unable to read homography data file " << HOMOGRAPHY_FILE << endl;
        return -1;
    }

    int iFrame = 0;
    // cout << *aProbNonFloorHS << endl;
    while (true)
    {

        // Block for next frame

        videoCapture.read(matFrameCapture);
        if (matFrameCapture.empty())
        {
            // End of video file
            break;
        }

        segmentFrame(matFrameCapture, aProbFloorHS, aProbNonFloorHS);
        // imshow("someother-window", matFrameCapture);
        displayFrame(matFrameCapture, iFrame, cFrames, homographyData);
        // displayFrame(matFrameCapture, iFrame, cFrames, &homographyData);

        cv::warpPerspective(matFrameCapture, matHomographyDisplay, homographyData.matH, matFrameCapture.size(), INTER_LINEAR);
        imshow("homography_result", matHomographyDisplay);
        int iKey;
        do
        {
            iKey = cv::waitKey(10);
            if (getWindowProperty(VIDEO_FILE, cv::WND_PROP_VISIBLE) == 0)
            {
                return 0;
            }
            if (iKey == (int)'q' || iKey == (int)'Q')
            {
                return 0;
            }
        } while (iKey != (int)' ');
        iFrame++;
    }

    return 0;
}