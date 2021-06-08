#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>

using namespace cv;
using namespace std;

void trackPoint(VideoCapture capture, Mat old_gray, vector<Point2f> p0, vector<Point2f> p1, Mat mask, vector<Scalar> colors)
{
    while (true)
    {
        Mat frame, frame_gray;
        capture >> frame;
        if (frame.empty())
        {
            cout << "frame break from here" << endl;
            break;
        }
        cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
        vector<uchar> status;
        vector<float> err;
        TermCriteria criteria = TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 10, 0.03);
        calcOpticalFlowPyrLK(old_gray, frame_gray, p0, p1, status, err, Size(30, 30), 2, criteria);
        vector<Point2f> good_new;
        for (uint i = 0; i < p0.size(); i++)
        {
            if (status[i] == 1)
            {
                good_new.push_back(p1[i]);
                line(mask, p1[i], p0[i], colors[i], 2);
                circle(frame, p1[i], 5, colors[i], -1);
            }
        }
        Mat img;
        cv::add(frame, mask, img);
        cv::imshow("Frame", img);
        int keyboard = waitKey(30);
        if (keyboard == 'q' || keyboard == 27)
            break;
        // Now update the previous frame and previous points
        old_gray = frame_gray.clone();
        p0 = good_new;
    }
}

int main(int argc, char **argv)
{
    const string keys =
        "{ h help |      | print this help message }"
        "{ @image | vtest.avi | path to image file }";
    CommandLineParser parser(argc, argv, keys);
    cout << parser.get<string>("@image") << endl;
    string name = parser.get<string>("@image");
    string filename = "";
    if (name != "vtest.avi")
    {
        filename = samples::findFile(parser.get<string>("@image"));
        if (!parser.check())
        {
            parser.printErrors();
            return 0;
        }
    }

    cv::VideoCapture capture;
    if (filename == "")
    {
        int deviceID = 0;
        int apiID = cv::CAP_ANY;
        capture.open(deviceID, apiID);
    }
    else
    {
        capture.open(filename);
    }

    if (!capture.isOpened())
    {
        cerr << "ERROR! Unable to open camera\n";
        return -1;
    }

    vector<Scalar> colors;
    RNG rng;

    for (int i = 0; i < 100; i++)
    {
        int r = rng.uniform(0, 256);
        int g = rng.uniform(0, 256);
        int b = rng.uniform(0, 256);
        colors.push_back(Scalar(r, g, b));
    }

    Mat old_frame, old_gray;
    vector<Point2f> p0, p1;

    capture >> old_frame;
    cvtColor(old_frame, old_gray, COLOR_BGR2GRAY);
    goodFeaturesToTrack(old_gray, p0, 100, 0.3, 7, Mat(), 7, false, 0.04);
    Mat mask = Mat::zeros(old_frame.size(), old_frame.type());

    try
    {
        trackPoint(capture, old_gray, p0, p1, mask, colors);
    }
    catch (Exception e)
    {
        cout << "caught an exception";
        goodFeaturesToTrack(old_gray, p0, 100, 0.3, 7, Mat(), 7, false, 0.04);
        Mat mask = Mat::zeros(old_frame.size(), old_frame.type());
        trackPoint(capture, old_gray, p0, p1, mask, colors);
    }
}