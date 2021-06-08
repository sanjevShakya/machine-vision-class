#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main()
{
    double point1[] = {2, 4, 2};
    double point2[] = {6, 3, 3};
    double point3[] = {1, 2, 0.5};
    double point4[] = {16, 8, 4};
    double testLine[] = {8, -4, 0};

    cv::Mat point1Mat(3, 1, CV_64F, point1);
    cv::Mat point2Mat(3, 1, CV_64F, point2);
    cv::Mat point3Mat(3, 1, CV_64F, point3);
    cv::Mat point4Mat(3, 1, CV_64F, point4);
    cv::Mat testLineMat(3, 1, CV_64F, testLine);

    cv::Mat result1 = point1Mat.t() * testLineMat;
    cv::Mat result2 = point2Mat.t() * testLineMat;
    cv::Mat result3 = point3Mat.t() * testLineMat;
    cv::Mat result4 = point4Mat.t() * testLineMat;

    cv::Mat points[] = {point1Mat, point2Mat, point3Mat, point4Mat};
    cv::Mat results[] = {result1, result2, result3, result4};

    for (int i = 0; i < 4; i++)
    {
        if (fabs(results[i].at<double>(0, 0)) < 1e-6)
        {
            cout << "Point " << points[i].t() << " is on line" << testLineMat.t() << " (P' * L = " << results[i].at<double>(0, 0) << ")" << endl;
        }
    }

    return 0;
}