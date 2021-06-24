#include <opencv2/opencv.hpp>

class HomographyData
{
public:
    cv::Mat matH;
    int widthOut;
    int heightOut;
    int cPoints;
    cv::Point2f aPoints[4];

    HomographyData();
    HomographyData(std::string homography_file);

    bool read(std::string homography_file);
    bool write(std::string homography_file);
};