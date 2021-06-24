#include "HomographyData.h"

HomographyData::HomographyData(std::string homography_file)
{
    read(homography_file);
}

HomographyData::HomographyData()
{
    cPoints = 0;
}

bool HomographyData::read(std::string homography_file)
{
    cv::FileStorage fileStorage(homography_file, cv::FileStorage::Mode::READ);
    if (!fileStorage.isOpened()) {
        return false;
    }
    cv::FileNode points = fileStorage["aPoints"];
    cv::FileNodeIterator it = points.begin(), it_end = points.end();
    cPoints = 0;
    for (int i = 0; it != it_end; it++, i++) {
        (*it) >> aPoints[i];
        cPoints++;
    }
    fileStorage["matH"] >> matH;
    fileStorage["widthOut"] >> widthOut;
    fileStorage["heightOut"] >> heightOut;
    fileStorage.release();
    return true;
}

bool HomographyData::write(std::string homography_file)
{
    cv::FileStorage fileStorage(homography_file, cv::FileStorage::Mode::WRITE);
    if (!fileStorage.isOpened()) {
        return false;
    }

    fileStorage << "aPoints" << "[";
    for (int i = 0; i < 4; i++)
    {
        fileStorage << aPoints[i];
    }
    fileStorage << "]";
    fileStorage << "matH" << matH;
    fileStorage << "widthOut" << widthOut;
    fileStorage << "heightOut" << heightOut;
    fileStorage.release();
    return true;
}