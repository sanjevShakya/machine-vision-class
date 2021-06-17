#include <opencv2/opencv.hpp>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

// In C++, you can define constants variable using #define
// #define VIDEO_FILE "robot.mp4"
#define ROTATE false

struct CloneWindow
{
    cv::Mat matPauseScreen, matResult, matFinal;
    vector<Point> pts;
    Point point;
    int var;
    int drag;
    string cloneWindowName;
};

void mouseHandler(int, int, int, int, void *);

void mouseHandler(int event, int x, int y, int, void *params)
{
    CloneWindow *pState = (CloneWindow *)params;
    // cout << "pState" << pState;

    if (pState->var >= 4) // If we already have 4 points, do nothing
        return;
    if (event == EVENT_LBUTTONDOWN) // Left button down
    {
        pState->drag = 1;                     // Set it that the mouse is in pressing down mode
        pState->matResult = pState->matFinal.clone(); // copy final image to draw image
        pState->point = Point(x, y);          // memorize current mouse position to point pState->var
        if (pState->var >= 1)                 // if the point has been added more than 1 points, draw a line
        {
            line(pState->matResult, pState->pts[pState->var - 1], pState->point, Scalar(0, 255, 0, 255), 2); // draw a green line with thickness 2
        }
        circle(pState->matResult, pState->point, 2, Scalar(0, 255, 0), -1, 8, 0); // draw a current green point
        imshow(pState->cloneWindowName, pState->matResult);               // show the current drawing
    }
    if (event == EVENT_LBUTTONUP && pState->drag) // When Press mouse left up
    {
        pState->drag = 0;                     // no more mouse pState->drag
        pState->pts.push_back(pState->point);         // add the current point to pState->pts
        pState->var++;                        // increase point number
        pState->matFinal = pState->matResult.clone(); // copy the current drawing image to final image
        if (pState->var >= 4)                 // if the homograpy points are done
        {
            line(pState->matFinal, pState->pts[0], pState->pts[3], Scalar(0, 255, 0, 255), 2); // draw the last line
            fillPoly(pState->matFinal, pState->pts, Scalar(0, 120, 0, 20), 8, 0);      // draw polygon from points

            setMouseCallback(pState->cloneWindowName, NULL, NULL); // remove mouse event handler
        }
        imshow(pState->cloneWindowName, pState->matFinal);
    }
    if (pState->drag) // if the mouse is dragging
    {
        pState->matResult = pState->matFinal.clone(); // copy final images to draw image
        pState->point = Point(x, y);          // memorize current mouse position to point pState->var
        if (pState->var >= 1)                 // if the point has been added more than 1 points, draw a line
        {
            line(pState->matResult, pState->pts[pState->var - 1], pState->point, Scalar(0, 255, 0, 255), 2); // draw a green line with thickness 2
        }
        circle(pState->matResult, pState->point, 2, Scalar(0, 255, 0), -1, 8, 0); // draw a current green point
        imshow(pState->cloneWindowName, pState->matResult);               // show the current drawing
    }
}

int main(int argc, char **argv)
{
    int var = 0;
    CloneWindow cw;
    cw.var = 0;
    cw.drag = 0;
    cw.cloneWindowName = "clone_window_name";

    CloneWindow *pState;
    pState = &cw;

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

    Mat matFrameCapture;
    Mat matFrameDisplay;
    int iKey = -1;

    // Open input video file
    VideoCapture videoCapture(filename);
    if (!videoCapture.isOpened())
    {
        cerr << "ERROR! Unable to open input video file " << filename << endl;
        return -1;
    }
    string window_name = "window_name";
    cv::namedWindow(window_name, cv::WINDOW_GUI_EXPANDED);
    cv::namedWindow(cw.cloneWindowName, cv::WINDOW_GUI_EXPANDED);

    // Capture loop
    int key = -1;
    while (key < 0) // play video until user presses <space>
    {
        // Get the next frame
        videoCapture.read(matFrameCapture);
        if (matFrameCapture.empty())
        {
            // End of video file
            break;
        }

        // We can rotate the image easily if needed.
#if ROTATE
        rotate(matFrameCapture, matFrameDisplay, RotateFlags::ROTATE_180); //rotate 180 degree and put the image to matFrameDisplay
#else
        matFrameDisplay = matFrameCapture;
#endif

        float ratio = 480.0 / matFrameDisplay.rows;
        resize(matFrameDisplay, matFrameDisplay, cv::Size(), ratio, ratio, INTER_LINEAR); // resize image to 480p for showing

        // Display
        imshow(window_name, matFrameDisplay); // Show the image in window named "robot.mp4"
        key = waitKey(30);
        // if (iKey != int(' '))
        // {
        //     waitKey(0);
        // }
        if (key >= 0)
        {
            cw.matPauseScreen = matFrameCapture;     // transfer the current image to process
            cw.matFinal = cw.matPauseScreen.clone(); // clone image to final image
        }
        if (key == int('q'))
        {
            break;
        }
    }
    // --------------------- [STEP 3: use mouse handler to select 4 points] ---------------------
    if (!matFrameCapture.empty())
    {
        var = 0;                                                    // reset number of saving points
        cw.pts.clear();                                             // reset all points
        namedWindow(cw.cloneWindowName, WINDOW_AUTOSIZE);           // create a windown named source
        setMouseCallback(cw.cloneWindowName, mouseHandler, pState); // set mouse event handler "mouseHandler" at Window "Source"
        imshow(cw.cloneWindowName, cw.matPauseScreen);              // Show the image
        waitKey(0);                                                 // wait until press anykey
        destroyWindow(cw.cloneWindowName);                          // destroy the window
    }
    else
    {
        cout << "You did not pause the screen before the video finish, the program will stop" << endl;
        return 0;
    }

    if (cw.pts.size() == 4)
    {
        Point2f src[4];
        for (int i = 0; i < 4; i++)
        {
            src[i].x = cw.pts[i].x * 1.0;
            src[i].y = cw.pts[i].y * 1.0;
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
        cv::warpPerspective(cw.matPauseScreen, cw.matResult, homography_matrix, cw.matPauseScreen.size(), cv::INTER_LINEAR);
        imshow("Source", cw.matPauseScreen);
        imshow("Result", cw.matResult);

        waitKey(0);
    }

    return 0;
}

// int main(int argc, char **argv)
// {
//     Mat matFrameCapture;
//     Mat matFrameDisplay;
//     int key = -1;

//     // --------------------- [STEP 1: Make video capture from file] ---------------------
//     // Open input video file
//     VideoCapture videoCapture(VIDEO_FILE);
//     if (!videoCapture.isOpened())
//     {
//         cerr << "ERROR! Unable to open input video file " << VIDEO_FILE << endl;
//         return -1;
//     }

//     // Capture loop
//     while (key < 0) // play video until press any key
//     {
//         // Get the next frame
//         videoCapture.read(matFrameCapture);
//         if (matFrameCapture.empty())
//         { // no more frame capture from the video
//             // End of video file
//             break;
//         }
//         cvtColor(matFrameCapture, matFrameCapture, COLOR_BGR2BGRA);

//         // Rotate if needed, some video has output like top go down, so we need to rotate it
// #if ROTATE
//         rotate(matFrameCapture, matFrameCapture, RotateFlags::ROTATE_180); //rotate 180 degree and put the image to matFrameDisplay
// #endif

//         float ratio = 640.0 / matFrameCapture.cols;
//         resize(matFrameCapture, matFrameDisplay, cv::Size(), ratio, ratio, INTER_LINEAR);

//         // Display
//         imshow(VIDEO_FILE, matFrameDisplay); // Show the image in window named "robot.mp4"
//         key = waitKey(30);

//         // --------------------- [STEP 2: pause the screen and show an image] ---------------------
//         if (key >= 0)
//         {
//             matPauseScreen = matFrameCapture;  // transfer the current image to process
//             matFinal = matPauseScreen.clone(); // clone image to final image
//         }
//     }

//     // --------------------- [STEP 3: use mouse handler to select 4 points] ---------------------
//     if (!matFrameCapture.empty())
//     {
//         var = 0;                                        // reset number of saving points
//         pts.clear();                                    // reset all points
//         namedWindow("Source", WINDOW_AUTOSIZE);         // create a windown named source
//         setMouseCallback("Source", mouseHandler, NULL); // set mouse event handler "mouseHandler" at Window "Source"
//         imshow("Source", matPauseScreen);               // Show the image
//         waitKey(0);                                     // wait until press anykey
//         destroyWindow("Source");                        // destroy the window
//     }
//     else
//     {
//         cout << "You did not pause the screen before the video finish, the program will stop" << endl;
//         return 0;
//     }

//     return 0;
// }