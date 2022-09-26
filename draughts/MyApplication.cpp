#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>

#define BINARY_THRESHOLD 95

using namespace std;
using namespace cv;

void MyApplication() {
    VideoCapture cap("Media/StayingInLane_MPEG4.avi");

    if(!cap.isOpened()) 
    {
        cout << "Error opening video stream" << endl;
    }

    double fps = cap.get(CAP_PROP_FPS);

    namedWindow("Staying In Lane", WINDOW_AUTOSIZE);

    while(1) 
    {
        Mat frame;
        
        if(!cap.read(frame))
        {
            cout << "The Video has ended" << endl;
            break;
        }

        Mat greyscaleFrame(frame.size(), CV_8U);

        cvtColor(frame, greyscaleFrame, COLOR_BGR2GRAY);

        Mat binaryFrame(greyscaleFrame.size(), greyscaleFrame.type());

        threshold(greyscaleFrame, binaryFrame, BINARY_THRESHOLD, 255, THRESH_BINARY);

        imshow("Staying In lane", binaryFrame);

        if(waitKey(30) == 27) 
        {
            break;
        }
    }
}

int main(int argc, char const *argv[]) {
    MyApplication();

    return 0;
}
