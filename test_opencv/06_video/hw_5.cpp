#include <iostream>
#include <ctype.h>
#include <opencv2/video.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;

#define WINDOW_NAME "Optical flow"
#define MAX_COUNT 500
#define MAX_MAGNITUDE 10

vector<Point2f> points_now, points_previous;

Point2f nUp(0.0, 1.0), nDown(0.0, -1.0), nRight(1.0, 0.0), nLeft(-1.0, 0);

Mat frame;

static void on_mouse( int event, int x, int y, int flags, void* param) {

    if( event == EVENT_LBUTTONDOWN )
        points_now.push_back(Point2f((float)x, (float)y));
}

Point2f apply_transformation(Point2f move, Point2f drawing)
{
	Point2f tmp = drawing + move;
	Point2f move2(0.0, 0.0);
	if (tmp.x <= 0 && tmp.y <= 0) { // top left
		move2 = -move;
	}
	else if (tmp.x >= frame.cols && tmp.y >= frame.rows) { // bottom right
		move2 = -move;
	}
	else if (tmp.x >= frame.cols && tmp.y <= 0) { // top right
		move2 = -move;
	}
	else if (tmp.x <= 0 && tmp.y >= frame.rows) { // bottom left
		move2 = -move;
	}
	else if (tmp.x > 0 && tmp.x < frame.cols && tmp.y < 0) { // top
		move2 = move - 2 * move.dot(nUp) * nUp;
	}
	else if (tmp.x > 0 && tmp.x < frame.cols && tmp.y >= frame.rows) { // bottom
		move2 = move - 2 * move.dot(nDown) * nDown;
	}
	else if (tmp.y > 0 && tmp.y < frame.rows && tmp.x < 0) { // left
		move2 = move - 2 * move.dot(nLeft) * nLeft;
	}
	else if (tmp.y > 0 && tmp.y < frame.rows && tmp.x >= frame.cols) { // right
		move2 = move - 2 * move.dot(nRight) * nRight;
	}
	else { //if (tmp.y > 0 && tmp.y < frame.rows && tmp.x < 0 && tmp.x >= frame.cols) { // OK
		move2 = move2;
	}
	tmp = drawing + move + move2;
	return tmp;
}

int main( int argc, char** argv ) {

    TermCriteria termcrit(TermCriteria::COUNT | TermCriteria::EPS, 20, 0.03);
    Size subPixWinSize(10, 10), winSize(31, 31);

    VideoCapture capture(0);

    if( !capture.isOpened() )
        return -1;
    
    cout << "Press r to initialize track-points automatically on good positions, c to clear the track-point set." << endl;
    cout << "Click on the image to initialize a track point manually." << endl;

    namedWindow(WINDOW_NAME, 1);
    //setMouseCallback(WINDOW_NAME, on_mouse, 0);

    Mat gray_now, gray_previous, image;
	capture >> frame;
	points_now = { Point2f(frame.cols / 2, frame.rows / 2) };
	Point2f move, drawing(frame.cols / 2, frame.rows / 2);
	double magnitude;

    while(true)
    {
        capture >> frame;
		flip(frame, frame, 1);
        if( frame.empty() )
            break;

        cvtColor(frame, gray_now, COLOR_BGR2GRAY);

        if( !points_previous.empty() && !gray_previous.empty() )
        {
            vector<uchar> status;
            vector<float> err;

            calcOpticalFlowPyrLK(gray_previous, gray_now, points_previous, points_now, status, err, winSize, 3, termcrit, 0, 0.001);

            for(unsigned int i = 0; i < points_now.size(); i++) {
				if (!status[i]) {
					points_now = { Point2f(frame.cols / 2, frame.rows / 2) };
					break;
				}
				move = points_now[i] - points_previous[i];
				drawing = apply_transformation(move, drawing);
            }
		}
		if(drawing.x < frame.cols)

		circle(frame, drawing, 10, Scalar(0, 255, 0), -1, 8);
        cv::imshow(WINDOW_NAME, frame);

    int c = waitKey(10);
        switch(c % 256)
        {
			case 27: case 'q':
			  return 0;
        }

        std::swap(points_now, points_previous);
        cv::swap(gray_now, gray_previous);
    }

    return 0;
}

