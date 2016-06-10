#include <opencv2/core/utility.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cstring>

#include "trackers.h"

using namespace std;
using namespace cv;

#define WINDOW_NAME "Object tracking"

#define TRACKER_NCC 0
#define TRACKER_CAMSHIFT 1
#define TRACKER_BOOSTING 2
#define TRACKER_MIL 3
#define TRACKER_KCF 4
#define TRACKER_MEDIANFLOW 5

int tracker_type = TRACKER_NCC;
Point2d point1(-1, -1), point2;
Rect2d initialization(0, 0, -1, -1);
Ptr<Tracker> tracker;
Rect2d region;
Mat frame;

static void on_mouse( int event, int x, int y, int flags, void* param) {

  if (event == EVENT_LBUTTONDOWN) {
    point1.x = (float)x;
    point1.y = (float)y;
  } else if (event == EVENT_MOUSEMOVE && point1.x > -1) {
    point2.x = (float)x;
    point2.y = (float)y;
    initialization = Rect2f(MIN(point1.x, point2.x), 
      MIN(point1.y, point2.y), MAX(point1.x, point2.x) - MIN(point1.x, point2.x), 
      MAX(point1.y, point2.y) - MIN(point1.y, point2.y));
  } else if (event == EVENT_LBUTTONUP) {
    if (initialization.width > 5 && initialization.height > 5) {

      switch (tracker_type) {
      case TRACKER_NCC:
        tracker = TrackerNCC::createTracker();
        break;
      case TRACKER_CAMSHIFT:
        tracker = TrackerCamShift::createTracker();
        break;
      case TRACKER_BOOSTING:
        tracker = TrackerBoosting::createTracker();
        break;
      case TRACKER_MIL:
        tracker = TrackerMIL::createTracker();
        break;
      case TRACKER_KCF:
        tracker = TrackerKCF::createTracker();
        break;
      case TRACKER_MEDIANFLOW:
        tracker = TrackerMedianFlow::createTracker();
        break;
      }

      tracker->init(frame, initialization);

    }
    point1.x = -1;
    point2.x = -1;
    initialization = Rect2f(0, 0, -1, -1);
  }
}

int main( int argc, char** argv ) {

  VideoCapture capture(0);

  if (!capture.isOpened())
    return -1;

  cout << "Press: \n * n: NCC tracker \n * c: CamShift tracker \n * b: Boosting tracker \n * m: MedianFlow tracker \n * i: MIL tracker \n * k: KCF tracker \n";

  namedWindow(WINDOW_NAME, WINDOW_NORMAL);
  setMouseCallback(WINDOW_NAME, on_mouse, 0);

  for (;;) {

    capture >> frame;

    if(frame.empty())
      break;

    int c = waitKey(10);
    switch(c % 256) {
      case 'q':
      case 27: {
        return 0;
      }
      case 'n': {
        tracker_type = TRACKER_NCC;
        tracker = Ptr<Tracker>();
        break;
      }
      case 'c': {
        tracker_type = TRACKER_CAMSHIFT;
        tracker = Ptr<Tracker>();
        break;
      }
      case 'k': {
        tracker_type = TRACKER_KCF;
        tracker = Ptr<Tracker>();
        break;
      }
      case 'm': {
        tracker_type = TRACKER_MEDIANFLOW;
        tracker = Ptr<Tracker>();
        break;
      }
      case 'b': {
        tracker_type = TRACKER_BOOSTING;
        tracker = Ptr<Tracker>();
        break;
      }
      case 'i': {
        tracker_type = TRACKER_MIL;
        tracker = Ptr<Tracker>();
        break;
      }
      case ' ': {
        tracker = Ptr<Tracker>();
        break;
      }
    }

    if (!tracker.empty()) {
      tracker->update(frame, region);
      rectangle(frame, region, Scalar(255, 0, 0), 2, 1);
    }

    if (initialization.width > -1) 
      rectangle(frame, initialization, Scalar(255, 255, 255), 3, 1 );

    string tracker_name;

    switch (tracker_type) {
    case TRACKER_NCC:
      tracker_name = "NCC";
      break;
    case TRACKER_CAMSHIFT:
      tracker_name = "CamShift";
      break;
    case TRACKER_BOOSTING:
      tracker_name = "Online Boosting";
      break;
    case TRACKER_MIL:
      tracker_name = "MIL";
      break;
    case TRACKER_KCF:
      tracker_name = "KCF";
      break;
    case TRACKER_MEDIANFLOW:
      tracker_name = "Median Flow";
      break;
    }

    putText(frame, tracker_name, Point(20, 20), FONT_HERSHEY_SIMPLEX, 0.5,
      Scalar(0, 0, 255), 1, 8);

    imshow(WINDOW_NAME, frame);

  }

}

