#include <iostream>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/video.hpp>

#include "models.h"

using namespace std;
using namespace cv;

#define WINDOW_NAME "Motion models"
#define TRACKBAR_NAME_MEASUREMENT "Measurement noise"
#define TRACKBAR_NAME_PROCESS "Process noise"
#define MAX_NOISE_VALUE 1000

#define DRAW_CROSS(image, center, color, d) \
  line(image, Point(center.x - d, center.y - d), \
  Point(center.x + d, center.y + d), color, 1, CV_AA, 0); \
  line(image, Point( center.x + d, center.y - d), \
  Point(center.x - d, center.y + d), color, 1, CV_AA, 0)

bool dragging = false;
Mat image;
Ptr<KalmanFilter> filter;
vector<Point> measurements;
vector<Point> predictions;
vector<Point> states;
int measurement_noise_value = 10, process_noise_value = 1;

static void recalculate_system() {

  if (measurements.empty())
    return;

  states.clear();
  predictions.clear();

  filter->statePost.at<float>(0) = measurements[0].x;
	filter->statePost.at<float>(1) = measurements[0].y;

  Mat_<float> measurement(2,1); measurement.setTo(Scalar(0));

  for (unsigned int i = 1; i < measurements.size(); i++) {
    Mat prediction = filter->predict();

    measurement(0) = measurements[i].x;
    measurement(1) = measurements[i].y;

    Mat estimated = filter->correct(measurement);
    Point state(estimated.at<float>(0), estimated.at<float>(1));

    predictions.push_back(Point(prediction.at<float>(0), prediction.at<float>(1)));
    states.push_back(state);

  }

}

static void process_measurement(Point p) {

  Mat prediction = filter->predict();
  Mat_<float> measurement(2,1); measurement.setTo(Scalar(0));
 
  measurement(0) = p.x;
  measurement(1) = p.y;

  Mat estimated = filter->correct(measurement);
  Point state(estimated.at<float>(0), estimated.at<float>(1));

  states.push_back(state);
  measurements.push_back(p);
  predictions.push_back(Point(prediction.at<float>(0), prediction.at<float>(1)));

}

static void draw_system() {

  image.setTo(0);

  if (measurements.empty())
    return;

  for (unsigned int i = 0; i < states.size()-1; i++) {
		line(image, states[i], states[i+1], Scalar(255,255,0), 1);
	}

  for (unsigned int i = 0; i < measurements.size()-1; i++) {
		line(image, measurements[i], measurements[i+1], Scalar(0,255,0), 1);
	}

  DRAW_CROSS(image, states[states.size()-1], Scalar(255,255,255), 2);
  DRAW_CROSS(image, measurements[measurements.size()-1], Scalar(0,0,255), 2);

}

void update_noise(int, void*) {
  filter = motion_model_ncc(measurement_noise_value, process_noise_value);
  
  recalculate_system();
  draw_system();
}


static void on_mouse( int event, int x, int y, int flags, void* param) {

  if (event == EVENT_LBUTTONDOWN) {
    measurements.clear();
    states.clear();
    predictions.clear();
    filter->statePost.at<float>(0) = x;
  	filter->statePost.at<float>(1) = y;
    dragging = true;
    draw_system();
  } else if (event == EVENT_MOUSEMOVE && dragging) {
    process_measurement(Point(x, y));
    draw_system();
  } else if (event == EVENT_LBUTTONUP) {
    dragging = false;
  }
}

int main( int argc, char** argv ) {

  image = Mat(400, 600, CV_8UC3);
  image.setTo(0);

  filter = motion_model_ncc(measurement_noise_value, process_noise_value);

  namedWindow(WINDOW_NAME, WINDOW_NORMAL);
  createTrackbar(TRACKBAR_NAME_MEASUREMENT, WINDOW_NAME, &measurement_noise_value, MAX_NOISE_VALUE, update_noise);
  createTrackbar(TRACKBAR_NAME_PROCESS, WINDOW_NAME, &process_noise_value, MAX_NOISE_VALUE, update_noise);
  setMouseCallback(WINDOW_NAME, on_mouse, 0);

  for (;;) {

    int c = waitKey(10);
    switch(c % 256) {
      case 'q':
      case 27: {
        return 0;
      }
    }

    imshow(WINDOW_NAME, image);

  }

}

