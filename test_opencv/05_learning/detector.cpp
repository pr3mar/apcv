#include <cstdio>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "utilities.h"

using namespace std;
using namespace cv;

#define WINDOW_NAME "Cascade detector"

inline Size operator * (Size size, float f) {
  Size result(size);
  result.width *= f;
  result.height *= f;
  return result;
}

CascadeClassifier cascade;
Mat frame, gray;

Size reference_size(100, 40);

int main( int argc, char** argv ) {

  string cascade_file("cascade.xml");

  if (argc > 1)
    cascade_file = argv[1];

  if( !cascade.load(cascade_file) ) {
      cerr << "Could not load classifier cascade" << endl;
      return -1;
  }

  VideoCapture capture;

  if (argc > 2) 
    capture.open(argv[2]);
  else
    capture.open(0);

  if (!capture.isOpened())
    return -1;

  while (true) {
    capture >> frame;

    if (frame.empty())
        break;

    cvtColor(frame, gray, COLOR_BGR2GRAY);

    vector<Rect> regions;

    cascade.detectMultiScale(gray, regions,
      1.1, 3, CASCADE_FIND_BIGGEST_OBJECT | CASCADE_DO_ROUGH_SEARCH | CASCADE_SCALE_IMAGE,
      reference_size * 0.8, reference_size * 1.2);

    for (unsigned int i = 0; i < regions.size(); i++) {
        Rect r = regions[i];

        rectangle(frame, Point(round(r.x), round(r.y)),
                   Point(round((r.x + r.width-1)), round((r.y + r.height-1))),
                   Scalar(0, 0, 255), 3, 8, 0);
    }
    imshow(WINDOW_NAME, frame);

    int c = waitKey(0);
    switch(c % 256) {
      case 'q':
      case 27: {
        return 0;
      }
      default: {
        break;
      }
    }
  }

  return 0;
}



