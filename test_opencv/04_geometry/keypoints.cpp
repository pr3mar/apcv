#include <stdio.h>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>

#include "features.h"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

#define WINDOW_NAME "Features"

int algorithm = FEATURES_SIFT;

void do_detect(Mat img) {

  Mat visualization;
  std::vector<KeyPoint> keypoints;
  Ptr<FeatureDetector> detector = get_detector(algorithm);

  detector->detect(img, keypoints);

  drawKeypoints(img, keypoints, visualization, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

  imshow(WINDOW_NAME, visualization);

}

int main( int argc, char** argv )
{
  if (argc < 2)
   return -1;

  Mat img = imread( argv[1], CV_LOAD_IMAGE_GRAYSCALE );

  if (img.empty())
    return -1;

	namedWindow(WINDOW_NAME, CV_WINDOW_AUTOSIZE);

  list_detectors();

  do_detect(img);

  while (true)
  {
    int c = waitKey(0);
    switch(c % 256) {
      case 'q':
      case 27: {
        return 0;
      }
      default: {
        algorithm = decode_detector(c % 256);
        break;
      }
    }
    do_detect(img);
  }

	return 0;

}


