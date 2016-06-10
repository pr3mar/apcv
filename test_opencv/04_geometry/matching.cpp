#include <cstdio>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>

#include "features.h"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

#define WINDOW_NAME "Matching"

int algorithm = FEATURES_SIFT;

void do_match(Mat img, Mat pattern) {

  Mat visualization, descriptors_img, descriptors_pattern;
  vector<KeyPoint> keypoints_img, keypoints_pattern;
  vector<DMatch> matches;

  Ptr<Feature2D> descriptor = get_descriptor(algorithm);
  Ptr<DescriptorMatcher> matcher = new BFMatcher();

  descriptor->detectAndCompute(img, Mat(), keypoints_img, descriptors_img);
  descriptor->detectAndCompute(pattern, Mat(), keypoints_pattern, descriptors_pattern);

  matcher->match(descriptors_pattern, descriptors_img, matches);

  drawMatches(pattern, keypoints_pattern, img, keypoints_img, matches, visualization);
  imshow(WINDOW_NAME, visualization);

}

int main( int argc, char** argv )
{
  if (argc < 3)
   return -1;

  Mat img = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);
  Mat pattern = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);

  if (img.empty() || pattern.empty())
    return -1;

	namedWindow(WINDOW_NAME, CV_WINDOW_AUTOSIZE);

  list_descriptors();

  do_match(img, pattern);

  while (true)
  {
    int c = waitKey(0);
    switch(c % 256) {
      case 'q':
      case 27: {
        return 0;
      }
      default: {
        algorithm = decode_descriptor(c % 256);
        break;
      }
    }
    do_match(img, pattern);
  }

	return 0;

}


