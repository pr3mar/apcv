#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/calib3d.hpp>

#include "features.h"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

#define WINDOW_NAME "Homography matching"
#define TRACKBAR_NAME "Number of matches"
#define MAX_MATCHES_COUNT 1000

int algorithm = FEATURES_SIFT;
int matches_count = 80;

Mat img, pattern;

bool compare_matches(DMatch first, DMatch second) {

  return (first.distance < second.distance);

}

void do_match( int, void* ) {

  Mat visualization, temp, descriptors_img, descriptors_pattern;
  vector<KeyPoint> keypoints_img, keypoints_pattern;
  vector<DMatch> matches, good_matches;

  Ptr<Feature2D> descriptor = get_descriptor(algorithm);
  Ptr<DescriptorMatcher> matcher = new BFMatcher();

  descriptor->detectAndCompute(img, Mat(), keypoints_img, descriptors_img);
  descriptor->detectAndCompute(pattern, Mat(), keypoints_pattern, descriptors_pattern);

  matcher->match(descriptors_pattern, descriptors_img, matches);

  double max_dist = 0; double min_dist = 100;

  for( int i = 0; i < (int)matches.size(); i++ ) {
    double dist = matches[i].distance;
    if( dist < min_dist ) min_dist = dist;
    if( dist > max_dist ) max_dist = dist;
  }

  cout << format("Total matches: %d", matches.size()) << endl;
  cout << format("Match distances minimum=%f, maximum=%f", min_dist, max_dist) << endl;

  for( int i = 0; i < MIN(matches_count, (int)matches.size()); i++ ) { 
      good_matches.push_back( matches[i]);
  }

  std::vector<Point2f> srcPoints;
  std::vector<Point2f> dstPoints;

  sort(matches.begin(), matches.end(), compare_matches);

  if (good_matches.size() < 4) {
    cout << "At least four matches required to compute homography." << endl;
    return;
  }

  for( int i = 0; i < (int)good_matches.size(); i++ ) {
    srcPoints.push_back(keypoints_pattern[good_matches[i].queryIdx].pt);
    dstPoints.push_back(keypoints_img[good_matches[i].trainIdx].pt);
  }

  vector<int> inliers;
  Mat H = findHomography(srcPoints, dstPoints, CV_RANSAC, 3, inliers);

  std::vector<Point2f> pattern_corners(4);
  pattern_corners[0] = Point2f(0,0); pattern_corners[1] = Point2f(pattern.cols, 0);
  pattern_corners[2] = Point2f(pattern.cols, pattern.rows); pattern_corners[3] = Point2f(0, pattern.rows);
  std::vector<Point2f> img_corners(4);

  perspectiveTransform(pattern_corners, img_corners, H);

  img.copyTo(temp);
  pattern.copyTo(temp(Rect(0, 0, pattern.cols, pattern.rows)));

  cvtColor(temp, visualization, COLOR_GRAY2BGR);

  for (int i = 0; i < (int)good_matches.size(); i++ ) {
    Scalar color = inliers[i] == 0 ? Scalar(255, 0, 0) : Scalar(0, 0, 255);
    int width = inliers[i] == 0 ? 1 : 2;
    line(visualization, keypoints_pattern[good_matches[i].queryIdx].pt,
       keypoints_img[good_matches[i].trainIdx].pt, color, width);
  }

  line(visualization, img_corners[0], img_corners[1], Scalar(0, 255, 0), 4);
  line(visualization, img_corners[1], img_corners[2], Scalar(0, 255, 0), 4);
  line(visualization, img_corners[2], img_corners[3], Scalar(0, 255, 0), 4);
  line(visualization, img_corners[3], img_corners[0], Scalar(0, 255, 0), 4);

  imshow(WINDOW_NAME, visualization);

}

int main( int argc, char** argv )
{
  if (argc < 3)
   return -1;

  img = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);
  pattern = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);

  if (img.empty() || pattern.empty())
    return -1;

	namedWindow(WINDOW_NAME, CV_WINDOW_AUTOSIZE);

  createTrackbar(TRACKBAR_NAME, WINDOW_NAME, &matches_count, MAX_MATCHES_COUNT, do_match);

  cout << "Press space to rotate pattern for 90 degrees" << endl;

  list_descriptors();

  do_match(0, 0);

  while (true)
  {
    int c = waitKey(0);
    switch(c % 256) {
      case 'q':
      case 27: {
        return 0;
      }
      case ' ': {
        Mat temp;
        transpose(pattern, temp);  
        flip(temp, pattern, 1);
        break;
      }
      default: {
        algorithm = decode_descriptor(c % 256);
        break;
      }
    }
    do_match(0, 0);
  }

	return 0;

}


