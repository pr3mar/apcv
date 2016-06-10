
#ifndef _FEATURES
#define _FEATURES

#include <stdio.h>
#include <iostream>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

#define FEATURES_SIFT 1
#define FEATURES_SURF 2
#define FEATURES_FAST 3
#define FEATURES_BRISK 4
#define FEATURES_ORB 5
#define FEATURES_MSER 6
#define FEATURES_AKAZE 7

bool detect_template_marker(Mat image, Mat pattern, float pattern_size, Mat intrinsics, Mat distortion, Mat& rotation, Mat& translation);

void list_detectors();

void list_descriptors();

int decode_detector(int c);

int decode_descriptor(int c);

Ptr<FeatureDetector> get_detector(int id);

Ptr<Feature2D> get_descriptor(int id);

#endif
