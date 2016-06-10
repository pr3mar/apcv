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
int matches_count = 50;

Size2f camera_size(8, 6);
Mat img, pattern, intrinsics, distortion, map1, map2, rectified, visualization;
float pattern_size = 50;
bool compare_matches(DMatch first, DMatch second) {

	return (first.distance < second.distance);

}

void do_match(int, void*) {

	Mat visualization, temp, descriptors_img, descriptors_pattern;
	vector<KeyPoint> keypoints_img, keypoints_pattern;
	vector<DMatch> matches, good_matches;
	double max_dist = 0; double min_dist = 100;
	int descs[] = { FEATURES_SIFT, FEATURES_SURF, FEATURES_ORB, FEATURES_AKAZE };

	std::vector<Point2f> srcPoints;
	std::vector<Point2f> dstPoints;

	for (int i = 0; i < 4; i++) {
		//if (i == 3) continue;
		Ptr<Feature2D> descriptor = get_descriptor(descs[i]);
		Ptr<DescriptorMatcher> matcher = new BFMatcher();
		descriptor->detectAndCompute(img, Mat(), keypoints_img, descriptors_img);
		//imshow("des", descriptors_img);
		descriptor->detectAndCompute(pattern, Mat(), keypoints_pattern, descriptors_pattern);

		matcher->match(descriptors_pattern, descriptors_img, matches);

		for (int j = 0; j < (int)matches.size(); j++) {
			double dist = matches[j].distance;
			if (dist < min_dist) min_dist = dist;
			if (dist > max_dist) max_dist = dist;
		}

		//keypoints_img.clear();
		//keypoints_pattern.clear();
		//matches.clear();
	}

	sort(matches.begin(), matches.end(), compare_matches);
	for (int j = 0; j < MIN(matches_count, (int)matches.size()); j++) {
		good_matches.push_back(matches[j]);
		srcPoints.push_back(keypoints_pattern[matches[j].queryIdx].pt);
		dstPoints.push_back(keypoints_img[matches[j].trainIdx].pt);
	}

	cout << format("Total matches: %d", matches.size()) << endl;
	cout << format("Match distances minimum=%f, maximum=%f", min_dist, max_dist) << endl;

	if (srcPoints.size() < 4) {
		cout << "At least four matches required to compute homography." << endl;
		return;
	}

	vector<int> inliers;
	Mat H = findHomography(srcPoints, dstPoints, CV_RANSAC, 3, inliers);

	std::vector<Point2f> pattern_corners(4);
	pattern_corners[0] = Point2f(0, 0);
	pattern_corners[1] = Point2f(pattern.cols, 0);
	pattern_corners[2] = Point2f(pattern.cols, pattern.rows);
	pattern_corners[3] = Point2f(0, pattern.rows);
	std::vector<Point2f> img_corners(4);

	perspectiveTransform(pattern_corners, img_corners, H);

	img.copyTo(temp);
	pattern.copyTo(temp(Rect(0, 0, pattern.cols, pattern.rows)));

	cvtColor(temp, visualization, COLOR_GRAY2BGR);

	for (int i = 0; i < (int)good_matches.size(); i++) {
		Scalar color = inliers[i] == 0 ? Scalar(255, 0, 0) : Scalar(0, 0, 255);
		int width = inliers[i] == 0 ? 1 : 2;
		line(visualization, keypoints_pattern[good_matches[i].queryIdx].pt,
			keypoints_img[good_matches[i].trainIdx].pt, color, width);
	}

	rectangle(visualization, img_corners[0], img_corners[2], Scalar(255, 255, 255),-1, 8, 0);
	imshow(WINDOW_NAME, visualization);

}

void capture_screen() {
	VideoCapture camera(0);
	if (!camera.isOpened()) {
		cout << "Unable to access camera" << endl;
		exit(-1);
	}
	Mat frame;
	camera.set(CV_CAP_PROP_CONTRAST, 1);
	while (true) {

		camera.read(frame);
		imshow("Camera", frame);

		if (waitKey(30) >= 0)
			break;
	}
	destroyWindow("Camera");
	cvtColor(frame, img, CV_BGR2GRAY);
}

int main(int argc, char** argv)
{
	if (argc < 3)
		return -1;
	string calibration_file = argv[1];
	FileStorage fs(calibration_file, FileStorage::READ);

	fs["intrinsics"] >> intrinsics;
	fs["distortion"] >> distortion;

	pattern = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);
	if (argc == 4)
		img = imread(argv[3], CV_LOAD_IMAGE_GRAYSCALE);
	else {
		capture_screen();
	}

	if (img.empty() || pattern.empty())
		return -1;

	Mat camera_matrix = getOptimalNewCameraMatrix(intrinsics, distortion, img.size(), 1);
	initUndistortRectifyMap(
		intrinsics, distortion, Mat(),
		camera_matrix, img.size(),
		CV_16SC2, map1, map2);
	remap(img, img, map1, map2, INTER_LINEAR);
	
	/*
	namedWindow(WINDOW_NAME, CV_WINDOW_AUTOSIZE);
	createTrackbar(TRACKBAR_NAME, WINDOW_NAME, &matches_count, MAX_MATCHES_COUNT, do_match);
	*/

	cout << "Press space to rotate pattern for 90 degrees" << endl;

	do_match(0, 0);

	while (true)
	{
		int c = waitKey(0);
		switch (c % 256) {
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


