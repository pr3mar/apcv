#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;
using namespace xfeatures2d;

int main(int argc, char** argv)
{
	if(argc != 3) return -1;

	Mat image1 = imread(argv[2]);
	Mat image2 = imread(argv[1]);
	Mat gray_image1;
	Mat gray_image2;

	cvtColor(image1, gray_image1, CV_RGB2GRAY);
	cvtColor(image2, gray_image2, CV_RGB2GRAY);

	if(!gray_image1.data || !gray_image2.data)
	{
		cout<<"Error reading images!!" <<endl; 
		return -1;
	}

	//-- Step 1: Detect the keypoints using SURF Detector and calculate descriptors (feature vectors)
	Mat descriptors_object, descriptors_scene;
	int minHessian = 400;

	Ptr<SURF> detector = SURF::create(minHessian);

	vector<KeyPoint> keypoints_object, keypoints_scene;

	detector->detectAndCompute(gray_image1, Mat(), keypoints_object,descriptors_object);
	detector->detectAndCompute(gray_image2, Mat(), keypoints_scene, descriptors_scene);


	//-- Step 2: Matching descriptor vectors using FLANN matcher
	FlannBasedMatcher matcher;
	vector<DMatch> matches;
	matcher.match(descriptors_object, descriptors_scene, matches);

	double max_dist = 0; double min_dist = 100;

	//-- Quick calculation of max and min distances between keypoints
	for(int i = 0; i <descriptors_object.rows; i++)
	{
		double dist = matches[i].distance;
		if(dist < min_dist) min_dist = dist;
		if(dist > max_dist) max_dist = dist;
	}

	//-- Use only "good" matches (i.e. whose distance is less than 3*min_dist)
	vector<DMatch> good_matches;

	for(int i = 0; i < descriptors_object.rows; i++)
	{
		if(matches[i].distance < 3 * min_dist)
			good_matches.push_back(matches[i]); 
	}
	vector<Point2f> obj;
	vector<Point2f> scene;

	for(int i = 0; i < good_matches.size(); i++)
	{
		//-- Get the keypoints from the good matches
		obj.push_back(keypoints_object[good_matches[i].queryIdx].pt);
		scene.push_back(keypoints_scene[good_matches[i].trainIdx].pt);
	}

	// Find the Homography Matrix
	Mat H = findHomography(obj, scene, CV_RANSAC);
	
	// Use the Homography Matrix to warp the images
	Mat result;
	warpPerspective(image1, result, H, Size(image1.cols + image2.cols, image1.rows));
	Mat half(result, Rect(0, 0, image2.cols, image2.rows));
	image2.copyTo(half);
	imshow("Result", result);

	waitKey(0);
	return 0;
}