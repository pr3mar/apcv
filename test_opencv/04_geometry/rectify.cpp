#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>

using namespace cv;
using namespace std;

#define WINDOW_NAME "Rectify"

int main(int argc, const char** argv) {
	int k;

	string calibration_file("extra/webcam.xml");
	int camera_id = 0;

	if (argc > 1) calibration_file = argv[1];
	if (argc > 2) camera_id = atoi(argv[2]);

	VideoCapture camera(camera_id); // ID of the camera

	if (!camera.isOpened()) {
		cout << "Unable to access camera" << endl;
		cin >> k;
		return -1;
	}

	Mat frame, rectified, map1, map2, intrinsics, distortion;

	FileStorage fs(calibration_file, FileStorage::READ);

	fs["intrinsics"] >> intrinsics;
	fs["distortion"] >> distortion;

	camera.read(frame);

	Mat camera_matrix = getOptimalNewCameraMatrix(intrinsics, distortion, frame.size(), 1);

	initUndistortRectifyMap(
		intrinsics, distortion, Mat(),
		camera_matrix, frame.size(),
		CV_16SC2, map1, map2);

	bool show_rectified = true;

	cout << "Press r to toggle between raw and rectified image.\n";

	while (true) {

		camera.read(frame);

		if (show_rectified) {
			remap(frame, rectified, map1, map2, INTER_LINEAR);
			imshow(WINDOW_NAME, rectified);
		}
		else
			imshow(WINDOW_NAME, frame);

		int c = waitKey(30);
		switch (c % 256) {
		case 'q':
		case 27: {
			return 0;
		}
		case 'r': {
			show_rectified = !show_rectified;
			break;
		}
		}
	}
	cin >> k;
	return 0;
}
