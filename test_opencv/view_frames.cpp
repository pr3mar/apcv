#include <iostream>
#include <opencv2\highgui.hpp>
#include <opencv2/videoio.hpp>

// window names
#define WIN_LEFT "left image"
#define WIN_RIGHT "right image"

using namespace std;
using namespace cv;

int main(int argc, char **argv) {
	// get the videos via arguments
	if (argc != 3) return -1;

	// video readers
	VideoCapture vid_1(argv[1]);
	VideoCapture vid_2(argv[2]);

	// frame holders
	Mat frame_1, frame_2;

	while (true) {
		// read frames
		vid_1 >> frame_1;
		vid_2 >> frame_2;
		// check if the frames are empty
		if (frame_2.empty() || frame_1.empty()) {
			break;
		}
		// view the frames
		imshow(WIN_LEFT, frame_1);
		imshow(WIN_RIGHT, frame_2);
		// wait
		waitKey(30);
	}

	return 0;
}