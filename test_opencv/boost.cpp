#include <iostream>

// Include standard OpenCV headers
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

// Include Boost headers for system time and threading
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/thread.hpp>

#include <csignal>
#include <windows.h>

using namespace std;
using namespace boost::posix_time;
using namespace cv;

bool loop;

BOOL CtrlHandler(DWORD fdwCtrlType)
{
	switch (fdwCtrlType)
	{
		// Handle the CTRL-C signal. 
	case CTRL_C_EVENT:
		printf("Ctrl-C event\n\n");
		Beep(750, 300);
		loop = false;
		return(TRUE);

		// CTRL-CLOSE: confirm that the user wants to exit. 
	case CTRL_CLOSE_EVENT:
		Beep(600, 200);
		printf("Ctrl-Close event\n\n");
		return(TRUE);

		// Pass other signals to the next handler. 
	case CTRL_BREAK_EVENT:
		Beep(900, 200);
		printf("Ctrl-Break event\n\n");
		return FALSE;

	case CTRL_LOGOFF_EVENT:
		Beep(1000, 200);
		printf("Ctrl-Logoff event\n\n");
		return FALSE;

	case CTRL_SHUTDOWN_EVENT:
		Beep(750, 500);
		printf("Ctrl-Shutdown event\n\n");
		return FALSE;

	default:
		return FALSE;
	}
}

void signalHandler(int signum) {
	cout << "Interrupt signal (" << signum << ") received.\n";
	loop = false;
}



// Code for capture thread
void captureFunc(Mat *frame_1, VideoCapture *cam_1) {
	//loop infinitely
	for (;;) {
		//capture from webcame to Mat frame
		(*cam_1) >> (*frame_1);
	}
}

//main
int main(int argc, char *argv[])
{
	signal(SIGBREAK, signalHandler);
	SetConsoleCtrlHandler((PHANDLER_ROUTINE)CtrlHandler, TRUE);
	loop = true;
	//vars
	time_duration td, td1;
	ptime nextFrameTimestamp, currentFrameTimestamp, initialLoopTimestamp, finalLoopTimestamp;
	int delayFound = 0;
	int totalDelay = 0;

	// initialize capture on default source
	VideoCapture cam_1, cam_2;
	cam_1 = VideoCapture(1);
	cam_2 = VideoCapture(2);

	// set framerate to record and capture at
	int framerate = 15;//capture.get(CV_CAP_PROP_FPS);

					   // Get the properties from the camera
	double width_1 = cam_1.get(CV_CAP_PROP_FRAME_WIDTH);
	double height_1 = cam_1.get(CV_CAP_PROP_FRAME_HEIGHT);

	double width_2 = cam_2.get(CV_CAP_PROP_FRAME_WIDTH);
	double height_2 = cam_2.get(CV_CAP_PROP_FRAME_HEIGHT);

	// print camera frame size
	cout << "Camera 1 properties" << endl;
	cout << "width = " << width_1 << endl << "height = " << height_1 << endl;

	cout << "Camera 2 properties" << endl;
	cout << "width = " << width_2 << endl << "height = " << height_2 << endl;


	// Create a matrix to keep the retrieved frame
	Mat frame_1, frame_2;

	// Create the video writer
	VideoWriter video_1("cam_1.avi", CV_FOURCC('D', 'I', 'V', 'X'), framerate, Size((int)width_1, (int)height_1));
	VideoWriter video_2("cam_2.avi", CV_FOURCC('D', 'I', 'V', 'X'), framerate, Size((int)width_2, (int)height_2));

	if (!video_1.isOpened() || !video_2.isOpened()) {
		cout << "error opening file!" << endl;
		return -1;
	}
	// initialize initial timestamps
	nextFrameTimestamp = microsec_clock::local_time();
	currentFrameTimestamp = nextFrameTimestamp;
	td = (currentFrameTimestamp - nextFrameTimestamp);

	// start thread to begin capture and populate Mat frame
	boost::thread captureThread_1(captureFunc, &frame_1, &cam_1);
	boost::thread captureThread_2(captureFunc, &frame_2, &cam_2);
	// loop infinitely
	while(loop)
	{

		// wait for X microseconds until 1second/framerate time has passed after previous frame write
		while (td.total_microseconds() < 1000000 / framerate && !loop) {
			//determine current elapsed time
			currentFrameTimestamp = microsec_clock::local_time();
			td = (currentFrameTimestamp - nextFrameTimestamp);
		}

		//	 determine time at start of write
		initialLoopTimestamp = microsec_clock::local_time();

		// Save frame to video
		video_1 << frame_1;
		video_2 << frame_2;

		//write previous and current frame timestamp to console
		cout << nextFrameTimestamp << " " << currentFrameTimestamp << " ";

		// add 1second/framerate time for next loop pause
		nextFrameTimestamp = nextFrameTimestamp + microsec(1000000 / framerate);

		// reset time_duration so while loop engages
		td = (currentFrameTimestamp - nextFrameTimestamp);

		//determine and print out delay in ms, should be less than 1000/FPS
		//occasionally, if delay is larger than said value, correction will occur
		//if delay is consistently larger than said value, then CPU is not powerful
		// enough to capture/decompress/record/compress that fast.
		finalLoopTimestamp = microsec_clock::local_time();
		td1 = (finalLoopTimestamp - initialLoopTimestamp);
		delayFound = td1.total_milliseconds();
		cout << delayFound << endl;

		//output will be in following format
		//[TIMESTAMP OF PREVIOUS FRAME] [TIMESTAMP OF NEW FRAME] [TIME DELAY OF WRITING]
	}
	// Exit
	return 0;
}
