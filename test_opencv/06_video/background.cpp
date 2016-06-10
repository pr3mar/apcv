#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/bgsegm.hpp>
#include <iostream>

using namespace std;
using namespace cv;
using namespace cv::bgsegm;

#define WINDOW_NAME "Background subtraction"

Ptr<BackgroundSubtractor> model;
Mat frame, mask, segmentation;

int main( int argc, char** argv ) {

  model = createBackgroundSubtractorGMG();

  VideoCapture capture(0);

  if (!capture.isOpened())
    return -1;

  cout << "Press: \n * k: KNN \n * m: Mixture of Gaussians \n * g: GMG \n";
  cout << "\nKeep in mind that the models need a few seconds of empty frames to learn the model.\n";

  namedWindow(WINDOW_NAME, WINDOW_NORMAL);

  while (true) {
    capture >> frame;

    if (frame.empty())
        break;

    model->apply(frame, mask);

    frame.convertTo(segmentation, CV_8U, 0.5);
    add(frame, Scalar(0, 0, 200), segmentation, mask);

    imshow(WINDOW_NAME, segmentation);
 
    int c = waitKey(30);
    switch(c % 256) {
      case 'q':
      case 27: {
        return 0;
      }
      case 'k': {
        model = createBackgroundSubtractorKNN();
        break;
      }
      case 'm': {
        model = createBackgroundSubtractorMOG2();
        break;
      }
      case 'g': {
        model = createBackgroundSubtractorGMG();
        break;
      }
    }
  }

  return 0;

}


