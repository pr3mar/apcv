#ifndef MOTION_MODELS_H
#define MOTION_MODELS_H

#include <opencv2/video.hpp>
#include <opencv2/tracking.hpp>

using namespace cv;

Ptr<KalmanFilter> motion_model_ncc(double measurement_noise, double process_noise);

#endif
