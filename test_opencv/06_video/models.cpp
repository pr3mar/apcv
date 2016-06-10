
#include "models.h"

Ptr<KalmanFilter> motion_model_ncc(double measurement_noise, double process_noise) {

  Ptr<KalmanFilter> filter = Ptr<KalmanFilter>(new KalmanFilter(4, 2, 0));

  filter->statePre.at<float>(0) = 0;
	filter->statePre.at<float>(1) = 0;
	filter->statePre.at<float>(2) = 0;
	filter->statePre.at<float>(3) = 0;
	filter->transitionMatrix = (Mat_<float>(4, 4) << 1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1);
	
  setIdentity(filter->measurementMatrix);
  setIdentity(filter->processNoiseCov, Scalar::all(1e-3 * process_noise));
  setIdentity(filter->measurementNoiseCov, Scalar::all(1e-3 * measurement_noise));
  setIdentity(filter->errorCovPost, Scalar::all(.1));

  return filter;
}

