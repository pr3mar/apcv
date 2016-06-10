#include <iostream>

#include "trackers.h"

using namespace std;

class TrackerNCCModel : public TrackerModel {
public:
  TrackerNCCModel() {}

  Mat p_template;

  Point2f p_position;

  Size p_size;

  int p_window;

  double p_update;

private:

  void modelEstimationImpl( const std::vector<Mat>& /*responses*/ ){}
  void modelUpdateImpl(){}

};

TrackerNCC::~TrackerNCC() {

}

bool TrackerNCC::initImpl(const Mat &image, const Rect2d &rect) {

  model = Ptr<TrackerModel>(new TrackerNCCModel());

  TrackerNCCModel* pmodel = ((TrackerNCCModel*)static_cast<TrackerModel*>(model));

  pmodel->p_window = MAX(rect.width, rect.height) * 2;

  Mat gray;
  cvtColor(image, gray, CV_BGR2GRAY);

  int left = MAX(rect.x, 0);
  int top = MAX(rect.y, 0);

  int right = MIN(rect.x + rect.width, gray.cols - 1);
  int bottom = MIN(rect.y + rect.height, gray.rows - 1);

  Rect roi(left, top, right - left, bottom - top);

  gray(roi).copyTo(pmodel->p_template);

  pmodel->p_position.x = (float)rect.x + (float)rect.width / 2;
  pmodel->p_position.y = (float)rect.y + (float)rect.height / 2;

  pmodel->p_size = Size2f(rect.width, rect.height);

  return true;

}

void TrackerNCC::read (const FileNode &fn) {

}

bool TrackerNCC::updateImpl(const Mat &image, Rect2d &result) {

  Mat gray;
  cvtColor(image, gray, CV_BGR2GRAY);

  TrackerNCCModel* pmodel = ((TrackerNCCModel*)static_cast<TrackerModel*>(model));


  float left = MAX(round(pmodel->p_position.x - (float)pmodel->p_window / 2), 0);
  float top = MAX(round(pmodel->p_position.y - (float)pmodel->p_window / 2), 0);

  float right = MIN(round(pmodel->p_position.x + (float)pmodel->p_window / 2), gray.cols - 1);
  float bottom = MIN(round(pmodel->p_position.y + (float)pmodel->p_window / 2), gray.rows - 1);

  Rect roi((int) left, (int) top, (int) (right - left), (int) (bottom - top));

  if (roi.width < pmodel->p_template.cols || roi.height < pmodel->p_template.rows) {

    result.x = pmodel->p_position.x - pmodel->p_size.width / 2;
    result.y = pmodel->p_position.y - pmodel->p_size.height / 2;
    result.width = pmodel->p_size.width;
    result.height = pmodel->p_size.height;
    return false;

  }

  Mat matches;
  Mat cut = gray(roi);

  matchTemplate(cut, pmodel->p_template, matches, CV_TM_CCOEFF_NORMED);

  Point matchLoc;
  minMaxLoc(matches, NULL, NULL, NULL, &matchLoc, Mat());

  pmodel->p_position.x = left + matchLoc.x + (float)pmodel->p_size.width / 2;
  pmodel->p_position.y = top + matchLoc.y + (float)pmodel->p_size.height / 2;

  result.x = left + matchLoc.x;
  result.y = top + matchLoc.y;
  result.width = pmodel->p_size.width;
  result.height = pmodel->p_size.height;

  return true;
}

void TrackerNCC::write(FileStorage &fs) const {

}

Ptr<Tracker> TrackerNCC::createTracker() {

  return Ptr<Tracker>(new TrackerNCC());

}

TrackerNCC::TrackerNCC() {
  isInit = false;
}


class TrackerCamShiftModel : public TrackerModel {
public:
  TrackerCamShiftModel() : vmin(10), vmax(256), smin(30), hsize(16) {

  }

  Mat mask;
  Mat hist;

  int vmin, vmax, smin, hsize;

  Rect selection;

private:

  void modelEstimationImpl( const std::vector<Mat>& /*responses*/ ){}
  void modelUpdateImpl(){}

};

TrackerCamShift::~TrackerCamShift() {

}

Rect box2rect(RotatedRect box) {

    Point pt0, pt1, pt2, pt3;
    Rect result;

    double _angle = box.angle*CV_PI/180.0; 
    float a = (float)cos(_angle)*0.5f; 
    float b = (float)sin(_angle)*0.5f; 
     
    pt0.x = box.center.x - a*box.size.height - b*box.size.width; 
    pt0.y = box.center.y + b*box.size.height - a*box.size.width; 
    pt1.x = box.center.x + a*box.size.height - b*box.size.width; 
    pt1.y = box.center.y - b*box.size.height - a*box.size.width; 
    pt2.x = 2 * box.center.x - pt0.x; 
    pt2.y = 2 * box.center.y - pt0.y; 
    pt3.x = 2 * box.center.x - pt1.x; 
    pt3.y = 2 * box.center.y - pt1.y; 

    result.x = floor(MIN(MIN(MIN(pt0.x, pt1.x), pt2.x), pt3.x)); 
    result.y = floor(MIN(MIN(MIN(pt0.y, pt1.y), pt2.y), pt3.y));

    result.width = ceil(MAX(MAX(MAX(pt0.x, pt1.x), pt2.x), pt3.x)) - result.x - 1; 
    result.height = ceil(MAX(MAX(MAX(pt0.y, pt1.y), pt2.y), pt3.y)) - result.y - 1; 

    return result; 
}

bool TrackerCamShift::initImpl(const Mat &image, const Rect2d &rect) {

  model = Ptr<TrackerModel>(new TrackerCamShiftModel());

  TrackerCamShiftModel* pmodel = ((TrackerCamShiftModel*)static_cast<TrackerModel*>(model));

  float hranges[] = {0,180};
  const float* phranges = hranges;

  cvtColor(image, hsv, COLOR_BGR2HSV);

  inRange(hsv, Scalar(0, pmodel->smin, MIN(pmodel->vmin, pmodel->vmax)),
          Scalar(180, 256, MAX(pmodel->vmin, pmodel->vmax)), pmodel->mask);
  int ch[] = {0, 0};
  hue.create(hsv.size(), hsv.depth());
  mixChannels(&hsv, 1, &hue, 1, ch, 1);

  Mat roi(hue, rect), maskroi(pmodel->mask, rect);
  calcHist(&roi, 1, 0, maskroi, pmodel->hist, 1, &pmodel->hsize, &phranges);
  normalize(pmodel->hist, pmodel->hist, 0, 255, NORM_MINMAX);

  pmodel->selection = rect;

  Mat buf(1, pmodel->hsize, CV_8UC3);
  for( int i = 0; i < pmodel->hsize; i++ )
      buf.at<Vec3b>(i) = Vec3b(saturate_cast<uchar>(i*180./pmodel->hsize), 255, 255);
  cvtColor(buf, buf, COLOR_HSV2BGR);

  return true;

}

void TrackerCamShift::read (const FileNode &fn) {

}

bool TrackerCamShift::updateImpl(const Mat &image, Rect2d &result) {

  Mat gray;
  cvtColor(image, gray, CV_BGR2GRAY);

  TrackerCamShiftModel* pmodel = ((TrackerCamShiftModel*)static_cast<TrackerModel*>(model));

  float hranges[] = {0,180};
  const float* phranges = hranges;

  cvtColor(image, hsv, COLOR_BGR2HSV);

  inRange(hsv, Scalar(0, pmodel->smin, MIN(pmodel->vmin, pmodel->vmax)),
          Scalar(180, 256, MAX(pmodel->vmin, pmodel->vmax)), pmodel->mask);
  int ch[] = {0, 0};
  hue.create(hsv.size(), hsv.depth());
  mixChannels(&hsv, 1, &hue, 1, ch, 1);


  calcBackProject(&hue, 1, 0, pmodel->hist, backproject, &phranges);
  backproject &= pmodel->mask;
  RotatedRect trackBox = CamShift(backproject, pmodel->selection,
                      TermCriteria( TermCriteria::EPS | TermCriteria::COUNT, 10, 1 ));
  if( pmodel->selection.area() <= 1 )
  {
      int cols = backproject.cols, rows = backproject.rows, r = (MIN(cols, rows) + 5)/6;
      pmodel->selection = Rect(pmodel->selection.x - r, pmodel->selection.y - r,
                         pmodel->selection.x + r, pmodel->selection.y + r) &
                    Rect(0, 0, cols, rows);
  }

  result = box2rect(trackBox);

  return true;
}

void TrackerCamShift::write(FileStorage &fs) const {

}

Ptr<Tracker> TrackerCamShift::createTracker() {

  return Ptr<Tracker>(new TrackerCamShift());

}

TrackerCamShift::TrackerCamShift() {
  isInit = false;
}

