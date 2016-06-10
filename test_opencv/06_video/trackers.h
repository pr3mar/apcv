#ifndef TRACKING_H
#define TRACKING_H

#include <opencv2/video.hpp>
#include <opencv2/tracking.hpp>

using namespace cv;

class TrackerNCC : public Tracker {
public:
  virtual ~TrackerNCC();
 
  virtual void read (const FileNode &fn);

  virtual void write(FileStorage &fs) const;

  static Ptr<Tracker> createTracker();

protected:

  virtual bool initImpl(const Mat &image, const Rect2d &boundingBox);
 
  virtual bool updateImpl(const Mat &image, Rect2d &boundingBox);
 
private:

  TrackerNCC();

};

class TrackerCamShift : public Tracker {
public:
  virtual ~TrackerCamShift();
 
  virtual void read (const FileNode &fn);

  virtual void write(FileStorage &fs) const;

  static Ptr<Tracker> createTracker();

protected:

  virtual bool initImpl(const Mat &image, const Rect2d &boundingBox);
 
  virtual bool updateImpl(const Mat &image, Rect2d &boundingBox);
 
private:

  Mat hsv;
  Mat hue;
  Mat backproject;

  TrackerCamShift();

};

#endif


