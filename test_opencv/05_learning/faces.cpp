#include <cstdio>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <opencv2/objdetect.hpp>
#include <opencv2/face.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "utilities.h"

using namespace std;
using namespace cv;
using namespace cv::face;

#define WINDOW_NAME "Face recognition"

CascadeClassifier cascade;
Size reference_size(100, 100);

Mat extract_face(Mat image) {

  Mat sample;

  vector<Rect> regions;

  cascade.detectMultiScale(image, regions,
    1.1, 3, CASCADE_FIND_BIGGEST_OBJECT | CASCADE_DO_ROUGH_SEARCH | CASCADE_SCALE_IMAGE,
    Size(130, 130), Size(150, 150) );

  if (regions.size() == 1) 
      resize(image(regions[0]), sample, reference_size);

  return sample;
}

void load_faces(string dataset, vector<string> classes, vector<Mat>& faces, vector<int>& ids) {

  int class_id = 0;

  cout << "Loading faces ..." << endl;

  for (vector<string>::iterator it = classes.begin(); it != classes.end(); it++) {

	  for (unsigned int i = 1; ; i++) {

		  string filename = join(join(dataset, "training"), join(*it, format("%03d.jpg", i)));

		  Mat image = imread(filename, IMREAD_GRAYSCALE);

		  if (image.empty())
			  break;

		  Mat sample = extract_face(image);

		  if (!sample.empty()) {

			  faces.push_back(sample);
			  ids.push_back(class_id);

		  }
	  }

	  class_id++;
  }
}

int main( int argc, char** argv ) {

  if( !cascade.load("face_cascade.xml") ) {
      cerr << "Could not load classifier cascade" << endl;
      return -1;
  }

  vector<Mat> faces;
  vector<int> ids;
  string dataset(argv[1]), raw;
  getline(ifstream(join(join(dataset, "training"), "people.txt")), raw);
  vector<string> classes = split(raw, ';');

  load_faces(dataset, classes, faces, ids);

  Ptr<FaceRecognizer> recognizer = createEigenFaceRecognizer();

  cout << "Training model ..." << endl;

  recognizer->train(faces, ids);

  cout << "Testing model ..." << endl;

  for (unsigned int i = 1; ; i++) {

    string filename = join(join(dataset, "testing"), format("%03d.jpg", i));

    Mat image = imread(filename, IMREAD_GRAYSCALE);

    if (image.empty())
      break;

    Mat sample = extract_face(image);

    if (!sample.empty()) {

      int label;
      double confidence;
      recognizer->predict(sample, label, confidence);

      Mat visualization;
      cvtColor(image, visualization, COLOR_GRAY2BGR); 

      putText(visualization, format("%s (%.3f)", classes[label].c_str(), confidence), Point(10, 10), FONT_HERSHEY_SIMPLEX, 0.3,
        Scalar(255, 0, 0), 1, 8);

      imshow(WINDOW_NAME, visualization);

      waitKey(0);

    }
  }

  return 0;
}



