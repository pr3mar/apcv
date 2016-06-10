#include <cstdio>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml.hpp>

#include "utilities.h"

using namespace std;
using namespace cv;
using namespace cv::ml;
using namespace cv::xfeatures2d;

#define WINDOW_NAME "Bag of words"

Mat train_vocabulary(string dataset, vector<string> classes, int size = 1000) {

  Mat features;
  Ptr<Feature2D> detector = SIFT::create();

  cout << "Generating data for BoW vocabulary" << endl;

  for (vector<string>::iterator it = classes.begin(); it != classes.end(); it++) {

    cout << format("Extracting features for class %s", (*it).c_str()) << endl;   

    for (unsigned int i = 1; ; i++) {

      string filename = join(dataset, join(*it, format("%03d.jpg", i)));

      Mat image = imread(filename, IMREAD_GRAYSCALE);

      if (image.empty())
        break;

      Mat descriptor;
      vector<KeyPoint> keypoints;
      detector->detectAndCompute(image, Mat(), keypoints, descriptor);

      features.push_back(descriptor);        

    }
  }

  cout << "Training BoW vocabulary" << endl;

  double start = (double)getTickCount();

  BOWKMeansTrainer trainer(size, TermCriteria(CV_TERMCRIT_ITER, 100, 0.001), 1, KMEANS_PP_CENTERS);

  Mat vocabulary = trainer.cluster(features);    

  cout << format("Training done in %.2f seconds", ((double)getTickCount() - start) / getTickFrequency()) << endl;

  return vocabulary;

}

Ptr<StatModel> train_classifier(string dataset, vector<string> classes, Mat& vocabulary) {

  cout << "Generating data for classifier training" << endl;

  Ptr<Feature2D> detector = SIFT::create();
  Ptr<DescriptorMatcher> matcher(new FlannBasedMatcher());
  BOWImgDescriptorExtractor extractor(detector, matcher);

  extractor.setVocabulary(vocabulary);
 
  Mat samples;
  Mat responses;

  int class_id = 0;

  for (vector<string>::iterator it = classes.begin(); it != classes.end(); it++) {

    cout << format("Extracting features for class %03s", (*it).c_str()) << endl;   

    for (unsigned int i = 1; ; i++) {

      string filename = join(dataset, join(*it, format("%03d.jpg", i)));

      Mat image = imread(filename, IMREAD_GRAYSCALE);

      if (image.empty())
        break;

      Mat descriptor;
      vector<KeyPoint> keypoints;
      detector->detect(image, keypoints);
      extractor.compute(image, keypoints, descriptor);

      samples.push_back(descriptor);        
      responses.push_back(class_id);
    }

    class_id++;
  }

  Ptr<TrainData> data = TrainData::create(samples, ROW_SAMPLE, responses);
  Ptr<SVM> classifier = SVM::create();

  cout << "Training classifier" << endl;

  double start = (double)getTickCount();

  classifier->trainAuto(data); // alternative train(data);

  cout << format("Training done in %.2f seconds", ((double)getTickCount() - start) / getTickFrequency()) << endl;

  return classifier;

}

string predict(Mat image, Ptr<StatModel> classifier, vector<string> classes, Mat& vocabulary) {

  Ptr<Feature2D> detector = SIFT::create();
  Ptr<DescriptorMatcher> matcher(new FlannBasedMatcher());
  BOWImgDescriptorExtractor extractor(detector, matcher);

  extractor.setVocabulary(vocabulary);

  Mat descriptor;
  vector<KeyPoint> keypoints;
  detector->detect(image, keypoints);
  extractor.compute(image, keypoints, descriptor);

  float class_id = classifier->predict(descriptor);

  return (class_id >= 0 && class_id < classes.size()) ? classes[class_id] : "Unknown";

}

int main( int argc, char** argv )
{
  if (argc < 2) {
    cout << "Usage:" << endl;
    cout << "./bow vocabulary <dataset>" << endl;
    cout << "./bow classifier <dataset> <vocabulary>" << endl;
    cout << "./bow predict <vocabulary> <classifier> <directory>" << endl;
    return -1;
  }
  string command(argv[1]);

  if (command == "vocabulary") {

    string dataset(argv[2]), raw;
    getline(ifstream(join(dataset, "classes.txt")), raw);
    vector<string> classes = split(raw, ';');

    Mat vocabulary = train_vocabulary(dataset, classes);

    FileStorage fs("vocabulary.yaml", FileStorage::WRITE);
    fs << "vocabulary" << vocabulary;
    fs.release();

  } else if (command == "classifier") {

    string dataset(argv[2]), raw;
    getline(ifstream(join(dataset, "classes.txt")), raw);
    vector<string> classes = split(raw, ';');

    Mat vocabulary;
    FileStorage fs(argv[3], FileStorage::READ);
    fs["vocabulary"] >> vocabulary;
    fs.release();

    Ptr<StatModel> classifier = train_classifier(dataset, classes, vocabulary);

    fs.open("classifier.yaml", FileStorage::WRITE);
    fs << "classifier" << "{";
    classifier->write(fs);
    fs << "}";
    fs << "classes" << classes;
    fs.release();

  } else if (command == "predict") {
    Mat vocabulary;
    vector<string> classes;
    string directory(argv[4]);

    FileStorage fs(argv[2], FileStorage::READ);
    fs["vocabulary"] >> vocabulary;
    fs.release();

    fs.open(argv[3], FileStorage::READ);
    Ptr<SVM> classifier = SVM::create();
    read(fs["classes"], classes);
    classifier->read(fs["classifier"]);
    fs.release();

    Mat visualization;

    for (unsigned int i = 1; ; i++) {
      Mat image = imread(join(directory, format("%03d.jpg", i)), IMREAD_GRAYSCALE);

      if (image.empty())
        break;

      string label = predict(image, classifier, classes, vocabulary);

      cvtColor(image, visualization, COLOR_GRAY2BGR);

      putText(visualization, label, Point(10, 10), FONT_HERSHEY_SIMPLEX, 0.5,
        Scalar(0, 0, 255), 1, 8);

      imshow(WINDOW_NAME, visualization);

      int c = waitKey(0);
      switch(c % 256) {
        case 'q':
        case 27: {
          return 0;
        }
        default: {
          break;
        }
      }

    }

  }

  return 0;

}

