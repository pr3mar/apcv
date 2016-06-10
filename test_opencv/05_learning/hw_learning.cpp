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

#define WINDOW_NAME "Face detection"

CascadeClassifier cascade;
Size reference_size(100, 100);

Mat extract_face(Mat image) {

	Mat sample;

	vector<Rect> regions;

	cascade.detectMultiScale(image, regions,
		1.1, 3, CASCADE_FIND_BIGGEST_OBJECT | CASCADE_DO_ROUGH_SEARCH | CASCADE_SCALE_IMAGE,
		Size(130, 130), Size(150, 150));

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

int main(int argc, char** argv) {
	// press on space to save an image of the face -> build a training set

	if (!cascade.load("face_cascade.xml")) {
		cerr << "Could not load classifier cascade" << endl;
		return -1;
	}

	VideoCapture camera(0); // ID of the camera

	if (!camera.isOpened()) {
		cerr << "Unable to access camera" << endl;
		return -1;
	}

	vector<Mat> faces_collection;
	vector<int> ids;
	string dataset(argv[1]), raw;
	getline(ifstream(join(join(dataset, "training"), "people.txt")), raw);
	vector<string> classes = split(raw, ';');
	load_faces(dataset, classes, faces_collection, ids);
	Ptr<FaceRecognizer> recognizer = createEigenFaceRecognizer();
	cout << "Training model ..." << endl;
	recognizer->train(faces_collection, ids);

	Mat frame, gray;
	vector<Mat> subImg;
	vector<Rect> faces;
	int key_pressed, face_count = 0;
	int label; double confidence;

	while (true) {

		camera.read(frame);
		flip(frame, frame, 1);

		cvtColor(frame, gray, COLOR_BGR2GRAY);

		faces.clear();
		subImg.clear();

		cascade.detectMultiScale(gray, faces,
			1.1, 3, CASCADE_FIND_BIGGEST_OBJECT | CASCADE_DO_ROUGH_SEARCH | CASCADE_SCALE_IMAGE,
			Size(120, 120), Size(180, 180));

		for (unsigned int i = 0; i < faces.size(); i++) {
			Rect r = faces[i];
			rectangle(frame, Point(round(r.x), round(r.y)),
				Point(round((r.x + r.width - 1)), round((r.y + r.height - 1))),
				Scalar(255, 255, 255), 3, 8, 0);
			subImg.push_back(gray(Rect(r.x, r.y, r.width, r.height)));
			Mat tmp;
			resize(subImg[i], tmp, Size(faces_collection[i].cols, faces_collection[i].rows));
			recognizer->predict(tmp, label, confidence);
			cout << format("Hello %s (%.3f)\n", classes[label].c_str(), confidence);
			imshow("check",tmp);
		}
		imshow(WINDOW_NAME, frame);

		key_pressed = waitKey(10);
		switch (key_pressed % 256) {
		case 'q': case 27:
			return 0;
		case ' ':
			cout << "Saving new faces." << endl;
			for (unsigned int i = 0; i < subImg.size(); i++) {
				string out = format("resources/faces/training/%03d.jpg", face_count);
				imwrite(out, subImg[i]);
				face_count++;
				imshow(to_string(face_count), subImg[i]);
			}
			break;
		}
	}

	return 0;
}



