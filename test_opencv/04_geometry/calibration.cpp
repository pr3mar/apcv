#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>

using namespace cv;
using namespace std;

#define TYPE_CHESSBOARD 0
#define TYPE_CIRCLES 1

typedef struct {
  Size pattern_grid;
  int pattern_type;
  float pattern_size;
} CalibrationParameters;

double reprojection_error(const vector<vector<Point3f> >& pattern_points,
                         const vector<vector<Point2f> >& image_points,
                         const vector<Mat>& rotation, const vector<Mat>& translation,
                         const Mat& intrinsics, const Mat& distortion)
{
    vector<Point2f> reprojected_points;
    int count = 0;
    double total = 0, error;

    for(size_t i = 0; i < image_points.size(); ++i) {

        projectPoints(pattern_points[i], rotation[i], translation[i], intrinsics, distortion, reprojected_points);

        error = norm(image_points[i], reprojected_points, NORM_L2);

        int n = pattern_points[i].size();
        total += error * error;
        count += n;
    }

    return std::sqrt(total / count);
}

bool calibration(CalibrationParameters& parameters, Size& size, Mat& intrinsics, Mat& distortion, vector<vector<Point2f> > image_points) {
    intrinsics = Mat::eye(3, 3, CV_64F);
    distortion = Mat::zeros(8, 1, CV_64F);
    
    vector<vector<Point3f> > pattern_points(1);

    if (parameters.pattern_type == TYPE_CHESSBOARD)  {
        for (int i = 0; i < parameters.pattern_grid.height; ++i)
            for (int j = 0; j < parameters.pattern_grid.width; ++j)
                pattern_points[0].push_back(Point3f(j * parameters.pattern_size, i * parameters.pattern_size, 0));
    } else {
        for (int i = 0; i < parameters.pattern_grid.height; i++)
            for (int j = 0; j < parameters.pattern_grid.width; j++)
                pattern_points[0].push_back(Point3f((2 * j + i % 2) * parameters.pattern_size, i * parameters.pattern_size, 0));
    }

    pattern_points.resize(image_points.size(), pattern_points[0]);

    vector<Mat> translation(image_points.size()), rotation(image_points.size());

    int flags = CALIB_FIX_K4 | CALIB_FIX_K5 | CALIB_FIX_ASPECT_RATIO;

    calibrateCamera(pattern_points, image_points, size, intrinsics, distortion, rotation, translation, flags);

    double reprojection = reprojection_error(pattern_points, image_points, rotation, translation, intrinsics, distortion);

    cout << format("Re-projection error: %.2fpx", reprojection) << endl;

    bool ok = checkRange(intrinsics) && checkRange(distortion);

    return ok;
}

bool process_image(Mat& image, CalibrationParameters& parameters, vector<Point2f>& points) {

  if (parameters.pattern_type != TYPE_CIRCLES) {
    if (!findChessboardCorners(image, parameters.pattern_grid, points, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE))
      return false; 
    cornerSubPix(image, points, Size(11,11), Size(-1,-1), TermCriteria( TermCriteria::EPS + TermCriteria::COUNT, 30, 0.1));
    return true;
  }
  else {
    return findCirclesGrid(image, parameters.pattern_grid, points, CALIB_CB_SYMMETRIC_GRID);
  }
}

int main(int argc, const char** argv) {

  CalibrationParameters parameters;

  CommandLineParser parser(argc, argv,
      "{help h    |             | Print this help}"
      "{camera c  | 0           | Camera ID}"
      "{pattern p | chessboard  | Pattern type (chessboard,circles)}"
      "{columns x | 5           | Number of points in width}"
      "{rows y    | 8           | Number of points in height}"
      "{size      | 50          | Size of rectangle or circle in millimeters}"
      "{output o  | camera.yaml | Size of rectangle or circle in millimeters}"
  );

  if (!parser.check()) {
    parser.printErrors();
    return -1;
  }

  if (parser.has("help"))
  {
    parser.printMessage();
    return 0;
  }

  VideoCapture camera(parser.get<int>("camera"));
 
  if (!camera.isOpened()) {
    cout << "Unable to access camera" << endl;
    return -1;
  }
 
  parameters.pattern_type = (parser.get<string>("pattern") == "circles") ? TYPE_CIRCLES : TYPE_CHESSBOARD;
  parameters.pattern_grid.width = parser.get<int>("columns");
  parameters.pattern_grid.height = parser.get<int>("rows");
  parameters.pattern_size = parser.get<float>("size");

  cout << format("Pattern type: %s, grid %dx%d", parameters.pattern_type == TYPE_CIRCLES ? 
      "circles" : "chessboard", parameters.pattern_grid.width, parameters.pattern_grid.height) << endl;

  string output = parser.get<string>("output");

  Mat frame;
  Mat snapshot;

  vector<vector<Point2f>> point_sets;

  int show_snapshot = 0;

  while (true) {
   
    camera.read(frame);

    if (show_snapshot > 0) show_snapshot--;
    imshow("Calibration", (show_snapshot > 0) ? snapshot : frame);

    int c = waitKey(30);
    switch(c % 256) {
      case 'q':
      case 27: // escape
	  {
        return 0;
      }
      case 32: { // space
        cout << "Capturing image" << endl;
        vector<Point2f> points;
        cvtColor(frame, snapshot, CV_BGR2GRAY);
        
        if (process_image(snapshot, parameters, points)) {
          point_sets.push_back(points);
          drawChessboardCorners(snapshot, parameters.pattern_grid, Mat(points), true);
          cout << format("Adding point set (total %d)", point_sets.size()) << endl;
          show_snapshot = 10;
        } else {
          cout << "Unable to find pattern" << endl;
        }
        break;
      }
      case 'r': {
        point_sets.clear();
        cout << "Removing all point sets" << endl;
        break;
      }
      case 'c': {
        if (point_sets.size() < 1) break;

        Mat distortion;
        Mat intrinsics;
        Size size(frame.cols, frame.rows);

        if (calibration(parameters, size, intrinsics, distortion, point_sets)) {
          cout << "Calibrated successfully. Writing parameters to file " << output << endl;

          FileStorage fs(output, FileStorage::WRITE);

          fs << "intrinsics" << intrinsics;
          fs << "distortion" << distortion;

          fs.release();

        } else 
          cout << "Calibrated unsuccessfully." << endl;

        break;
      }
    }
  
  }

  return 0;
}
