#include <iostream>

#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

using namespace cv;
using namespace std;

Mat src;
Mat src_gray;
int thresh = 100;
RNG rng(12345);
void thresh_callback(int, void*);

int main(int argc, char** argv) {
  CommandLineParser parser(argc, argv, "{@input | stuff.jpg | input image}");
  src = imread(samples::findFile(parser.get<String>("@input")));
  if (src.empty()) {
    cout << "Could not open or find the image!\n" << endl;
    cout << "usage: " << argv[0] << " <Input image>" << endl;
    return -1;
  }
  cvtColor(src, src_gray, COLOR_BGR2GRAY);
  blur(src_gray, src_gray, Size(3, 3));
  const char* source_window = "Source";
  namedWindow(source_window);
  const int max_thresh = 255;
  createTrackbar("Canny thresh:", source_window, &thresh, max_thresh,
                 thresh_callback);
  thresh_callback(0, 0);
  waitKey();
  return 0;
}

void thresh_callback(int, void*) {
  Mat canny_output;
  Canny(src_gray, canny_output, thresh, thresh * 2);
  vector<vector<Point>> contours;
  findContours(canny_output, contours, RETR_LIST, CHAIN_APPROX_NONE);
  const auto by_size = [](const auto& l, const auto& r) { return l.size() < r.size(); };
  const auto largest_contour = std::max_element(contours.begin(), contours.end(), by_size);

  Mat drawing = Mat::zeros(canny_output.size(), CV_8UC3);

  const auto thickness = 3;

  const auto bounding_box = boundingRect(*largest_contour);
  const auto red = Scalar(0, 0, 255);
  rectangle(drawing, bounding_box, red, thickness);

  const auto color = Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
  const auto draw_all_contours = -1;
  vector<Point> contour_poly;
  approxPolyDP(*largest_contour, contour_poly, 3, false);
  drawContours(drawing, std::vector<decltype(contour_poly)>{ std::move(contour_poly) }, draw_all_contours, color, thickness);

  imshow("Source", src + drawing);
}
