#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>
#include <string>
#include <numeric>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <vtkActor.h>
#include <vtkCellArray.h>
#include <vtkFloatArray.h>
#include <vtkImageViewer.h>
#include <vtkInteractorStyleImage.h>
#include <vtkLight.h>
#include <vtkLightCollection.h>
#include <vtkPLYWriter.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkSmartPointer.h>
#include <vtkTriangle.h>

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

#include "verify.h"

void displayMesh(int width, int height, cv::Mat Z, const std::string& filename) {
  /* creating visualization pipeline which basically looks like this:
     vtkPoints -> vtkPolyData -> vtkPolyDataMapper -> vtkActor -> vtkRenderer */
  vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
  vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New();
  vtkSmartPointer<vtkPolyDataMapper> modelMapper =
      vtkSmartPointer<vtkPolyDataMapper>::New();
  vtkSmartPointer<vtkActor> modelActor = vtkSmartPointer<vtkActor>::New();
  vtkSmartPointer<vtkCellArray> vtkTriangles =
      vtkSmartPointer<vtkCellArray>::New();

  /* insert x,y,z coords */
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      points->InsertNextPoint(x, y, Z.at<float>(y, x));
    }
  }

  /* setup the connectivity between grid points */
  vtkSmartPointer<vtkTriangle> triangle = vtkSmartPointer<vtkTriangle>::New();
  triangle->GetPointIds()->SetNumberOfIds(3);
  for (int i = 0; i < height - 1; i++) {
    for (int j = 0; j < width - 1; j++) {
      triangle->GetPointIds()->SetId(0, j + (i * width));
      triangle->GetPointIds()->SetId(1, (i + 1) * width + j);
      triangle->GetPointIds()->SetId(2, j + (i * width) + 1);
      vtkTriangles->InsertNextCell(triangle);
      triangle->GetPointIds()->SetId(0, (i + 1) * width + j);
      triangle->GetPointIds()->SetId(1, (i + 1) * width + j + 1);
      triangle->GetPointIds()->SetId(2, j + (i * width) + 1);
      vtkTriangles->InsertNextCell(triangle);
    }
  }
  polyData->SetPoints(points);
  polyData->SetPolys(vtkTriangles);

  /* meshlab-ish background */
  modelMapper->SetInputData(polyData);
  modelActor->SetMapper(modelMapper);

  /* setting some properties to make it look just right */
  modelActor->GetProperty()->SetSpecularColor(1, 1, 1);
  modelActor->GetProperty()->SetAmbient(0.2);
  modelActor->GetProperty()->SetDiffuse(0.2);
  modelActor->GetProperty()->SetInterpolationToPhong();
  modelActor->GetProperty()->SetSpecular(0.8);
  modelActor->GetProperty()->SetSpecularPower(8.0);

  /* export mesh */
  vtkSmartPointer<vtkPLYWriter> plyExporter =
      vtkSmartPointer<vtkPLYWriter>::New();
  plyExporter->SetInputData(polyData);
  plyExporter->SetFileName(filename.c_str());
  plyExporter->SetColorModeToDefault();
  plyExporter->SetArrayName("Colors");
  plyExporter->Update();
  plyExporter->Write();
}

cv::Mat globalHeights(cv::Mat Pgrads, cv::Mat Qgrads) {
  cv::Mat P(Pgrads.rows, Pgrads.cols, CV_32FC2, cv::Scalar::all(0));
  cv::Mat Q(Pgrads.rows, Pgrads.cols, CV_32FC2, cv::Scalar::all(0));
  cv::Mat Z(Pgrads.rows, Pgrads.cols, CV_32FC2, cv::Scalar::all(0));

  float lambda = 1.0f;
  float mu = 1.0f;

  cv::dft(Pgrads, P, cv::DFT_COMPLEX_OUTPUT);
  cv::dft(Qgrads, Q, cv::DFT_COMPLEX_OUTPUT);
  for (int i = 0; i < Pgrads.rows; i++) {
    for (int j = 0; j < Pgrads.cols; j++) {
      if (i != 0 || j != 0) {
        float u = sin((float)(i * 2 * CV_PI / Pgrads.rows));
        float v = sin((float)(j * 2 * CV_PI / Pgrads.cols));

        float uv = pow(u, 2) + pow(v, 2);
        float d = (1.0f + lambda) * uv + mu * pow(uv, 2);
        Z.at<cv::Vec2f>(i, j)[0] =
            (u * P.at<cv::Vec2f>(i, j)[1] + v * Q.at<cv::Vec2f>(i, j)[1]) / d;
        Z.at<cv::Vec2f>(i, j)[1] =
            (-u * P.at<cv::Vec2f>(i, j)[0] - v * Q.at<cv::Vec2f>(i, j)[0]) / d;
      }
    }
  }

  /* setting unknown average height to zero */
  Z.at<cv::Vec2f>(0, 0)[0] = 0.0f;
  Z.at<cv::Vec2f>(0, 0)[1] = 0.0f;

  cv::dft(Z, Z, cv::DFT_INVERSE | cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);

  return Z;
}

cv::Vec3f getLightDirFromSphere(cv::Mat Image, cv::Rect boundingbox) {
  const int THRESH = 254;
  const float radius = boundingbox.width / 2.0f;

  cv::Mat Binary;
  cv::threshold(Image, Binary, THRESH, 255, cv::THRESH_BINARY);
  cv::Mat SubImage(Binary, boundingbox);

  /* calculate center of pixels */
  cv::Moments m = cv::moments(SubImage, false);
  cv::Point center(m.m10 / m.m00, m.m01 / m.m00);

  /* x,y are swapped here */
  float x = (center.y - radius) / radius;
  float y = (center.x - radius) / radius;
  float z = std::sqrt(1.0 - std::pow(x, 2.0) - std::pow(y, 2.0));

  return cv::Vec3f(x, y, z);
}

cv::Rect getBoundingBox(cv::Mat Mask) {
  std::vector<std::vector<cv::Point>> v;
  cv::findContours(Mask.clone(), v, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
  assert(v.size() > 0);
  const auto by_size = [](const auto& l, const auto& r) { return l.size() < r.size(); };
  const auto largest_contour = std::max_element(v.begin(), v.end(), by_size);
  return cv::boundingRect(*largest_contour);
}

bool processCommandLineArguments(int argc, char** argv, std::string& calibration_path, std::string& model_path) {
	boost::program_options::options_description command_line_options("OPTIONS", 10000);
	command_line_options.add_options()
		("help,h", "produce this help message")
    ("calibration,c", boost::program_options::value<std::string>(&calibration_path)->required(),
        "Path to a directory with calibration sphere images.\n"
        "Names of images should satisfy next requirement:\n"
        "for given name `name` image name must next string `name.%u.ext`, where `ext` is your image extention.\n"
        "Among the images there should be an image with name, which contains substring `.mask.` instead of a `number` of the image.\n"
        "This image should contain mask of the object, so that the program could differentiate where the desired object is placed on the image.\n"
        "Images or files, which names does not satisfy this requirement are ignored.\n")
    ("model,m",       boost::program_options::value<std::string>(&model_path)->required(),
        "Path to a directory with model images.\n"
        "Requirements are the same as for calibration images.\n"
        "Amount of model images must be exact as the amount of calibration images.\n"
        "Each calibration image must correspond with one model image.\n")
		;

	boost::program_options::variables_map var_map;
	const auto opts = boost::program_options::command_line_parser(argc, argv)
		.options(command_line_options)
    .run();
	boost::program_options::store(opts, var_map);

	if (var_map.count("help")) {
    std::cout
      << "USAGE:"
      << "\t" << argv[0] << " [OPTIONS]\n"
      << command_line_options << '\n';
    return false;
  }

	boost::program_options::notify(var_map);
  return true;
}

struct parseDirectoryResult {
  boost::filesystem::path mask;
  std::vector<boost::filesystem::path> images;
};

parseDirectoryResult parseDirectory(const boost::filesystem::path& directory) {
  using namespace boost::filesystem;
  parseDirectoryResult result;
  const auto substr_between_dots = [](const path& path) -> std::string {
    const auto name_without_extention = path.stem().string();
    const auto dot_pos = name_without_extention.find('.');
    if (dot_pos == std::string::npos)
      return {};
    return name_without_extention.substr(dot_pos + 1);
  };
  const auto extract_number = [](const std::string& s) -> size_t {
    size_t pos;
    if (std::sscanf(s.c_str(), "%lu", &pos) != 1)
      return std::string::npos;
    return pos;
  };
  for (const auto& entry : boost::make_iterator_range(directory_iterator{ directory }, directory_iterator{})) {
    const auto image_path = absolute(entry.path());
    const auto after_dot = substr_between_dots(image_path);
    if (after_dot == "mask") {
      if (!result.mask.empty())
        throw std::logic_error("found several masks in directory " + directory.string());
      result.mask = absolute(entry.path());
      continue;
    }

    if (extract_number(after_dot) == std::string::npos)
      continue;

    result.images.push_back(image_path);
  }
  const auto by_image_number = [&](const auto& lhs, const auto& rhs) {
    return extract_number(substr_between_dots(lhs)) < extract_number(substr_between_dots(rhs));
  };
  std::sort(result.images.begin(), result.images.end(), by_image_number);
  std::cout << "found " << result.images.size() << " images in "  << directory << ":\n";
  for (const auto& image : result.images) {
    std::cout << "\t" << image << "\n";
  }
  return result;
}

int main(int argc, char** argv) {
  std::string calibration_path;
  std::string model_path;
  if (!processCommandLineArguments(argc, argv, calibration_path, model_path))
    return 1;

  const auto [calibration_mask_file, calibration_images] = parseDirectory(calibration_path);
  VERIFY_LOG_RETURN(!calibration_images.empty(), "error: no calibration images which satisfy requirement were found, abort", 1);
  VERIFY_LOG_RETURN(!calibration_mask_file.empty(), "error: no calibration mask was found, abort", 1);

  const auto [model_mask_file, model_images] = parseDirectory(model_path);
  VERIFY_LOG_RETURN(!model_images.empty(), "error: no model images which satisfy requirement were found, abort", 1);
  VERIFY_LOG_RETURN(!model_mask_file.empty(), "error: no model mask was found, abort", 1);

  VERIFY_LOG_RETURN(model_images.size() == calibration_images.size(), "error: expected equal amount of images in calibration and model directiories, abort", 2);

  const auto NUM_IMGS = model_images.size();

  std::vector<cv::Mat> calibImages;
  std::vector<cv::Mat> modelImages;
  cv::Mat Lights(NUM_IMGS, 3, CV_32F);
  auto Mask      = cv::imread(calibration_mask_file.string(), cv::IMREAD_GRAYSCALE);
  VERIFY_LOG_RETURN(Mask.data != nullptr, "error: failed to read calibration mask file", 3);
  auto ModelMask = cv::imread(model_mask_file.string(), cv::IMREAD_GRAYSCALE);
  VERIFY_LOG_RETURN(ModelMask.data != nullptr, "error: failed to read model mask file", 3);
  cv::Rect bb = getBoundingBox(Mask);
  for (auto i = 0u; i < NUM_IMGS; i++) {
    cv::Mat Calib = cv::imread(calibration_images[i].string(), cv::IMREAD_GRAYSCALE);
    VERIFY_LOG_RETURN(Calib.data != nullptr, "error: failed to read calibration file: " << calibration_images[i], 3);
    cv::Mat tmp   = cv::imread(model_images[i].string(), cv::IMREAD_GRAYSCALE);
    VERIFY_LOG_RETURN(tmp.data != nullptr, "error: failed to read model file: " << model_images[i], 3);
    cv::Mat Model;
    tmp.copyTo(Model, ModelMask);
    cv::Vec3f light = getLightDirFromSphere(Calib, bb);
    Lights.at<float>(i, 0) = light[0];
    Lights.at<float>(i, 1) = light[1];
    Lights.at<float>(i, 2) = light[2];
    calibImages.push_back(Calib);
    modelImages.push_back(Model);
  }

  const int height = calibImages[0].rows;
  const int width = calibImages[0].cols;
  /* light directions, surface normals, p,q gradients */
  cv::Mat LightsInv;
  cv::invert(Lights, LightsInv, cv::DECOMP_SVD);

  cv::Mat Normals(height, width, CV_32FC3, cv::Scalar::all(0));
  cv::Mat Pgrads(height, width, CV_32F, cv::Scalar::all(0));
  cv::Mat Qgrads(height, width, CV_32F, cv::Scalar::all(0));
  /* estimate surface normals and p,q gradients */
  for (int x = 0; x < width; x++) {
    for (int y = 0; y < height; y++) {
      std::vector<float> I(NUM_IMGS);
      for (auto i = 0u; i < NUM_IMGS; i++) {
        I[i] = modelImages[i].at<uchar>(cv::Point(x, y));
      }

      cv::Mat n = LightsInv * cv::Mat(I);
      float p = std::sqrt(cv::Mat(n).dot(n));
      if (p > 0) {
        n = n / p;
      }

      if (std::abs(n.at<float>(2, 0)) < std::numeric_limits<float>::epsilon()) {
        n.at<float>(2, 0) = 1.0;
      }

      Normals.at<cv::Vec3f>(cv::Point(x, y)) = n;
      Pgrads.at<float>(cv::Point(x, y)) =
          n.at<float>(0, 0) / n.at<float>(2, 0);
      Qgrads.at<float>(cv::Point(x, y)) =
          n.at<float>(1, 0) / n.at<float>(2, 0);
    }
  }

  cv::Mat Normalmap;
  cv::cvtColor(Normals, Normalmap, cv::COLOR_BGR2RGB);
  const auto normal_map_output = boost::filesystem::path{ model_path } / std::string{ "normal" + model_images[0].extension().string() };
  // The `cv::imwrite(...)` function cannot handle images stored with float values in the range [0, 1]
  // See https://stackoverflow.com/a/54165573/15751315.
  cv::imwrite(normal_map_output.string(), 255 * Normalmap);

  /* global integration of surface normals */
  cv::Mat Z = globalHeights(Pgrads, Qgrads);

  /* display reconstruction */
  const auto reconstruction_output = boost::filesystem::path{ model_path } / "reconstruction.ply";
  displayMesh(Pgrads.cols, Pgrads.rows, Z, reconstruction_output.string());

  return 0;
}
