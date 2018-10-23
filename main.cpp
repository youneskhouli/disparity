#include <iostream>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/structured_light.hpp>
#include <opencv2/opencv_modules.hpp>
#include <opencv2/opencv.hpp>   
#include <opencv2/rgbd.hpp>

#define DEBUG_SL 0

static const char *keys = {
    "{@images_list              | | Image list where the captured pattern images are saved}"
    "{@calib_param_path         | | Calibration_parameters}"
    "{@proj_width               | | The projector width used to acquire the pattern}"
    "{@proj_height              | | The projector height used to acquire the pattern}"
    "{@white_thresh             | | The white threshold height (optional)}"
    "{@black_thresh             | | The black threshold (optional)}"
};

int main(int argc, char **argv)
{
    cv::structured_light::GrayCodePattern::Params params;
    cv::CommandLineParser parser(argc, argv, keys);
    // Get the path of where the folders containing left and right should be
    std::string images_path = parser.get<std::string>(0);
    std::string calib_file = parser.get<std::string>(1);
    params.width = parser.get<int>(2);
    params.height = parser.get<int>(3);

    if (images_path.empty() || calib_file.empty() || params.width < 1 || params.height < 1 || argc < 5 || argc > 7)
    {
        std::cout << "Error in the arguments passed... at least one of them is incorrect" << std::endl;
        return -1;
    }

    cv::Ptr<structured_light::GrayCodePattern> graycode = cv::structured_light::GrayCodePattern::create(params);
    size_t white_thresh = 0;
    size_t black_thresh = 0;

    if (argc == 7)
    {
        // If passed, setting the white and black threshold, otherwise using default values
        white_thresh = parser.get<unsigned>(4);
        black_thresh = parser.get<unsigned>(5);
        graycode->setWhiteThreshold(white_thresh);
        graycode->setBlackThreshold(black_thresh);
    }

    cv::FileStorage file_storage(images_path, cv::FileStorage::READ);
    std::string images_path, file_format;
    // TODO: Make a function to have split foderls
    file_storage["path"] >> images_path;
    file_storage["format"] >> file_format;
    images_path += "/*." + file_format;

    std::vector<cv::String> files;
    std::vector<cv::Mat> image_list;
    cv::glob(images_path, files, true);

    for (size_t i = 0; i < files.size(); i++) {
        cv::Mat image = cv::imread(files[i]);
        if (image.empty()) continue;
        image_list.push_back(image);

#if DEBUG_SL
        std::cout << "Image: " << files[i] << " pushed back!" << std::endl;
#endif
    }

    cv::FileStorage file_storage(calib_file, cv::FileStorage::READ);
    Mat cam1_intr, cam1_dist, cam2_intr, cam2_dist, R, T;
    Mat cam1intrinsics, cam1distCoeffs, cam2intrinsics, cam2distCoeffs, R, T;
    fs["cam1_intrinsics"] >> cam1intrinsics;
    fs["cam2_intrinsics"] >> cam2intrinsics;
    fs["cam1_distorsion"] >> cam1distCoeffs;
    fs["cam2_distorsion"] >> cam2distCoeffs;
    fs["R"] >> R;
    fs["T"] >> T;

#if DEBUG_SL
    cout << "cam1intrinsics" << endl << cam1intrinsics << endl;
    cout << "cam1distCoeffs" << endl << cam1distCoeffs << endl;
    cout << "cam2intrinsics" << endl << cam2intrinsics << endl;
    cout << "cam2distCoeffs" << endl << cam2distCoeffs << endl;
    cout << "T" << endl << T << endl << "R" << endl << R << endl;
#endif

    size_t num_pattern_images = graycode->getNumberOfPatternImages();
    vector<vector<cv::Mat>> captured_pattern;
}