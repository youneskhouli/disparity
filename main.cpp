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
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/ximgproc.hpp>

// Project size
#define PROJECT_WIDTH 1023
#define PROJECT_HEIGHT 624
// Resizing
#define RESIZE_WIDTH 1224
#define RESIZE_HEIGHT 816
// Tresh values
#define W_TRESH 0
#define B_TRESH 0
// Param file
#define PARAM_FILE "cam_param.yml"
// Filtering parameters
#define WLS_LAMBDA 8000.0
#define WLS_SIGMA 0.1
// Stereo matching parameters
#define MAX_DISPARITY 160
#define WINDOW_SIZE 7 // Recommended sizes are 3, 7 and 15

static const char *keys = {
    "{@images_path              | | Path for the images }"
    "{@white_thresh             | | The white threshold height (optional)}"
    "{@black_thresh             | | The black threshold (optional)}"};

int main(int argc, char **argv)
{
    cv::structured_light::GrayCodePattern::Params params;
    cv::CommandLineParser parser(argc, argv, keys);
    std::string images_path = parser.get<std::string>(0);
    std::string calib_file = PARAM_FILE;
    params.width = PROJECT_WIDTH;
    params.height = PROJECT_HEIGHT;

    if (images_path.empty())
    {
        std::cout << "No path argument passed or wrong argument" << std::endl;
        return -1;
    }

    cv::Ptr<cv::structured_light::GrayCodePattern> graycode = cv::structured_light::GrayCodePattern::create(params);
    size_t white_thresh = W_TRESH;
    size_t black_thresh = B_TRESH;

    if (argc == 3)
    {
        // If passed, setting the white and black threshold, otherwise using default values
        white_thresh = parser.get<unsigned>(1);
        black_thresh = parser.get<unsigned>(2);
        graycode->setWhiteThreshold(white_thresh);
        graycode->setBlackThreshold(black_thresh);
    }

    std::string right_path, left_path;
    right_path = images_path + "/right/*.JPG";
    left_path = images_path + "/left/*.JPG";

    std::vector<cv::String> r_files;
    std::vector<cv::String> l_files;
    std::vector<cv::String> files;

    cv::glob(left_path, l_files, true);
    cv::glob(right_path, r_files, true);

    files.insert(files.end(), l_files.begin(), l_files.end());
    files.insert(files.end(), r_files.begin(), r_files.end());

    cv::FileStorage calibration_fs(calib_file, cv::FileStorage::READ);
    cv::Mat cam1intrinsics, cam1distCoeffs, cam2intrinsics, cam2distCoeffs, R, T;
    calibration_fs["cam1_intrinsics"] >> cam1intrinsics;
    calibration_fs["cam2_intrinsics"] >> cam2intrinsics;
    calibration_fs["cam1_distorsion"] >> cam1distCoeffs;
    calibration_fs["cam2_distorsion"] >> cam2distCoeffs;
    calibration_fs["R"] >> R;
    calibration_fs["T"] >> T;

    std::cout << "Params found!" << std::endl;

    size_t numberOfPatternImages = graycode->getNumberOfPatternImages();
    std::vector<std::vector<cv::Mat>> captured_pattern;

    captured_pattern.resize(2);
    captured_pattern[0].resize(numberOfPatternImages);
    captured_pattern[1].resize(numberOfPatternImages);

    cv::Mat color = imread(files[numberOfPatternImages], cv::IMREAD_COLOR);
    cv::Size imagesSize = color.size();

    std::cout << "Rectifying images..." << std::endl;

    cv::Mat R1, R2, P1, P2, Q;
    cv::Rect validRoi[2];
    cv::stereoRectify(cam1intrinsics, cam1distCoeffs, cam2intrinsics, cam2distCoeffs, imagesSize,
                      R, T, R1, R2, P1, P2, Q, 0, -1, imagesSize, &validRoi[0], &validRoi[1]);
    cv::Mat map1x, map1y, map2x, map2y;
    cv::initUndistortRectifyMap(cam1intrinsics, cam1distCoeffs, R1, P1, imagesSize, CV_32FC1, map1x, map1y);
    cv::initUndistortRectifyMap(cam2intrinsics, cam2distCoeffs, R2, P2, imagesSize, CV_32FC1, map2x, map2y);

    for (size_t i = 0; i < numberOfPatternImages; i++)
    {
        captured_pattern[0][i] = imread(files[i], cv::IMREAD_GRAYSCALE);
        captured_pattern[1][i] = imread(files[i + numberOfPatternImages + 2], cv::IMREAD_GRAYSCALE);

        if ((!captured_pattern[0][i].data) || (!captured_pattern[1][i].data))
        {
            std::cout << "Empty images" << std::endl;
            return -1;
        }

        cv::remap(captured_pattern[1][i], captured_pattern[1][i], map1x, map1y, cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar());
        cv::remap(captured_pattern[0][i], captured_pattern[0][i], map2x, map2y, cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar());
    }

    std::cout << "Done!" << std::endl;

    std::vector<cv::Mat> blackImages, whiteImages;
    blackImages.resize(2);
    whiteImages.resize(2);

    cv::cvtColor(color, whiteImages[0], cv::COLOR_RGB2GRAY);
    whiteImages[1] = cv::imread(files[2 * numberOfPatternImages + 2], cv::IMREAD_GRAYSCALE);
    blackImages[0] = cv::imread(files[numberOfPatternImages + 1], cv::IMREAD_GRAYSCALE);
    blackImages[1] = cv::imread(files[2 * numberOfPatternImages + 2 + 1], cv::IMREAD_GRAYSCALE);
    cv::remap(color, color, map2x, map2y, cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar());
    cv::remap(whiteImages[0], whiteImages[0], map2x, map2y, cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar());
    cv::remap(whiteImages[1], whiteImages[1], map1x, map1y, cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar());
    cv::remap(blackImages[0], blackImages[0], map2x, map2y, cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar());
    cv::remap(blackImages[1], blackImages[1], map1x, map1y, cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar());

    std::cout << std::endl
              << "Decoding pattern..." << std::endl;
    cv::Mat disparityMap;
    bool decoded = graycode->decode(captured_pattern, disparityMap, blackImages, whiteImages, cv::structured_light::DECODE_3D_UNDERWORLD);

    if (decoded)
    {
        std::cout << std::endl
                  << "Done!" << std::endl;

        double min;
        double max;
        cv::minMaxIdx(disparityMap, &min, &max);
        cv::Mat cm_disp, scaledDisparityMap;
        std::cout << "Min: " << min << std::endl
                  << "Max: " << max << std::endl;
        cv::convertScaleAbs(disparityMap, scaledDisparityMap, 255 / (max - min));
        cv::applyColorMap(scaledDisparityMap, cm_disp, cv::COLORMAP_JET);

        cv::resize(cm_disp, cm_disp, cv::Size(RESIZE_WIDTH, RESIZE_HEIGHT));

        disparityMap.convertTo(disparityMap, CV_32FC1);

        cv::Mat dst, thresholded_disp;
        cv::threshold(scaledDisparityMap, thresholded_disp, 0, 255, cv::THRESH_OTSU + cv::THRESH_BINARY);
        cv::resize(thresholded_disp, dst, cv::Size(RESIZE_WIDTH, RESIZE_HEIGHT));

        std::cout << "Converting the mats..." << std::endl;
        dst.convertTo(dst, CV_32FC1);

        std::cout << "Saving exr files..." << std::endl;
        cv::imwrite("disparity_map.exr", disparityMap);
        cv::imwrite("dst.exr", dst);

        // There's still some work to do bellow this point...
        cv::Mat filtered_disparity, filtered_dst;
        cv::Ptr<cv::ximgproc::DisparityWLSFilter> wls_filter;
        cv::Ptr<cv::StereoBM> left_matcher = cv::StereoBM::create(MAX_DISPARITY, WINDOW_SIZE);
        // Automatically set up the relevant filter parameters
        wls_filter = cv::ximgproc::createDisparityWLSFilter(left_matcher);
        // Set the lambda and sigma values
        wls_filter->setLambda(WLS_LAMBDA);
        wls_filter->setSigmaColor(WLS_SIGMA);
        // Need to convert the Mats to CV_16S for filtering (I think...)
        dst.convertTo(dst, CV_16S);
        disparityMap.convertTo(disparityMap, CV_16S);
        // Filter it, passing an unmodified image for size reference
        cv::Mat size_ref = imread(files[0], cv::IMREAD_COLOR);
        std::cout << "Filtering the results..." << std::endl;
        std::cout << "Filtering dst..." << std::endl;
        // for some reason not having the "optional" right disparity, makes the program fail
        // Gotta look into it.
        wls_filter->filter(dst, size_ref, filtered_dst, dst);
        cv::imshow("dst", filtered_dst);
        std::cout << "Filtering disparity map..." << std::endl;
        wls_filter->filter(disparityMap, size_ref, filtered_disparity, disparityMap);
        cv::imshow("disparity", filtered_disparity);
        // Convert and save results
        std::cout << "Converting the filtered results..." << std::endl;
        filtered_disparity.convertTo(filtered_disparity, CV_32FC1);
        filtered_dst.convertTo(filtered_dst, CV_32FC1);
        std::cout << "Saving the filtered results..." << std::endl;
        cv::imwrite("filtered_disparity.exr", filtered_disparity);
        cv::imwrite("filtered_dst.exr", filtered_dst);
    }
}
