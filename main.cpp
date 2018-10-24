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

static const char *keys = {
    "{@images_path               | | Path for the images }"
    "{@white_thresh             | | The white threshold height (optional)}"
    "{@black_thresh             | | The black threshold (optional)}"
};

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

    cv::FileStorage fs(calib_file, cv::FileStorage::READ);
    cv::Mat cam1intrinsics, cam1distCoeffs, cam2intrinsics, cam2distCoeffs, R, T;
    fs["cam1_intrinsics"] >> cam1intrinsics;
    fs["cam2_intrinsics"] >> cam2intrinsics;
    fs["cam1_distorsion"] >> cam1distCoeffs;
    fs["cam2_distorsion"] >> cam2distCoeffs;
    fs["R"] >> R;
    fs["T"] >> T;

    std::cout << "Params found!" << std::endl;

    size_t numberOfPatternImages = graycode->getNumberOfPatternImages();
    std::vector<std::vector<cv::Mat> > captured_pattern;

    captured_pattern.resize(2);
    captured_pattern[0].resize(numberOfPatternImages);
    captured_pattern[1].resize(numberOfPatternImages);

    cv::Mat color = imread(files[numberOfPatternImages], cv::IMREAD_COLOR);
    cv::Size imagesSize = color.size();

    std::cout << "Rectifying images..." << std::endl;

    cv::Mat R1, R2, P1, P2, Q;
    cv::Rect validRoi[2];
    cv::stereoRectify(cam1intrinsics, cam1distCoeffs, cam2intrinsics, cam2distCoeffs, imagesSize, R, T, R1, R2, P1, P2, Q, 0, -1, imagesSize, &validRoi[0], &validRoi[1]);
    cv::Mat map1x, map1y, map2x, map2y;
    initUndistortRectifyMap(cam1intrinsics, cam1distCoeffs, R1, P1, imagesSize, CV_32FC1, map1x, map1y);
    initUndistortRectifyMap(cam2intrinsics, cam2distCoeffs, R2, P2, imagesSize, CV_32FC1, map2x, map2y);

    for (size_t i = 0; i < numberOfPatternImages; i++)
    {
        captured_pattern[0][i] = imread(files[i], cv::IMREAD_GRAYSCALE);
        captured_pattern[1][i] = imread(files[i + numberOfPatternImages + 2], cv::IMREAD_GRAYSCALE);

        if ((!captured_pattern[0][i].data) || (!captured_pattern[1][i].data))
        {
            std::cout << "Empty images" << std::endl;
            return -1;
        }

        remap(captured_pattern[1][i], captured_pattern[1][i], map1x, map1y, cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar());
        remap(captured_pattern[0][i], captured_pattern[0][i], map2x, map2y, cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar());
    }

    std::cout << "Done!" << std::endl;

    std::vector<cv::Mat> blackImages, whiteImages;
    blackImages.resize(2);
    whiteImages.resize(2);

    cvtColor(color, whiteImages[0], cv::COLOR_RGB2GRAY);
    whiteImages[1] = imread(files[2 * numberOfPatternImages + 2], cv::IMREAD_GRAYSCALE);
    blackImages[0] = imread(files[numberOfPatternImages + 1], cv::IMREAD_GRAYSCALE);
    blackImages[1] = imread(files[2 * numberOfPatternImages + 2 + 1], cv::IMREAD_GRAYSCALE);
    remap(color, color, map2x, map2y, cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar());
    remap(whiteImages[0], whiteImages[0], map2x, map2y, cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar());
    remap(whiteImages[1], whiteImages[1], map1x, map1y, cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar());
    remap(blackImages[0], blackImages[0], map2x, map2y, cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar());
    remap(blackImages[1], blackImages[1], map1x, map1y, cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar());

    std::cout << std::endl << "Decoding pattern..." << std::endl;
    cv::Mat disparityMap;
    bool decoded = graycode->decode(captured_pattern, disparityMap, blackImages, whiteImages, cv::structured_light::DECODE_3D_UNDERWORLD);

    if (decoded)
    {
        std::cout << std::endl << "Done!" << std::endl;

        double min;
        double max;
        minMaxIdx(disparityMap, &min, &max);
        cv::Mat cm_disp, scaledDisparityMap;
        std::cout << "Min: " << min << std::endl << "Max: " << max << std::endl;
        convertScaleAbs(disparityMap, scaledDisparityMap, 255 / (max - min));
        applyColorMap(scaledDisparityMap, cm_disp, cv::COLORMAP_JET);

        resize(cm_disp, cm_disp, cv::Size(RESIZE_WIDTH, RESIZE_HEIGHT));

        disparityMap.convertTo(disparityMap, CV_32FC1);

        cv::Mat dst, thresholded_disp;
        threshold(scaledDisparityMap, thresholded_disp, 0, 255, cv::THRESH_OTSU + cv::THRESH_BINARY);
        resize(thresholded_disp, dst, cv::Size(RESIZE_WIDTH, RESIZE_HEIGHT));

        std::cout << "Converting the mats" << std::endl;
        dst.convertTo(dst, CV_32FC1);

        std::cout << "Saving exr files..." << std::endl;
        imwrite("disparity_map.exr", disparityMap);
        imwrite("dst.exr", dst);
    }
}