#pragma once

#include <opencv2/opencv.hpp>

class ImageStitcher {
public:
    ImageStitcher() {};
    ~ImageStitcher() {};

    void setHomography(const cv::Mat& homog);

    const cv::Mat getHomography();
    const bool getHomography(cv::Mat& homog);

    const bool computeHomography(const std::pair<cv::Mat, cv::Mat>& imgs,
                                 cv::Mat& homog);
    const bool computeHomography(const std::pair<cv::Mat, cv::Mat>& imgs,
                                 float roiWidthPerc,
                                 float roiHeightPerc,
                                 cv::Mat& homog);

    const bool manualStitch(const cv::Mat& homog,
                            const std::vector<std::pair<cv::Mat, cv::Mat>>& imgPairs,
                            std::vector<cv::Mat>& stitchedImgs);

private:
    cv::Mat _homography;
};
