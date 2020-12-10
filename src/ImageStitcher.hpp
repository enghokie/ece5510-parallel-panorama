#pragma once

#include <opencv2/opencv.hpp>

class ImageStitcher {
public:

    ImageStitcher() {};
    ~ImageStitcher() {};
    
    void setHomography(const cv::Mat& homog);
    cv::Mat getHomography();
    bool getHomography(cv::Mat& homog);

    bool computeHomography(const std::pair<cv::Mat, cv::Mat>& imgs,
                           std::pair<cv::Mat, unsigned int>& homog);

    bool computeHomography(const std::pair<cv::Mat, cv::Mat>& imgs,
                           float roiWidthPerc,
                           float roiHeighPerc,
                           std::pair<cv::Mat, unsigned int>& homog);

    bool stitchImages(const std::pair<cv::Mat, unsigned int>& homog,
                      const std::vector<std::pair<cv::Mat, cv::Mat>>& imgPairs,
                      std::vector<cv::Mat>& stitchedImgs);
    
private:
    cv::Mat _homography;
};
