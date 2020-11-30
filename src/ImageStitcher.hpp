#pragma once

#include <opencv2/opencv.hpp>

class ImageStitcher {
public:

    ImageStitcher() {};
    ~ImageStitcher() {};
    
    bool stitchImages(cv::Mat& src, cv::Mat& dst, cv::Mat& stitchedImg);
    
private:
    
};
