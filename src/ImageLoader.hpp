#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

typedef std::pair<unsigned int, std::shared_ptr<std::vector<cv::Mat>>> ImgIdPair;

class ImageLoader {
public:
    ImageLoader()
        : _maxLoadedImgId(0)
    {}

    const unsigned int getMaxImgId() { return _maxLoadedImgId; }

    bool loadImages(std::string imgDirPath);
    bool loadImages(std::vector<std::string> imgDirPaths);
    
    const void getImgPairs(std::vector<ImgIdPair>& imgPairs);
    const bool getImgPairs(unsigned int id, std::vector<ImgIdPair>& imgPairs);
    const bool getImages(unsigned int id, std::vector<cv::Mat>& imgs);

    bool addImage(unsigned int id, cv::Mat& img);
    bool addImages(unsigned int id, std::vector<cv::Mat>& imgs);

private:
    std::vector<ImgIdPair> _imgPairs;
    unsigned int _maxLoadedImgId;
};