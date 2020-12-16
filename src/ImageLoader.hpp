/***
ParallelPanorama: Concurrently stitches together images from files and displays them.
Copyright (C) 2020 Braedon Dickerson and Amir Kimiyaie
   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 3 of the License, or
   (at your option) any later version.
   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.
   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software Foundation,
   Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301  USA
***/

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
    const bool getImage(unsigned int id, cv::Mat& img);

    const bool popImage(unsigned int id, cv::Mat& img);

    bool addImage(unsigned int id, cv::Mat& img);
    bool addImages(unsigned int id, std::vector<cv::Mat>& imgs);

private:
    std::vector<ImgIdPair> _imgPairs;
    unsigned int _maxLoadedImgId;
};