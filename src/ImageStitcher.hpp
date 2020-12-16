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

#include <opencv2/opencv.hpp>

class ImageStitcher {
public:
    enum StitcherMode
    {
        StitcherMode_Manual = 0,
        StitcherMode_OpenCV = 1
    };

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
