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

#include <vector>
#include <opencv2/opencv.hpp>

#include "ThreadSafeDequeue.hpp"
#include "ImageStitcher.hpp"

typedef std::pair<cv::Mat, cv::Mat> ImgPair;
typedef std::pair<unsigned int, std::vector<cv::Mat>> JobIdPair;
typedef std::pair<unsigned int, cv::Mat> ResIdPair;

class StitcherWorker 
{
public:
    StitcherWorker(ThreadSafeDequeue<JobIdPair>& jobQueue,
                   ThreadSafeDequeue<ResIdPair>& resQueue,
                   ImageStitcher::StitcherMode stitcherMode)
        : _jobQueue(jobQueue)
        , _resQueue(resQueue)
        , _stitcherMode(stitcherMode)
        , _quit(false)
    {
        if (_stitcherMode == ImageStitcher::StitcherMode::StitcherMode_OpenCV)
        {
            _cvStitcher = cv::Stitcher::create();
            _cvStitcher->setRegistrationResol(-1);
            _cvStitcher->setSeamEstimationResol(-1);
            _cvStitcher->setCompositingResol(-1);
            _cvStitcher->setPanoConfidenceThresh(0.3);
        }
    }
    ~StitcherWorker() { _quit = true; }

    void run();
    void quit();

    bool stitchImgs(std::vector<cv::Mat>& curImages, cv::Mat& stitchedImg);

    bool manualStitchImgs(const ImgPair& imgPairs,
                          float roiWidthPerc,
                          float roiHeightPerc,
                          cv::Mat& stitchedImg);

private:
    ThreadSafeDequeue<JobIdPair>& _jobQueue;
    ThreadSafeDequeue<ResIdPair>& _resQueue;
    ImageStitcher::StitcherMode _stitcherMode;
    ImageStitcher _stitcher;
    cv::Ptr<cv::Stitcher> _cvStitcher;
    volatile bool _quit;
};