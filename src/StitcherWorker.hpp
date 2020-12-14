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