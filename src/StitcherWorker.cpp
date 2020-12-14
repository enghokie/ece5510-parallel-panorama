#include "StitcherWorker.hpp"

const float STITCH_WIDTH_PERCENTAGE = 0.60;
const float STITCH_HEIGHT_PERCENTAGE = 1.0;

void StitcherWorker::run()
{
    while (!_quit)
    {
        JobIdPair job(std::move(_jobQueue.pop()));
        if (job.second.empty())
            continue;

        cv::Mat stitchedImg;
        if (!stitchImgs(job.second, stitchedImg))
            continue; // implement spdlog to do thread safe logging

         ResIdPair pair(job.first, std::move(stitchedImg));
        _resQueue.push(pair);
    }
}

void StitcherWorker::quit()
{
    // Do not call from worker thread
    _quit = true;
}

bool StitcherWorker::stitchImgs(std::vector<cv::Mat>& curImages, cv::Mat& stitchedImg)
{
    static int imgNum = 0;
    std::vector<cv::Mat> nextImages;
    for (int i = 0; i < curImages.size(); i += 2)
    {
        cv::Mat curStitchedImg;
        if (_stitcherMode == ImageStitcher::StitcherMode_OpenCV)
        {
            std::vector<cv::Mat> imgs = { curImages[i], curImages[i + 1], };
            if (_cvStitcher.empty())
            {
                std::cerr << "Error(stitchAllImgs): OpenCV Stitcher is null." << std::endl;
                return false;
            }

            cv::Stitcher::Status res = _cvStitcher->stitch(imgs, curStitchedImg);
            if (res != cv::Stitcher::OK)
            {
                std::cerr << "Error(stitchAllImgs): Failed to stitch images with opencv for index i - "
                    << i << ", Error code: " << res << std::endl;
                continue;
            }
        }
        else
        {
            //std::cout << "BDUB(stitchImgs): Manually stitching images." << std::endl;
            ImgPair imgPair(curImages[i], curImages[i + 1]);
            if (!manualStitchImgs(imgPair, STITCH_WIDTH_PERCENTAGE, STITCH_HEIGHT_PERCENTAGE, curStitchedImg))
            {
                std::cerr << "Error(stitchAllImgs): Failed to manually stitch images for index i - " << i << std::endl;
                continue;
            }
            //std::cout << "BDUB(stitchImgs): Done manually stitching images." << std::endl;
        }

        nextImages.push_back(std::move(curStitchedImg));
    }
    curImages.clear();

    // Check if we're done
    //std::cout << "BDUB(stitchImgs): nextImages size " << nextImages.size() << std::endl;
    if (nextImages.size() < 2)
    {
        if (nextImages.empty())
            return false;

        stitchedImg = std::move(nextImages.back());
        return true;
    }

    if (!stitchImgs(nextImages, stitchedImg))
        return false;

    return true;
}

bool StitcherWorker::manualStitchImgs(const ImgPair& imgPair,
                                      float roiWidthPerc,
                                      float roiHeightPerc,
                                      cv::Mat& stitchedImg)
{
    cv::Mat homography;
    if (roiWidthPerc <= 0.0 || roiHeightPerc <= 0.0)
    {
        if (!_stitcher.computeHomography(imgPair, homography))
        {
            std::cerr << "Error(manualStitchImgs): Failed to compute homography for images." << std::endl;
            return false;
        }
    }
    else
    {
        if (!_stitcher.computeHomography(imgPair, roiWidthPerc, roiHeightPerc, homography))
        {
            std::cerr << "Error(manualStitchImgs): Failed to compute homography-roi for images." << std::endl;
            return false;
        }
    }

    std::vector<ImgPair> imgPairs = { imgPair };
    std::vector<cv::Mat> curStitchedImgs;
    if (!_stitcher.manualStitch(homography, imgPairs, curStitchedImgs))
    {
        std::cerr << "Error(manualStitchImgs): Failed to stitch images." << std::endl;
        return false;
    }

    stitchedImg = std::move(curStitchedImgs.front());
    if (stitchedImg.empty())
        return false;

    return true;
}
