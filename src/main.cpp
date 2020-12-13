#include <iostream>
#include <chrono>
#include <thread>
#include <filesystem>
#include <vector>

#include <opencv2/opencv.hpp>

#include "ImageStitcher.hpp"
#include "ImageLoader.hpp"

typedef std::pair<cv::Mat, cv::Mat> ImgPair;

const float DISPLAY_PERCENTAGE = 0.3;
const float STITCH_WIDTH_PERCENTAGE = 0.3;
const float STITCH_HEIGHT_PERCENTAGE = 1.0;

enum StitcherMode
{
    StitcherMode_Manual = 0,
    StitcherMode_OpenCV = 1
};

bool stitchAllImgs(StitcherMode mode,
                   ImageLoader& imgLoader,
                   ImageStitcher& stitcher,
                   cv::Ptr<cv::Stitcher> cvStitcher,
                   cv::Mat& stitchedImg);

bool manualStitchImgs(ImageStitcher& stitcher,
                      const ImgPair& imgPairs,
                      float roiWidthPerc,
                      float roiHeightPerc,
                      cv::Mat& stitchedImg);

void printUsage();

int main(int argc, char* argv[])
{
    if (argc != 3) {
        printUsage();
        return 1;
    }

    StitcherMode stitchMode = StitcherMode_Manual;
    cv::Ptr<cv::Stitcher> cvStitcher;
    ImageStitcher stitcher;
    if (std::string(argv[1]).find("opencv") != std::string::npos)
    {
        stitchMode = StitcherMode_OpenCV;
        cvStitcher = cv::Stitcher::create();
        cvStitcher->setRegistrationResol(-1);
        cvStitcher->setSeamEstimationResol(-1);
        cvStitcher->setCompositingResol(-1);
        cvStitcher->setPanoConfidenceThresh(0.3);
    }

    ImageLoader initImgLoader;
    for (const auto& entry : std::filesystem::directory_iterator(argv[2]))
    {
        if (!initImgLoader.loadImages(entry.path().string()))
            std::cerr << "Error(main): Failed to load images from directory - " << entry.path() << std::endl;
        else
            std::cout << "Loaded images from - " << entry.path() << std::endl;
    }

    cv::Mat stitchedImg;
    if (!stitchAllImgs(stitchMode, initImgLoader, stitcher, cvStitcher, stitchedImg))
    {
        std::cerr << "Error(main): Could not stitch images!" << std::endl;
        return 1;
    }

    cv::resize(stitchedImg, stitchedImg, cv::Size(), DISPLAY_PERCENTAGE, DISPLAY_PERCENTAGE);
    cv::imshow("Stitched Image", stitchedImg);
    
    cv::waitKey(0);
    return 0;
}


bool stitchAllImgs(StitcherMode mode,
                   ImageLoader& imgLoader,
                   ImageStitcher& stitcher,
                   cv::Ptr<cv::Stitcher> cvStitcher,
                   cv::Mat& stitchedImg)
{
    static int imgNum = 0;
    ImageLoader nextImages;
    std::vector<cv::Mat> curImages;
    while (true)
    {
        for (int i = 0; i < imgLoader.getMaxImgId(); i++)
        {
            cv::Mat img;
            if (!imgLoader.popImage(i + 1, img))
                break;

            curImages.push_back(std::move(img));
        }

        if (curImages.empty())
            break;

        std::cout << "Stitching " << curImages.size() << " images with "
                  << (mode == StitcherMode::StitcherMode_Manual ? "manual" : "opencv") << " mode." << std::endl;

        std::chrono::high_resolution_clock time;
        auto start = time.now();
        for (int i = 0; i < curImages.size(); i += 2)
        {
            cv::Mat curStitchedImg;
            if (mode == StitcherMode_OpenCV)
            {
                std::vector<cv::Mat> imgs = { curImages[i], curImages[i + 1], };
                if (cvStitcher.empty())
                {
                    std::cerr << "Error(stitchAllImgs): OpenCV Stitcher is null." << std::endl;
                    return false;
                }

                cv::Stitcher::Status res = cvStitcher->stitch(imgs, curStitchedImg);
                if (res != cv::Stitcher::OK)
                {
                    std::cerr << "Error(stitchAllImgs): Failed to stitch images with opencv for index i - "
                              << i << ", Error code: " << res << std::endl;
                    continue;
                }
            }
            else
            {
                ImgPair imgPair(curImages[i], curImages[i + 1]);
                if (!manualStitchImgs(stitcher, imgPair, STITCH_WIDTH_PERCENTAGE, STITCH_HEIGHT_PERCENTAGE, curStitchedImg))
                {
                    std::cerr << "Error(stitchAllImgs): Failed to manually stitch images for index i - " << i << std::endl;
                    continue;
                }
            }

            std::vector<cv::Mat> curStitchedImgs = { curStitchedImg };
            nextImages.addImages(nextImages.getMaxImgId() + 1, curStitchedImgs);
        }
        auto end = time.now();
        std::cout << "Total stitch time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << std::endl;
        curImages.clear();
    }

    // Check if we're done
    if (nextImages.getMaxImgId() < 2)
    {
        std::vector<cv::Mat> images;
        if (!nextImages.getImages(1, images))
            return false;

        // Acquired the final stitched image
        stitchedImg = std::move(images.front());
        return true;
    }

    return stitchAllImgs(mode, nextImages, stitcher, cvStitcher, stitchedImg);
}

bool manualStitchImgs(ImageStitcher& stitcher,
                      const ImgPair& imgPair,
                      float roiWidthPerc,
                      float roiHeightPerc,
                      cv::Mat& stitchedImg)
{
    cv::Mat homography;
    if (roiWidthPerc <= 0.0 || roiHeightPerc <= 0.0)
    {
        if (!stitcher.computeHomography(imgPair, homography))
        {
            std::cerr << "Error(manualStitchImgs): Failed to compute homography for images." << std::endl;
            return false;
        }
    }
    else
    {
        if (!stitcher.computeHomography(imgPair, roiWidthPerc, roiHeightPerc, homography))
        {
            std::cerr << "Error(manualStitchImgs): Failed to compute homography-roi for images." << std::endl;
            return false;
        }
    }

    std::vector<ImgPair> imgPairs = { imgPair };
    std::vector<cv::Mat> curStitchedImgs;
    if (!stitcher.manualStitch(homography, imgPairs, curStitchedImgs))
    {
        std::cerr << "Error(manualStitchImgs): Failed to stitch images." << std::endl;
        return false;
    }

    stitchedImg = std::move(curStitchedImgs.front());
    if (stitchedImg.empty())
        return false;

    return true;
}

void printUsage() {
    printf("ParallelPanorama <stitcher-mode | (manual) (opencv)> <top-level-img-directory-path>\n");
    printf("NOTE: Top-level image diretory must contain subdirectories that contain images\n");
    printf("\tand are named with a numeric value to represent the image stitch position");
}
