#include <iostream>
#include <chrono>
#include <thread>
#include <filesystem>
#include <vector>

#include <opencv2/opencv.hpp>

#include "ImageStitcher.hpp"
#include "ImageLoader.hpp"

typedef std::vector<std::pair<cv::Mat, cv::Mat>> ImgPairs;

bool stitchAllImgs(ImageLoader& imgLoader, cv::Mat& stitchedImg);
bool stitchImgs(const ImgPairs& imgPairs, float roiWidthPerc, float roiHeightPerc, std::vector<cv::Mat>& stitchedImgs);
void printUsage();

int main(int argc, char* argv[])
{
    if (argc != 2) {
        printUsage();
        return 1;
    }

    ImageLoader initImgLoader;
    for (const auto& entry : std::filesystem::directory_iterator(argv[1]))
    {
        if (!initImgLoader.loadImages(entry.path().string()))
            std::cerr << "Error(main): Failed to load images from directory - " << entry.path() << std::endl;
        else
            std::cout << "Loaded images from - " << entry.path() << std::endl;
    }

    cv::Mat stitchedImg;
    if (!stitchAllImgs(initImgLoader, stitchedImg))
    {
        std::cerr << "Error(main): Could not stitch images!" << std::endl;
        return 1;
    }

    cv::resize(stitchedImg, stitchedImg, cv::Size(), 0.25, 0.25);
    cv::imshow("Stitched Image", stitchedImg);
    
    cv::waitKey(0);
    return 0;
}

bool stitchAllImgs(ImageLoader& imgLoader, cv::Mat& stitchedImg)
{
    static int imgNum = 0;
    ImageLoader nextImages;
    for (int i = 0; i < imgLoader.getMaxImgId(); i += 2)
    {
        std::vector<cv::Mat> leftImgs;
        if (!imgLoader.getImages(i + 1, leftImgs))
        {
            std::cerr << "Error(main): No images for id - " << i + 1 << std::endl;
            continue;
        }

        std::vector<cv::Mat> rightImgs;
        if (!imgLoader.getImages(i + 2, rightImgs))
        {
            std::cerr << "Error(main): No images for id - " << i + 1 << std::endl;
            continue;
        }

        size_t numImgs(std::min(leftImgs.size(), rightImgs.size()));
        ImgPairs imgPairs(numImgs);
        for (int i = 0; i < numImgs; i++)
        {
            imgPairs.at(i).first = std::move(leftImgs.at(i));
            imgPairs.at(i).second = std::move(rightImgs.at(i));
        }

        std::vector<cv::Mat> stitchedImgs;
        if (stitchImgs(imgPairs, 0.5, 1.0, stitchedImgs))
            nextImages.addImages(nextImages.getMaxImgId() + 1, stitchedImgs);
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

    return stitchAllImgs(nextImages, stitchedImg);
}

bool stitchImgs(const ImgPairs& imgPairs, float roiWidthPerc, float roiHeightPerc, std::vector<cv::Mat>& stitchedImgs)
{
    std::vector<cv::Mat> homogs;
    ImageStitcher imgStitcher;
    std::chrono::high_resolution_clock time;

    for (size_t i = 0; i < imgPairs.size(); i++)
    {
        printf("Computing homography ROI\n");
        cv::Mat homography;
        auto start = time.now();
        if (roiWidthPerc <= 0.0 || roiHeightPerc <= 0.0)
        {
            if (!imgStitcher.computeHomography(imgPairs.at(i), homography))
            {
                std::cerr << "Error(stitchImgs): Failed to compute homography for images at index" << i << std::endl;
                continue;
            }
        }
        else
        {
            if (!imgStitcher.computeHomography(imgPairs.at(i), roiWidthPerc, roiHeightPerc, homography))
            {
                std::cerr << "Error(stitchImgs): Failed to compute homography-roi for images at index" << i << std::endl;
                continue;
            }
        }
        auto end = time.now();
        homogs.push_back(homography);
        printf("Done computing homography ROI: %lldms\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());

        printf("Stitching Images\n");
        std::vector<std::pair<cv::Mat, cv::Mat>> curPair = { imgPairs.at(i) };
        std::vector<cv::Mat> curStitchedImgs;
        start = time.now();
        if (!imgStitcher.stitchImages(homogs.at(i), curPair, curStitchedImgs))
        {
            std::cerr << "Error(stitchImgs): Failed to stitch images at index " << i << std::endl;
            continue;
        }
        end = time.now();
        printf("Done stitching images: %lldms\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
        stitchedImgs.push_back(curStitchedImgs.front());
    }

    return true;
}

void printUsage() {
    printf("ParallelPanorama <top-level-img-directory-path>\n");
    printf("NOTE: Top-level image diretory must contain subdirectories that contain images\n");
    printf("\tand are named with a numeric value to represent the image stitch position");
}
