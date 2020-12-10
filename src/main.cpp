#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <thread>

#include "ImageStitcher.hpp"

void testStitching(const std::string& leftImgPath, const std::string& rightImgPath,
                   int numIterations, float roiWidthPerc, float roiHeightPerc);
void printUsage();

int main(int argc, char* argv[])
{
    if (argc != 3) {
        printUsage();
        return 0;
    }

    testStitching(argv[1], argv[2], 4, 0.3, 0.5);
    
    return 0;
}

void testStitching(const std::string& leftImgPath, const std::string& rightImgPath,
                   int numIterations, float roiWidthPerc, float roiHeightPerc)
{
    std::vector<std::pair<cv::Mat, cv::Mat>> imgPairs;
    for (int i = 0; i < 4; i++)
    {
        cv::Mat leftImgImg = cv::imread(leftImgPath);
        cv::Mat dstImg = cv::imread(rightImgPath);
        imgPairs.push_back(std::pair<cv::Mat, cv::Mat>(leftImgImg, dstImg));
    }

    std::vector<cv::Mat> stitchedImgs;
    std::vector<std::pair<cv::Mat, unsigned int>> homogs;
    ImageStitcher imgStitcher;
    std::chrono::high_resolution_clock time;

    for (size_t i = 0; i < imgPairs.size(); i++)
    {
        printf("Computing homography ROI\n");
        std::pair<cv::Mat, unsigned int> homography;
        auto start = time.now();
        if (roiWidthPerc <= 0.0 || roiHeightPerc <= 0.0)
            imgStitcher.computeHomography(imgPairs.at(i), homography);
        else
            imgStitcher.computeHomography(imgPairs.at(i), roiWidthPerc, roiHeightPerc, homography);
        auto end = time.now();
        homogs.push_back(homography);
        printf("Done computing homography ROI: %lldms\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());

        printf("Stitching Images\n");
        std::vector<std::pair<cv::Mat, cv::Mat>> curPair = { imgPairs.at(i) };
        std::vector<cv::Mat> curStitchedImgs;
        start = time.now();
        imgStitcher.stitchImages(homogs.at(i), curPair, curStitchedImgs);
        end = time.now();
        printf("Done stitching images: %lldms\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
        stitchedImgs.push_back(curStitchedImgs.front());
    }

    cv::imshow("Left Image", imgPairs.front().first);
    cv::imshow("Right Image", imgPairs.front().second);
    for (int i = 0; i < imgPairs.size(); i++)
    {
        cv::imshow("Stitched Image - " + std::to_string(i), stitchedImgs.at(i));
    }

    cv::waitKey(0);
}

void printUsage() {
    printf("ParallelPanorama <left-img-path> <right-img-path>\n");
}
