#include "ImageStitcher.hpp"
#include "CustomMatcher.hpp"

const int MAX_FEATURES = 500;
const float GOOD_MATCH_PERCENT = 0.15f;

void ImageStitcher::setHomography(const cv::Mat& homog)
{
    _homography = homog;
}

const cv::Mat ImageStitcher::getHomography()
{
    return _homography;
}

const bool ImageStitcher::getHomography(cv::Mat& homog)
{
    if (!_homography.empty())
    {
        homog = _homography;
        return true;
    }

    return false;
}

const bool ImageStitcher::computeHomography(const std::pair<cv::Mat, cv::Mat>& imgs,
                                            cv::Mat& homog)
{
    return computeHomography(imgs, 1.0, 1.0, homog);
}

const bool ImageStitcher::computeHomography(const std::pair<cv::Mat, cv::Mat>& imgs,
                                            float roiWidthPerc,
                                            float roiHeightPerc,
                                            cv::Mat& homog)
{
    if (roiWidthPerc <= 0.0 || roiHeightPerc <= 0.0)
        return false;

    if (imgs.first.empty() || imgs.second.empty())
    {
        std::cerr << "Error(computeHomography): One or both of the images are empty." << std::endl;
        return false;
    }

    cv::Mat leftImg = imgs.first;
    cv::Mat rightImg = imgs.second;

    // Get ROI
    unsigned int leftImgWidthRoi = leftImg.cols * roiWidthPerc;
    unsigned int leftImgHeightRoi = leftImg.rows * roiHeightPerc;
    unsigned int leftImgWidthOffset = leftImg.cols - leftImgWidthRoi;
    unsigned int rightImgWidthRoi = rightImg.cols * roiWidthPerc;
    unsigned int rightImgHeightRoi = rightImg.rows * roiHeightPerc;
    unsigned int minImgHeight = std::min(leftImgHeightRoi, rightImgHeightRoi);
    cv::Rect leftImgRoi(leftImgWidthOffset, 0, leftImgWidthRoi, minImgHeight);
    cv::Rect rightImgRoi(0, 0, rightImgWidthRoi, minImgHeight);

    // Convert image to grayscale
    cv::Mat leftGray, rightGray;
    if (leftImg.type() != CV_8UC1)
        cv::cvtColor(leftImg(leftImgRoi), leftGray, cv::COLOR_BGR2GRAY);
    else
        leftGray = leftImg;
    if (rightImg.type() != CV_8UC1)
        cv::cvtColor(rightImg(rightImgRoi), rightGray, cv::COLOR_BGR2GRAY);
    else
        rightGray = rightImg;

    // Variables to store keypoints and descriptors
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;

    // Detect ORB features and compute descriptors.
    cv::Ptr<cv::Feature2D> orb = cv::ORB::create(MAX_FEATURES);
    orb->detectAndCompute(leftGray, cv::Mat(), keypoints1, descriptors1);
    orb->detectAndCompute(rightGray, cv::Mat(), keypoints2, descriptors2);
    
    // Match features.
    std::vector<cv::DMatch> matches;
    CustomMatcher *matcher = new CustomMatcher(150,150, 200, 200, "BruteForce-Hamming");
    matcher -> set_boundaries(rightGray, leftGray);
    matches = matcher->match(keypoints1, keypoints2, descriptors1, descriptors2);
    // cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
    // matcher->match(descriptors1, descriptors2, matches, cv::Mat());
    // Sort matches by score
    std::sort(matches.begin(), matches.end());

    // Remove not so good matches
    const int numGoodMatches = matches.size() * GOOD_MATCH_PERCENT;
    matches.erase(matches.begin() + numGoodMatches, matches.end());

    // Extract location of good matches
    std::vector<cv::Point2f> points1, points2;
    for (size_t i = 0; i < matches.size(); i++)
    {
        cv::Point2f queryPt = keypoints1[matches[i].queryIdx].pt;
        queryPt.x += leftImgWidthOffset;

        cv::Point2f trainPt = keypoints2[matches[i].trainIdx].pt;
        points1.push_back(queryPt);
        points2.push_back(trainPt);
    }

    // Find homography
    homog = cv::findHomography(points2, points1, cv::RANSAC);

    return true;
}

const bool ImageStitcher::manualStitch(const cv::Mat& homog,
                                       const std::vector<std::pair<cv::Mat, cv::Mat>>& imgPairs,
                                       std::vector<cv::Mat>& stitchedImgs)
{
    if (homog.empty() || imgPairs.empty())
    {
        std::cerr << "Error(stitchImages): No homography or img pairs provided." << std::endl;
        return false;
    }

    for (int i = 0; i < imgPairs.size(); i++)
    {
        std::pair<cv::Mat, cv::Mat> imgPair = imgPairs.at(i);
        cv::Mat leftImg = imgPair.first;
        cv::Mat rightImg = imgPair.second;

        if (leftImg.empty() || rightImg.empty())
        {
            std::cerr << "Error(stitchImages): left or right image is empty for pair at idx - " << i << std::endl;
            continue;
        }

        // Warp right image to left image based on homography
        unsigned int minImgHeight = std::min(leftImg.rows, rightImg.rows);
        unsigned int totalImgWidth = leftImg.cols + rightImg.cols;
        cv::Rect rightImgRoi(0, 0, rightImg.cols, minImgHeight);
        cv::Mat stitchedImg(cv::Size(totalImgWidth, minImgHeight), rightImg.type());
        cv::warpPerspective(rightImg(rightImgRoi), stitchedImg, homog, stitchedImg.size(),
                            cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0, 255, 0));

        // Now find the extra pixels
        int widthEndIdx = totalImgWidth;
        int widthStartIdx = 0;
        int widthPadding = totalImgWidth * 0.10;
        int initImgHeight = minImgHeight * 0.10;
        int maxHeightIdx = minImgHeight - initImgHeight;
        int heightIdx = initImgHeight;
        for (; heightIdx < maxHeightIdx; heightIdx++)
        {
            // Find the extra end pixels
            for (; widthEndIdx > leftImg.cols; widthEndIdx--)
            {
                if (stitchedImg.at<cv::Vec3b>(heightIdx, widthEndIdx - 1) != cv::Vec3b(0, 255, 0))
                {
                    break;
                }
            }

            // Find the extra beginning pixels
            for (; widthStartIdx < leftImg.cols + widthPadding; widthStartIdx++)
            {
                if (stitchedImg.at<cv::Vec3b>(heightIdx, widthStartIdx) != cv::Vec3b(0, 255, 0))
                {
                    break;
                }
            }
        }
        widthStartIdx = widthStartIdx > leftImg.cols ? widthStartIdx - leftImg.cols : 0;

        // Copy the left image onto the canvas
        cv::Rect leftImgRoi(widthStartIdx, 0, leftImg.cols, minImgHeight);
        leftImg(leftImgRoi).copyTo(stitchedImg(leftImgRoi));

        stitchedImgs.push_back(std::move(stitchedImg(cv::Rect(widthStartIdx, 0, widthEndIdx, minImgHeight))));
    }

    return true;
}
