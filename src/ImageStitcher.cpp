#include "ImageStitcher.hpp"

const int MAX_FEATURES = 500;
const float GOOD_MATCH_PERCENT = 0.15f;

void ImageStitcher::setHomography(const cv::Mat& homog)
{
    _homography = homog;
}

cv::Mat ImageStitcher::getHomography()
{
    return _homography;
}

bool ImageStitcher::getHomography(cv::Mat& homog)
{
    if (!_homography.empty())
    {
        homog = _homography;
        return true;
    }

    return false;
}

bool ImageStitcher::computeHomography(const std::pair<cv::Mat, cv::Mat>& imgs,
                                      std::pair<cv::Mat, unsigned int>& homog)
{
    return computeHomography(imgs, 1.0, 1.0, homog);
}

bool ImageStitcher::computeHomography(const std::pair<cv::Mat, cv::Mat>& imgs,
                                      float roiWidthPerc,
                                      float roiHeightPerc,
                                      std::pair<cv::Mat, unsigned int>& homog)
{
    if (roiWidthPerc <= 0.0 || roiHeightPerc <= 0.0)
        return false;

    cv::Mat leftImg = imgs.first;
    cv::Mat rightImg = imgs.second;

    // Get ROI
    unsigned int leftImgWidthRoi = leftImg.cols * roiWidthPerc;
    unsigned int leftImgHeightRoi = leftImg.rows * roiHeightPerc;
    unsigned int leftImgWidthOffset = leftImg.cols - leftImgWidthRoi;
    unsigned int rightImgWidthRoi = rightImg.cols * roiWidthPerc;
    unsigned int rightImgHeightRoi = rightImg.rows * roiHeightPerc;
    cv::Rect leftImgRoi(leftImgWidthOffset, 0, leftImgWidthRoi, leftImgHeightRoi);
    cv::Rect rightImgRoi(0, 0, rightImgWidthRoi, rightImgHeightRoi);

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
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
    matcher->match(descriptors1, descriptors2, matches, cv::Mat());

    // Sort matches by score
    std::sort(matches.begin(), matches.end());

    // Remove not so good matches
    const int numGoodMatches = matches.size() * GOOD_MATCH_PERCENT;
    matches.erase(matches.begin() + numGoodMatches, matches.end());


    // Draw top matches
    //cv::Mat imMatches;
    //cv::drawMatches(leftImg, keypoints1, dst, keypoints2, matches, imMatches);
    //cv::imwrite("matches.jpg", imMatches);


    // Extract location of good matches
    unsigned int maxMatchWidth(0);
    std::vector<cv::Point2f> points1, points2;
    for (size_t i = 0; i < matches.size(); i++)
    {
        cv::Point2f queryPt = keypoints1[matches[i].queryIdx].pt;
        queryPt.x += leftImgWidthOffset;

        cv::Point2f trainPt = keypoints2[matches[i].trainIdx].pt;
        points1.push_back(queryPt);
        points2.push_back(trainPt);

        unsigned int dist = std::abs(queryPt.x - trainPt.x);
        maxMatchWidth = std::max(dist, maxMatchWidth);
    }

    // Find homography
    homog.first = cv::findHomography(points2, points1, cv::RANSAC);
    homog.second = maxMatchWidth;

    return true;
}

bool ImageStitcher::stitchImages(const std::pair<cv::Mat, unsigned int>& homog,
                                 const std::vector<std::pair<cv::Mat, cv::Mat>>& imgPairs,
                                 std::vector<cv::Mat>& stitchedImgs)
{
    if (homog.first.empty() || imgPairs.empty())
        return false;

    for (int i = 0; i < imgPairs.size(); i++)
    {
        std::pair<cv::Mat, cv::Mat> imgPair = imgPairs.at(i);
        cv::Mat leftImg = imgPair.first;
        cv::Mat rightImg = imgPair.second;

        // Warp source image to destination based on homography
        stitchedImgs.push_back(cv::Mat());
        unsigned int totalImgWidth = leftImg.cols + rightImg.cols - homog.second;

        //std::cout << "Stitched image size - " << cv::Size(totalImgWidth, rightImg.rows) << std::endl;
        cv::warpPerspective(rightImg, stitchedImgs.back(), homog.first, cv::Size(totalImgWidth, rightImg.rows));
        cv::Rect leftImgRoi(0, 0, leftImg.cols, rightImg.rows);
        leftImg(leftImgRoi).copyTo(stitchedImgs.back()(leftImgRoi));
    }

    return true;
}
