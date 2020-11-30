#include "ImageStitcher.hpp"

const int MAX_FEATURES = 500;
const float GOOD_MATCH_PERCENT = 0.15f;

bool ImageStitcher::stitchImages(cv::Mat& src, cv::Mat& dst, cv::Mat& stitchedImg)
{
    // Variables to store keypoints and descriptors
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;
    
    // Detect ORB features and compute descriptors.
    cv::Ptr<cv::Feature2D> orb = cv::ORB::create(MAX_FEATURES);
    orb->detectAndCompute(src, cv::Mat(), keypoints1, descriptors1);
    orb->detectAndCompute(dst, cv::Mat(), keypoints2, descriptors2);
    
    // Match features.
    std::vector<cv::DMatch> matches;
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
    matcher->match(descriptors1, descriptors2, matches, cv::Mat());
    
    // Sort matches by score
    std::sort(matches.begin(), matches.end());
    
    // Remove not so good matches
    const int numGoodMatches = matches.size() * GOOD_MATCH_PERCENT;
    matches.erase(matches.begin()+numGoodMatches, matches.end());
    
    
    // Draw top matches
    cv::Mat imMatches;
    cv::drawMatches(src, keypoints1, dst, keypoints2, matches, imMatches);
    cv::imwrite("matches.jpg", imMatches);
    
    
    // Extract location of good matches
    int maxMatchWidth(0);
    std::vector<cv::Point2f> points1, points2;
    for( size_t i = 0; i < matches.size(); i++ )
    {
        cv::Point2f queryPt = keypoints1[ matches[i].queryIdx ].pt;
        cv::Point2f trainPt = keypoints2[ matches[i].trainIdx ].pt;
        points1.push_back(queryPt);
        points2.push_back(trainPt);
        
        int dist = queryPt.x - trainPt.x;
        dist = dist < 0 ? -dist : dist;
        if (dist > maxMatchWidth)
            maxMatchWidth = dist;
    }
    
    // Find homography
    cv::Mat homog = cv::findHomography( points1, points2, cv::RANSAC );
    
    // Warp source image to destination based on homography
    int totalImgWidth = src.cols + dst.cols - maxMatchWidth;
    cv::warpPerspective(src, stitchedImg, homog, cv::Size(totalImgWidth, src.rows));
    cv::Rect dstRoi(0, 0, dst.cols, src.rows);    
    dst(dstRoi).copyTo(stitchedImg(dstRoi));
  
    return true;
}