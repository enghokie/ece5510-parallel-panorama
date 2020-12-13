#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <opencv2/opencv.hpp>

class CustomMatcher{

    public:

    CustomMatcher(int neighbour_x, int neighbour_y, int move_x, int move_y, std::string norm_type) ; 
    ~CustomMatcher() {}
    void set_boundaries(cv::Mat & query, cv::Mat & train);

    std::vector<cv::DMatch> match(std::vector<cv::KeyPoint> & query,
            std::vector<cv::KeyPoint> & train, cv::Mat &descriptors1,
            cv::Mat &descriptors2);

    private:
    cv::Ptr<cv::DescriptorMatcher> matcher;
    int block_size_x;
    int block_size_y;
    int move_x, move_y;
    int h, w;
    int query_width, query_height, train_width, train_height = 0;
    std::vector<int> query_idx_group;
    std::vector<int> train_idx_group;
    std::vector<int> ** query_kpnt_idx, ** train_kpnt_idx;
    cv::Mat ** query_descriptors, ** train_descriptors;
    void cluster_keypoints(std::vector<cv::KeyPoint> & query, std::vector<cv::KeyPoint> & train,
            cv::Mat &query_desc, cv::Mat &train_desc);
};