#include "CustomMatcher.hpp"
#include <thread>

CustomMatcher::CustomMatcher(
     int block_size_x, int block_size_y, int move_x, int move_y, std::string norm_type):
     block_size_x(block_size_x), block_size_y(block_size_y), move_x(move_x), move_y(move_y)
{
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(norm_type);

}

void CustomMatcher::set_boundaries(cv::Mat & query, cv::Mat & train)
{
    query_height = query.rows;
    query_width = query.cols;
    train_height = train.rows;
    train_width = train.cols;
    std::cout<<"IMG DIMENSIONS QUERY:"<<query.rows<<"X"<<query.cols<<std::endl;
    std::cout<<"IMG DIMENSIONS TRAIN:"<<train.rows<<"X"<<train.cols<<std::endl;
    w = train_height / block_size_x;
    h = train_width / block_size_y;
    query_kpnt_idx = new std::vector<int> *[h];
    train_kpnt_idx = new std::vector<int> *[h];
    query_descriptors = new cv::Mat *[h];
    train_descriptors = new cv::Mat *[h];
    for(int i=0; i < h; i++)
    {
        query_kpnt_idx[i] = new std::vector<int>[w];
        train_kpnt_idx[i] = new std::vector<int>[w];
        query_descriptors[i] = new cv::Mat [w];
        train_descriptors[i] = new cv::Mat [w];
    }
}

void CustomMatcher::cluster_keypoints(std::vector<cv::KeyPoint> & query, std::vector<cv::KeyPoint> & train,
            cv::Mat &query_desc, cv::Mat &train_desc)
{
    int initials_x = move_x;
    int initial_y = move_y;
    #pragma omp parallel for
    for(int i=0;i<query.size(); i++)
    {
        int h = (query[i].pt.x / block_size_x) > this->h - 1 ? this->h - 1 : (query[i].pt.x / block_size_x) ;
        int w = (query[i].pt.y / block_size_y) > this->w - 1 ? this->w - 1 : (query[i].pt.y / block_size_y) ;
        query_descriptors[h][w].push_back(query_desc.row(i));
    }
    #pragma omp parallel for
    for(int j=0;j<train.size();j++)
    {
        int shifted_x = (train[j].pt.x - move_x);
        int shifted_y = (train[j].pt.y - move_y);
        int h = (shifted_x > 0 ? shifted_x : 0) / block_size_x;
        int w = (shifted_y > 0 ? shifted_y : 0) / block_size_y;
        h = h > this->h - 1 ? this->h - 1 : h;
        w = w > this->w - 1 ? this->w - 1 : w;
        train_kpnt_idx[h][w].push_back(j);
        train_descriptors[h][w].push_back(train_desc.row(j));  
    }
}

std::vector<cv::DMatch> CustomMatcher::match(std::vector<cv::KeyPoint> & query, std::vector<cv::KeyPoint> & train, cv::Mat &descriptors1, cv::Mat &descriptors2)
{
    cluster_keypoints(query, train, descriptors1, descriptors2);
    std::vector<cv::DMatch> matches;
    std::vector<cv::DMatch> temp;
    //#pragma omp parallel for
    for(int i=0; i < this->h; i++)
        for(int j=0; j < this->w; j++)
        {
            if (query_descriptors[i][j].rows == 0 || train_descriptors[i][j].rows == 0) 
               continue;
            query_descriptors[i][j].reshape(0, descriptors1.cols);
            train_descriptors[i][j].reshape(0, descriptors2.cols);
            matcher-> match(train_descriptors[i][j], query_descriptors[i][j], temp, cv::Mat());
            for(int k=0; k<temp.size() ; k++)
            {
                temp[k].queryIdx = query_kpnt_idx[i][j][temp[k].queryIdx];
                temp[k].trainIdx = train_kpnt_idx[i][j][temp[k].trainIdx];
            }
            matches.insert(matches.end(), temp.begin(), temp.end());
            temp.clear();
        }
    return matches;
}
