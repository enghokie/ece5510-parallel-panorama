#include <iostream>
#include <chrono>
#include <thread>
#include <filesystem>
#include <vector>
#include <opencv2/opencv.hpp>

#include "ImageStitcher.hpp"
#include "ImageLoader.hpp"
#include "StitcherWorker.hpp"
#include "ThreadSafeDequeue.hpp"

bool QUIT_PROCESSING = false;
const float DISPLAY_PERCENTAGE = 0.3;
unsigned long TOTAL_STITCH_TIME = 0;
unsigned long TOTAL_FINAL_STITCHES = 0;
std::chrono::high_resolution_clock TIME;
std::chrono::steady_clock::time_point START_TIME;

bool stitchAllImgs(ThreadSafeDequeue<JobIdPair>& jobQueue,
                   ThreadSafeDequeue<ResIdPair>& resQueue,
                   ImageLoader& imgLoader,
                   const bool& quit);

void printUsage();

int main(int argc, char* argv[])
{
    if (argc != 4) {
        printUsage();
        return 1;
    }

    unsigned int numStitcherWorkerThreads = std::stoul(argv[1]);
    if (numStitcherWorkerThreads == 0 || numStitcherWorkerThreads > std::thread::hardware_concurrency())
    {
        std::cerr << "Error(main): Number of workers threads must be between 1 and <number-of-physical-cores>.";
        return 1;
    }

    // Setup stitcher mode
    ImageStitcher::StitcherMode stitchMode = ImageStitcher::StitcherMode_Manual;
    if (std::string(argv[2]).find("opencv") != std::string::npos)
        stitchMode = ImageStitcher::StitcherMode_OpenCV;

    // Load images
    ImageLoader initImgLoader;
    for (const auto& entry : std::filesystem::directory_iterator(argv[3]))
    {
        if (!initImgLoader.loadImages(entry.path().string()))
            std::cerr << "Error(main): Failed to load images from directory - " << entry.path() << std::endl;
        else
            std::cout << "Loaded images from - " << entry.path() << std::endl;
    }

    // Setup stitcher worker threads and start them
    ThreadSafeDequeue<JobIdPair> jobQueue;
    ThreadSafeDequeue<ResIdPair> resQueue;
    std::vector<std::pair<std::thread, StitcherWorker>> stitcherWorkers;
    for (int i = 0; i < numStitcherWorkerThreads; i++)
    {
        StitcherWorker stitcherWorker(jobQueue, resQueue, stitchMode);
        std::thread workerThread(&StitcherWorker::run, stitcherWorker);
        stitcherWorkers.emplace_back(std::pair<std::thread, StitcherWorker>(std::move(workerThread), std::move(stitcherWorker)));
    }

    // Send all stitch jobs
    if (!stitchAllImgs(jobQueue, resQueue, initImgLoader, QUIT_PROCESSING))
    {
        std::cerr << "Error(main): Could not stitch images!" << std::endl;
        return 1;
    }

    // Make sure the workers are done
    for (auto& worker : stitcherWorkers)
    {
        worker.second.quit();
        worker.first.join();
    }

    return 0;
}


bool stitchAllImgs(ThreadSafeDequeue<JobIdPair>& jobQueue,
                   ThreadSafeDequeue<ResIdPair>& resQueue,
                   ImageLoader& imgLoader,
                   const bool& quit)
{
    // Send all the images to the job queue to be stitched together
    std::cout << "Sending all stitch jobs to job queue." << std::endl;
    unsigned int jobId(0);
    while (!quit)
    {
        // Acquire the next group of images to stitch together
        std::vector<cv::Mat> curImages;
        for (int i = 0; i < imgLoader.getMaxImgId(); i++)
        {
            cv::Mat img;
            if (!imgLoader.popImage(i + 1, img))
                break;

            curImages.push_back(std::move(img));
        }

        // Check if we're done acquiring images
        if (curImages.size() != imgLoader.getMaxImgId())
            break;
    
        // Send the group of images to the job queue
        JobIdPair pair(++jobId, std::move(curImages));
        jobQueue.push(pair);
    }
    std::cout << "Finished sending all stitch jobs to job queue." << std::endl;

    // Get the result images
    std::cout << "Acquiring all stitched images from result queue." << std::endl;
    jobId = 1;
    std::vector<ResIdPair> jobResults;
    while (!quit)
    {
        // Track the time acquired to get result images
        START_TIME = TIME.now();
        ResIdPair jobRes = std::move(resQueue.pop());

        // Make sure the image is valid
        if (jobRes.second.empty())
        {
            std::cerr << "Error(stitchAllImgs): Acquired stitched image for id - " << jobRes.first << " is empty." << std::endl;
            return false;
        }

        // Get the time taken to acquire this final stitched image
        auto end = TIME.now();
        TOTAL_STITCH_TIME += std::chrono::duration_cast<std::chrono::milliseconds>(end - START_TIME).count();
        if ((++TOTAL_FINAL_STITCHES % 10) == 0)
        {
            std::cout << "Average time elapsed from last final stitched image: " << TOTAL_STITCH_TIME / TOTAL_FINAL_STITCHES << "ms" << std::endl;
        }

        // Ensure this job result should be displayed next
        bool foundResToDisplay = jobRes.first == jobId ? true : false;
        if (!foundResToDisplay)
        {
            jobResults.push_back(std::move(jobRes));

            // See if we have the next job result stored already
            for (auto itr = jobResults.begin(); itr != jobResults.end(); itr++)
            {
                if (itr->first == jobId)
                {
                    jobRes = std::move(*itr);
                    jobResults.erase(itr);
                    foundResToDisplay = true;
                    break;
                }
            }
        }

        if (foundResToDisplay)
        {
            // Display the stitched image
            cv::resize(jobRes.second, jobRes.second, cv::Size(), DISPLAY_PERCENTAGE, DISPLAY_PERCENTAGE);
            cv::imshow("Stitched Image", jobRes.second);
            cv::waitKey(1);
            ++jobId;
        }
    }
    std::cout << "Finished acquiring all stitch jobs from result queue." << std::endl;
    return true;
}

void printUsage() {
    printf("ParallelPanorama <num-stitcher-worker-threads> <stitcher-mode | (manual) (opencv)> <top-level-img-directory-path>\n");
    printf("NOTE: Top-level image diretory must contain subdirectories that contain images\n");
    printf("\tand are named with a numeric value to represent the image stitch position");
}
