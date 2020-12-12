#include <filesystem>

#include "ImageLoader.hpp"

bool ImageLoader::loadImages(std::string imgDirPath)
{
    while (imgDirPath.back() == '/' || imgDirPath.back() == '\\')
        imgDirPath.pop_back();
    size_t subDirBeginIdx = imgDirPath.rfind("/");
    subDirBeginIdx = subDirBeginIdx != std::string::npos ? subDirBeginIdx : imgDirPath.rfind("\\");
    if (subDirBeginIdx == std::string::npos)
    {
        std::cerr << "Error(loadImages): Invalid image directory path - " << imgDirPath << std::endl;
        return false;
    }
    else
    {
        ++subDirBeginIdx;
    }

    unsigned int id = std::stoul(imgDirPath.substr(subDirBeginIdx, imgDirPath.size() - subDirBeginIdx));
    std::vector<cv::Mat> imgs;
    for (const auto &entry : std::filesystem::directory_iterator(imgDirPath))
    {
        std::string ext(entry.path().extension().string());
        if (ext != ".jpg" && ext != ".png")
            continue;

        cv::Mat img = cv::imread(entry.path().string());
        if (img.empty())
        {
            std::cerr << "Error(loadImages): Could not load image - " << entry.path() << std::endl;
            return false;
        }
        imgs.push_back(std::move(img));
    }

    if (imgs.empty())
    {
        std::cerr << "Error(loadImages): No images could be loaded from directory path - " << imgDirPath << std::endl;
        return false;
    }

    int idIdx(-1);
    for (int i = 0; i < _imgPairs.size(); i++)
    {
        if (_imgPairs[i].first == id)
        {
            idIdx = i;
            break;
        }
    }

    std::shared_ptr<std::vector<cv::Mat>> storedImgs;
    if (idIdx != -1)
    {
        storedImgs = _imgPairs.at(idIdx).second;
    }
    else
    {
        storedImgs = std::make_shared<std::vector<cv::Mat>>();
        _imgPairs.emplace_back(ImgIdPair(id, storedImgs));
        _maxLoadedImgId = std::max(_maxLoadedImgId, id);
    }

    for (auto& img : imgs)
        storedImgs->push_back(std::move(img));

    return true;
}

bool ImageLoader::loadImages(std::vector<std::string> imgDirPaths)
{
    for (std::string& dirPath : imgDirPaths)
    {
        if (!loadImages(dirPath))
            return false;
    }

    return true;
}

const void ImageLoader::getImgPairs(std::vector<ImgIdPair>& imgPairs)
{
    for (auto& imgPair : _imgPairs)
        imgPairs.push_back(std::move(imgPair));
}

const bool ImageLoader::getImgPairs(unsigned int id, std::vector<ImgIdPair>& imgPairs)
{
    for (auto& imgPair : _imgPairs)
    {
        if (imgPair.first == id)
        {
            imgPairs.push_back(std::move(imgPair));
            return true;
        }
    }

    return false;
}

const bool ImageLoader::getImages(unsigned int id, std::vector<cv::Mat>& imgs)
{
    for (auto imgPair : _imgPairs)
    {
        if (imgPair.first == id)
        {
            if (imgPair.second->size() < 0)
                return false;

            for (auto img : *imgPair.second)
                imgs.push_back(std::move(img));
            return true;
        }
    }

    return false;
}

bool ImageLoader::addImage(unsigned int id, cv::Mat& img)
{    
    if (img.empty())
        return false;

    int idIdx(-1);
    for (int i = 0; i < _imgPairs.size(); i++)
    {
        if (_imgPairs[i].first == id)
        {
            idIdx = i;
            break;
        }
    }

    std::shared_ptr<std::vector<cv::Mat>> storedImgs;
    if (idIdx != -1)
    {
        storedImgs = _imgPairs.at(idIdx).second;
    }
    else
    {
        storedImgs = std::make_shared<std::vector<cv::Mat>>();
        _imgPairs.emplace_back(ImgIdPair(id, storedImgs));
        _maxLoadedImgId = std::max(_maxLoadedImgId, id);
    }

    storedImgs->push_back(std::move(img));

    return true;
}

bool ImageLoader::addImages(unsigned int id, std::vector<cv::Mat>& imgs)
{
    if (imgs.empty())
        return false;

    int idIdx(-1);
    for (int i = 0; i < _imgPairs.size(); i++)
    {
        if (_imgPairs[i].first == id)
        {
            idIdx = i;
            break;
        }
    }

    std::shared_ptr<std::vector<cv::Mat>> storedImgs;
    if (idIdx != -1)
    {
        storedImgs = _imgPairs.at(idIdx).second;
    }
    else
    {
        storedImgs = std::make_shared<std::vector<cv::Mat>>();
        _imgPairs.emplace_back(ImgIdPair(id, storedImgs));
        _maxLoadedImgId = std::max(_maxLoadedImgId, id);
    }

    for (auto& img : imgs)
        storedImgs->push_back(std::move(img));

    return true;
}
