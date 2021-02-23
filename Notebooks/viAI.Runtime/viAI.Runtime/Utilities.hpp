#pragma once

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <sstream>
#include <utility>
#include <vector>
#include <fstream>
#include <memory>

#include <inference_engine.hpp>
#include <samples/common.hpp>

namespace idf {

    /**
     * \brief Parse image size provided as string in format WIDTHxHEIGHT
     * @return parsed width and height
     */
    std::pair<size_t, size_t> parseImageSize(const std::string& size_string) {
        auto delimiter_pos = size_string.find("x");
        if (delimiter_pos == std::string::npos
            || delimiter_pos >= size_string.size() - 1
            || delimiter_pos == 0) {
            std::stringstream err;
            err << "Incorrect format of image size parameter, expected WIDTHxHEIGHT, "
                "actual: " << size_string;
            throw std::runtime_error(err.str());
        }

        size_t width = static_cast<size_t>(
            std::stoull(size_string.substr(0, delimiter_pos)));
        size_t height = static_cast<size_t>(
            std::stoull(size_string.substr(delimiter_pos + 1, size_string.size())));

        if (width == 0 || height == 0) {
            throw std::runtime_error(
                "Incorrect format of image size parameter, width and height must not be equal to 0");
        }

        if (width % 2 != 0 || height % 2 != 0) {
            throw std::runtime_error("Unsupported image size, width and height must be even numbers");
        }

        return { width, height };
    }

    /**
     * \brief Read image data from file
     * @return buffer containing the image data
     */
    std::unique_ptr<unsigned char[]> readImageDataFromFile(const std::string& image_path, size_t size) {
        std::ifstream file(image_path, std::ios_base::ate | std::ios_base::binary);
        if (!file.good() || !file.is_open()) {
            std::stringstream err;
            err << "Cannot access input image file. File path: " << image_path;
            throw std::runtime_error(err.str());
        }

        const size_t file_size = file.tellg();
        if (file_size < size) {
            std::stringstream err;
            err << "Invalid read size provided. File size: " << file_size << ", to read: " << size;
            throw std::runtime_error(err.str());
        }
        file.seekg(0);

        std::unique_ptr<unsigned char[]> data(new unsigned char[size]);
        file.read(reinterpret_cast<char*>(data.get()), size);
        return data;
    }

    /**
     * \brief Sets batch size of the network to the specified value
     */
    //void setBatchSize(CNNNetwork& network, size_t batch) {
    //    ICNNNetwork::InputShapes inputShapes = network.getInputShapes();
    //    for (auto& shape : inputShapes) {
    //        auto& dims = shape.second;
    //        if (dims.empty()) {
    //            throw std::runtime_error("Network's input shapes have empty dimensions");
    //        }
    //        dims[0] = batch;
    //    }
    //    network.reshape(inputShapes);
    //}
};