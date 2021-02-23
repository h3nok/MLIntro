#pragma once
#include <string>
#include <pqxx/pqxx>
#include <opencv2/opencv.hpp>


class NetInput
{
public:
	NetInput(const pqxx::row& row);
	~NetInput();

	const std::string Id() const { return this->_id; }
	const std::string GT() const { return this->_groundtruth; }
	cv::Mat Data() { return this->_data; }

public:
	cv::Mat _data;
	std::string _groundtruth;
	std::string _id;
};

