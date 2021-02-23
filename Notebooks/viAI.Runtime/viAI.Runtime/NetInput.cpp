#include "NetInput.h"



NetInput::NetInput(const pqxx::row & row)
{
	this->_id = row[0].as<std::string>();
	this->_groundtruth = row[1].as<std::string>();
	auto imageData = pqxx::binarystring(row[2]);
	auto array = cv::_InputArray(imageData.data(), static_cast<int>(imageData.size()));
	this->_data = cv::imdecode(array, cv::IMREAD_COLOR);
}


NetInput::~NetInput()
{
	_data.release();
}

