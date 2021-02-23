#pragma once
#include <string>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <cassert>

#include "NetConfig.h"
#include "NetInput.h"


class OpenCVInference
{
public:
	OpenCVInference(){}
	OpenCVInference(NNetConfig net);
	~OpenCVInference();
	cv::dnn::Net Net() { return this->_net; }
	bool Predict(NetInput &input);

private:
	NNetConfig _netConfig;
	cv::dnn::Net _net;
	cv::Mat _output;
	std::vector <std::string> _classificationName;

	bool _Load();
	bool _SetInput(NetInput& input);
	bool _PropagateForward();
	void _GetMostProbablClass(int& classId, double& prob);
};

