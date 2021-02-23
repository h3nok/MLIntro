#include "OpenCV.h"



OpenCVInference::OpenCVInference(NNetConfig net)
{
	this->_netConfig = net;
	this->_Load();
}

OpenCVInference::~OpenCVInference()
{
}

bool OpenCVInference::Predict(NetInput & input)
{
	if (input.Data().empty())
	{
		std::cerr << "Failed to decode image for frame: " << input.Id() << ": " << "cv::imdecode returned empty image" << std::endl;
		return false;
	}
	
	this->_SetInput(input);
	this->_PropagateForward();

	int classId;
	double confidence;

	this->_GetMostProbablClass(classId, confidence);

	std::cout << "GT: " << input.GT() << " PRED: " << this->_classificationName.at(classId) << "PROB: " << confidence << std::endl;

	return true;
}

bool OpenCVInference::_Load()
{
	try
	{
		this->_net = cv::dnn::readNetFromTensorflow(this->_netConfig.ModelPath());
		this->_classificationName = this->_netConfig.Classes();
	}
	catch(cv::Exception ex)
	{
		std::cerr << ex.what() << std::endl; 
		return false;
	}

	return true;
}

bool OpenCVInference::_SetInput(NetInput & input)
{
	try
	{
		auto blob = cv::dnn::blobFromImage(input.Data(), this->_netConfig.ScaleFactor(), 
			this->_netConfig.InputSize(), this->_netConfig.Mean(), this->_netConfig.SwapRB(), 
			this->_netConfig.Crop());
		this->_net.setInput(blob, input.Id());
	}
	catch (cv::Exception ex)
	{
		std::cerr << ex.what() << std::endl;
		return false;
	}

	return true;
}

bool OpenCVInference::_PropagateForward()
{
	try 
	{
		auto output = this->_net.forward(this->_netConfig.OutputNode());
		this->_output = output.reshape(1, 1);
		return true;
	}
	catch (cv::Exception ex) 
	{
		std::cerr << ex.what() << std::endl;
		return false;
	}

	return false;
}

void OpenCVInference::_GetMostProbablClass(int &classId, double & prob)
{
	auto start = _output.begin<float>();
	auto end = _output.end<float>();

	if (start == end)
	{
		classId = -1;
		prob = 0.0;
		return;
	}

	auto maxProb = std::max_element(start, end);
	classId = static_cast<int>(std::distance(start, maxProb));

	prob = *maxProb;
}
