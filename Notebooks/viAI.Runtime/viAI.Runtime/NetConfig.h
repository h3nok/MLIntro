#pragma once
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>

#include <inference_engine.hpp>

enum Order { Sync = 0, Async = 1 };

class NNetConfig
{
public:
	NNetConfig(){}
	explicit NNetConfig(const std::string &configFile);
	NNetConfig(std::string model_path, std::string classmap, 
		const std::string& inputNode="input", const std::string &outputNode="output",
		const std::string& ir_model_path = "", const std::string& _ir_bin_path = {},
		const std::string& device_name="CPU", const Order& transmission_order = Order::Sync);
	~NNetConfig();

	//Mutators 
	void SetScaleFactor(const float& factor) { this->_scaleFactor = factor; }
	void SetMean(const float& mean) { this->_mean = mean; }
	void MustSwapRB(const bool& swapRb) { this->_swapRB = swapRb; }
	void CanCrop(const bool& crop){this->_crop = crop; }

	// Accessors 
	const std::string &ModelPath() const { return this->_model_path; }
	const std::string &IRModelPath() const { return this->_ir_model_path; }
	const std::string &IRBinPath() const { return this->_ir_bin_path; }
	const std::string &ClassMap() const { return this->_classmap; }
	const std::string &InputNode() const { return this->_inputNode; }
	const std::string &OutputNode() const { return this->_outputNode; }
	const float &ScaleFactor() const { return this->_scaleFactor; }
	const cv::Size &InputSize() const { return this->_netInputSize; }
	const float &Mean() const { return this->_mean; }
	const bool &SwapRB() const { return this->_swapRB; }
	const bool &Crop() const { return this->_crop; }
	const std::string& DeviceName() const { return this->_device_name; }
	const Order& TransmissionOrder() const { return this->_transmission_order;  }
	InferenceEngine::Precision Precision() { return this->_precision; }
	InferenceEngine::Layout Layout() { return this->_layout; }



	std::vector<std::string> Classes();

private:
	std::string _model_path;
	std::string _ir_model_path;
	std::string _classmap;
	std::string _inputNode;
	std::string _outputNode;
	std::string _configFile;
	std::string _device_name;
	std::string _ir_bin_path;
	Order _transmission_order;
	InferenceEngine::Precision _precision = InferenceEngine::Precision::FP32;
	InferenceEngine::Layout _layout = InferenceEngine::Layout::NCHW;

	float _scaleFactor;
	cv::Size _netInputSize = cv::Size(224, 224);
	float _mean;
	bool _swapRB;
	bool _crop;
};

