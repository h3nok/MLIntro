#include "NetConfig.h"


NNetConfig::NNetConfig(const std::string & configFile)
{
	this->_configFile = configFile;


}

NNetConfig::NNetConfig(std::string model_path, std::string classmap,
	const std::string& inputNode, const std::string& outputNode, 
	const std::string& ir_model_path, const std::string& _ir_bin_path,
	const std::string& device_name, const Order& transmission_order)
{
	this->_classmap = classmap.c_str();
	this->_model_path = model_path;
	this->_inputNode = inputNode;
	this->_outputNode = outputNode;
	this->_scaleFactor = 1.0f;
	this->_netInputSize = cv::Size(224, 224);
	this->_mean = 127.0f;
	this->_swapRB = true;
	this->_crop = false;
	this->_ir_model_path = ir_model_path;
	this->_ir_bin_path = _ir_bin_path;
	this->_device_name = device_name;
	this->_transmission_order = transmission_order;
}

NNetConfig::~NNetConfig()
{
}

std::vector<std::string> NNetConfig::Classes()
{
	std::vector<std::string> classes;
	std::ifstream ifs(this->_classmap);

	if (!ifs.is_open())
		std::cerr << "Unable to open classifications map file\n";

	std::string line;

	while (std::getline(ifs, line))
	{
		classes.emplace_back(line.substr(line.find(' ') + 1));
	}

	ifs.close();
	return classes;
}
