#include <iostream>
#include <opencv2/highgui.hpp>
#include "OpenCV.h"
#include "Common.h"
#include "DatabaseFactory.h"
#include "OpenVINO.h"
#include <cstdio>
#include <samples/slog.hpp>

using namespace cv;
using namespace idf;

enum Engine {OPENCV=0, OPENVINO, TF};

std::string keys ="{ help  h | | Print help message. }"
    "{ engine| 0 | Choose one of inference engines: " 
					"0: opencv, "
					"1: openvino, "
					"2: tensorflow }"
	"{model | | Path to model file }"
	"{xml | | Path to xml file }"
	"{classes || Path to classifications file}"
	"{dataset || Path to images directory}"
	"{host h |12.1.1.91| Database host name or IP}"
	"{database db |viNet| Database}"
	"{user u| | Database user}"
	"{password p | | Database password}"
	"{ir_model | | Openvino (IR model}";

int main(int argc, char** argv)
{
	try
	{
		CommandLineParser parser(argc, argv, keys);

		if (argc == 1 || parser.has("help"))
		{
			parser.printMessage();
			return 0;
		}

		if (!parser.check())
		{
			parser.printErrors();
			return -1;
		}

		auto module = parser.get<int>("engine");
		auto model_path = parser.get<std::string>("model");
		auto class_path = parser.get<std::string>("classes");
		auto dataset = parser.get<std::string>("dataset");
		auto host = parser.get<std::string>("host");
		auto database = parser.get <std::string>("database");
		auto user = parser.get<std::string>("user");
		auto pwd = parser.get<std::string>("password");
		auto ir_model_path = parser.get<std::string>("ir_model");

		auto connectionStr = "host=" + host + " dbname=" + database + " user=" + user + " password=" + pwd;
		auto dbf = DatabaseFactory(connectionStr);
		auto category = "Hawk";
		const int limit = 10;
		auto framesCursor = dbf.CreateSelectFramesByCategoryCursor(category, limit);

		//assert(file_exists(model_path));
		//assert(file_exists(class_path));
		//assert(file_exists(ir_model_path));

		OpenCVInference ocv;
		auto net = NNetConfig(model_path, class_path, "", "", ir_model_path);

		switch (module)
		{
			case Engine::OPENCV:
			{
				ocv = OpenCVInference(net);
				return  1;
			}
			case Engine::OPENVINO:
			{
				OpenVINOInference open_vino(net);
				open_vino.Predict(framesCursor);
				return 1;
			}
			case Engine::TF:
			{
				return 1;
			}
			default:
			{
				std::cerr << "Unknown inference engine!\n";
				return -1;
			}
		}

	}
	catch (const std::exception& exc)
	{
		slog::err << exc.what() << slog::endl;
		return -1;
	}
	
	return 1;
}