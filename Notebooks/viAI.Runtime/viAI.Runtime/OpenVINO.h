#pragma once
#include "NetInput.h"
#include "NetConfig.h"

#include <inference_engine.hpp>
#include <windows.h>
#include <vector>
#include <memory>
#include <string>
#include <map>
#include <opencv2/opencv.hpp>
#include <samples/slog.hpp>
#include <samples/ocv_common.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/core.hpp>



using namespace std;
using namespace InferenceEngine;


namespace idf
{

    class OpenVINOInference
    {

    private:
        Core _ie;
        CNNNetwork _network;
        ExecutableNetwork _executable_network;
        InferRequest _infer_request;
        InputsDataMap _input_info;
        size_t _batch_size{ 0 };
        vector<string> _class_labels;
        vector<cv::Mat> _images_data;
        NNetConfig _net_config;

        void _PrepareInputsAndOuts();
        void _LoadModel();
        void _CreateInferRequest();
        void _PrepareInputs();
        void _InferenceAsync();
        void _InferenceSync();
        void _ProcessOutput();
        void _ClearData();
        void _Predict();

    public:

        OpenVINOInference(NNetConfig net);
        ~OpenVINOInference();

        void Predict(NetInput& input);
        void Predict(std::vector<NetInput>& input);

        static bool ConvertProtobufMLModelToIRMLModel(
            const string model_path,
            const string model_name,
            const string output_dir,
            const map<string, string> params);
    };
    
};