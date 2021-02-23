#include "OpenVINO.h"
#include <iostream>
#include <inference_engine.hpp>
#include <vector>
#include <memory>
#include <windows.h>
#include <string>
#include <map>
#include <samples/slog.hpp>
#include <samples/classification_results.h>
#include <format_reader_ptr.h>
#include <iostream>
#include <samples/common.hpp>
#include <details/os/os_filesystem.hpp>
#include <opencv2/opencv.hpp>


#define ClassificationResult_t ClassificationResultW

using namespace std;
using namespace InferenceEngine;


namespace idf
{

    OpenVINOInference::OpenVINOInference(NNetConfig net)
    {
        try
        {
            this->_net_config = net;
            slog::info << "InferenceEngine: " << InferenceEngine::GetInferenceEngineVersion() << slog::endl;
            slog::info << this->_ie.GetVersions(this->_net_config.DeviceName()) << slog::endl;
            slog::info << "Loading network files" << slog::endl;
            this->_network = this->_ie.ReadNetwork(net.IRModelPath(), net.IRBinPath());
        }
        catch (const std::exception& error)
        {
            slog::err << error.what() << slog::endl; // output exception message
            throw error;
        }
    }


    OpenVINOInference::~OpenVINOInference()
	{
	}


    void OpenVINOInference::_Predict()
    {
        try
        {
            this->_PrepareInputsAndOuts();
            this->_LoadModel();
            this->_CreateInferRequest();
            this->_PrepareInputs();
            if (this->_net_config.TransmissionOrder() == Order::Sync)
            {
                this->_InferenceSync();
            }
            else if (this->_net_config.TransmissionOrder() == Order::Async)
            {
                this->_InferenceAsync();
            }
            this->_ProcessOutput();
        }
        catch (const std::exception& error) {
            slog::err << error.what() << slog::endl;
            throw error;
        }
    }


    void OpenVINOInference::Predict(NetInput& input)
    {
        this->_ClearData();
        cv::cvtColor(input.Data(), input.Data(), cv::COLOR_RGB2BGR);
        cv::Mat out;
        cv::resize(input.Data(), out, this->_net_config.InputSize());
        this->_images_data.push_back(input.Data());
        this->_class_labels.push_back(input.GT());
        this->_Predict();
    }


    void OpenVINOInference::Predict(std::vector<NetInput>& input)
    {
        _ClearData();
        for (auto i : input) {
            try {
                if (i.Data().empty())
                {
                    cout << "Cannot open image!" << endl;
                    continue;
                }

                cv::cvtColor(i.Data(), i.Data(), cv::COLOR_RGB2BGR);
                cv::Mat out;
                cv::resize(i.Data(), out, this->_net_config.InputSize());
                this->_images_data.push_back(out);
                this->_class_labels.push_back(i.GT());
            }
            catch (const std::exception& exc)
            {
                slog::err << exc.what() << slog::endl;
            }
        }
        this->_Predict();
    }


    void OpenVINOInference::_ClearData()
    {
        this->_images_data.clear();
        this->_class_labels.clear();
    }
   

    void OpenVINOInference::_PrepareInputsAndOuts()
    {
        slog::info << "Preparing input blobs" << slog::endl;

        /** Taking information about all topology inputs **/
        InputsDataMap inputInfo(_network.getInputsInfo());
        this->_input_info = inputInfo;
        if (this->_input_info.size() != 1) throw std::logic_error("Sample supports topologies with 1 input only");

        auto inputInfoItem = *this->_input_info.begin();

        /** Specifying the precision and layout of input data provided by the user.
         * This should be called before load of the network to the device **/
        inputInfoItem.second->setPrecision(this->_net_config.Precision());
        inputInfoItem.second->setLayout(this->_net_config.Layout());

        /** Setting batch size using image count **/
        this->_network.setBatchSize(this->_images_data.size());
        this->_batch_size = _network.getBatchSize();
        slog::info << "Batch size is " << std::to_string(this->_batch_size) << slog::endl;
    }


    void OpenVINOInference::_LoadModel()
    {
        slog::info << "Loading model to the device" << slog::endl;
        ExecutableNetwork executable_network = this->_ie.LoadNetwork(_network, this->_net_config.DeviceName());
        this->_executable_network = executable_network;
    }


    void OpenVINOInference::_CreateInferRequest()
    {
        slog::info << "Create infer request" << slog::endl;
        InferRequest inferRequest = this->_executable_network.CreateInferRequest();
        this->_infer_request = inferRequest;
    }


    void OpenVINOInference::_PrepareInputs()
    {
        for (auto& item : _input_info) {

            Blob::Ptr inputBlob = this->_infer_request.GetBlob(item.first);
            SizeVector dims = inputBlob->getTensorDesc().getDims();
            /** Fill input tensor with images. First b channel, then g and r channels **/
            size_t num_channels = dims[1];
            size_t image_size = dims[3] * dims[2];

            MemoryBlob::Ptr minput = as<MemoryBlob>(inputBlob);
            if (!minput) {
                slog::err << "We expect MemoryBlob from inferRequest, but by fact we were not able to cast inputBlob to MemoryBlob" << slog::endl;
            }
            //   locked memory holder should be alive all time while access to its buffer happens
            auto minputHolder = minput->wmap();

            auto data = minputHolder.as<PrecisionTrait<Precision::FP32>::value_type*>();
            /** Iterate over all input images **/
            for (size_t image_id = 0; image_id < this->_images_data.size(); ++image_id) {
                /** Iterate over all pixel in image (b,g,r) **/
                for (size_t pid = 0; pid < image_size; pid++) {
                    /** Iterate over all channels **/
                    for (size_t ch = 0; ch < num_channels; ++ch) {
                        /**          [images stride + channels stride + pixel id ] all in bytes            **/
                        data[image_id * image_size * num_channels + ch * image_size + pid] = this->_images_data.at(image_id).at<cv::Vec3b>(pid)[ch];
                    }
                }
            }
        }
    }


    void OpenVINOInference::_InferenceAsync()
    {
        size_t numIterations = 10;
        size_t curIteration = 0;
        std::condition_variable condVar;

        this->_infer_request.SetCompletionCallback(
            [&] {
            curIteration++;
            slog::info << "Completed " << curIteration << " async request execution" << slog::endl;
            if (curIteration < numIterations) {
                /* here a user can read output containing inference results and put new input
                   to repeat async request again */
                this->_infer_request.StartAsync();
            }
            else {
                /* continue sample execution after last Asynchronous inference request execution */
                condVar.notify_one();
            }}
        );

        /* Start async request for the first time */
        slog::info << "Start inference (" << numIterations << " asynchronous executions)" << slog::endl;
        this->_infer_request.StartAsync();

        /* Wait all repetitions of the async request */
        std::mutex mutex;
        std::unique_lock<std::mutex> lock(mutex);
        condVar.wait(lock, [&] { return curIteration == numIterations; });

    }


    void OpenVINOInference::_InferenceSync()
    {
        slog::info << "Start inference " << slog::endl;
        this->_infer_request.Infer();
    }


    void OpenVINOInference::_ProcessOutput()
    {
        slog::info << "Processing output blobs" << slog::endl;
        OutputsDataMap output_info(_network.getOutputsInfo());
        if (output_info.size() != 1) throw std::logic_error("Sample supports topologies with 1 output only");
        Blob::Ptr outputBlob = this->_infer_request.GetBlob(output_info.begin()->first);

        ///** Validating -nt value **/
        int nt = this->_net_config.Classes().size();
        const size_t resultsCnt = outputBlob->size() / _batch_size;
        if (nt > resultsCnt || nt < 1) {
            slog::warn << "-nt " << nt << " is not available for this network (-nt should be less than " \
                << resultsCnt + 1 << " and more than 0)\n            will be used maximal value : " << resultsCnt << slog::endl;
            nt = resultsCnt;
        }

        ClassificationResult classificationResult(outputBlob, this->_class_labels,
            this->_batch_size, nt,
            this->_net_config.Classes());
        classificationResult.print();
    }


    bool OpenVINOInference::ConvertProtobufMLModelToIRMLModel(const string model_path, const string model_name, const string output_dir, const map<string, string> params)
    {
        try
        {
            string mo_tf_call;
            mo_tf_call += R"(python "C:\Program Files (x86)\IntelSWTools\openvino_2020.4.287\deployment_tools\model_optimizer\mo_tf.py")";
            mo_tf_call += " --input_model \"" + model_path + "\"";
            mo_tf_call += " --model_name \"" + model_name + "\"";
            mo_tf_call += " --output_dir \"" + output_dir + "\"";
            mo_tf_call += " --reverse_input_channels ";

            for (auto item : params) {

                mo_tf_call += " --" + item.first + " \"" + item.second + "\"";
            }
            slog::err << mo_tf_call << slog::endl;
            system(mo_tf_call.c_str());
            return true;
        }
        catch (details::InferenceEngineException& e)
        {
            slog::err << e.what() << slog::endl;
            return false;
        }
    }

};