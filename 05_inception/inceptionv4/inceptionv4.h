#ifndef TRTX_INCEPTION_NETWORK_H
#define TRTX_INCEPTION_NETWORK_H


#include <memory>
#include <vector>
#include <chrono>
#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include "../../lib_cpp/logging.h"
#include "../../lib_cpp/com_function.h"
#include "utils.h"
#include "layers_api.h"

static Logger gLogger;
using namespace trtxlayers;
using namespace nvinfer1;

namespace trtx {
    struct InceptionV4Params {
        // data
        int32_t batchSize{1};
        bool int8{false};
        bool fp16{false};
        const char *inputTensorName = "data";
        const char *outputTensorName = "prob";

        int inputH, inputW, outputSize;
        std::string weightFilePath;
        std::string trtEnginePath;
    };

    class InceptionV4 {
    public:
        InceptionV4(const InceptionV4Params &modelParams);
        ~InceptionV4() {};

        bool serializeEngine();
        bool deserializeCudaEngine();

        void doInference(float* input, float* output, int batchSize);
        bool cleanUp();
    
    private:
        bool buildEngine(IBuilder *builder, IBuilderConfig *config);
        InceptionV4Params mParams;
        ICudaEngine *mEngine;
        std::map<std::string, Weights> weightMap;
        IExecutionContext* mContext;
        std::string inception;
        DataType dt{DataType::kFLOAT};
    };
}

#endif