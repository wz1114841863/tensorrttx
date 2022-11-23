#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <map>
#include <chrono>
#include <iostream>
#include <assert.h>
#include <cmath>

#ifndef COM_FUNCTION_H
#define COM_FUNCTION_H

// 返回值检查
#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

using namespace nvinfer1;

// Load weights from files shared with TensorRT samples.
// TensorRT weight files have a simple space delimited format:
// [type] [size] <data x size in hex>
// std::map<std::string, nvinfer1::Weights> loadWeights(const std::string file);

// IScaleLayer* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, 
//     ITensor &input, std::string lname, float eps);

// IActivationLayer *basicConv2d(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, 
//     ITensor &input, int outch, int ksize, int s, int p, std::string lname);

// IScaleLayer *normalizedLayer(INetworkDefinition *network, ITensor &data);

// void doInference(IExecutionContext& context, float* input, float* output, int batchSize, 
//         const char* input_blob_name, const char* output_blob_name, const int input_h, const int input_w, const int output_size);


// std::map<std::string, nvinfer1::Weights> loadWeights(const std::string file){
//     /*
//     load network weights from .wts file.
//     */
//     std::cout << "Loading Weights: " << file << std::endl;
//     std::map<std::string, nvinfer1::Weights> weightMap;

//     // Open weights file
//     std::ifstream input(file);
//     assert(input.is_open() && "Unable to load weight file.");

//     // Read number of weight blobs
//     int32_t count;
//     input >> count;
//     assert(count > 0 && "Invalid weight map file.");

//     while (count--) {
//         Weights wt{DataType::kFLOAT, nullptr, 0};
//         uint32_t size;

//         // Read name and type of blob
//         std::string name;
//         input >> name >> std::dec >> size;
        
//         // Load blob
//         uint32_t *val = reinterpret_cast<uint32_t *>(malloc(sizeof(val) * size));
//         for (uint32_t x = 0, y = size; x < y; x++) {
//             input >> std::hex >> val[x];
//         }
//         wt.values = val;
//         wt.count = size;
//         weightMap[name] = wt;
//     }
//     return weightMap;
// }

// IScaleLayer* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, 
//         ITensor &input, std::string lname, float eps) {
//     float *gamma = (float *)weightMap[lname + ".weight"].values;
//     float *beta = (float *)weightMap[lname + ".bias"].values;
//     float *mean = (float *)weightMap[lname + ".running_mean"].values;
//     float *var = (float *)weightMap[lname + ".running_var"].values;
//     int len = weightMap[lname + ".running_var"].count;
//     // std::cout << "len " << len << std::endl;

//     float *scval = reinterpret_cast<float *>(malloc(sizeof(float) * len));
//     for (int i = 0; i < len; i++) {
//         scval[i] = gamma[i] / sqrt(var[i] + eps);
//     }
//     Weights scale{DataType::kFLOAT, scval, len};

//     float *shval = reinterpret_cast<float *>(malloc(sizeof(float) * len));
//     for (int i = 0; i < len; i++) {
//         shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
//     }
//     Weights shift{DataType::kFLOAT, shval, len};

//     float *pval = reinterpret_cast<float *>(malloc(sizeof(float) * len));
//     for (int i = 0; i < len; i++) {
//         pval[i] = 1.0;
//     }
//     Weights power{DataType::kFLOAT, pval, len};

//     weightMap[lname + ".scale"] = scale;
//     weightMap[lname + ".shift"] = shift;
//     weightMap[lname + ".power"] = power;
//     IScaleLayer *scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
//     assert(scale_1);
//     return scale_1;
// }

// IActivationLayer *basicConv2d(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, 
//         ITensor &input, int outch, int ksize, int s, int p, std::string lname) {
//     Weights emptywts{DataType::kFLOAT, nullptr, 0};

//     IConvolutionLayer *conv1 = network->addConvolutionNd(input, outch, DimsHW{ksize, ksize}, weightMap[lname + "conv.weight"], emptywts);
//     assert(conv1);
//     conv1->setStrideNd(DimsHW{s, s});
//     conv1->setPaddingNd(DimsHW{p, p});

//     IScaleLayer *bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "bn", 1e-3);

//     IActivationLayer *relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
//     assert(relu1);
//     return relu1;
// }

// IScaleLayer *normalizedLayer(INetworkDefinition *network, ITensor &data) {
//     /*
//     Normalized layer
//     */
//     float shval[3] = {(0.485 - 0.5) / 0.5, (0.456 - 0.5) / 0.5, (0.406 - 0.5) / 0.5};
//     float scval[3] = {0.229 / 0.5, 0.224 / 0.5, 0.225 / 0.5};
//     float pval[3] = {1.0, 1.0, 1.0};
//     Weights shift{DataType::kFLOAT, shval, 3};
//     Weights scale{DataType::kFLOAT, scval, 3};
//     Weights power{DataType::kFLOAT, pval, 3};
//     IScaleLayer* scale1 = network->addScale(data, ScaleMode::kCHANNEL, shift, scale, power);
//     assert(scale1);
    
//     return scale1;
// }

// void doInference(IExecutionContext& context, float* input, float* output, int batchSize, 
//         const char* input_blob_name, const char* output_blob_name, const int input_h, const int input_w, const int output_size) {
//     const ICudaEngine& engine = context.getEngine();

//     // Pointers to input and output device buffers to pass to engine.
//     // Engine requires exactly IEngine::getNbBindings() number of buffers.
//     assert(engine.getNbBindings() == 2);
//     void* buffers[2];

//     // In order to bind the buffers, we need to know the names of the input and output tensors.
//     // Note that indices are guaranteed to be less than IEngine::getNbBindings()
//     const int inputIndex = engine.getBindingIndex(input_blob_name);
//     const int outputIndex = engine.getBindingIndex(output_blob_name);

//     // Create GPU buffers on device
//     CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * input_h * input_w * sizeof(float)));
//     CHECK(cudaMalloc(&buffers[outputIndex], batchSize * output_size * sizeof(float)));

//     // Create stream
//     cudaStream_t stream;
//     CHECK(cudaStreamCreate(&stream));

//     // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
//     CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * input_h * input_w * sizeof(float), cudaMemcpyHostToDevice, stream));
//     context.enqueue(batchSize, buffers, stream, nullptr);
//     CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * output_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
//     cudaStreamSynchronize(stream);

//     // Release stream and buffers
//     cudaStreamDestroy(stream);
//     CHECK(cudaFree(buffers[inputIndex]));
//     CHECK(cudaFree(buffers[outputIndex]));
// }

#endif