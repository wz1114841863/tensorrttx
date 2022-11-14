#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include "../common/common.hpp"
#include "../common/logging.h"

static Logger gLogger;
#define DEVICE 0;  // GPU id
#define BATCH_SIZE 1;

const char *INPUT_BLOB_NAME = "image";
const char *OUTPUT_BLOB_NAME = "output";
static const int INPUT_H = 224;
static const int INPUT_W = 224;
static const int OUTPUT_SIZE = 1000;

// Create the engine using the API
ICudaEngine *createEngine(unsigned int maxBatchSize, IBuilder *builder, IBuilderConfig *config, DataType dt) {
    INetworkDefinition *network = builder->createNetworkV2(0U);

    // Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
    ITensor *data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{3, INPUT_H, INPUT_W});
    assert(data);

    std::string wts_path = "/home/ubuntu/tensorrt/samples/tensorrtx/hrnet/HRNt_w18_classify.wts";
    std::map<std::string, Weights> weightMap = loadWeights(wts_path);
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    auto id_993 = convBnLeaky(network, weightMap, *data, 64, 3, 2, 1, "conv1", "bn1");
    auto id_996 = convBnLeaky(network, weightMap, *id_993->getOutput(0), 64, 3, 2, 1, "conv2", "bn2");
    auto id_1008 = ResBlock2Conv(network, weightMap, *id_996->getOutput(0), 64, 256, 1, "layer1.0");
    auto id_1018 = ResBlock(network, weightMap, *id_1008->getOutput(0), 256, 64, 1, "layer1.1");

    // transition1-1
    auto id_1021 = convBnLeaky(network, weightMap, *id_1018->getOutput(0), 18, 3, 1, 1, "transition1.0.0", "transition1.0.1");
    auto id_1031 = liteResBlock(network, weightMap, *id_1021->getOutput(0), 18, "stage2.0.branches.0.0");
    auto id_1038 = liteResBlock(network, weightMap, *id_1031->getOutput(0), 18, "stage2.0.branches.0.1");

    // 右侧分支
    auto id_1024 = convBnLeaky(network, weightMap, *id_1018->getOutput(0), 36, 3, 2, 1, "transition1.1.0.0", "transition1.0.1");
    auto id_1045 = liteResBlock(network, weightMap, *id_1024->getOutput(0), 36, "stage2.0.branches.1.0");
    auto id_1052 = liteResBlock(network, weightMap, *id_1045->getOutput(0), 36, "stage2.0.branches.1.1");

    IConvolutionLayer *id_1053 = network->addConvolutionNd(*id_1052->getOutput(0), 18, DimsHW{1, 1}, weightMap["stage2.0.fuse_layers.0.1.0.weight"], emptywts);
    assert(id_1053);
    id_1053->setStrideNd(DimsHW{1, 1});
    id_1053->setPaddingNd(DimsHW{0, 0});

    
}