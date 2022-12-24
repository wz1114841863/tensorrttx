#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include <cmath>
#include <opencv2/opencv.hpp>

#include "../lib_cpp/logging.h"
#include "../lib_cpp/com_function.h"
#include "yolayer.h"

#define USE_FP16
#define DEVICE 0
#define NMS_THRESH 0.5
#define BBOX_CONF_THRESH 0.4

using namespace nvinfer1;
// stuff we know about the network and the input/output blobs
static const int INPUT_H = Yolo::INPUT_H;
static const int INPUT_W = Yolo::INPUT_W;
static const int OUTPUT_SIZE = 1000 * 7 + 1;
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";

static Logger gLogger;

cv::Mat preprocess_img(cv::Mat &img) {
    int w, h, x, y;
    float r_w = INPUT_W / (img.cols * 1.0);
    float r_h = INPUT_H / (img.rows * 1.0);
    if (r_h > r_w) {
        w = INPUT_W;
        h = r_w * img.rows;
        x = 0;
        y = (INPUT_H - h) / 2;

    }else {
        w = r_h * img.cols;
        h = INPUT_H;
        x = (INPUT_W - w) / 2;
        y = 0;
    }

    cv::Mat re(h, w, CV_8UC3);
    cv::resize(img, re, re.size(), 0, 0, cv::INTER_CUBIC);
    cv::Mat out(INPUT_H, INPUT_W, CV_8UC3, cv::Scalar(128, 128, 128));
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
    return out;
}

cv::Rect get_rect(cv::Mat &img, float bbox[4]) {
    int l, r, t, b;
    float r_w = INPUT_W / (img.cols * 1.0);
    float r_h = INPUT_H / (img.rows * 1.0);
    if (r_h > r_w) {
        l = bbox[0] - bbox[2] / 2.f;
        r = bbox[0] + bbox[2] / 2.f;
        t = bbox[1] - bbox[3] / 2.f - (INPUT_H - r_w * img.rows) / 2;
        b = bbox[1] + bbox[3] / 2.f - (INPUT_H - r_w * img.rows) / 2;
        l = l / r_w;
        r = r / r_w;
        t = t / r_w;
        b = b / r_w;
    }else {
        l = bbox[0] - bbox[2] / 2.f - (INPUT_W - r_h * img.cols) / 2;
        r = bbox[0] + bbox[2] / 2.f - (INPUT_W - r_h * img.cols) / 2;
        t = bbox[1] - bbox[3]/2.f;
        b = bbox[1] + bbox[3]/2.f;
        l = l / r_h;
        r = r / r_h;
        t = t / r_h;
        b = b / r_h;
    }
    
    return cv::Rect(l, t, r - l, b - t);
}

float iou(float lbox[4], float rbox[4]) {
    float interBox[] = {
        std::max(lbox[0] - lbox[2] / 2.f, rbox[0] - rbox[2] / 2.f),
        std::min(lbox[0] + lbox[2] / 2.f, rbox[0] + rbox[2] / 2.f),
        std::max(lbox[1] - lbox[3] / 2.f, rbox[1] - rbox[3] / 2.f),
        std::min(lbox[1] + lbox[3] / 2.f, rbox[1] + rbox[3] / 2.f),
    };

    if (interBox[2] > interBox[3] || interBox[0] > interBox[1]) return 0.0f;

    float interBoxS = (interBox[1] - interBox[0]) * (interBox[3] - interBox[2]);
    return interBoxS / (lbox[2] * lbox[3] + rbox[2] * rbox[3] - interBoxS);
}

bool cmp(const Yolo::Detection &a, const Yolo::Detection &b) {
    return a.det_confidence > b.det_confidence;
}

void nms(std::vector<Yolo::Detection> &res, float *output, float nms_thresh = NMS_THRESH) {
    std::map<float, std::vector<Yolo::Detection>> m;
    for (int i = 0; i < output[0] && i < 1000; ++i) {
        if (output[1 + 7 * i + 4] <= BBOX_CONF_THRESH) 
            continue;
        Yolo::Detection det;
        memcpy(&det, &output[1 + 7 * i], 7 * sizeof(float));
        if (m.count(det.class_id) == 0) 
            m.emplace(det.class_id, std::vector<Yolo::Detection>());
        m[det.class_id].push_back(det);
    }

    for (auto it = m.begin(); it != m.end(); ++it) {
        auto &dets = it->second;
        std::sort(dets.begin(), dets.end(), cmp);
        for (size_t m = 0; m < dets.size(); ++m) {
            auto &item = dets[m];
            res.push_back(item);
            for (size_t n = m + 1; n < dets.size(); ++n) {
                if (iou(item.bbox, dets[n].bbox) > nms_thresh) {
                    dets.erase(dets.begin() + n);
                    --n;
                }
            }
        }
    }
}

std::map<std::string, Weights> loadWeights(const std::string file) {
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, Weights> weightMap;

    // Open weights file
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file.");

    // Read number of weight blobs
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--) {
        Weights wt{DataType::kFLOAT, nullptr, 0};
        uint32_t size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> size;
        wt.type = DataType::kFLOAT;

        // Load blob
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x)
        {
            input >> std::hex >> val[x];
        }
        wt.values = val;
        
        wt.count = size;
        weightMap[name] = wt;
    }

    return weightMap;
}

IScaleLayer* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, 
        ITensor& input, std::string lname, float eps) {
    float *gamma = (float*)weightMap[lname + ".weight"].values;
    float *beta = (float*)weightMap[lname + ".bias"].values;
    float *mean = (float*)weightMap[lname + ".running_mean"].values;
    float *var = (float*)weightMap[lname + ".running_var"].values;
    int len = weightMap[lname + ".running_var"].count;

    float *scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{DataType::kFLOAT, scval, len};
    
    float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{DataType::kFLOAT, shval, len};

    float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        pval[i] = 1.0;
    }
    Weights power{DataType::kFLOAT, pval, len};

    weightMap[lname + ".scale"] = scale;
    weightMap[lname + ".shift"] = shift;
    weightMap[lname + ".power"] = power;
    IScaleLayer* scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);
    return scale_1;
}

ILayer *convBnLeaky(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, 
        ITensor &input, int outch, int ksize, int s, int p, int linx) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    IConvolutionLayer *conv1 = network->addConvolutionNd(input, outch, DimsHW{ksize, ksize}, 
            weightMap["module_list." + std::to_string(linx) + ".Conv2d.weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{s, s});
    conv1->setPaddingNd(DimsHW{p, p});

    IScaleLayer *bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), "module_list." + std::to_string(linx) + ".BatchNorm2d", 1e-4);
    auto lr = network->addActivation(*bn1->getOutput(0), ActivationType::kLEAKY_RELU);
    lr->setAlpha(0.1);

    return lr;
}

ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt) {
    INetworkDefinition* network = builder->createNetworkV2(0U);

    // Create input tensor of shape {3, INPUT_H, INPUT_W}
    ITensor *data = network->addInput(INPUT_BLOB_NAME, dt, Dims{3, INPUT_H, INPUT_W});
    assert(data);

    auto weightMap = loadWeights("./yolov3_tiny.wts");
    Weights emptyWts{DataType::kFLOAT, nullptr, 0};

    auto lr0 = convBnLeaky(network, weightMap, *data, 16, 3, 1, 1, 0);
    auto pool1 = network->addPoolingNd(*lr0->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
    pool1->setStrideNd(DimsHW{2, 2});

    auto lr2 = convBnLeaky(network, weightMap, *pool1->getOutput(0), 32, 3, 1, 1, 2);
    auto pool3 = network->addPoolingNd(*lr2->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
    pool3->setStrideNd(DimsHW{2, 2});

    auto lr4 = convBnLeaky(network, weightMap, *pool3->getOutput(0), 64, 3, 1, 1, 4);
    auto pool5 = network->addPoolingNd(*lr4->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
    pool5->setStrideNd(DimsHW{2, 2});

    auto lr6 = convBnLeaky(network, weightMap, *pool5->getOutput(0), 128, 3, 1, 1, 6);
    auto pool7 = network->addPoolingNd(*lr6->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
    pool7->setStrideNd(DimsHW{2, 2});

    auto lr8 = convBnLeaky(network, weightMap, *pool7->getOutput(0), 256, 3, 1, 1, 8);
    auto pool9 = network->addPoolingNd(*lr8->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
    pool9->setStrideNd(DimsHW{2, 2});

    auto lr10 = convBnLeaky(network, weightMap, *pool9->getOutput(0), 512, 3, 1, 1, 10);
    auto pad11 = network->addPaddingNd(*lr10->getOutput(0), DimsHW{0, 0}, DimsHW{1, 1});
    auto pool11 = network->addPoolingNd(*pad11->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
    pool11->setStrideNd(DimsHW{1, 1});

    auto lr12 = convBnLeaky(network, weightMap, *pool11->getOutput(0), 1024, 3, 1, 1, 12);
    auto lr13 = convBnLeaky(network, weightMap, *lr12->getOutput(0), 256, 1, 1, 0, 13);
    auto lr14 = convBnLeaky(network, weightMap, *lr13->getOutput(0), 512, 3, 1, 1, 14);

    IConvolutionLayer* conv15 = network->addConvolutionNd(*lr14->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{1, 1}, 
            weightMap["module_list.15.Conv2d.weight"], weightMap["module_list.15.Conv2d.bias"]);
    
    // 16 is yolo layer
    auto lr17 = lr13;
    auto lr18 = convBnLeaky(network, weightMap, *lr17->getOutput(0), 128, 1, 1, 0, 18);
    
}