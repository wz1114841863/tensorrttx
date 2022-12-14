#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <map>
#include <chrono>

#include "../../lib_cpp/logging.h"
#include "../../lib_cpp/com_function.h"

// stuff we know about the network and the input/output blobs
static const int INPUT_H = 299;
static const int INPUT_W = 299;
static const int OUTPUT_SIZE = 1000;

const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";

using namespace nvinfer1;

static Logger gLogger;

std::map<std::string, nvinfer1::Weights> loadWeights(const std::string file){
    /*
    load network weights from .wts file.
    */
    std::cout << "Loading Weights: " << file << std::endl;
    std::map<std::string, nvinfer1::Weights> weightMap;

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
        
        // Load blob
        uint32_t *val = reinterpret_cast<uint32_t *>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; x++) {
            input >> std::hex >> val[x];
        }
        wt.values = val;
        wt.count = size;
        weightMap[name] = wt;
    }
    return weightMap;
}

IScaleLayer* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, 
        ITensor &input, std::string lname, float eps) {
    float *gamma = (float *)weightMap[lname + ".weight"].values;
    float *beta = (float *)weightMap[lname + ".bias"].values;
    float *mean = (float *)weightMap[lname + ".running_mean"].values;
    float *var = (float *)weightMap[lname + ".running_var"].values;
    int len = weightMap[lname + ".running_var"].count;
    // std::cout << "len " << len << std::endl;

    float *scval = reinterpret_cast<float *>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{DataType::kFLOAT, scval, len};

    float *shval = reinterpret_cast<float *>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{DataType::kFLOAT, shval, len};

    float *pval = reinterpret_cast<float *>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        pval[i] = 1.0;
    }
    Weights power{DataType::kFLOAT, pval, len};

    weightMap[lname + ".scale"] = scale;
    weightMap[lname + ".shift"] = shift;
    weightMap[lname + ".power"] = power;
    IScaleLayer *scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);
    return scale_1;
}

IActivationLayer *basicConv2d(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, 
        ITensor &input, int outch, DimsHW ksize, int s, DimsHW p, std::string lname) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    IConvolutionLayer *conv1 = network->addConvolutionNd(input, outch, ksize, weightMap[lname + "conv.weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{s, s});
    conv1->setPaddingNd(p);

    IScaleLayer *bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "bn", 1e-3);
    assert(bn1);

    IActivationLayer *relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);
    return relu1;
}

IConcatenationLayer *inceptionA(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, 
        ITensor &input, std::string lname, int pool_proj) {

    IActivationLayer *relu1 = basicConv2d(network, weightMap, input, 64, DimsHW{1, 1}, 1, DimsHW{0, 0}, lname + "branch1x1.");

    IActivationLayer *relu2 = basicConv2d(network, weightMap, input, 48, DimsHW{1, 1}, 1, DimsHW{0, 0}, lname + "branch5x5_1.");
    relu2 = basicConv2d(network, weightMap, *relu2->getOutput(0), 64, DimsHW{5, 5}, 1, DimsHW{2, 2}, lname + "branch5x5_2.");

    IActivationLayer *relu3 = basicConv2d(network, weightMap, input, 64, DimsHW{1, 1}, 1, DimsHW{0, 0}, lname + "branch3x3dbl_1.");
    relu3 =  basicConv2d(network, weightMap, *relu3->getOutput(0), 96, DimsHW{3, 3}, 1, DimsHW{1, 1}, lname + "branch3x3dbl_2.");
    relu3 =  basicConv2d(network, weightMap, *relu3->getOutput(0), 96, DimsHW{3, 3}, 1, DimsHW{1, 1}, lname + "branch3x3dbl_3.");

    IPoolingLayer *pool1 = network->addPoolingNd(input, PoolingType::kAVERAGE, DimsHW{3, 3});
    assert(pool1);
    pool1->setStrideNd(DimsHW{1, 1});
    pool1->setPaddingNd(DimsHW{1, 1});
    pool1->setAverageCountExcludesPadding(false);
    IActivationLayer* relu4 = basicConv2d(network, weightMap, *pool1->getOutput(0), pool_proj, DimsHW{1, 1}, 1, DimsHW{0, 0}, lname + "branch_pool.");

    ITensor *inputTensors[] = {relu1->getOutput(0),  relu2->getOutput(0), relu3->getOutput(0), relu4->getOutput(0)};
    IConcatenationLayer* cat1 = network->addConcatenation(inputTensors, 4);
    assert(cat1);
    return cat1;
}

IConcatenationLayer* inceptionB(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname) {
    IActivationLayer* relu1 = basicConv2d(network, weightMap, input, 384, DimsHW{3, 3}, 2, DimsHW{0, 0}, lname + "branch3x3.");

    IActivationLayer* relu2 = basicConv2d(network, weightMap, input, 64, DimsHW{1, 1}, 1, DimsHW{0, 0}, lname + "branch3x3dbl_1.");
    relu2 = basicConv2d(network, weightMap, *relu2->getOutput(0), 96, DimsHW{3, 3}, 1, DimsHW{1, 1}, lname + "branch3x3dbl_2.");
    relu2 = basicConv2d(network, weightMap, *relu2->getOutput(0), 96, DimsHW{3, 3}, 2, DimsHW{0, 0}, lname + "branch3x3dbl_3.");

    IPoolingLayer* pool1 = network->addPoolingNd(input, PoolingType::kMAX, DimsHW{3, 3});
    assert(pool1);
    pool1->setStrideNd(DimsHW{2, 2});

    ITensor* inputTensors[] = {relu1->getOutput(0), relu2->getOutput(0), pool1->getOutput(0)};
    IConcatenationLayer* cat1 = network->addConcatenation(inputTensors, 3);
    assert(cat1);
    return cat1;
}

IConcatenationLayer* inceptionC(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname,
        int c7) {
    IActivationLayer* relu1 = basicConv2d(network, weightMap, input, 192, DimsHW{1, 1}, 1, DimsHW{0, 0}, lname + "branch1x1.");

    IActivationLayer* relu2 = basicConv2d(network, weightMap, input, c7, DimsHW{1, 1}, 1, DimsHW{0, 0}, lname + "branch7x7_1.");
    relu2 = basicConv2d(network, weightMap, *relu2->getOutput(0), c7, DimsHW{1, 7}, 1, DimsHW{0, 3}, lname + "branch7x7_2.");
    relu2 = basicConv2d(network, weightMap, *relu2->getOutput(0), 192, DimsHW{7, 1}, 1, DimsHW{3, 0}, lname + "branch7x7_3.");

    IActivationLayer* relu3 = basicConv2d(network, weightMap, input, c7, DimsHW{1, 1}, 1, DimsHW{0, 0}, lname + "branch7x7dbl_1.");
    relu3 = basicConv2d(network, weightMap, *relu3->getOutput(0), c7, DimsHW{7, 1}, 1, DimsHW{3, 0}, lname + "branch7x7dbl_2.");
    relu3 = basicConv2d(network, weightMap, *relu3->getOutput(0), c7, DimsHW{1, 7}, 1, DimsHW{0, 3}, lname + "branch7x7dbl_3.");
    relu3 = basicConv2d(network, weightMap, *relu3->getOutput(0), c7, DimsHW{7, 1}, 1, DimsHW{3, 0}, lname + "branch7x7dbl_4.");
    relu3 = basicConv2d(network, weightMap, *relu3->getOutput(0), 192, DimsHW{1, 7}, 1, DimsHW{0, 3}, lname + "branch7x7dbl_5.");

    IPoolingLayer* pool1 = network->addPoolingNd(input, PoolingType::kAVERAGE, DimsHW{3, 3});
    assert(pool1);
    pool1->setStrideNd(DimsHW{1, 1});
    pool1->setPaddingNd(DimsHW{1, 1});
    pool1->setAverageCountExcludesPadding(false);
    IActivationLayer* relu4 = basicConv2d(network, weightMap, *pool1->getOutput(0), 192, DimsHW{1, 1}, 1, DimsHW{0, 0}, lname + "branch_pool.");

    ITensor* inputTensors[] = {relu1->getOutput(0), relu2->getOutput(0), relu3->getOutput(0), relu4->getOutput(0)};
    IConcatenationLayer* cat1 = network->addConcatenation(inputTensors, 4);
    assert(cat1);
    return cat1;
}

IConcatenationLayer* inceptionD(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname) {
    IActivationLayer* relu1 = basicConv2d(network, weightMap, input, 192, DimsHW{1, 1}, 1, DimsHW{0, 0}, lname + "branch3x3_1.");
    relu1 = basicConv2d(network, weightMap, *relu1->getOutput(0), 320, DimsHW{3, 3}, 2, DimsHW{0, 0}, lname + "branch3x3_2.");

    IActivationLayer* relu2 = basicConv2d(network, weightMap, input, 192, DimsHW{1, 1}, 1, DimsHW{0, 0}, lname + "branch7x7x3_1.");
    relu2 = basicConv2d(network, weightMap, *relu2->getOutput(0), 192, DimsHW{1, 7}, 1, DimsHW{0, 3}, lname + "branch7x7x3_2.");
    relu2 = basicConv2d(network, weightMap, *relu2->getOutput(0), 192, DimsHW{7, 1}, 1, DimsHW{3, 0}, lname + "branch7x7x3_3.");
    relu2 = basicConv2d(network, weightMap, *relu2->getOutput(0), 192, DimsHW{3, 3}, 2, DimsHW{0, 0}, lname + "branch7x7x3_4.");

    IPoolingLayer* pool1 = network->addPoolingNd(input, PoolingType::kMAX, DimsHW{3, 3});
    assert(pool1);
    pool1->setStrideNd(DimsHW{2, 2});

    ITensor* inputTensors[] = {relu1->getOutput(0), relu2->getOutput(0), pool1->getOutput(0)};
    IConcatenationLayer* cat1 = network->addConcatenation(inputTensors, 3);
    assert(cat1);
    return cat1;
}

IConcatenationLayer *inceptionE(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname) {

    IActivationLayer *relu1 = basicConv2d(network, weightMap, input, 320, DimsHW{1, 1}, 1, DimsHW{0, 0}, lname + "branch1x1.");

    IActivationLayer *relu2 = basicConv2d(network, weightMap, input, 384, DimsHW{1, 1}, 1, DimsHW{0, 0}, lname + "branch3x3_1.");
    IActivationLayer *relu2a = basicConv2d(network, weightMap, *relu2->getOutput(0), 384, DimsHW{1, 3}, 1, DimsHW{0, 1}, lname + "branch3x3_2a.");
    IActivationLayer *relu2b = basicConv2d(network, weightMap, *relu2->getOutput(0), 384, DimsHW{3, 1}, 1, DimsHW{1, 0}, lname + "branch3x3_2b.");
    ITensor *inputTensors[] = {relu2a->getOutput(0), relu2b->getOutput(0)};
    IConcatenationLayer *cat1 = network->addConcatenation(inputTensors, 2);
    assert(cat1);

    IActivationLayer* relu3 = basicConv2d(network, weightMap, input, 448, DimsHW{1, 1}, 1, DimsHW{0, 0}, lname + "branch3x3dbl_1.");
    relu3 = basicConv2d(network, weightMap, *relu3->getOutput(0), 384, DimsHW{3, 3}, 1, DimsHW{1, 1}, lname + "branch3x3dbl_2.");
    IActivationLayer* relu3a = basicConv2d(network, weightMap, *relu3->getOutput(0), 384, DimsHW{1, 3}, 1, DimsHW{0, 1}, lname + "branch3x3dbl_3a.");
    IActivationLayer* relu3b = basicConv2d(network, weightMap, *relu3->getOutput(0), 384, DimsHW{3, 1}, 1, DimsHW{1, 0}, lname + "branch3x3dbl_3b.");
    ITensor* inputTensors1[] = {relu3a->getOutput(0), relu3b->getOutput(0)};
    IConcatenationLayer* cat2 = network->addConcatenation(inputTensors1, 2);
    assert(cat2);
    std::cout << "test4";
    IPoolingLayer* pool1 = network->addPoolingNd(input, PoolingType::kAVERAGE, DimsHW{3, 3});
    assert(pool1);
    pool1->setStrideNd(DimsHW{1, 1});
    pool1->setPaddingNd(DimsHW{1, 1});
    pool1->setAverageCountExcludesPadding(false);
    IActivationLayer* relu4 = basicConv2d(network, weightMap, *pool1->getOutput(0), 192, DimsHW{1, 1}, 1, DimsHW{0, 0}, lname + "branch_pool.");

    ITensor* inputTensors2[] = {relu1->getOutput(0), cat1->getOutput(0), cat2->getOutput(0), relu4->getOutput(0)};
    IConcatenationLayer* cat3 = network->addConcatenation(inputTensors2, 4);
    assert(cat3);
    return cat3;
}

IScaleLayer *normalizedLayer(INetworkDefinition *network, ITensor &data) {
    /*
    Normalized layer
    */
    float shval[3] = {(0.485 - 0.5) / 0.5, (0.456 - 0.5) / 0.5, (0.406 - 0.5) / 0.5};
    float scval[3] = {0.229 / 0.5, 0.224 / 0.5, 0.225 / 0.5};
    float pval[3] = {1.0, 1.0, 1.0};
    Weights shift{DataType::kFLOAT, shval, 3};
    Weights scale{DataType::kFLOAT, scval, 3};
    Weights power{DataType::kFLOAT, pval, 3};
    IScaleLayer* scale1 = network->addScale(data, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale1);
    
    return scale1;
}

ICudaEngine *createEngine(unsigned int maxBatchSize, IBuilder *builder, IBuilderConfig *config, DataType dt) {
    INetworkDefinition *network = builder->createNetworkV2(0U);
    assert(network);

    // Create input tensor of shape {1, 1, 32, 32} with name INPUT_BLOB_NAME
    ITensor *data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{3, INPUT_H, INPUT_W});
    assert(data);

    auto weightMap = loadWeights("../inceptionv3.wts");
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    auto scale1 = normalizedLayer(network, *data);

    IActivationLayer* relu1 = basicConv2d(network, weightMap, *scale1->getOutput(0), 32, DimsHW{3, 3}, 2, DimsHW{0, 0}, "Conv2d_1a_3x3.");
    relu1 = basicConv2d(network, weightMap, *relu1->getOutput(0), 32, DimsHW{3, 3}, 1, DimsHW{0, 0}, "Conv2d_2a_3x3.");
    relu1 = basicConv2d(network, weightMap, *relu1->getOutput(0), 64, DimsHW{3, 3}, 1, DimsHW{1, 1}, "Conv2d_2b_3x3.");
    IPoolingLayer* pool1 = network->addPoolingNd(*relu1->getOutput(0), PoolingType::kMAX, DimsHW{3, 3});
    assert(pool1);

    pool1->setStrideNd(DimsHW{2, 2});
    relu1 = basicConv2d(network, weightMap, *pool1->getOutput(0), 80, DimsHW{1, 1}, 1, DimsHW{0, 0}, "Conv2d_3b_1x1.");
    relu1 = basicConv2d(network, weightMap, *relu1->getOutput(0), 192, DimsHW{3, 3}, 1, DimsHW{0, 0}, "Conv2d_4a_3x3.");
    pool1 = network->addPoolingNd(*relu1->getOutput(0), PoolingType::kMAX, DimsHW{3, 3});
    pool1->setStrideNd(DimsHW{2, 2});

    auto cat1 = inceptionA(network, weightMap, *pool1->getOutput(0), "Mixed_5b.", 32);
    cat1 = inceptionA(network, weightMap, *cat1->getOutput(0), "Mixed_5c.", 64);
    cat1 = inceptionA(network, weightMap, *cat1->getOutput(0), "Mixed_5d.", 64);
    cat1 = inceptionB(network, weightMap, *cat1->getOutput(0), "Mixed_6a.");
    cat1 = inceptionC(network, weightMap, *cat1->getOutput(0), "Mixed_6b.", 128);
    cat1 = inceptionC(network, weightMap, *cat1->getOutput(0), "Mixed_6c.", 160);
    cat1 = inceptionC(network, weightMap, *cat1->getOutput(0), "Mixed_6d.", 160);
    cat1 = inceptionC(network, weightMap, *cat1->getOutput(0), "Mixed_6e.", 192);
    cat1 = inceptionD(network, weightMap, *cat1->getOutput(0), "Mixed_7a.");
    cat1 = inceptionE(network, weightMap, *cat1->getOutput(0), "Mixed_7b.");
    cat1 = inceptionE(network, weightMap, *cat1->getOutput(0), "Mixed_7c.");
    IPoolingLayer* pool2 = network->addPoolingNd(*cat1->getOutput(0), PoolingType::kAVERAGE, DimsHW{8, 8});
    assert(pool2);

    IFullyConnectedLayer* fc1 = network->addFullyConnected(*pool2->getOutput(0), 1000, weightMap["fc.weight"], weightMap["fc.bias"]);
    assert(fc1);
    fc1->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    std::cout << "set name out" << std::endl;
    network->markOutput(*fc1->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(1 << 20);
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "build out" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*) (mem.second.values));
    }

    return engine;
}

void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream)
{
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = createEngine(maxBatchSize, builder, config, DataType::kFLOAT);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
    config->destroy();
}

void doInference(IExecutionContext& context, float* input, float* output, int batchSize)
{
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

int main(int argc, char** argv)
{
    if (argc != 2) {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./inception -s   // serialize model to plan file" << std::endl;
        std::cerr << "./inception -d   // deserialize plan file and run inference" << std::endl;
        return -1;
    }

    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};

    if (std::string(argv[1]) == "-s") {
        IHostMemory* modelStream{nullptr};
        APIToModel(1, &modelStream);
        assert(modelStream != nullptr);

        std::ofstream p("inception_cpp.engine", std::ios::binary);
        if (!p)
        {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        return 1;
    } else if (std::string(argv[1]) == "-d") {
        std::ifstream file("inception_cpp.engine", std::ios::binary);
        if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
        }
    } else {
        return -1;
    }


    // Subtract mean from image
    static float data[3 * INPUT_H * INPUT_W];
    for (int i = 0; i < 3 * INPUT_H * INPUT_W; i++)
        data[i] = 1.0;

    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;

    // Run inference
    static float prob[OUTPUT_SIZE];
    for (int i = 0; i < 100; i++) {
        auto start = std::chrono::system_clock::now();
        doInference(*context, data, prob, 1);
        auto end = std::chrono::system_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    }

    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    // Print histogram of the output distribution
    std::cout << "\nOutput:\n\n";
    for (unsigned int i = 0; i < OUTPUT_SIZE; i++)
    {
        std::cout << prob[i] << ", ";
        if (i % 10 == 0) std::cout << i / 10 << std::endl;
    }
    std::cout << std::endl;

    return 0;
}