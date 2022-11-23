#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <NvInferPlugin.h>
#include <iostream>
#include <map>
#include <fstream>
#include <chrono>

#include "../lib_cpp/logging.h"


// provided by nvidia for using TensorRT APIS
using namespace nvinfer1;

// Logger from TRT API
static Logger gLogger;

const int INPUT_SIZE = 1;
const int OUTPUT_SIZE = 1;

std::map<std::string, Weights> loadWeights(const std::string filePath) {
    /*
    Parse the .wts file and store weights in dict format.
    filePath: the file path of .wts file.
    */
    std::cout << "[INFO]: Loading weights..." << filePath << std::endl;
    std::map<std::string, Weights> weightMap;

    // Open Weight File
    std::ifstream input(filePath);
    assert(input.is_open() && "[ERROR]: Unable to load weight file...");

    // Read nunmber of weights
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    // Loop through number of line, actually number of weights & biases
    while (count--) {
        // TensorRT weights
        Weights wt{DataType::kFLOAT, nullptr, 0};
        uint32_t size;
        // Read name and type of weights 
        std::string w_name;
        input >> w_name >> std::dec >> size;
        wt.type = DataType::kFLOAT;

        uint32_t *val = reinterpret_cast<uint32_t *>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; x++) {
            input >> std::hex >> val[x];
        }
        wt.values = val;
        wt.count = size;

        weightMap[w_name] = wt;
    }

    return weightMap;
}

ICudaEngine *createMLPEngine(unsigned int maxBatchSize, IBuilder *builder, IBuilderConfig *config, DataType dt) {
    /*
    Create Multi-Layer Perception using the TRT Builder ang Configurations
    maxBatchSize: batch size for built TRT model
    builder: to build engine and networks
    config: configuration related to Hardware
    dt: datatype for model layers
    engine: TRT model
    */
    std::cout << "[INFO]: Creating MLP using TensorRT..." << std::endl;

    // Load Weights from relevant file
    std::map<std::string, Weights> weightMap = loadWeights("../mlp.wts");

    // Create an empty network
    INetworkDefinition *network = builder->createNetworkV2(0U);

    // Create an input with proper *name
    ITensor *data = network->addInput("data", DataType::kFLOAT, Dims3{1, 1, 1});
    assert(data);

    // Add Layer for MLP
    IFullyConnectedLayer *fc1 = network->addFullyConnected(*data, 
        1, weightMap["linear.weight"],  weightMap["linear.bias"]);
    assert(fc1);

    // set output with *name
    fc1->getOutput(0)->setName("out");

    // mark the output
    network->markOutput(*fc1->getOutput(0));

    // Set configurations
    builder->setMaxBatchSize(1);

    // Set workspace size
    config->setMaxWorkspaceSize(1 << 20);

    // Build CUDA Engine using network and configurations
    ICudaEngine *engine = builder->buildEngineWithConfig(*network, *config);
    assert(engine != nullptr);

    // Don't need the network any more
    // free captured memory
    network->destroy();

    // Release host memory
    for (auto &mem: weightMap) {
        free((void *) (mem.second.values));
    }

    return engine;
}

void APIToModel(unsigned int maxBatchSize, IHostMemory **modelStream) {
    /*
    Create engine using TensorRT APIs
    maxBatchSize: for the deploy model configs
    modelStream: shared memory to store serialized model
    */

    // Create builder with the help of logger
    IBuilder *builder = createInferBuilder(gLogger);
    
    // Create hardware configs
    IBuilderConfig *config = builder->createBuilderConfig();

    // Build an engine
    ICudaEngine *engine = createMLPEngine(maxBatchSize, builder, config, DataType::kFLOAT);
    assert(engine != nullptr);

    // serialize the engine into binary stream
    (*modelStream) = engine->serialize();

    // free up the memory
    engine->destroy();
    builder->destroy();
}

void performSerialization() {
    /*
    Serialization Function
    */
    // shared memory object
    IHostMemory *modelStream{nullptr};

    // Write model into stream
    APIToModel(1, &modelStream);
    assert(modelStream != nullptr);

    std::cout << "[INFO]: Writing engine into binary..." << std::endl;

    // open the file and write the contents there in binary format
    std::ofstream p("./mlp_cpp.engine", std::ios::binary);
    if (!p) {
        std::cerr << "could not open plan output file" << std::endl;
        return ;
    }

    p.write(reinterpret_cast<const char *>(modelStream->data()), modelStream->size());

    // Release the memory
    modelStream->destroy();

    std::cout << "[INFO]: Successfully created TensorRT engine..." << std::endl;
    std::cout << "\n\t Run inference using ./mlp -d " << std::endl;
}

void doInference(IExecutionContext &context, float *input, float *output, int batchSize) {
    /*
    Perform inference using CUDA context
    context: context creates by engine.
    input: input from the host.
    output: output to save on host.
    batchSize: batch size for TRT model.
    */
    // Get the engine from the context
    const ICudaEngine &engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void *buffers[2];
    const int inputIndex = engine.getBindingIndex("data");
    const int outputIndex = engine.getBindingIndex("out");

    // Create GPU buffers on device -- allocate memory for input and output
    cudaMalloc(&buffers[inputIndex], batchSize * INPUT_SIZE * sizeof(float));
    cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float));

    // Create CUDA stream for simultaneous CUDA operations
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // copy input from host to device in stream
    cudaMemcpyAsync(buffers[inputIndex], input, batchSize * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice, stream);

    // execute inference using context provided by engine
    context.enqueue(batchSize, buffers, stream, nullptr);

    // copy output back from device to host
    cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream);

    // synchronize the stream to prevent issues
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    cudaFree(buffers[inputIndex]);
    cudaFree(buffers[outputIndex]);
}

void performInference() {
    /*
    Get inference using the pre-trained model
    */

    // stream to write model
    char *trtModelStream{nullptr};
    size_t size{0};

    // read model from the engine file
    std::ifstream file("./mlp_cpp.engine", std::ios::binary);
    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }

    //create a runtime (required for deserialization of model) with NVIDIA's logger
    IRuntime *runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);

    // deserialize engine for using the char->stream
    ICudaEngine *engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
    assert(engine != nullptr);

    // create execution context -- required for inference executions
    IExecutionContext *context = engine->createExecutionContext();
    assert(context != nullptr);

    float out[1];
    float data[1];
    for (float &i: data) {
        i = 4.0;
    }

    // time the execution
    auto start = std::chrono::system_clock::now();

    // do Inference using the parameters
    doInference(*context, data, out, 1);

    // time the execution
    auto end = std::chrono::system_clock::now();
    std::cout << "\n[INFO]: Time taken by execution: " 
        << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    // free the captured space
    context->destroy();
    engine->destroy();
    runtime->destroy();
    std::cout << "\nInput:\t" << data[0];
    std::cout << "\nOutput:\t";
    for (float i: out) {
        std::cout << i;
    }
    std::cout << std::endl;
}

int checkArgs(int argc, char **argv, std::string filename) {
    /**
     * Parse command line arguments
     *
     * @param argc: argument count
     * @param argv: arguments vector
     * @return int: a flag to perform operation
     */

    if (argc != 2) {
        std::cerr << "[ERROR]: Arguments not right!" << std::endl;
        std::cerr << "./" << filename << " -s   // serialize model to plan file" << std::endl;
        std::cerr << "./" << filename << " -d   // deserialize plan file and run inference" << std::endl;
        return -1;
    }
    if (std::string(argv[1]) == "-s") {
        return 1;
    } else if (std::string(argv[1]) == "-d") {
        return 2;
    }
    return -1;
}

int main(int argc, char **argv) {
    int args = checkArgs(argc, argv, "mlp");
    if (args == 1)
        performSerialization();
    else if (args == 2)
        performInference();
    return 0;
}