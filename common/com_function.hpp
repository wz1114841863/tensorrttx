#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <map>
#include <chrono>


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




