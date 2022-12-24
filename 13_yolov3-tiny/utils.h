#ifndef __TRT_UTILS_H
#define __TRT_UTILS_H

#include <iostream>
#include <vector>
#include <algorithm>
#include <cudnn.h>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include "macros.h"

#ifndef CUDA_CHECK

// 宏定义，检查返回编号
#define CUDA_CHECK(callstr) {                                                                   \
    cudaError_t error_code = callstr;                                                           \
    if (error_code != cudaSuccess) {                                                            \
        std::cerr << "CUDA error " << error_code << " at " <<  __FILE__ << " : " << __LINE__;   \
        assert(0);                                                                              \
    }                                                                                           \
}

#endif

namespace Tn {
    // time profile
    class Profiler : public nvinfer1::IProfiler {
    public:
        void printLayerTimes(int itrationsTimes) {
            float totalTime = 0;
            for (size_t i = 0; i < mProfile.size(); ++i) {
                printf("%-40.40s %4.3fms\n", mProfile[i].first.c_str(), mProfile[i].second / itrationsTimes);
                totalTime += mProfile[i].second;
            }
            printf("Time over all layers: %4.3f\n", totalTime / itrationsTimes);
        }
    private:
        typedef std::pair<std::string, float> Record;
        std::vector<Record> mProfile;

        virtual void reportLayerTime(const char *layerName, float ms) TRT_NOEXCEPT {
            auto record = std::find_if(mProfile.begin(), mProfile.end(), [&](const Record& r){ return r.first == layerName; });
            if (record == mProfile.end()) {
                mProfile.push_back(std::make_pair(layerName, ms));
            }else {
                record->second += ms;
            }
        }
    };

    // log report
    class Logger : public nvinfer1::ILogger {
    public:
        Logger(): Logger(Severity::kWARNING) {}
        Logger(Severity severity): reportableSeverity(severity) {}

        void log(Severity severity, const char *msg) TRT_NOEXCEPT override {
            if (severity > reportableSeverity) return ;

            switch (severity) {
                case Severity::kINTERNAL_ERROR: std::cerr << "INTERNAL_ERROR: "; break;
                case Severity::kERROR: std::cerr << "ERROR: "; break;
                case Severity::kWARNING: std::cerr << "WARNING: "; break;
                case Severity::kINFO: std::cerr << "INFO: ";  break;
                default: std::cerr << "UNKNOWN: "; break;
            }
            std::cerr << msg << std::endl;
        }

        Severity reportableSeverity{Severity::kWARNING};
    };

    // write and read
    template<typename T>
    void write(char *&buffer, const T &val) {
        *reinterpret_cast<T *>(buffer) = val;  // 赋值
        buffer += sizeof(T);  // 地址移动
    }

    template<typename T>
    void read(const char *&buffer, T &val) {
        val = *reinterpret_cast<const T*>(buffer);
        buffer += sizeof(T);
    }
} // namespace Tn


#endif