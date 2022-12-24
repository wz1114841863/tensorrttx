#ifndef _YOLO_LAYER_H
#define _YOLO_LAYER_H

#include <vector>
#include <string>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include "macros.h"


namespace Yolo {
    static constexpr int CHECK_COUNT = 3;
    static constexpr float IGNORE_THRESH = 0.1f;
    static constexpr int MAX_OUTPUT_BBOX_COUNT = 1000;
    static constexpr int CLASS_NUM = 80;
    static constexpr int INPUT_H = 608;
    static constexpr int INPUT_W = 608;

    struct YoloKernel {
        int width;
        int height;
        float anchors[CHECK_COUNT * 2];

        // YoloKernel(int width, int height, float (&anchors)[CHECK_COUNT * 2])
        //     :width(width), height(height), anchors{anchors}{

        // }
    };

    static constexpr YoloKernel yolo1 = {
        INPUT_W / 32,
        INPUT_H / 32,
        {81, 82, 135, 160, 344, 319}
    };

    static constexpr YoloKernel yolo2 = {
        INPUT_W / 16,
        INPUT_H / 16,
        {23, 27, 37, 58, 81, 82}
    };

    static constexpr int LOCATIONS = 4;
    struct alignas(float) Detection {
        // x, y, w, h
        float bbox[LOCATIONS];
        float det_confidence;
        float class_id;
        float class_confidence;
    };

}


namespace nvinfer1 {
    class YoloLayerPlugin: public IPluginV2IOExt {
    public:
        explicit YoloLayerPlugin();
        YoloLayerPlugin(const void *data, size_t length);

        ~YoloLayerPlugin();

        int getNbOutputs() const TRT_NOEXCEPT override {
            return 1;
        }

        Dims getOutputDimensions(int index, const Dims* inputs, int nbInoutDims) TRT_NOEXCEPT override;

        int initialize() TRT_NOEXCEPT override;

        virtual void terminate() TRT_NOEXCEPT override {};

        virtual size_t getWorkspaceSize(int maxBatchSize) const TRT_NOEXCEPT override {return 0;}

        virtual int enqueue(int bathcSize, const void *const *inputs, 
            void *TRT_CONST_ENQUEUE *outputs, void *workspace, cudaStream_t stream) TRT_NOEXCEPT override;
        
        virtual size_t getSerializationSize() const TRT_NOEXCEPT override;

        virtual void serialize(void *buffer) const TRT_NOEXCEPT override;

        bool supportsFormatCombination(int pos, const PluginTensorDesc *inOut, int nbInputs, int nbOutputs) const TRT_NOEXCEPT override {
            return inOut[pos].format == TensorFormat::kLINEAR && inOut[pos].type == DataType::kFLOAT;
        }

        const char *getPluginType() const TRT_NOEXCEPT override;

        const char *getPluginVersion() const TRT_NOEXCEPT override;

        void destroy() TRT_NOEXCEPT override;

        IPluginV2IOExt *clone() const TRT_NOEXCEPT override;

        void setPluginNamespace(const char *pluginNamespace) TRT_NOEXCEPT override;

        const char *getPluginNamespace() const TRT_NOEXCEPT override;

        DataType getOutputDataType(int index, const nvinfer1::DataType *intputTypes, int nbInputs) const TRT_NOEXCEPT override;

        bool isOutputBroadcastAcrossBatch(int outputIndex, const bool *inputIsBroadcasted, int nbInputs) const TRT_NOEXCEPT override;

        bool canBroadcastInputAcrossBatch(int inoutIndex) const TRT_NOEXCEPT override;

        void attachToContext(cudnnContext *cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) TRT_NOEXCEPT override;

        void configurePlugin(const PluginTensorDesc *in, int nbInput, const PluginTensorDesc *out, int nbOutput) TRT_NOEXCEPT override;

        void detachFromContext() TRT_NOEXCEPT override;

    private:
        void forwardGpu(const float *const *inputs, float *output, cudaStream_t stream, int batchSize=1);
        int mClassCount;
        int mKernelCount;
        std::vector<Yolo::YoloKernel> mYolokernel;
        int mThreadCount = 256;
        const char *mPluginNamespace;
    };

    class YoloPluginCreator : public IPluginCreator {
    public:
        YoloPluginCreator();
        
        ~YoloPluginCreator() override = default;

        const char *getPluginName() const TRT_NOEXCEPT override;

        const char *getPluginVersion() const TRT_NOEXCEPT override;

        const PluginFieldCollection *getFieldNames() TRT_NOEXCEPT override;

        IPluginV2IOExt *createPlugin(const char *name, const PluginFieldCollection *fc) TRT_NOEXCEPT override;

        IPluginV2IOExt *deserializePlugin(const char* name, const void *serialData, size_t serialLength) TRT_NOEXCEPT override;

        void setPluginNamespace(const char *libNamespace) TRT_NOEXCEPT override {
            mNamespace = libNamespace;
        };

        const char* getPluginNamespace() const TRT_NOEXCEPT override {
            return mNamespace.c_str();
        }

    private:
        std::string mNamespace;
        static PluginFieldCollection mFC;
        static std::vector<PluginField> mPluginAttributes;
    };
    REGISTER_TENSORRT_PLUGIN(YoloPluginCreator);
};

#endif