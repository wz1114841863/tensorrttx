import argparse
import os
import struct
import sys

import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt

# override
INPUT_H = 32
INPUT_W = 32
OUTPUT_SIZE = 10
INPUT_BLOB_NAME = ""
OUTPUT_BLOB_NAME = ""

# weight_path = ""
# engine_path = ""

gLogger = trt.Logger(trt.Logger.INFO)

def load_weights(file_path, weight_path):
    print(f"Loading weights: {file_path}")
    assert os.path.exists(file_path), "Unable to load weight file."

    weight_map = {}
    with open(weight_path, "r") as fp:
        lines = [line.strip() for line in fp]  # 移除字符串头尾指定的字符（默认为空格）
    count = int(lines[0])
    assert count == len(lines) - 1
    for i in range(1, count + 1):
        splits = lines[i].split(" ")
        name = splits[0]
        cur_count = int(splits[1])
        assert cur_count + 2 == len(splits)
        values = []
        for j in range(2, len(splits)):
            # hex string to bytes to float
            values.append(struct.unpack(">f", bytes.fromhex(splits[j])))
        weight_map[name] = np.array(values, dtype=np.float32)
    
    return weight_map

def createLenetEngine(maxBatchSize, builder, config, dt, weight_path):
    pass

def APIToModel(maxBatchSize, engine_path):
    builder = trt.Builder(gLogger)
    config = builder.create_builder_config()
    engine = createLenetEngine(maxBatchSize, builder, config, trt.float32)
    assert engine

    with open(engine_path, "wb") as fp:
        fp.write(engine.serialize())

    del engine
    del builder

def doInference(context, host_in, host_out, batchsize):
    engine = context.engine
    assert engine.num_bindings == 2

    device_in = cuda.mem_alloc(host_in.nbytes)
    device_out = cuda.mem_alloc(host_out.nbytes)
    bindings = [int(device_in), int(device_out)]
    stream = cuda.Stream()

    cuda.memcpy_htod_async(device_in, host_in, stream)
    context.execute_async(bindings=bindings, stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(host_out, device_out, stream)
    stream.synchronize()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", action='store_true')
    parser.add_argument("-d", action='store_true')
    args = parser.parse_args()

    if not (args.s ^ args.d):
        print("arguments not right!")
        print("python lenet.py -s   # serialize model to plan file")
        print("python lenet.py -d   # deserialize plan file and run inference")
        sys.exit()
    return args

def main(engine_path):
    args = get_args()
    if args.s:
        APIToModel(1)
    else:
        runtime = trt.Runtime(gLogger)
        assert runtime

        with open(engine_path, "rb") as fp:
            engine = runtime.deserialize_cuda_engine(fp.read())
        assert engine

        context = engine.create_execution_context()
        assert context

        data = np.ones((INPUT_H * INPUT_W), dtype=np.float32)
        host_in = cuda.pagelocked_empty(INPUT_H * INPUT_W, dtype=np.float32)
        np.copyto(host_in, data.ravel())
        host_out = cuda.pagelocked_empty(OUTPUT_SIZE, dtype=np.float32)

        doInference(context, host_in, host_out, 1)
        print(f'Output: {host_out}')