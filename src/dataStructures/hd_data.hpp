#pragma once

#include "cstdio"
#include "cuda_runtime.h"

#include "constants.hpp"
#include <helper/cuda/cuda_error_check.h>

template <typename DataType> class hd_data {
  public:
    DataType *_host;
    DataType *_device;
    hd_data() {
        cudaMalloc(&_device, sizeof(DataType));
        _host = new DataType();
    };
    hd_data(DataType *data, bool itsDevice) : hd_data() {
        Set(data, itsDevice);
    }
    hd_data(DataType data) : hd_data() { Set(&data, false); }

    __host__ __device__ DataType &operator()(bool device = false) {
        if (device)
            return *_device;
        else
            return *_host;
    }

    void SetDevice(DataType *data, bool itsDevice = true) {
        if (itsDevice) {
            gpuErrchk(cudaMemcpy(_device, data, sizeof(DataType),
                                 cudaMemcpyDeviceToDevice));
        } else {
            gpuErrchk(cudaMemcpy(_device, data, sizeof(DataType),
                                 cudaMemcpyHostToDevice));
        }
    }
    void SetDevice() { SetDevice(_host, false); }

    void SetHost(DataType *data, bool itsDevice = false) {
        if (itsDevice) {
            gpuErrchk(cudaMemcpy(_host, data, sizeof(DataType),
                                 cudaMemcpyDeviceToHost));
        } else {
            gpuErrchk(cudaMemcpy(_host, data, sizeof(DataType),
                                 cudaMemcpyHostToHost));
        }
    }
    void SetHost() { SetHost(_device, true); }

    void Set(DataType *data, bool itsDevice = false) {
        SetHost(data, itsDevice);
        SetDevice(data, itsDevice);
    }

    ~hd_data() {
        cudaFree(_device);
        delete _host;
    }
};