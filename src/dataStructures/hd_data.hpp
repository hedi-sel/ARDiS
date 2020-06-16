#pragma once

#include "cstdio"
#include "cuda_runtime.h"

#include "constants.hpp"
#include <hediHelper/cuda/cuda_error_check.h>

template <typename DataType> class HDData {
  public:
    DataType *_host;
    DataType *_device;
    HDData() {
        cudaMalloc(&_device, sizeof(DataType));
        _host = new DataType();
    };
    HDData(DataType *data, bool itsDevice) : HDData() { Set(data, itsDevice); }
    HDData(DataType data) : HDData() { Set(&data, false); }
    DataType &operator()(bool device = false) {
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

    ~HDData() {
        cudaFree(_device);
        delete _host;
    }
};