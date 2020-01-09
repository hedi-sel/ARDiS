#pragma once

#include "cuda_runtime.h"

#include "constants.hpp"

template <typename DataType> class HDData {
  public:
    DataType _host;
    DataType *_device;
    HDData() { cudaMalloc(&_device, sizeof(DataType)); };
    HDData(DataType &data, bool itsDevice = false) : HDData() {
        Set(data, itsDevice);
    }
    HDData(DataType data) : HDData() { Set(data, false); }
    DataType &operator()(bool device = false) {
        if (device)
            return *_device;
        else
            return _host;
    }
    void SetDevice(DataType &data, bool itsDevice = true) {
        if (itsDevice)
            cudaMemcpy(_device, &data, sizeof(DataType),
                       cudaMemcpyDeviceToDevice);
        else
            cudaMemcpy(_device, &data, sizeof(DataType),
                       cudaMemcpyHostToDevice);
    }
    void SetDevice() { SetDevice(_host, false); }

    void SetHost(DataType &data, bool itsDevice = false) {
        if (itsDevice)
            cudaMemcpy(&_host, &data, sizeof(DataType), cudaMemcpyDeviceToHost);
        else
            _host = data;
    }
    void SetHost() { SetHost(*_device, true); }

    void Set(DataType &data, bool itsDevice = false) {
        SetHost(data, itsDevice);
        SetDevice(data, itsDevice);
    }

    ~HDData() { cudaFree(_device); }
};