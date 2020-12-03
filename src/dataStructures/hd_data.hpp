#pragma once

#include "cstdio"
#include "cuda_runtime.h"

#include "constants.hpp"
#include <helper/cuda/cuda_error_check.h>

template <typename dtype> class hd_data {
  public:
    dtype *_host;
    dtype *_device;
    hd_data() {
        cudaMalloc(&_device, sizeof(dtype));
        _host = new dtype();
    };
    hd_data(dtype *data, bool itsDevice) : hd_data() { set(data, itsDevice); }
    hd_data(dtype data) : hd_data() { set(&data, false); }

    __host__ __device__ dtype &operator()(bool device = false) {
        if (device)
            return *_device;
        else
            return *_host;
    }

    void set_dev(dtype *data, bool itsDevice = true) {
        if (itsDevice) {
            gpuErrchk(cudaMemcpy(_device, data, sizeof(dtype),
                                 cudaMemcpyDeviceToDevice));
        } else {
            gpuErrchk(cudaMemcpy(_device, data, sizeof(dtype),
                                 cudaMemcpyHostToDevice));
        }
    }
    void update_dev() { set_dev(_host, false); }

    void set_host(dtype *data, bool itsDevice = false) {
        if (itsDevice) {
            gpuErrchk(
                cudaMemcpy(_host, data, sizeof(dtype), cudaMemcpyDeviceToHost));
        } else {
            gpuErrchk(
                cudaMemcpy(_host, data, sizeof(dtype), cudaMemcpyHostToHost));
        }
    }
    void update_host() { set_host(_device, true); }

    void set(dtype *data, bool itsDevice = false) {
        set_host(data, itsDevice);
        set_dev(data, itsDevice);
    }

    ~hd_data() {
        cudaFree(_device);
        delete _host;
    }
};