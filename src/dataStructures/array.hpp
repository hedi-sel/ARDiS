#pragma once

#include <cuda_runtime.h>
#include <cusparse.h>
#include <nvfunctional>

#include "constants.hpp"
#include "helper/cuda/cuda_error_check.h"
#include "helper/cuda/cuda_thread_manager.hpp"
#include "helper/cuda/cusparse_error_check.h"

template <typename C> class d_array {
  public:
    int n;
    const bool is_device;

    C *data;

    d_array *_device;

    // Constructors
    __host__ d_array(int = 0, bool = true);
    __host__ d_array(const d_array &, bool copyToOtherMem = false);
    __host__ d_array(d_array<C> &&);

    // Manipulation
    __host__ void operator=(const d_array &);
    __host__ void resize(int);
    __host__ void fill(C value);

    // Accessors
    __host__ __device__ C &at(int i);
    __host__ __device__ int size();
    __host__ cusparseDnVecDescr_t make_descriptor();
    __host__ __device__ void print(int printCount = 5) const;

    __host__ ~d_array();

  private:
    __host__ void mem_alloc();
    __host__ void mem_free();
    int *n_dataholders = nullptr;
};

// typedef d_array<T> d_vector;

template class d_array<T>;
template class d_array<bool>;
template class d_array<int>;

class d_vector : public d_array<T> {
  public:
    using d_array<T>::d_array;
    __host__ std::string to_string();
    __host__ void prune(T value = 0);
    __host__ void prune_under(T value = 0);
};

template class d_array<d_vector *>;

enum AccessError { AccessHostOnDevice, AccessDeviceOnHost };