#include <cstdio>

template <typename F> __global__ void kernel(F f) { f(threadIdx.x); }

__host__ __device__ void function_hostdevice(int i) { printf("cpu, %d\n", i); }

__device__ void function_onlydevice(int i) { printf("gpu, %d\n", i); }

int main() {
    auto func_gpu = [=] __device__(int i) { function_hostdevice(i); };

    auto func_cpu = [=](int i) { function_hostdevice(i); };

    auto func_gpu2 = [=] __device__(int i) { function_onlydevice(i); };

    kernel<<<1, 16>>>(func_gpu2);
}