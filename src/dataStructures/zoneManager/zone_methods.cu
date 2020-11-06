#include "dataStructures/helper/apply_operation.h"
#include "hediHelper/cuda/cuda_device_converter.h"
#include "matrixOperations/basic_operations.hpp"
#include "rectangle_zone.hpp"
#include <dataStructures/array.hpp>

__global__ void IsInsideArrayK(D_Vector &mesh_x, D_Vector &mesh_y, Zone &zone,
                               D_Vector &is_inside, T value) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= is_inside.n)
        return;
    is_inside.data[i] =
        (zone.IsInside(mesh_x.data[i], mesh_y.data[i])) ? value : 0;
}
D_Vector IsInsideArray(D_Vector &mesh_x, D_Vector &mesh_y, Zone &zone,
                       T value = 1) {
    assert(mesh_x.n == mesh_y.n);
    Zone *d_zone;
    cudaMalloc(&d_zone, sizeof(zone));
    cudaMemcpy(d_zone, &zone, sizeof(zone), cudaMemcpyHostToDevice);

    D_Vector is_inside(mesh_x.n);
    auto tb = Make1DThreadBlock(mesh_x.n);
    IsInsideArrayK<<<tb.block, tb.thread>>>(
        *(D_Vector *)mesh_x._device, *(D_Vector *)mesh_y._device, *d_zone,
        *(D_Vector *)is_inside._device, value);
    return is_inside;
    cudaFree(d_zone);
}

void FillZone(D_Vector &u, D_Vector &mesh_x, D_Vector &mesh_y, Zone &zone,
              T value) {
    auto setToVal = [value] __device__(T & a) { a = value; };
    auto is_inside = IsInsideArray(mesh_x, mesh_y, zone);
    ApplyFunctionConditional(u, is_inside, setToVal);
}

void FillOutsideZone(D_Vector &u, D_Vector &mesh_x, D_Vector &mesh_y,
                     Zone &zone, T value) {
    auto setToVal = [value] __device__(T & a) { a = value; };
    D_Vector is_outside(u.n);
    is_outside.Fill(1);
    auto is_inside = IsInsideArray(mesh_x, mesh_y, zone);
    HDData<T> m1(-1);
    VectorSum(is_outside, is_inside, is_outside, m1(true));
    ApplyFunctionConditional(u, is_inside, setToVal);
}

T GetMinZone(D_Vector &u, D_Vector &mesh_x, D_Vector &mesh_y, Zone &zone) {
    auto min = [] __device__(T & a, T & b) { return (a < b) ? a : b; };
    D_Vector u_copy(u);
    auto is_inside = IsInsideArray(mesh_x, mesh_y, zone);
    ReductionFunctionConditional(u_copy, is_inside, min);
    T result = -1;
    cudaMemcpy(&result, u_copy.data, sizeof(T), cudaMemcpyDeviceToHost);
    return result;
};

T GetMaxZone(D_Vector &u, D_Vector &mesh_x, D_Vector &mesh_y, Zone &zone) {
    auto max = [] __device__(T & a, T & b) { return (a > b) ? a : b; };
    D_Vector u_copy(u);
    auto is_inside = IsInsideArray(mesh_x, mesh_y, zone);
    ReductionFunctionConditional(u_copy, is_inside, max);
    T result = -1;
    cudaMemcpy(&result, u_copy.data, sizeof(T), cudaMemcpyDeviceToHost);
    return result;
};

T GetMeanZone(D_Vector &u, D_Vector &mesh_x, D_Vector &mesh_y, Zone &zone) {
    D_Vector ones(u.n);
    ones.Fill(1);
    auto is_inside = IsInsideArray(mesh_x, mesh_y, zone, (T)(1.0));
    HDData<T> n(0);
    Dot(ones, is_inside, n(true));
    n.SetHost();
    HDData<T> result(0);
    Dot(u, is_inside, result(true));
    result.SetHost();
    return result() / n();
};