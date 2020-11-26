#include "dataStructures/helper/apply_operation.h"
#include "helper/cuda/cuda_device_converter.h"
#include "matrixOperations/basic_operations.hpp"
#include "zone_methods.hpp"

__global__ void IsInsideArrayK(D_Vector &mesh_x, D_Vector &mesh_y,
                               RectangleZone &zone, D_Array<bool> &is_inside) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= is_inside.n)
        return;
    is_inside.data[i] = zone.IsInside(mesh_x.data[i], mesh_y.data[i]);
}

D_Array<bool> IsInsideArray(D_Mesh &mesh, RectangleZone &zone) {
    RectangleZone *d_zone;
    cudaMalloc(&d_zone, sizeof(RectangleZone));
    cudaMemcpy(d_zone, &zone, sizeof(zone), cudaMemcpyHostToDevice);

    RectangleZone *zone2 = new RectangleZone();
    cudaMemcpy(zone2, d_zone, sizeof(zone), cudaMemcpyDeviceToHost);

    D_Array<bool> is_inside(mesh.Size());
    auto tb = Make1DThreadBlock(mesh.Size());

    IsInsideArrayK<<<tb.block, tb.thread>>>(
        *(D_Vector *)mesh.X._device, *(D_Vector *)mesh.Y._device, *d_zone,
        *(D_Array<bool> *)is_inside._device);
    cudaFree(d_zone);
    return is_inside;
}

void FillZone(D_Vector &u, D_Mesh &mesh, RectangleZone &zone, T value) {
    assert(u.n == mesh.Size());
    auto setToVal = [value] __device__(T & a) { a = value; };
    auto is_inside = IsInsideArray(mesh, zone);
    ApplyFunctionConditional(u, is_inside, setToVal);
}

void FillOutsideZone(D_Vector &u, D_Mesh &mesh, RectangleZone &zone, T value) {
    assert(u.n == mesh.Size());
    auto setToVal = [value] __device__(T & a) { a = value; };
    D_Vector is_outside(u.n);
    auto is_inside = IsInsideArray(mesh, zone);
    HDData<T> m1(-1);
    ApplyFunction(is_inside, [] __device__(bool &a) { a = !a; });
    ApplyFunctionConditional(u, is_inside, setToVal);
}

T GetMinZone(D_Vector &u, D_Mesh &mesh, RectangleZone &zone) {
    assert(u.n == mesh.Size());
    auto min = [] __device__(T & a, T & b) { return (a < b) ? a : b; };
    D_Vector u_copy(u);
    auto is_inside = IsInsideArray(mesh, zone);
    ReductionFunctionConditional(u_copy, is_inside, min);
    T result = -1;
    cudaMemcpy(&result, u_copy.data, sizeof(T), cudaMemcpyDeviceToHost);
    return result;
};

T GetMaxZone(D_Vector &u, D_Mesh &mesh, RectangleZone &zone) {
    assert(u.n == mesh.Size());
    auto max = [] __device__(T & a, T & b) { return (a > b) ? a : b; };
    D_Vector u_copy(u);
    auto is_inside = IsInsideArray(mesh, zone);
    ReductionFunctionConditional(u_copy, is_inside, max);
    T result = -1;
    cudaMemcpy(&result, u_copy.data, sizeof(T), cudaMemcpyDeviceToHost);
    return result;
};

T GetMeanZone(D_Vector &u, D_Mesh &mesh, RectangleZone &zone) {
    assert(u.n == mesh.Size());
    D_Array<int> ones(u.n);
    ones.Fill(1);
    auto is_inside = IsInsideArray(mesh, zone);
    ReductionFunctionConditional(
        ones, is_inside, [] __device__(int &a, int &b) { return a + b; });
    HDData<int> n_vals(ones.data, true);
    ReductionFunctionConditional(u, is_inside,
                                 [] __device__(T & a, T & b) { return a + b; });
    HDData<T> total_sum(u.data, true);
    return total_sum() / n_vals();
};