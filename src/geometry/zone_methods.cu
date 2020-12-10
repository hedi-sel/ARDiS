#include "dataStructures/helper/apply_operation.h"
#include "helper/cuda/cuda_device_converter.h"
#include "matrixOperations/basic_operations.hpp"
#include "zone_methods.hpp"

__global__ void is_inside_arrayK(d_vector &mesh_x, d_vector &mesh_y,
                                 rect_zone &zone, d_array<bool> &is_inside) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= is_inside.n)
        return;
    is_inside.data[i] = zone.is_inside(mesh_x.data[i], mesh_y.data[i]);
}

d_array<bool> is_inside_array(d_mesh &mesh, rect_zone &zone) {
    rect_zone *d_zone;
    cudaMalloc(&d_zone, sizeof(rect_zone));
    cudaMemcpy(d_zone, &zone, sizeof(zone), cudaMemcpyHostToDevice);

    d_array<bool> is_inside(mesh.size());
    auto tb = make1DThreadBlock(mesh.size());

    is_inside_arrayK<<<tb.block, tb.thread>>>(
        *(d_vector *)mesh.X._device, *(d_vector *)mesh.Y._device, *d_zone,
        *(d_array<bool> *)is_inside._device);
    cudaFree(d_zone);
    return is_inside;
}

void fill_zone(d_vector &u, d_mesh &mesh, rect_zone &zone, T value) {
    assert(u.n == mesh.size());
    auto setToVal = [value] __device__(T & a) { a = value; };
    auto is_inside = is_inside_array(mesh, zone);
    apply_func_cond(u, is_inside, setToVal);
}

void fill_outside_zone(d_vector &u, d_mesh &mesh, rect_zone &zone, T value) {
    assert(u.n == mesh.size());
    auto setToVal = [value] __device__(T & a) { a = value; };
    d_vector is_outside(u.n);
    auto is_inside = is_inside_array(mesh, zone);
    hd_data<T> m1(-1);
    apply_func(is_inside, [] __device__(bool &a) { a = !a; });
    apply_func_cond(u, is_inside, setToVal);
}

T min_zone(d_vector &u, d_mesh &mesh, rect_zone &zone) {
    assert(u.n == mesh.size());
    auto min = [] __device__(T & a, T & b) { return (a < b) ? a : b; };
    d_vector u_copy(u);
    auto is_inside = is_inside_array(mesh, zone);
    reduction_func_cond(u_copy, is_inside, min);
    T result = -1;
    cudaMemcpy(&result, u_copy.data, sizeof(T), cudaMemcpyDeviceToHost);
    return result;
};

T max_zone(d_vector &u, d_mesh &mesh, rect_zone &zone) {
    assert(u.n == mesh.size());
    auto max = [] __device__(T & a, T & b) { return (a > b) ? a : b; };
    d_vector u_copy(u);
    auto is_inside = is_inside_array(mesh, zone);
    reduction_func_cond(u_copy, is_inside, max);
    T result = -1;
    cudaMemcpy(&result, u_copy.data, sizeof(T), cudaMemcpyDeviceToHost);
    return result;
};

T mean_zone(d_vector &u, d_mesh &mesh, rect_zone &zone) {
    assert(u.n == mesh.size());
    d_array<int> ones(u.n);
    ones.fill(1);
    auto is_inside = is_inside_array(mesh, zone);
    reduction_func_cond(ones, is_inside,
                        [] __device__(int &a, int &b) { return a + b; });
    hd_data<int> n_vals(ones.data, true);
    reduction_func_cond(u, is_inside,
                        [] __device__(T & a, T & b) { return a + b; });
    hd_data<T> total_sum(u.data, true);
    return total_sum() / n_vals();
};