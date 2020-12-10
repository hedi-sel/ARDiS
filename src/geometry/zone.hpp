#pragma once

#include "point_2d.hpp"
#include "pybind11_include.hpp"
#include <constants.hpp>
#include <dataStructures/hd_data.hpp>

struct zone {
    // __device__ __host__ virtual bool is_inside(T x, T y) = 0;
    // __device__ __host__ virtual bool is_inside(point2d p) = 0;
};

struct simple_zone : zone {
    bool always_return;
    simple_zone(bool b) : always_return(b){};
    __device__ __host__ bool is_inside(T x, T y) { return always_return; }
    __device__ __host__ bool is_inside(point2d p) { return always_return; }

    static simple_zone all;
    static simple_zone none;
};

struct rect_zone : zone {
    T x0, x1, y0, y1;

    rect_zone();
    rect_zone(T x0, T y0, T x1, T y1);
    rect_zone(point2d p0, point2d p1);

    __device__ __host__ bool is_inside(T x, T y);
    __device__ __host__ bool is_inside(point2d p);
};

struct tri_zone : zone {
    T x0, x1, x2, y0, y1, y2;

    tri_zone();
    tri_zone(T x0, T y0, T x1, T y1, T x2, T y2);
    tri_zone(point2d p0, point2d p1, point2d p2);

    __device__ __host__ bool is_inside(T x, T y);
    __device__ __host__ bool is_inside(point2d p);

    void print() {
        printf("tri_zone: P0(%f,%f) P1(%f,%f) P2(%f,%f) \n", x0, y0, x1, y1, x2,
               y2);
    }
};

struct circle_zone : zone {
    T x0, y0, r;

    circle_zone();
    circle_zone(T x0, T y0, T r);
    circle_zone(point2d center, T radius);

    __device__ __host__ bool is_inside(T x, T y);
    __device__ __host__ bool is_inside(point2d p);

    void print() {
        printf("circle_zone: Center(%f,%f) Radius(%f) \n", x0, y0, r);
    }
};