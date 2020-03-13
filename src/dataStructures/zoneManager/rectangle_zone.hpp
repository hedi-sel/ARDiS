#pragma once

#include "point_2d.hpp"
#include <constants.hpp>
#include <dataStructures/hd_data.hpp>

struct Zone {
    __device__ __host__ bool IsInside(T x, T y) const { return false; }
    __device__ __host__ bool IsInside(Point2D p) { return false; }
};

struct RectangleZone : Zone {
    T x0, x1, y0, y1;

    RectangleZone();
    RectangleZone(T x0, T y0, T x1, T y1);
    RectangleZone(Point2D p0, Point2D p1);

    __device__ __host__ bool IsInside(T x, T y) const;
    __device__ __host__ bool IsInside(Point2D p);
};

struct TriangleZone : Zone {
    T x0, x1, x2, y0, y1, y2;

    TriangleZone();
    TriangleZone(T x0, T y0, T x1, T y1, T x2, T y2);
    TriangleZone(Point2D p0, Point2D p1, Point2D p2);

    __device__ __host__ bool IsInside(T x, T y) const;
    __device__ __host__ bool IsInside(Point2D p);

    void Print() {
        printf("TriangleZone: P0(%f,%f) P1(%f,%f) P2(%f,%f) \n", x0, y0, x1, y1,
               x2, y2);
    }
};