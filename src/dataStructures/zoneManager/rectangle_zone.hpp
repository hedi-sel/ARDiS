#pragma once

#include "point_2d.hpp"
#include <constants.hpp>
#include <dataStructures/hd_data.hpp>

struct RectangleZone {
    T x0, x1, y0, y1;

    RectangleZone();
    RectangleZone(T x0, T x1, T y0, T y1);
    RectangleZone(Point2D p0, Point2D p1);

    __device__ __host__ bool IsInside(T x, T y) const;
    __device__ __host__ bool IsInside(Point2D p);
};
