#include "rectangle_zone.hpp"

RectangleZone::RectangleZone() : RectangleZone(0, 0, 0, 0){};
RectangleZone::RectangleZone(T x0, T x1, T y0, T y1)
    : x0(x0), x1(x1), y0(y0), y1(y1){};
RectangleZone::RectangleZone(Point2D p0, Point2D p1)
    : RectangleZone(p0.x, p1.x, p0.y, p1.y){};

__device__ __host__ bool RectangleZone::IsInside(T x, T y) const {
    return x0 <= x && x1 >= x && y0 <= y && y1 >= y;
}
__device__ __host__ bool RectangleZone::IsInside(Point2D p) {
    return IsInside(p.x, p.y);
}
