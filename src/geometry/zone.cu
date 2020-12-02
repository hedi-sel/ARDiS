#include "zone.hpp"

SimpleZone SimpleZone::all = SimpleZone(true);
SimpleZone SimpleZone::none = SimpleZone(false);

rect_zone::rect_zone() : rect_zone(0, 0, 0, 0){};
rect_zone::rect_zone(T x0, T y0, T x1, T y1) : x0(x0), x1(x1), y0(y0), y1(y1){};
rect_zone::rect_zone(point2d p0, point2d p1)
    : rect_zone(p0.x, p0.y, p1.x, p1.y){};

__device__ __host__ bool rect_zone::is_inside(T x, T y) {
    return x0 <= x && x1 >= x && y0 <= y && y1 >= y;
}
__device__ __host__ bool rect_zone::is_inside(point2d p) {
    return is_inside(p.x, p.y);
}

tri_zone::tri_zone() : tri_zone(0, 0, 0, 0, 0, 0){};
tri_zone::tri_zone(T x0, T y0, T x1, T y1, T x2, T y2)
    : x0(x0), x1(x1), y0(y0), y1(y1), x2(x2), y2(y2){};
tri_zone::tri_zone(point2d p0, point2d p1, point2d p2)
    : tri_zone(p0.x, p0.y, p1.x, p1.y, p2.x, p2.y){};

__device__ __host__ T Sign(T x0, T y0, T x1, T y1, T x2, T y2) {
    return (x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2);
}

__device__ __host__ bool tri_zone::is_inside(T x, T y) {
    bool b1 = Sign(x, y, x0, y0, x1, y1) < 0.0;
    bool b2 = Sign(x, y, x1, y1, x2, y2) < 0.0;
    bool b3 = Sign(x, y, x2, y2, x0, y0) < 0.0;
    return ((b1 == b2) && (b2 == b3));
}

__device__ __host__ bool tri_zone::is_inside(point2d p) {
    return is_inside(p.x, p.y);
}
