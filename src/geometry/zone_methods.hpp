#pragma once

#include "mesh.hpp"
#include "zone.hpp"
#include <dataStructures/array.hpp>

void fill_zone(d_vector &u, d_mesh &mesh, rect_zone &rect_zone, T value);

void fill_outside_zone(d_vector &u, d_mesh &mesh, rect_zone &zone, T value);

T min_zone(d_vector &u, d_mesh &mesh, rect_zone &zone);

T max_zone(d_vector &u, d_mesh &mesh, rect_zone &zone);

T mean_zone(d_vector &u, d_mesh &mesh, rect_zone &zone);