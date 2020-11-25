#pragma once

#include "mesh.hpp"
#include "zone.hpp"
#include <dataStructures/array.hpp>

void FillZone(D_Vector &u, D_Mesh &mesh, RectangleZone &RectangleZone, T value);

void FillOutsideZone(D_Vector &u, D_Mesh &mesh, RectangleZone &zone, T value);

T GetMinZone(D_Vector &u, D_Mesh &mesh, RectangleZone &zone);

T GetMaxZone(D_Vector &u, D_Mesh &mesh, RectangleZone &zone);

T GetMeanZone(D_Vector &u, D_Mesh &mesh, RectangleZone &zone);