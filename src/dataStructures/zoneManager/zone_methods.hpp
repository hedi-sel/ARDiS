#include "rectangle_zone.hpp"
#include <dataStructures/array.hpp>

void FillZone(D_Array &u, D_Array &mesh_x, D_Array &mesh_y, RectangleZone &zone,
              T value);

void FillOutsideZone(D_Array &u, D_Array &mesh_x, D_Array &mesh_y,
                     RectangleZone &zone, T value);

T GetMinZone(D_Array &u, D_Array &mesh_x, D_Array &mesh_y, RectangleZone &zone);

T GetMaxZone(D_Array &u, D_Array &mesh_x, D_Array &mesh_y, RectangleZone &zone);

T GetMeanZone(D_Array &u, D_Array &mesh_x, D_Array &mesh_y,
              RectangleZone &zone);