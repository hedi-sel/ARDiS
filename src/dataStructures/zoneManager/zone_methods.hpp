#include "rectangle_zone.hpp"
#include <dataStructures/array.hpp>

void FillZone(D_Vector &u, D_Vector &mesh_x, D_Vector &mesh_y, Zone &zone,
              T value);

void FillOutsideZone(D_Vector &u, D_Vector &mesh_x, D_Vector &mesh_y,
                     Zone &zone, T value);

T GetMinZone(D_Vector &u, D_Vector &mesh_x, D_Vector &mesh_y, Zone &zone);

T GetMaxZone(D_Vector &u, D_Vector &mesh_x, D_Vector &mesh_y, Zone &zone);

T GetMeanZone(D_Vector &u, D_Vector &mesh_x, D_Vector &mesh_y, Zone &zone);