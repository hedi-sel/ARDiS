#!/usr/bin/env wolframscript

Needs["NDSolve`FEM`"]
name = "maze"
region = Region@ImageMesh@Import["maze.png"]
(* region = ImageMesh@Graphics@{Thickness[0.1], PeanoCurve[1]};*)
Export["region.png",RegionImage[region]]

Print["Shape built"]


(*Make mesh*)
nr=ToElementMesh[ BoundaryDiscretizeRegion@region , MaxCellMeasure->20]
Print["Mesh"]


vd=NDSolve`VariableData[{"DependentVariables"->{u},"Space"->{x,y}}];
sd=NDSolve`SolutionData[{"Space"->nr}];
coefficients={"DiffusionCoefficients"->{{IdentityMatrix[2]}},"DampingCoefficients"->{{1}}};
initCoeffs=InitializePDECoefficients[vd,sd,coefficients];
methodData=InitializePDEMethodData[vd,sd];

Print["Initialized"]

(*Assembly of matrices*)
discretePDE=DiscretizePDE[initCoeffs,methodData,sd];
{load,stiffness,damping,mass}=discretePDE["SystemMatrices"];

Print["Matrices built"]

(*Export matrices*)
Export["matrixLabyrinth/stiffness_"<>name<>".mtx",stiffness]
Export["matrixLabyrinth/damping_"<>name<>".mtx",damping]
Export["matrixLabyrinth/mesh_"<>name<>".dat",nr["Coordinates"]]

Print["Matrices exported"]
