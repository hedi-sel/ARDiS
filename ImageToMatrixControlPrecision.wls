#!/usr/bin/env wolframscript

Needs["NDSolve`FEM`"]
name = $ScriptCommandLine[[2]]
meshStepSize = ToExpression[$ScriptCommandLine[[3]]]

region = Region@ImageMesh@Import["maze.png"]
(* region = ImageMesh@Graphics@{Thickness[0.1], PeanoCurve[1]};*)
Export["region.png",RegionImage[region]]

(*Make mesh*)
nr=ToElementMesh[ BoundaryDiscretizeRegion@region , MaxCellMeasure->meshStepSize]
Print["Mesh number of points : "<>ToString@Length@nr["Coordinates"]]

vd=NDSolve`VariableData[{"DependentVariables"->{u},"Space"->{x,y}}];
sd=NDSolve`SolutionData[{"Space"->nr}];
coefficients={"DiffusionCoefficients"->{{IdentityMatrix[2]}},"DampingCoefficients"->{{1}}};
initCoeffs=InitializePDECoefficients[vd,sd,coefficients];
methodData=InitializePDEMethodData[vd,sd];

(*Assembly of matrices*)
discretePDE=DiscretizePDE[initCoeffs,methodData,sd];
{load,stiffness,damping,mass}=discretePDE["SystemMatrices"];

(*Export matrices*)
Export["matrixLabyrinth/stiffness_"<>name<>".mtx",stiffness]
Export["matrixLabyrinth/damping_"<>name<>".mtx",damping]
Export["matrixLabyrinth/mesh_"<>name<>".dat",nr["Coordinates"]]