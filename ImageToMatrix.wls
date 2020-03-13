#!/usr/bin/env wolframscript

Needs["NDSolve`FEM`"]

region = Region@ImageMesh@Import["maze.png"]
Export["region.png",RegionImage[region]]

(* 
branches = 
 RegionUnion[ Rectangle[{-1, branchInt}, {cInt, branchExt}], 
  Rectangle[{branchInt, -1}, {branchExt, cExt}]]

anneau = RegionDifference[Disk[{cExt, cExt}, rayonExt], 
  Disk[{cInt, cInt}, rayonInt]] 
RegionPlot@anneau
virage = RegionIntersection[anneau, 
  Rectangle[{0, 0}, {branchExt, branchExt}]]
region = RegionUnion[virage, branches] *)

(*Make mesh*)
nr=ToElementMesh[#,MaxCellMeasure->.001]&@BoundaryDiscretizeRegion@region;

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
