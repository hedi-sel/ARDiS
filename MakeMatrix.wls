#!/usr/bin/env wolframscript

Needs["NDSolve`FEM`"]

name = $ScriptCommandLine[[2]]

arg1 = ToExpression[$ScriptCommandLine[[3]]]
arg2 = ToExpression[$ScriptCommandLine[[4]]]
rayonExt = If[arg1>arg2, arg1, arg2]
rayonInt = If[arg1>arg2, arg2, arg1]

(*Define Region*)
branchInt = 1
branchExt = 2
branches = 
 RegionUnion[ Rectangle[{-1, branchInt}, {0, branchExt}], 
  Rectangle[{branchInt, -1}, {branchExt, 0}]]

Xext = (branchExt - Sqrt[2*rayonExt^2 - branchExt^2])/2
Xint = (branchInt - Sqrt[2*rayonInt ^2 - branchInt^2])/2
anneau = RegionDifference[Disk[{Xext, Xext}, rayonExt], 
  Disk[{Xint, Xint}, rayonInt]] 
RegionPlot@anneau
virage = RegionIntersection[anneau, 
  Rectangle[{0, 0}, {branchExt, branchExt}]]
region = RegionUnion[virage, branches]

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
Export["matrixLabyrinth/stiffness "<>name<>".mtx",stiffness]
Export["matrixLabyrinth/damping "<>name<>".mtx",damping]
Export["matrixLabyrinth/mesh "<>name<>".dat",nr["Coordinates"]]
