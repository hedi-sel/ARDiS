#!/usr/bin/env wolframscript

Needs["NDSolve`FEM`"]

name = $ScriptCommandLine[[2]]

rayonExt = ToExpression[$ScriptCommandLine[[3]]]
rayonInt = ToExpression[$ScriptCommandLine[[4]]]
thickness = ToExpression[$ScriptCommandLine[[5]]]

(*Define Region*)
branchInt = (4 - thickness)/2
branchExt = (4 + thickness)/2
cInt = branchInt - rayonInt
cExt = branchExt - rayonExt

centerX = 0
centerY = 0 (*the case X!=Y has not been tested !*)

includeReg = RegionIntersection[Disk[{cExt, cExt}, rayonExt],Rectangle[{cExt, cExt}, {cExt + rayonExt,cExt + rayonExt}]]
If[cExt > centerX, includeReg = RegionUnion[ includeReg, Rectangle[{centerX, centerY}, {cExt, branchExt}] , Rectangle[{cExt, centerY}, {branchExt, cExt}] ]]
If[cExt < centerX, includeReg = RegionIntersection[includeReg,Rectangle[{0,0},{branchExt,branchExt}]]]
(* 
Export["region1.png",RegionImage[includeReg]] *)

excludeReg = RegionIntersection[Disk[{cInt, cInt}, rayonInt],Rectangle[{cInt, cInt}, {cInt + rayonInt,cInt + rayonInt}]]
If[cInt > centerX, excludeReg = RegionUnion[ excludeReg, Rectangle[{centerX, centerY}, {cInt, branchInt}] , Rectangle[{cInt, centerY}, {branchInt, cInt}] ]]
(* 
Export["region2.png",RegionImage[excludeReg]] *)

region = RegionDifference[includeReg,excludeReg]
(* Export["region.png",RegionImage[region]] *)

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
