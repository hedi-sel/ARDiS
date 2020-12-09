#!/usr/bin/env wolframscript

(*Give the path of the image to convert as an argument*)

Needs["NDSolve`FEM`"]

imagePath = $ScriptCommandLine[[2]]
If[!StringContainsQ[imagePath, ".png"], Throw["File extension has to be .png"]]
region = RegionResize[Region@ImageMesh@Import[imagePath], {{0,100}, {0,100}}]


(*Make mesh*)
nr=ToElementMesh[ BoundaryDiscretizeRegion@region , MaxCellMeasure->1 ]

vd=NDSolve`VariableData[{"DependentVariables"->{u},"Space"->{x,y}}];
sd=NDSolve`SolutionData[{"Space"->nr}];
coefficients={"DiffusionCoefficients"->{{IdentityMatrix[2]}},"DampingCoefficients"->{{1}}};
initCoeffs=InitializePDECoefficients[vd,sd,coefficients];
methodData=InitializePDEMethodData[vd,sd];

(*Assembly of matrices*)
discretePDE=DiscretizePDE[initCoeffs,methodData,sd];
{load,stiffness,damping,mass}=discretePDE["SystemMatrices"];

(*Export matrices*)
stiffnessPath = StringReplace[imagePath, ".png"->"_stiffness.mtx"]
dampingPath = StringReplace[imagePath, ".png"->"_damping.mtx"]
meshPath = StringReplace[imagePath, ".png"->"_mesh.dat"]

Export[stiffnessPath,stiffness]
Export[dampingPath,damping]
Export[meshPath,nr["Coordinates"]]

Print["Data has been created for the given image:"]
Print[stiffnessPath]
Print[dampingPath]
Print[meshPath]
