#!/usr/bin/env wolframscript

Needs["NDSolve`FEM`"]

outputPath = $ScriptCommandLine[[2]]
size = ToExpression[$ScriptCommandLine[[3]]]

X = size; Y = size;
(* custom styling *)
style = {Background -> GrayLevel[0], 
   BaseStyle -> {Directive[White, EdgeForm[], Opacity[1]]}, 
   VertexShapeFunction -> (Rectangle[#1 + .16, #1 - .16] &), 
   EdgeShapeFunction -> (Rectangle[#1[[1]] + .16, #1[[2]] - .16] &)};
embedding = GraphEmbedding[GridGraph[{X, Y}]];

g = GridGraph[{X, Y}, EdgeWeight -> RandomReal[10, X*Y*2-X-Y]];
tree = FindSpanningTree[{g, 1}];
maze = Graph[tree, VertexCoordinates -> embedding, style];

region =Region@ImageMesh@Image@maze

(*Make mesh*)
nr=ToElementMesh[ BoundaryDiscretizeRegion@region , MaxCellMeasure->20]

vd=NDSolve`VariableData[{"DependentVariables"->{u},"Space"->{x,y}}];
sd=NDSolve`SolutionData[{"Space"->nr}];
coefficients={"DiffusionCoefficients"->{{IdentityMatrix[2]}},"DampingCoefficients"->{{1}}};
initCoeffs=InitializePDECoefficients[vd,sd,coefficients];
methodData=InitializePDEMethodData[vd,sd];

(*Assembly of matrices*)
discretePDE=DiscretizePDE[initCoeffs,methodData,sd];
{load,stiffness,damping,mass}=discretePDE["SystemMatrices"];

(*Export matrices*)
imagePath = outputPath <> ".png"
stiffnessPath = outputPath <> "_stiffness.mtx"
dampingPath =  outputPath <> "_damping.mtx"
meshPath =  outputPath <> "_mesh.dat"

Export[imagePath, region]
Export[stiffnessPath,stiffness]
Export[dampingPath,damping]
Export[meshPath,nr["Coordinates"]]

Print["A maze has been generated and data has been created at:"]
Print[imagePath]
Print[stiffnessPath]
Print[dampingPath]
Print[meshPath]