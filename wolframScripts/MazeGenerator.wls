#!/usr/bin/env wolframscript

Needs["NDSolve`FEM`"]

name = $ScriptCommandLine[[2]]
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
(* Export["region.png",RegionImage[region]] *)


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
Export["matrixLabyrinth/stiffness_"<>name<>".mtx",stiffness]
Export["matrixLabyrinth/damping_"<>name<>".mtx",damping]
Export["matrixLabyrinth/mesh_"<>name<>".dat",nr["Coordinates"]]
