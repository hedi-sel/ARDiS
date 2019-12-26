import modulePython.dna as dna

dna.LoadMatrixFromFile("matrix/test.mtx", True)
dna.PrintMatrix()

print(dna.SolveLinEq([1, 1, 1, 1, 1]))
