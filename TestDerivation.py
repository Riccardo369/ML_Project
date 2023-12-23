import sympy as sp
import numpy as np

from TestBasedTools import DerivationLambda

M1=sp.matrices.Matrix([
    [1],
    [2],
    [3]
])
M2=sp.matrices.Matrix([
    [4],
    [5],
    [6]
])

print(M1.dot(M2))



