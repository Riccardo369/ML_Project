import numpy as np
from types import LambdaType
import sympy as sp

#This function check number of lambda function's parameters
def CheckParametersFunction(Function, n):
  if(Function.__code__.co_argcount != n): raise ValueError(f"This function has {n} parameters, but it has {Function.__code__.co_argcount} arguments")

#This function take lambda function with more variables and get own derivate function, with one variable
def DerivationLambda(Lambda, i):
  VariablesName = Lambda.__code__.co_varnames[:Lambda.__code__.co_argcount]
  Variables = sp.symbols(' '.join(map(str, VariablesName)))
  if Lambda.__code__.co_argcount == 1:
    Expression=sp.sympify(Lambda(Variables))
  else:
    Expression = sp.sympify(Lambda(*Variables))
  Derivate = sp.diff(Expression, VariablesName[i])
  Result = sp.lambdify(VariablesName, Derivate)
  return Result

#This class represents a vector gradient
class Gradient:
  def __init__(self, Function: LambdaType):
    self.__GradientVector = np.array([DerivationLambda(Function, i) for i in range(Function.__code__.co_argcount)])

  def __call__(self, *InputVector):
    return np.array([i(*InputVector) for i in self.__GradientVector])
  
class HyperParameter:
  def __init__(self, Value):
    self.Value = Value
    
  def __call__(self):
    return self.Value
  