from types import LambdaType
import random 
import math
from BasedTools import *
import Bridge
import numpy as np

#Responsabilità, rappresentare un neurone tenendo conto degl' archi di uscita, di entrata e della bias, in più calcolare la propria funzione di attivazione
class Neuron:

  def __init__(self, UpdateWeightsFunction: LambdaType, LossFunction: LambdaType, *args):

    CheckParametersFunction(UpdateWeightsFunction, 2)
    CheckParametersFunction(LossFunction, 2)
    
    self._InputBridges = []
    self._OutputBridges = []
    
    self._InputVector = np.array([1]) #Lista degl' input che il neurone può avere, già inserito l' input della bias
    
    #Bias già inserita, la bias non è altro che un altro peso aggiuntivo, con input fisso a 1
    if(len(args) > 0): self._Weights = np.array([args[0]])
    else: self._Weights = np.array([random.randint(-10, 10)/100])

    self.__ActivedState = True

    self.__UpdateWeightsFunction = UpdateWeightsFunction
    self.__BeforeUpdateWeightsFunction = lambda x, y: (x, y)

    self.__LossFunction = LossFunction
    self.__BeforeLossFunction = lambda x, y: (x, y)
    
    try:  self._GradientLossFunction = DerivationLambda(self.__LossFunction, 0)
    except: self._GradientLossFunction = lambda x, y: 1
    
    self.__BeforeGradientLossFunction = lambda x, y: (x, y)
    
  #Implementazione intersecata con l' algoritmo di aggiornamento di momentum 
  def CalculateUpdatedWeights(self, LossGradientValue, OldGradientValue):
    LossGradientValue, NeuronOutputs = self.__BeforeUpdateWeightsFunction(LossGradientValue, [])
    return np.float64(self.__UpdateWeightsFunction(self._Weights, LossGradientValue, OldGradientValue))

  def CalculateLoss(self, CalculatedOutput, TargetOutput):
    CalculatedOutput, TargetOutput = self.__BeforeLossFunction(CalculatedOutput, TargetOutput)
    return np.float64(self.__LossFunction(CalculatedOutput, TargetOutput))

  def CalculateDerivationLoss(self, CalculatedOutput, TargetOutput):
    CalculatedOutput, TargetOutput = self.__BeforeGradientLossFunction(CalculatedOutput, TargetOutput)
    return np.float64(self._GradientLossFunction(CalculatedOutput, TargetOutput))

  def SetUpdateWeightsFunction(self, Function: LambdaType):
    CheckParametersFunction(Function, 3)
    self.__UpdateWeightsFunction = Function

  def SetLossFunction(self, Function: LambdaType):
    CheckParametersFunction(Function, 2)
    self.__LossFunction = Function
    self._GradientLossFunction = DerivationLambda(Function, 0)
  
  def SetGradientLossFunction(self, Function):
    CheckParametersFunction(Function, 2)
    self._GradientLossFunction = Function
     
  def SetBeforeUpdateWeightsFunction(self, Function: LambdaType):
    CheckParametersFunction(Function, 2)
    self.__BeforeUpdateWeightsFunction = Function

  def SetBeforeLossFunction(self, Function: LambdaType):
    CheckParametersFunction(Function, 2)
    self.__BeforeLossFunction = Function
    
  def SetBeforeGradientLossFunction(self, Function: LambdaType):
    CheckParametersFunction(Function, 2)
    self.__BeforeGradientLossFunction = Function
    
  def TurnOn(self):
    self.__ActivedState = True

  def TurnOff(self):
    self.__ActivedState = False

  def GetStateActived(self):
    return self.__ActivedState

  def GetInputVector(self):
    return list(self._InputVector)
  
  def GetBiasValue(self):
    return self._Weights[0]
  
  def GetWeightsVector(self):
    return list(self._Weights)

  def ResetInputVector(self):
    Bias = self._InputVector[0]
    self._InputVector[:] = None
    self._InputVector[0] = Bias
    
  def LoadWeights(self, WeightsVector):
    if(len(WeightsVector) != len(self._Weights)): return False
    for i, Weight in enumerate(WeightsVector): self._Weights[i] = Weight
    return True
  
  def LoadInputFromBridge(self, b, x):
    
    if(b not in self._InputBridges): return False
    self._InputVector[b.GetIndexNeuron()] = x
    return True
  
  def AddEnterBridge(self, b):
    
    self._InputBridges.append(b)
    self._InputVector = np.append(self._InputVector, None)
    self._Weights = np.append(self._Weights, random.randint(-10, 10)/100)
    
  def RemoveEnterBridge(self, b):
    
    self._InputBridges.remove(b)
    self._InputVector = np.remove(self._InputVector, b.GetIndexNeuron())
    self._Weights = np.remove(self._Weights, b.GetIndexNeuron())

  def AddConnectionTo(self, FinalNeuron):
    
    if(self.GetExitBridgeToNeuron(FinalNeuron) != None): return False   #Se l' arco esiste già
    
    Arc = Bridge.Bridge(self, FinalNeuron, len(FinalNeuron.GetSetEnterBridge()))
    
    self._OutputBridges.append(Arc)
    FinalNeuron.AddEnterBridge(Arc)
    
    return True

  def RemoveConnectionTo(self, FinalNeuron): 
    
    Arc = self.GetExitBridgeToNeuron(FinalNeuron)  
    
    if(Arc == None): return False   #Se l' arco non esiste ma dovrebbe esistere 
    
    self._OutputBridges.remove(Arc) 
    FinalNeuron.RemoveEnterBridge(Arc)
    
    return True

  def GetEnterBridgeFromNeuron(self, StartNeuron):
    try: return list(filter(lambda x: x.GetStartNeuron() == StartNeuron, self._InputBridges))[0]
    except: return None

  def GetExitBridgeToNeuron(self, FinalNeuron):
    try: return list(filter(lambda x: x.GetFinishNeuron() == FinalNeuron, self._OutputBridges))[0]
    except: return None

  def GetSetEnterBridge(self):
    return list(self._InputBridges)

  def GetSetExitBridge(self):
    return list(self._OutputBridges)

  def Calculate(self):
    raise NotImplemented

  def __eq__(self, obj):
    return self is obj

  def __str__(self):
    return f"Neuron ({self._Weights[0]})"

  def __hash__(self):
    return id(self)

#Responsabilità, rappresentare un neurone tenendo conto degl' archi di uscita, di entrata e della bias, in più calcolare la propria funzione di attivazione
class ActivationNeuron(Neuron):

  def __init__(self, ActivationFunction: LambdaType, UpdateWeightsFunction: LambdaType, LossFunction: LambdaType, *args):
    super().__init__(UpdateWeightsFunction, LossFunction, *args)

    CheckParametersFunction(ActivationFunction, 1)
    self._ActivationFunction = ActivationFunction
    
    try: self._ActivationDerative = DerivationLambda(self._ActivationFunction, 0)
    except: self._ActivationDerative = lambda x: 1
    
  def SetActivationDerivative(self, Function):
    CheckParametersFunction(Function, 1)
    self._ActivationDerivative = Function
    
  def SetActivationDerative(self, DerivationFunction):
    CheckParametersFunction(DerivationFunction, 1)
    self._GradientLossFunction = DerivationFunction
    
  def SetActivationFunction(self, Function):
    CheckParametersFunction(Function, 1)
    self._ActivationFunction = Function
    
  def CalculateActionDerivative(self):
    return self._ActivationDerivative(self._Net)
  
  def CalculateDerivationLoss(self):
    return self._GradientLossFunction(self._Net)
    
  def Calculate(self):
    if(not self.GetStateActived()):  return 0
    
    self._Net = np.dot(self._InputVector, self._Weights)
    return self._ActivationFunction(self._Net)

class InputNeuron(ActivationNeuron):
  def __init__(self, ActivationFunction: LambdaType, UpdateWeightsFunction: LambdaType, LossFunction: LambdaType):
    super().__init__(ActivationFunction, UpdateWeightsFunction, LossFunction, 0) 
    
class OutputNeuron(ActivationNeuron):
  def __init__(self,ActivationFunction: LambdaType, UpdateWeightsFunction: LambdaType, LossFunction: LambdaType):
    super().__init__(ActivationFunction, UpdateWeightsFunction, LossFunction, 0)
    
class Perceptron(ActivationNeuron):
  def __init__(self, threshold, UpdateWeightsFunction: LambdaType, LossFunction: LambdaType, *args):
    self.Threshold = threshold
    super().__init__(lambda x: x if x >= self.Threshold else 0, UpdateWeightsFunction, LossFunction, *args)