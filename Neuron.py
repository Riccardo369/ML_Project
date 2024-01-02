from types import LambdaType
import random 
import math
from BasedTools import *
import Bridge
#Responsabilità, rappresentare un neurone tenendo conto degl' archi di uscita, di entrata e della bias, in più calcolare la propria funzione di attivazione
class Neuron:

  def __init__(self, UpdateWeightsFunction: LambdaType, BiasUpdateFunction: LambdaType, LossFunction: LambdaType, *args):

    CheckParametersFunction(UpdateWeightsFunction, 2)
    CheckParametersFunction(BiasUpdateFunction, 2)
    CheckParametersFunction(LossFunction, 2)

    self._Bridges = []  #Lista che contiene i ponti in uscita (per la feedforward) ed  i ponti in entrata (per la backpropagation)
    self._InputVector = [] #Lista degl' input che il neurone può avere

    self.__ActivedState = True

    if(len(args) > 0): self.BiasValue = np.longdouble(args[0])                        #Se la nostra bias è già scelta
    else: self.BiasValue = np.longdouble(random.randint(-10000, 10000) / 1000)        #Valore scelto per convenienza da -10 ad 10 ma si può cambiare il range volendo

    self.__UpdateWeightsFunction = UpdateWeightsFunction
    self.__BeforeUpdateWeightsFunction = lambda x: x

    self.__UpdateBiasFunction = BiasUpdateFunction
    self.__BeforeUpdateBiasFunction = lambda x: x

    self.__LossFunction = LossFunction
    self.__BeforeLossFunction = lambda x, y: (x, y)

    self.__GradientLossFunction = DerivationLambda(self.__LossFunction, 0)
    self.__BeforeGradientLossFunction = lambda x, y: (x, y)


  def CalculateUpdatedWeights(self, LossGradientValue):
    LossGradientValue = self.__BeforeUpdateWeightsFunction(LossGradientValue)
    return np.longdouble(self.__UpdateWeightsFunction(list(map(lambda r: r.Weight, self.GetSetEnterBridge())), LossGradientValue))

  def CalculateUpdateBias(self, LossGradientValue):
    LossGradientValue = self.__BeforeUpdateBiasFunction(LossGradientValue)
    return np.longdouble(self.__UpdateBiasFunction(self.BiasValue, LossGradientValue))

  def CalculateLoss(self, CalculatedOutput, TargetOutput):
    CalculatedOutput, TargetOutput = self.__BeforeLossFunction(CalculatedOutput, TargetOutput)
    return np.longdouble(self.__LossFunction(CalculatedOutput, TargetOutput))

  def CalculateDerivationLoss(self, CalculatedOutput, TargetOutput):
    CalculatedOutput, TargetOutput = self.__BeforeGradientLossFunction(CalculatedOutput, TargetOutput)
    return np.longdouble(self.__GradientLossFunction(CalculatedOutput, TargetOutput))

  def SetUpdateWeightsFunction(self, Function: LambdaType):
    CheckParametersFunction(Function, 2)
    self.__UpdateWeightsFunction = Function

  def SetUpdateBiasFunction(self, Function: LambdaType):
    CheckParametersFunction(Function, 2)
    self.__UpdateBiasFunction = Function

  def SetLossFunction(self, Function: LambdaType):
    CheckParametersFunction(Function, 2)
    self.__LossFunction = Function
    self.__GradientLossFunction = DerivationLambda(Function, 0)
  
  def SetGradientLossFunction(self, Function):
    CheckParametersFunction(Function, 2)
    self.__GradientLossFunction = Function
     
  def SetBeforeUpdateWeightsFunction(self, Function: LambdaType):
    CheckParametersFunction(Function, 1)
    self.__BeforeUpdateWeightsFunction = Function

  def SetBeforeUpdateBiasFunction(self, Function: LambdaType):
    CheckParametersFunction(Function, 1)
    self.__BeforeUpdateBiasFunction = Function

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

  def ResetInputVector(self):
    self._InputVector = [None for i in range(len(self._InputVector))]

  def LoadInputFromBridge(self, b, x):
    assert np.isfinite(b.Weight)
    EnterBridgeList = self.GetSetEnterBridge()
    SelectedBridge = None

    for i in range(len(EnterBridgeList)):
      if(EnterBridgeList[i] == b):
        SelectedBridge = EnterBridgeList[i]
        break
    if(SelectedBridge == None): return False
    self._InputVector[i] = SelectedBridge.Weight * x
    return True

  def AddBridge(self, b):
    self._Bridges.append(b)
    self._InputVector.append(None)

  def RemoveBridge(self, b):
    i = self.GetSetEnterBridge().index(b)
    self._Bridges.remove(b)
    self._InputVector.pop(i)

  def AddConnectionTo(self, FinalNeuron):
    if(self.GetExitBridgeToNeuron(FinalNeuron) != None): return False   #Se l' arco esiste già
    CreatedBridge = Bridge.Bridge(self, FinalNeuron)
    self._Bridges.append(CreatedBridge)
    FinalNeuron.AddBridge(CreatedBridge)
    return True

  def RemoveConnectionTo(self, FinalNeuron):
    if(self.GetExitBridgeToNeuron(FinalNeuron) == None): return False   #Se l' arco non esiste ma dovrebbe esistere
    CreatedBridge = Bridge.Bridge(self, FinalNeuron)
    self._Bridges.remove(CreatedBridge)
    FinalNeuron.RemoveBridge(CreatedBridge)
    return True

  def GetEnterBridgeFromNeuron(self, StartNeuron):
    try: return self._Bridges[self._Bridges.index(Bridge.Bridge(StartNeuron, self))]
    except: return None

  def GetExitBridgeToNeuron(self, FinalNeuron):
    try: return self._Bridges[self._Bridges.index(Bridge.Bridge(self, FinalNeuron))]
    except: return None

  def GetSetEnterBridge(self):
    return list(filter(lambda i: self == i.GetFinishNeuron(), self._Bridges))

  def GetSetExitBridge(self):
    return list(filter(lambda i: self == i.GetStartNeuron(), self._Bridges))

  def Calculate(self):
    raise NotImplemented

  def __eq__(self, obj):
    return self is obj

  def __str__(self):
    return f"Neuron ({self.BiasValue})"

  def __hash__(self):
    return id(self)

#Responsabilità, rappresentare un neurone tenendo conto degl' archi di uscita, di entrata e della bias, in più calcolare la propria funzione di attivazione
class ActivationNeuron(Neuron):

  def __init__(self, ActivationFunction: LambdaType, UpdateWeightsFunction: LambdaType, BiasUpdateFunction: LambdaType, LossFunction: LambdaType, *args):
    super().__init__(UpdateWeightsFunction, BiasUpdateFunction, LossFunction, *args)

    CheckParametersFunction(ActivationFunction, 1)
    self.__ActivationFunction = ActivationFunction

    try:
      self.__GradientActivationFunction = DerivationLambda(ActivationFunction, 0)
      self.__GradientActivationFunction(0)
    except:
      self.__GradientActivationFunction = lambda x: 1
  def SetActivationFunction(self,Function):
    CheckParametersFunction(Function, 1)
    self.__ActivationFunction = Function
  def CalculateGradientActivationFunction(self):
    return self.__GradientActivationFunction(sum(self._InputVector) + self.BiasValue)
  def Calculate(self):
    if(not self.GetStateActived()): return 0

    Net = sum(self._InputVector) + self.BiasValue
    return self.__ActivationFunction(Net)

class InputNeuron(ActivationNeuron):
  def __init__(self):
    super().__init__(lambda x: x, lambda x, y: x, lambda x, y: 0, lambda x, y: (1/2)*((x - y)**2), 0) 
    
class OutputNeuron(ActivationNeuron):
  def __init__(self, UpdateWeightsFunction: LambdaType):
    super().__init__(lambda x: x, UpdateWeightsFunction, lambda x, y: 0, lambda x, y: (1/2)*((x - y)**2), 0)
    
class Perceptron(ActivationNeuron):
  def __init__(self, threshold, UpdateWeightsFunction: LambdaType, BiasUpdateFunction: LambdaType, LossFunction: LambdaType, *args):
    self.__Threshold = threshold
    super().__init__(lambda x: x if x>self.__Threshold else 0, UpdateWeightsFunction, BiasUpdateFunction, LossFunction, *args)

  def GetThreshold(self):
    return self.__Threshold

  def SetThreshold(self, x):
    self.__Threshold = x