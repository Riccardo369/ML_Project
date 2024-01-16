import json
import numpy as np
from types import LambdaType

from BasedTools import *
from Layer import *
from Bridge import *

class NeuralNetwork:
  
  def __init__(self, InputNumber: int, OutputNumber: int, LossLambdaFunctionEvaluation: LambdaType):

    CheckParametersFunction(LossLambdaFunctionEvaluation, 2)

    self.__InputNeuronVector = InputLayer(InputNumber)
    self.__OutputNeuronVector = OutputLayer(OutputNumber, lambda x, y: x)

    for i in self.__InputNeuronVector: i.AddBridge(Bridge(None, i, 1))     #For creating a starting bridge to inject input neurons
    
    self.__LambdaLossFunctionEvaluation = LossLambdaFunctionEvaluation
    self.__GradientLossFunction = DerivationLambda(LossLambdaFunctionEvaluation, 0)
    
    self.__BeforeLambdaLossFunctionEvaluation = lambda x, y: (x, y)
    self.__BeforeGradientLossFunction = lambda x, y: (x, y)
    
    self.__NeuronsLastOutput = dict()

    self.CalculateAllStructure()
    
  def GetInputLayer(self):
    return self.__InputNeuronVector
  
  def GetOutputLayer(self):
    return self.__OutputNeuronVector
  
  def GetLossLambdaFunctionEvaluation(self):
    return self.__LambdaLossFunctionEvaluation

  def TurnOnAllNeurons(self):
    for i in self.GetAllNeurons(): i.TurnOn()

  def GetAllInputNeurons(self):
    return self.__InputNeuronVector.GetNeurons()

  def GetAllHiddenNeurons(self):
    return list(self.__HiddeNeurons)

  def GetAllOutputNeurons(self):
    return list(self.__OutputNeuronVector.GetNeurons())

  def GetAllNeurons(self):
    return self.GetAllInputNeurons() + self.GetAllHiddenNeurons() + self.GetAllOutputNeurons()

  def GetAllBridges(self):
    return list(self.__AllBridges)

  def GetInputNeuron(self, i):
    return self.__InputNeuronVector[i]

  def GetOutputNeuron(self, i):
    return self.__InputNeuronVector[i]

  def CalculateAllStructure(self):

    CapturedNeurons = []
    CapturedBridges = []

    #Every input neuron is added inside CapturedNeurons
    for i in self.__InputNeuronVector: CapturedNeurons.append(i)

    i = 0

    #Until there is one more neuron to still explore
    while(i<len(CapturedNeurons)):
      Bridges = CapturedNeurons[i].GetSetExitBridge() #Calculate every bridge
      for r in Bridges:
        if(r.GetFinishNeuron() not in CapturedNeurons and not isinstance(r, InputNeuron) and not isinstance(r, OutputNeuron)): CapturedNeurons.append(r.GetFinishNeuron()) #Add neuron if not still exists in list
        if(r not in CapturedBridges): CapturedBridges.append(r) #Add bridge if not still exists in list
      i += 1

    self.__HiddeNeurons = CapturedNeurons
    self.__AllBridges = CapturedBridges

  def Clear(self):
    for i in self.GetAllNeurons(): i.ResetInputVector()
    for i in self.__AllBridges: i.ResetUsedCount()
    self.__NeuronsLastOutput.clear()

  def Predict(self, InputVector):

    if(len(InputVector) != len(self.__InputNeuronVector)): raise ValueError(f"Input vector must be {len(self.__InputNeuronVector)}, but is {len(InputVector)}")

    self.Clear() #Clear data of count bridge and vectore neuron input

    NeuronsToActive = []

    #Load input data and Load neuron to activate
    for i, NeuronInput in enumerate(self.__InputNeuronVector):

      BridgeToStart = NeuronInput.GetEnterBridgeFromNeuron(None)
      BridgeToStart.ResetUsedCount()
      BridgeToStart.IncrementUsedCount()
      
      NeuronInput.LoadInputFromBridge(BridgeToStart, InputVector[i])

      NeuronsToActive.append(NeuronInput)
    
    Result = [None for _ in self.__OutputNeuronVector]

    i = 0

    while(len(NeuronsToActive) > 0):

      ActualNeuron = NeuronsToActive[i]    #Choosed neuron to analyze

      try:
        if(min(map(lambda x: x.GetUsedCount(), ActualNeuron.GetSetEnterBridge())) == 0): #If all bridges have not been activated, go to the next neuron
          i = (i+1) % len(NeuronsToActive)
          continue
      except: pass

      Value = ActualNeuron.Calculate()
      
      #Calculate output, spread output to next neuron, add next neuron and delete this neuron from NeuronToActivate
      self.__NeuronsLastOutput[ActualNeuron] = Value

      #If ActualNeuron is OutputNeuron, save output value to final list according to his index in self.__OutputNeuronVector
      # it is an output neuron if it has no exit bridges
      if(ActualNeuron in self.__OutputNeuronVector.GetNeurons()): Result[self.__OutputNeuronVector.GetNeurons().index(ActualNeuron)] = Value

      #For each bridge from ActualNeuron
      for b in ActualNeuron.GetSetExitBridge():

        if(b.GetFinishNeuron() not in NeuronsToActive): NeuronsToActive.append(b.GetFinishNeuron())       #Next neuron is added to list
        
        b.GetFinishNeuron().LoadInputFromBridge(b, Value)                                                 #Value is spread to new neuron
        b.IncrementUsedCount()                                                                            #Updates Bridge's counter

      del NeuronsToActive[i]

    return np.array(Result)
  
  def Learn(self, LossValueVector):
    
    #Check length parameter
    if(len(LossValueVector) != len(self.__OutputNeuronVector)): raise ValueError(f"{len(self.__OutputNeuronVector)} Output neurons but {len(LossValueVector)} loss values insert")
    
    #Save which neurons must be still updated

    #Save all loss values of all neurons
    NeuronsLoss = dict()
    for i in self.GetAllNeurons(): NeuronsLoss[i] = None

    #Calculate Signal error Sk for output neuron (k index of output neuron)
    for k in enumerate(self.GetAllOutputNeurons()): 
      NeuronsLoss[k[1]] = LossValueVector[k[0]] * k[1].CalculateGradientActivationFunction()
    
    
    NeuronsToUpdate = self.GetAllHiddenNeurons()# + self.GetAllInputNeurons()
    #Index using for control list of neurons to still update
    i = 0
    #FIRST STEP: assign for each neuron (input and hidden) own signal error

    #For each neuron to still update
    while(len(NeuronsToUpdate) > 0):
      
      i %= len(NeuronsToUpdate)

      ActualNeuron = NeuronsToUpdate[i]   #Choose neuron to analyze
      
      #Bridges to consider
      Bridges = ActualNeuron.GetSetExitBridge()
      
      #If actual neuron can't get own signal error
      if(None in list(map(lambda w: NeuronsLoss[w.GetFinishNeuron()], Bridges))):
        i += 1 
        continue
      
      
      #Calculate Signal error Sj for hidden and input neuron (Actual neuron = neuron j, w is bridge to consider)
      SignalError = sum(list(map(lambda w: 0 if w.GetStartNeuron() == None else w.Weight * NeuronsLoss[w.GetFinishNeuron()], Bridges))) * ActualNeuron.CalculateGradientActivationFunction() 
      
      NeuronsLoss[ActualNeuron] = SignalError
      
      for w in Bridges: w.ResetUsedCount()
      # remove neuron from the list
      del NeuronsToUpdate[i]
    
    #Extract all neurons to update
    for Neuron in self.GetAllOutputNeurons()+self.GetAllHiddenNeurons():
      if any(map(lambda x:x.GetStartNeuron()==None, Neuron.GetSetEnterBridge())):
        continue
      #Update bias
      Neuron.BiasValue = Neuron.CalculateUpdateBias(NeuronsLoss[Neuron]) 

      EnterNeuronsOutput= map(lambda x:self.__NeuronsLastOutput[x],map(lambda x:x.GetStartNeuron(),Neuron.GetSetEnterBridge()))

      #Update weights
      WeightsNewValue = Neuron.CalculateUpdatedWeights(NeuronsLoss[Neuron],list(EnterNeuronsOutput))
      for i, Weight in enumerate(Neuron.GetSetEnterBridge()): Weight.Weight = WeightsNewValue[i]

  def SetLossFunctionEvaluation(self, Function):
    CheckParametersFunction(Function, 2)
    self.__LambdaLossFunctionEvaluation = Function
    self.__GradientLossFunction = DerivationLambda(Function, 0)
    
  def SetGradientLossFunction(self, Function):
    CheckParametersFunction(Function, 2)
    self.__GradientLossFunction = Function
    
  def SetBeforeLossFunctionEvaluation(self, Function):
    CheckParametersFunction(Function, 2)
    self.__BeforeLambdaLossFunctionEvaluation = Function
    
  def SetBeforeGradientLoss(self, Function):
    CheckParametersFunction(Function, 2)
    self.__BeforeGradientLossFunction = Function

  def LossFunctionEvaluation(self, OutputCalculatedVector: np.ndarray, OutputTargetVector: np.ndarray):

    #Check all requirements of 2 vectors
    if(len(OutputCalculatedVector[0]) != len(self.__OutputNeuronVector)): raise ValueError(f"Output calculated vector must be {len(self.__OutputNeuronVector)}, but is {len(OutputCalculatedVector)}")
    if(len(OutputTargetVector[0]) != len(self.__OutputNeuronVector)): raise ValueError(f"Output target vector must be {len(self.__OutputNeuronVector)}, but is {len(OutputTargetVector)}")
    if(len(OutputCalculatedVector) != len(OutputTargetVector)): raise ValueError(f"There are different values between calculated output {len(OutputCalculatedVector)} and {len(OutputTargetVector)}")
    
    #for x, y in zip(OutputCalculatedVector, OutputTargetVector): print(self.__LambdaLossFunctionEvaluation(*self.__BeforeLambdaLossFunctionEvaluation(x, y)))
    
    return np.longdouble(sum(map(lambda x, y: self.__LambdaLossFunctionEvaluation(*self.__BeforeLambdaLossFunctionEvaluation(x, y)), OutputCalculatedVector, OutputTargetVector)) / len(OutputCalculatedVector))

  def GradientDirectionLoss(self, OutputCalculated, OutputTarget):
    
    #import time
    
    #print(OutputCalculated, OutputTarget)
    OutputCalculated, OutputTarget = self.__BeforeGradientLossFunction(OutputCalculated, OutputTarget)
    #print(OutputCalculated, OutputTarget)
    Result = np.longdouble(self.__GradientLossFunction(OutputCalculated, OutputTarget))
    #print(Result)
    #print("")
    
    #time.sleep(0.5)
    
    return Result

  #NeuralNetwork State
  class NeuralNetworkState:
    def __init__(self, Weights: list[float], Biases: list[float]):
      self.__Weights = Weights
      self.__Biases = Biases

    def GetWeights(self):
      return list(self.__Weights)

    def GetBiases(self):
      return list(self.__Biases)

    def SaveToTxtForm(self):
      Result = {}
      Result["Weights"] = self.__Weights
      Result["Biases"] = self.__Biases

      return json.dumps(Result)

    def LoadFromTxtForm(self, String):
      Result = json.loads(String)

      self.__Weights = Result["Weights"]
      self.__Biases = Result["Biases"]

    def __eq__(self, obj):
      if(not isinstance(obj, self.__class__.__name__)): return False
      if(len(self.__Weights) != len(obj.__Weights) and len(self.__Biases) != len(obj.__Biases)): return False

      for i in range(len(self.__Weights)):
        if(len(self.__Weights[i]) != len(obj.__Weights[i])): return False

      for i in range(len(self.__Biases)):
        if(len(self.__Biases[i]) != len(obj.__Biases[i])): return False

      return True

  def ExtractLearningState(self):
    Weights = list(map(lambda i: i.Weight, self.__AllBridges))
    Biases = list(map(lambda i: i.BiasValue, self.GetAllNeurons()))

    return NeuralNetwork.NeuralNetworkState(Weights, Biases)

  def LoadLearningState(self, State: NeuralNetworkState):

    Weights = State.GetWeights()
    Biases = State.GetBiases()

    for i in range(len(self.__AllBridges)): self.__AllBridges[i].Weight = Weights[i]
    for i in enumerate(self.GetAllNeurons()): i[1].BiasValue = Biases[i[0]]
    
class MLP(NeuralNetwork):
  def __init__(self, InputNumber: int, OutputNumber: int, LossLambdaFunctionEvaluation: LambdaType):
    super().__init__(InputNumber, OutputNumber, LossLambdaFunctionEvaluation)
