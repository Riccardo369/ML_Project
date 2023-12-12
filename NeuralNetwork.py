class NeuralNetwork:
  def __init__(self, InputNumber: int, OutputNumber: int, LossLambdaFunctionEvaluation: LambdaType):

    CheckParametersFunction(LossLambdaFunctionEvaluation, 2)

    self.__InputNeuronVector = InputLayer(InputNumber)
    self.__OutputNeuronVector = OutputLayer(OutputNumber, lambda x, y: x)

    for i in self.__InputNeuronVector: i.AddBridge(Bridge(None, i, 1))                       #For creating a starting bridge to inject input neurons
    for i in self.__OutputNeuronVector: i.SetLossFunction(LossLambdaFunctionEvaluation)      #For setting loss function of output neurons

    self.__LambdaLossFunctionEvaluation = LossLambdaFunctionEvaluation
    self.__GradientLossFunction = DerivationLambda(LossLambdaFunctionEvaluation, 0)

    self.__NeuronsLastOutput = dict()

    self.CalculateAllStructure()

  def GetLossLambdaFunctionEvaluation(self):
    return self.__LambdaLossFunctionEvaluation

  def SetAllUpdateWeightsFunctionInputNeurons(self, Function: LambdaType):
    CheckParametersFunction(Function, 2)
    for i in self.GetInputLayer(): i.SetUpdateWeightsFunction(Function)

  def SetAllBiasUpdateFunctionInputNeurons(self, Function: LambdaType):
    CheckParametersFunction(Function, 2)
    for i in self.GetInputLayer(): i.SetBiasUpdateFunction(Function)

  def SetAllLossFunctionInputNeurons(self, Function: LambdaType):
    CheckParametersFunction(Function, 2)
    for i in self.GetInputLayer(): i.SetLossFunction(Function)

  def SetAllUpdateWeightsFunctionHiddenNeurons(self, Function: LambdaType):
    CheckParametersFunction(Function, 2)
    for i in self.GetAllHiddenNeurons(): i.SetUpdateWeightsFunction(Function)

  def SetAllBiasUpdateFunctionHiddenNeurons(self, Function: LambdaType):
    CheckParametersFunction(Function, 2)
    for i in self.GetAllHiddenNeurons(): i.SetBiasUpdateFunction(Function)

  def SetAllLossFunctionHiddenNeurons(self, Function: LambdaType):
    CheckParametersFunction(Function, 2)
    for i in self.GetAllHiddenNeurons(): i.SetLossFunction(Function)

  def SetAllUpdateWeightsFunctionOutputNeurons(self, Function: LambdaType):
    CheckParametersFunction(Function, 2)
    for i in self.GetOutputLayer(): i.SetUpdateWeightsFunction(Function)

  def SetAllBiasUpdateFunctionOutputNeurons(self, Function: LambdaType):
    CheckParametersFunction(Function, 2)
    for i in self.GetOutputLayer(): i.SetBiasUpdateFunction(Function)

  def SetAllLossFunctionOutputNeurons(self, Function: LambdaType):
    CheckParametersFunction(Function, 2)
    for i in self.GetOutputLayer(): i.SetLossFunction(Function)

  def TurnOnAllNeurons(self):
    for i in self.__AllNeurons: i.TurnOn()

  def GetAllInputNeurons(self):
    return self.__InputNeuronVector.GetNeurons()

  def GetAllHiddenNeurons(self):
    return list(filter(lambda i: i not in self.__InputNeuronVector and i not in self.__OutputNeuronVector, self.__AllNeurons))

  def GetAllOutputNeurons(self):
    return self.__OutputNeuronVector.GetNeurons()

  def GetAllNeurons(self):
    return list(self.__AllNeurons)

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
        if(r.GetFinishNeuron() not in CapturedNeurons): CapturedNeurons.append(r.GetFinishNeuron()) #Add neuron if not still exists in list
        if(r not in CapturedBridges): CapturedBridges.append(r) #Add bridge if not still exists in list
      i += 1

    self.__AllNeurons = CapturedNeurons
    self.__AllBridges = CapturedBridges

  def Clear(self):
    for i in self.__AllNeurons: i.ResetInputVector()
    for i in self.__AllBridges: i.ResetUsedCount()
    self.__NeuronsLastOutput.clear()

  def Predict(self, InputVector):

    if(len(InputVector) != len(self.__InputNeuronVector)): raise ValueError(f"Input vector must be {len(self.__InputNeuronVector)}, but is {len(InputVector)}")

    self.Clear() #Clear data of count bridge and vectore neuron input

    NeuronsToActive = []

    #Load input data and Load neuron to active
    for i in range(len(self.__InputNeuronVector)):

      BridgeToStart = self.__InputNeuronVector[i].GetEnterBridgeFromNeuron(None)
      BridgeToStart.ResetUsedCount()
      BridgeToStart.IncrementUsedCount()

      self.__InputNeuronVector[i].LoadInputFromBridge(BridgeToStart, InputVector[i])

      NeuronsToActive.append(self.__InputNeuronVector[i])

    Result = [None for i in self.__OutputNeuronVector]

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
      if(isinstance(ActualNeuron, OutputNeuron)): Result[self.__OutputNeuronVector.GetNeurons().index(ActualNeuron)] = Value

      #For each bridge from ActualNeuron
      for b in ActualNeuron.GetSetExitBridge():

        if(b.GetFinishNeuron() not in NeuronsToActive): NeuronsToActive.append(b.GetFinishNeuron())       #Next neuron is added to list

        b.GetFinishNeuron().LoadInputFromBridge(b, Value)                                                 #Value is spread to new neuron
        b.IncrementUsedCount()                                                                            #Updates Bridge's counter

      del NeuronsToActive[i]

    return np.array(Result)

  def Learn1(self,LossValue:np.array,Hyperparameters):
    if(len(LossValue)!=len(self.__OutputNeuronVector)): raise ValueError("the loss values must be as long as the number of output neurons")

    NeuronsToUpdate = list(self.__OutputNeuronVector)  #Save which neurons must be still updated

    #Save all loss values of all neurons
    NeuronsLoss = dict()
    for neuron in self.GetAllNeurons(): NeuronsLoss[neuron] = 0

    #Update loss associated to output neuron
    for i,neuron in enumerate(self.__OutputNeuronVector):
      NeuronsLoss[neuron] = LossValue[i] * neuron.CalculateGradientActivationFunction()

    i = 0

    while(len(NeuronsToUpdate) > 0):

      ActualNeuron = NeuronsToUpdate[i]    #Choosed neuron to analyze

      try:
        if(sum(map(lambda r: r.Weight, ActualNeuron.GetSetEnterBridge())) != 0):   #If actual neuron hasn't received all updates of the loss, pass to the next one
          i = (i+1) % len(NeuronsToUpdate)
          continue
      except: pass

      #Calculates bridges
      ActualBridges = ActualNeuron.GetSetEnterBridge()

      #Calculate GradientDirection
      #GradientDirection = - ActualNeuron.CalculateDerivationLoss(self.__NeuronsLastOutput[ActualNeuron], NeuronsLoss[ActualNeuron])
      #Spread GradientDirection from current neuron to other neurons throudh hidden layer
      for r in ActualBridges:

        if(r.GetStartNeuron() == None): continue

        NeuronsLoss[r.GetStartNeuron()] += NeuronsLoss[r.GetFinishNeuron()] * r.Weight
        r.ResetUsedCount()

        if(r.GetStartNeuron() not in NeuronsToUpdate):
          NeuronsToUpdate.append(r.GetStartNeuron())

        NeuronsLoss[r.GetStartNeuron()] *= -r.GetStartNeuron().CalculateGradientActivationFunction()

      #Calculate new weights
      NewWeights = ActualNeuron.CalculateUpdatedWeights(NeuronsLoss[ActualNeuron],ActualNeuron.Calculate(),*Hyperparameters)

      #Update weights
      for r in range(len(ActualBridges)):
        if(ActualBridges[r] not in ActualBridges): continue
        ActualBridges[r].Weight = NewWeights[r]

      #Update bias
      ActualNeuron.BiasValue = ActualNeuron.CalculateUpdateBias(NeuronsLoss[ActualNeuron])

      #Reset Neuron loss
      NeuronsLoss[ActualNeuron] = 0

      del NeuronsToUpdate[i]
#////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  def Learn(self, LossValue):

    NeuronsToUpdate = [i for i in self.__OutputNeuronVector]  #Save which neurons must be still updated

    #Save all loss values of all neurons
    NeuronsLoss = dict()
    for i in self.GetAllNeurons(): NeuronsLoss[i] = 0

    #Update loss associated to output neuron
    # nota : sarebbe il signal error nel pdf della backprop
    for i in self.__OutputNeuronVector: NeuronsLoss[i] = LossValue * i.CalculateGradientActivationFunction()

    i = 0

    while(len(NeuronsToUpdate) > 0):

      ActualNeuron = NeuronsToUpdate[i]    #Choosed neuron to analyze

      try:
        if(sum(map(lambda r: r.Weight, ActualNeuron.GetSetEnterBridge())) != 0):   #If actual neuron hasn't received all updates of the loss, pass to the next one
          i = (i+1) % len(NeuronsToUpdate)
          continue
      except: pass

      #Calculates bridges
      ActualBridges = ActualNeuron.GetSetEnterBridge()

      #Calculate GradientDirection
      GradientDirection = - ActualNeuron.CalculateDerivationLoss(self.__NeuronsLastOutput[ActualNeuron], NeuronsLoss[ActualNeuron])
      #Spread GradientDirection from current neuron
      for r in ActualBridges:

        if(r.GetStartNeuron() == None): continue

        NeuronsLoss[r.GetStartNeuron()] += GradientDirection * r.Weight
        r.ResetUsedCount()

        if(r.GetStartNeuron() not in NeuronsToUpdate):
          NeuronsToUpdate.append(r.GetStartNeuron())

        NeuronsLoss[r.GetStartNeuron()] *= r.GetStartNeuron().CalculateGradientActivationFunction()

      #Calculate new weights
      NewWeights = ActualNeuron.CalculateUpdatedWeights(NeuronsLoss[ActualNeuron])

      #Update weights
      for r in range(len(ActualBridges)):
        if(ActualBridges[r] not in ActualBridges): continue
        ActualBridges[r].Weight = NewWeights[r]

      #Update bias
      ActualNeuron.BiasValue = ActualNeuron.CalculateUpdateBias(NeuronsLoss[ActualNeuron])

      #Reset Neuron loss
      NeuronsLoss[ActualNeuron] = 0

      del NeuronsToUpdate[i]


#////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  def LossFunctionEvaluation(self, OutputCalculatedVector: np.ndarray, OutputTargetVector: np.ndarray):

    #Check all requirements of 2 list of vector
    for i in OutputCalculatedVector:
      if((len(i) != len(self.__OutputNeuronVector))): raise ValueError(f"Output calculated vector must be {len(self.__OutputNeuronVector)}, but is {len(i)}")

    for i in OutputTargetVector:
      if(len(i) != len(self.__OutputNeuronVector)): raise ValueError(f"Output target vector must be {len(self.__OutputNeuronVector)}, but is {len(i)}")

    if(len(OutputCalculatedVector) != len(OutputTargetVector)): raise ValueError(f"There are different values between calculated output {len(OutputCalculatedVector)} and {len(OutputTargetVector)}")

    return sum(map(lambda x, y: self.__LambdaLossFunctionEvaluation(x, y), OutputCalculatedVector, OutputTargetVector)) / len(OutputCalculatedVector)


  def GradientDirectionLoss(self, OutputCalculated, OutputTarget):
    return self.__GradientLossFunction(OutputCalculated, OutputTarget)


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
      if(not isinstance(obj, NeuralNetworkState)): return False
      if(len(self.__Weights) != len(obj.__Weights) and len(self.__Biases) != len(obj.__Biases)): return False

      for i in range(len(self.__Weights)):
        if(len(self.__Weights[i]) != len(obj.__Weights[i])): return False

      for i in range(len(self.__Biases)):
        if(len(self.__Biases[i]) != len(obj.__Biases[i])): return False

      return True

  def ExtractLearningState(self):
    Weights = list(map(lambda i: i.Weight, self.__Bridges))
    Biases = list(map(lambda i: i.BiasValue, self.__Neurons))

    return NeuralNetwork.NeuralNetworkState(Weights, Biases)

  def LoadLearningState(self, State: NeuralNetworkState):

    Weights = State.GetWeights()
    Biases = State.GetBiases()

    for i in range(len(self.__AllBridges)): self.__AllBridges[i].Weight = Weights[i]
    for i in range(len(self.__AllNeurons)): self.__AllNeurons[i].BiasValue = Biases[i]
    
class MLP(NeuralNetwork):
  def __init__(self, InputNumber: int, OutputNumber: int, LossLambdaFunctionEvaluation: LambdaType):
    super().__init__(InputNumber, OutputNumber, LossLambdaFunctionEvaluation)
    
    
    

