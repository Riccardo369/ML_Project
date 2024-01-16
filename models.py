from Bridge import Bridge
from Layer import Layer
from NeuralNetwork import MLP
from Neuron import ActivationNeuron, InputNeuron, Neuron, OutputNeuron, Perceptron


def BuildTwoLevelFeedForward(InputSize, HiddenSize, OutputSize, LossFunction, WeightsUpdateFunction,Threshold=0):
  mlp = MLP(0,0,LossFunction)
  InputLayer = mlp.GetInputLayer()
  for i in range(InputSize):
    InputLayer.InsertNeuron(Perceptron(Threshold,WeightsUpdateFunction,lambda x,y: 0, LossFunction), i)
  for i in InputLayer: i.AddBridge(Bridge(None, i, 1))

  OutputLayer = mlp.GetOutputLayer()  
  for i in range(OutputSize):
    OutputLayer.InsertNeuron(Perceptron(Threshold, WeightsUpdateFunction, lambda x,y: 0, LossFunction), i)
    
  HiddenLayer = Layer(Perceptron, HiddenSize, Threshold, WeightsUpdateFunction, lambda x, y: 0, LossFunction)
  
  #Set all derivation loss function
  for i in list(InputLayer) + list(OutputLayer) + list(HiddenLayer):
    i.SetDerivationLoss(lambda x: 1 if x >= Threshold else 0)
  
  InputLayer.ConnectTo(HiddenLayer)
  HiddenLayer.ConnectTo(OutputLayer) 
      
  mlp.CalculateAllStructure() 
  
  return mlp
def BuildTwoLevelFeedForwardMonk(InputSize, HiddenSize, OutputSize, LossFunction, WeightsUpdateFunction,Threshold=0,classification_threshold=0):
  mlp = MLP(0,0,LossFunction)
  InputLayer = mlp.GetInputLayer()
  for i in range(InputSize):
    neuron=InputNeuron(lambda x: x if x>Threshold else 0,WeightsUpdateFunction,lambda x,y:0,LossFunction)
    neuron.SetDerivationLoss(lambda x:1 if x> Threshold else 0)
    InputLayer.InsertNeuron(neuron,i)

  for i in InputLayer: i.AddBridge(Bridge(None, i, 1))

  OutputLayer = mlp.GetOutputLayer()  
  for i in range(OutputSize):
    neuron=OutputNeuron(lambda x:1 if x >  classification_threshold else 0,WeightsUpdateFunction,lambda x,y:0,LossFunction)
    neuron.SetDerivationLoss(lambda x:1 if x> Threshold else 0)
    OutputLayer.InsertNeuron(neuron,i)
    
  HiddenLayer = Layer(Perceptron, HiddenSize, Threshold, WeightsUpdateFunction, lambda x, y: 0, LossFunction)
  for neuron in HiddenLayer.GetNeurons():
    neuron.SetDerivationLoss(lambda x:1 if x> Threshold else 0)
  InputLayer.ConnectTo(HiddenLayer)
  HiddenLayer.ConnectTo(OutputLayer) 
      
  mlp.CalculateAllStructure() 
  
  return mlp