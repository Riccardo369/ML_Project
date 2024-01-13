from Bridge import Bridge
from Layer import Layer
from NeuralNetwork import MLP
from Neuron import Perceptron


def BuildTwoLevelFeedForward(InputSize, HiddenSize, OutputSize, LossFunction, WeightsUpdateFunction,Threshold=0):
  mlp = MLP(0,0,LossFunction)
  InputLayer = mlp.GetInputLayer()
  for i in range(InputSize):
    InputLayer.InsertNeuron(Perceptron(Threshold,WeightsUpdateFunction,lambda x,y:0,LossFunction),i)
  for i in InputLayer: i.AddBridge(Bridge(None, i, 1))

  OutputLayer = mlp.GetOutputLayer()  
  for i in range(OutputSize):
    OutputLayer.InsertNeuron(Perceptron(Threshold,WeightsUpdateFunction,lambda x,y:0,LossFunction),i)
    
  HiddenLayer = Layer(Perceptron, HiddenSize, Threshold, WeightsUpdateFunction, lambda x, y: 0, LossFunction)
  
  InputLayer.ConnectTo(HiddenLayer)
  HiddenLayer.ConnectTo(OutputLayer) 
      
  mlp.CalculateAllStructure() 
  
  return mlp