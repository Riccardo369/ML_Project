import random

from Neuron import *

class Layer:
  def __init__(self, Type: type, Number, *args):
    if(Type.__init__.__code__.co_argcount-1 > len(args) and Number!= 0):

      ParametersNameList = list(Type.__init__.__code__.co_varnames)[len(args)+1:]
      ParametersNameList[-1] = "*args"

      raise ValueError(f"Missing ({', '.join(ParametersNameList)}) arguments for create '{Type.__name__}' neurons")

    self.__Neurons = [Type(*args) for i in range(Number)]

  def GetNeurons(self):
    return list(self.__Neurons)
  
  def InsertNeuron(self, Neuron, i):
    self.__Neurons.insert(i, Neuron)
    
  def RemoveNeuron(self, i):
    del self.__Neurons[i]

  def ConnectTo(self, layer):
    for i in self.__Neurons:
      for r in layer.GetNeurons():
        i.AddConnectionTo(r)

  def ApplyDropOut(self, Percentual):
    Neurons = random.sample(list(self.__Neurons), len(self.__Neurons))   #Getting shuffled neuron list
    for i in range(int(len(Neurons)*Percentual)): Neurons[i].TurnOff()   #Turn off neurons

  def TurnOnAllNeurons(self):
    for i in self.__Neurons: i.TurnOn()
    
  def __contains__(self, x):
    return x in self.__Neurons

  def __getitem__(self, i):
    return self.__Neurons[i]

  def __len__(self):
    return len(self.__Neurons)

  def __iter__(self):
    self.__i = 0
    return self

  def __next__(self):
    if self.__i < len(self.__Neurons):
      Result = self.__Neurons[self.__i]
      self.__i += 1
      return Result
    else:
      del self.__i
      raise StopIteration

  def __list__(self):
    return list(self.__Neurons)

class PerceptronLayer(Layer):
  def __init__(self, Number, *args):
    super().__init__(Perceptron, Number, *args)


    
class InputLayer(Layer):
  def __init__(self, Number, *args):
    super().__init__(InputNeuron, Number, *args)
    
class OutputLayer(Layer):
  def __init__(self, Number, *args):
    super().__init__(OutputNeuron, Number, *args)