import random
import numpy as np

import Neuron

#Responsabilità, rappresentare un ponte, tenendo conto del neurone di partenza, di fine e del suo peso
class Bridge:

  def __init__(self, StartNeuron, FinishNeuron, IndexNeuron):
    self.__StartNeuron = StartNeuron
    self.__FinishNeuron = FinishNeuron
    self.__IndexNeuron = IndexNeuron
    
  def GetIndexNeuron(self):
    return self.__IndexNeuron
    
  def GetStartNeuron(self):
    return self.__StartNeuron

  def GetFinishNeuron(self):
    return self.__FinishNeuron

  def __eq__(self, obj):
    if(not isinstance(obj, Bridge)): return False
    return ((obj.GetStartNeuron() == self.__StartNeuron) and
            (obj.GetFinishNeuron() == self.__FinishNeuron) and
            (obj.GetIndexNeuron() == self.GetIndexNeuron()))

  def __str__(self):
    return f"({self.__StartNeuron} --- ({self.__FinishNeuron.Get}) --> {self.__FinishNeuron})"