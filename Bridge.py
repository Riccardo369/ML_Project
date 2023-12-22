import random

from Neuron import *

#Responsabilità, rappresentare un ponte, tenendo conto del neurone di partenza, di fine e del suo peso
class Bridge:

  def __init__(self, StartNeuron, FinishNeuron, *args):
    self.__UsedCount = 0
    self.__StartNeuron = StartNeuron
    self.__FinishNeuron = FinishNeuron

    if(len(args) > 0): self.Weight = args[0]                           #Se il valore del peso è già scelto
    else: self.Weight = random.randint(-10000, 10000) / 1000           #Valore scelto per convenienza da -1 ad 1 ma si può cambiare il range volendo

  def GetStartNeuron(self):
    return self.__StartNeuron

  def GetFinishNeuron(self):
    return self.__FinishNeuron

  def GetUsedCount(self):
    return self.__UsedCount

  def ResetUsedCount(self):
    self.__UsedCount = 0

  def IncrementUsedCount(self):
    self.__UsedCount += 1

  def __eq__(self, obj):
    if(not isinstance(obj, Bridge)): return False
    return ((obj.GetStartNeuron() == self.__StartNeuron) and (obj.GetFinishNeuron() == self.__FinishNeuron))

  def __str__(self):
    return f"({self.__StartNeuron} --- ({self.Weight}) --> {self.__FinishNeuron})"