import random
import numpy as np
from types import LambdaType

class CrossValidation:
  def __init__(self, Dataset, K):
    Data = random.sample(list(Dataset), len(Dataset))          #Shuffle data
    self.__Folds = np.split(np.array(Data,dtype=object),K)           #Split data in k folds
    del Data                                                   #It is not necessary in memory anymore

  def GetFoldsNum(self):
    return len(self.__Folds)

  def GetKFold(self, K):
    ListFolds = list(self.__Folds)
    ValidationSet = np.array(ListFolds[K])
    del ListFolds[K]
    TrainingSet = np.concatenate(ListFolds)
    return np.array(TrainingSet, dtype=object), np.array(ValidationSet, dtype=object)
  
  
  
#responsabilità: selezionare un modello, la classe è astratta e da essa poi dovranno ereditare altre classi che implemetano una strategia di validazione
class ModelSelection:
  def __init__(self,Model,TrSet,Retraining=False):
    self.__Model=Model
    self.__TrSet=TrSet
    self.__Retraining=Retraining
    # data una serie di iperparametri e un regola per selezionare i valori dalla epoch restituisce il risultato della selezione del modello
  def Select(self,EpochRule,Hyperparameters):
    raise NotImplemented

