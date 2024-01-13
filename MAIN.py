import random
import json
import numpy as np
from types import LambdaType
import matplotlib.pyplot as plt
import sympy as sp
import itertools as it

from BasedTools import *
from Bridge import *
from Dataset import *
from Evaluation import *
from Layer import *
from NeuralNetwork import *
from Neuron import *
from Phase import *
from Regularization import *
from Result import *

from models import BuildTwoLevelFeedForward
from model_selection import ModelSelection, make_grid

#Get all metrics for all K-folds considering: "LearningRate", "WeightDecay", "FoldsNumber"
#Print a graph of KFold


#instance desired model
#load dataset
#Data=TakeDataset('FilesData/ML-CUP23-TR.csv')[:200]

import Dataset
import numpy as np
monk1_tr=Dataset.TakeMonksDataSet("FilesData/monks-1.train")

monk_features={
  "a1":[1,2,3],
  "a2":[1,2,3],
  "a3":[1,2],
  "a4":[1,2,3],
  "a5":[1,2,3,4],
  "a6":[1,2],
  "class":[0,1]
}

monk_encoding=dict()

for k,v in monk_features.items():
  monk_encoding[k]=Dataset.one_hot_encoding(v)

Data=convert_to_one_hot(["a1","a2","a3","a4","a5","a6"],monk_encoding,monk1_tr)

#set hyperparamters values
LearningRate=np.linspace(0.01,0.1,3)
WeightDecay=np.linspace(0,1,3)# a.k.a. Lambda in Tikohonv regularization
FoldsNumber= np.array([2])

#grid creation
ParameterGrid = make_grid(LearningRate, WeightDecay, FoldsNumber)
print(f"performing grid search on {len(ParameterGrid[0])} hyperparamters with {len(ParameterGrid)} combinations")
print(f"using a data set with {len(Data)} examples, {len(Data[0][0])} input features and {len(Data[0][1])} output features")

#Choose best model
BestModelIndex=-1
BestModelError=np.inf
BestModelParameters=None
BestModelPerformance=None


Model=BuildTwoLevelFeedForward(17, 4, 1, lambda op, tp: (op-tp)**2, lambda x, y: 0)
Model.SetBeforeLossFunctionEvaluation(lambda op, tp: ( ((op[0]-tp[0])**2), 0))
InitialState=Model.ExtractLearningState()

for i,hyperparameters in enumerate(ParameterGrid):
  print(f">> training model no.{i+1} hyperparameters {hyperparameters}")
  
  Model.LoadLearningState(InitialState)
  
  ModelPerformance=ModelSelection(Model,
                                  Data,
                                  LearningRate=hyperparameters[0],
                                  WeightDecay=hyperparameters[1],
                                  FoldsNumber=hyperparameters[2])
  
  ValidationError=0 
  FoldsPerformance=list(map(lambda fold: fold["Loss"][-1], ModelPerformance["validation"]))

  ValidationError=sum(FoldsPerformance)/len(FoldsPerformance)
  print(f"<< finished model no.{i+1}, with a validation error of {ValidationError}")
  if ValidationError < BestModelError:
    BestModelError=ValidationError
    BestModelParameters=hyperparameters
    BestModelPerformance=ModelPerformance
    BestModelIndex=i

assert BestModelError != float('inf') 
FoldsGraphVal=np.array(np.sum( list(map(lambda fold: fold["Loss"], BestModelPerformance["validation"])),axis=0,dtype=np.float64 ))/BestModelParameters[2]
FoldsGraphTrain=np.array(np.sum( list(map(lambda fold: fold["Loss"], BestModelPerformance["training"])),axis=0,dtype=np.float64 ))/BestModelParameters[2]
print(f"the best model is no.{BestModelIndex} with a validation error of {BestModelError} and the following hyperparameters: {BestModelParameters}")
del BestModelPerformance["hyperparameters"]

Graph({"validation":FoldsGraphVal,"training":FoldsGraphTrain},["red","blue"],"epochs",f"test on model no. {BestModelIndex} with hyperparameters {hyperparameters}")