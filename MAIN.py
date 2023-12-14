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

def weights_update_function(weigths,GradientLoss,LearningRate=0.1,WeightDecay=0):
  return list(map(lambda w: w +LearningRate*(GradientLoss + WeightDecay*w),weigths)) # nota : 0.1 = eta


def BuildTwoLevelFeedForward(InputSize,HiddenSize,OutputSize):
  LossFunction= lambda yo, yt: 1/2 * sum((yo - yt)**2)
  NN=NeuralNetwork(InputSize,OutputSize,LossFunction)
  InputLayer=NN.GetAllInputNeurons()
  OutputLayer=NN.GetAllOutputNeurons()
  HiddenLayer = [ ActivationNeuron(lambda x:x,weights_update_function,lambda x,y:0,LossFunction) for _ in range(HiddenSize) ]
  for InputNeuron in InputLayer:
    for HiddenNeuron in HiddenLayer:
      InputNeuron.AddConnectionTo(HiddenNeuron)
  for OutputNeuron in OutputLayer:
    for HiddenNeuron in HiddenLayer:
      HiddenNeuron.AddConnectionTo(OutputNeuron)
  NN.CalculateAllStructure()
  return NN

TwoLevelFF=BuildTwoLevelFeedForward(10,5,3)

DataSet=np.loadtxt('sample_data/california_housing_train.csv',delimiter=',',skiprows=1,dtype=np.float64)

K=6

# codice di esempio non ancroa funzionante





training=TrainPhase(TwoLevelFF,Data[:100])

for _ in range(100):
  training.Work()
  print(training.GetMetrics())
prova=dict()
prova["Training loss"]=training.GetMetrics()["Loss"]

Graph(prova,["red"],"","")







def ModelSelection(Model,DataSet,folds,*hyperparams):
  ModelSelectionPerformance=dict()
  TrainingPerformance=[]
  ValidationPerformance=[]
  cv=CrossValidation(DataSet,folds)
  for fold in range(cv.GetFoldsNum()):
    tr,vl = cv.GetKFold(fold)
    training=TrainPhase(Model,tr)
    for i in range(50):# fine training dopo un numero fisso di epoche / verrà cambiato con un criterio di fermata
      training.Work()
    evaluation=EvaluationPhase(Model,vl)
    evaluation.Work()
    TrainingPerformance.append(training.GetMetrics())
    ValidationPerformance.append(evaluation.GetMetrics())
  ModelSelectionPerformance["training"]=TrainingPerformance
  ModelSelectionPerformance["validation"]=ValidationPerformance
  ModelSelectionPerformance["hyperparameters"]=dict()
  return ModelSelectionPerformance





def DisplayKFoldPerformance(ModelSelectionData):
  if(len(ModelSelectionData["training"])!=len(ModelSelectionData["training"])): raise ValueError("folds metrics bust be of the same size")
  XLen = max(max(map(len,ModelSelectionData["training"])), max(map(len,ModelSelectionData["validation"])))
  Xdata=np.linspace(0,XLen,XLen)


# hyperparameters

LearningRate=np.linspace(0.1,0.9,10)
WeightDecay=np.linspace(0,2,10)# a.k.a. Lambda in Tikohonv regularization
FoldsNumber= [6]
BatchDimension= [10]

ParameterGrid=list(it.product(LearningRate,WeightDecay,FoldsNumber,BatchDimension))

print(f"performing grid search on {len(ParameterGrid[0])} hyperparamters with {len(ParameterGrid)} combinations")

Model=TwoLevelFF
Data=Data


for hyperparameters in ParameterGrid:
  Results=ModelSelection(Model,Data,*hyperparameters)



