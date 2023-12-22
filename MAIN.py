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


def BuildTwoLevelFeedForward(InputSize,HiddenSize,OutputSize,LossFunction,WeightsUpdateFunction):
  NN=NeuralNetwork(InputSize,OutputSize,LossFunction)
  InputLayer=NN.GetAllInputNeurons()
  OutputLayer=NN.GetAllOutputNeurons()
  HiddenLayer = [ ActivationNeuron(lambda x:x,WeightsUpdateFunction,lambda x,y:0,LossFunction) for _ in range(HiddenSize) ]
  for InputNeuron in InputLayer:
    for HiddenNeuron in HiddenLayer:
      InputNeuron.AddConnectionTo(HiddenNeuron)
  for OutputNeuron in OutputLayer:
    for HiddenNeuron in HiddenLayer:
      HiddenNeuron.AddConnectionTo(OutputNeuron)
  NN.CalculateAllStructure()
  return NN

def ModelSelection(Model,DataSet,LearningRate,WeightDecay,FoldsNumber,BatchDimension,Steps=50):
  ModelSelectionPerformance=dict()
  TrainingPerformance=[]
  ValidationPerformance=[]
  cv=CrossValidation(DataSet,FoldsNumber)
  def weights_update_function(weigths,GradientLoss):
    return list(map(lambda w: w +LearningRate*(GradientLoss + WeightDecay*w),weigths)) # nota : 0.1 = eta
  Model.SetAllUpdateWeightsFunctionInputNeurons(weights_update_function)
  Model.SetAllUpdateWeightsFunctionOutputNeurons(weights_update_function)
  Model.SetAllUpdateWeightsFunctionHiddenNeurons(weights_update_function)
  for fold in range(cv.GetFoldsNum()):
    tr,vl = cv.GetKFold(fold)
    training=TrainPhase(Model,tr)
    evaluation=EvaluationPhase(Model,vl)
    for i in range(Steps):# fine training dopo un numero fisso di epoche / verrà cambiato con un criterio di fermata
      training.Work(BatchDimension)#train model 
      evaluation.Work(BatchDimension)#evaluate after change of parameters
    
    TrainingPerformance.append(training.GetMetrics())
    ValidationPerformance.append(evaluation.GetMetrics())

  
  ModelSelectionPerformance["training"]=TrainingPerformance
  ModelSelectionPerformance["validation"]=ValidationPerformance

  ModelSelectionPerformance["hyperparameters"]=dict()
  return ModelSelectionPerformance

def KFoldGraph(MetricsData: dict, Colors, LabelX, Title):
  
  XAxis= np.linspace(0, len(MetricsData["training"][0]),len(MetricsData["training"][0]))
  fig,axs= plt.subplots(len(MetricsData),1)

  fig.suptitle(f"Kfold cv with {len(MetricsData['training'])} folds")
  for i in len(MetricsData["training"]):
    axs[i].plot(XAxis,MetricsData["training"],linestyle='-', color = "red")
    axs[i].plot(XAxis,MetricsData["validation"],linestyle='-', color = "blue")
    axs[i].set_title(f'fold no. {i+1}')
  
  plt.show()





#istance desired model
LossFunction= lambda yo, yt: 1/2 * sum((yo - yt)**2)
Model=BuildTwoLevelFeedForward(10,5,3,LossFunction,lambda x:x)
#load dataset
Data=TakeDataset('FilesData/ML-CUP23-TR.csv')

#set hyperparamters values
LearningRate=np.linspace(0.1,0.9,10)
WeightDecay=np.linspace(0,2,10)# a.k.a. Lambda in Tikohonv regularization
FoldsNumber= [6]
BatchDimension= [10]#TODO: cambiare con perncuatle fra 0 ed 1
threshold= np.linspace(0.1,0.9,10)

#grid creation
ParameterGrid=list(it.product(LearningRate,WeightDecay,FoldsNumber,BatchDimension))

print(f"performing grid search on {len(ParameterGrid[0])} hyperparamters with {len(ParameterGrid)} combinations")
print(f"using a data set with {len(Data)} with {len(Data[0][0])} input features and {len(Data[0][1])}")
#grid search
BestModelIndex=-1
BestModelError=float('inf')
BestModelParameters=None
BestModelPerformance=None
for hyperparameters in ParameterGrid:
  ModelPerformance=ModelSelection(Model,Data,*hyperparameters)
  ValidationError=0
  FoldsPerformance=list(map(lambda fold:fold[-1],ModelPerformance["validation"]))
  ValidationError=sum(FoldsPerformance)/len(FoldsPerformance)
  if ValidationError < BestModelError:
    BestModelError=ValidationError
    BestModelParameters=hyperparameters
    BestModelPerformance=ModelPerformance

assert BestModelError != float('inf') and BestModelParameters!=None and BestModelPerformance != None

print(f"the best model is no.{BestModelIndex} with a validation error of {BestModelError}")