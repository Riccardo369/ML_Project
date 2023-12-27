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

#Build the model
def BuildTwoLevelFeedForward(InputSize, HiddenSize, OutputSize, LossFunction, WeightsUpdateFunction):
  
  NN = NeuralNetwork(InputSize,OutputSize,LossFunction)
  
  InputLayer = NN.GetInputLayer()  
  OutputLayer = NN.GetOutputLayer()
  HiddenLayer = Layer(ActivationNeuron, HiddenSize, lambda x: x, WeightsUpdateFunction, lambda x, y: 0, lambda x, y: 0)
  
  InputLayer.ConnectTo(HiddenLayer)
  HiddenLayer.ConnectTo(OutputLayer) 
      
  NN.CalculateAllStructure() 
  
  return NN

#Get all metrics for all K-folds considering: "LearningRate", "WeightDecay", "FoldsNumber"
def ModelSelection(Model, DataSet, LearningRate, WeightDecay, FoldsNumber, Steps=2):
  
  ModelSelectionPerformance=dict()
  
  TrainingPerformance=[]
  ValidationPerformance=[]
  
  cv=CrossValidation(DataSet,FoldsNumber)
  
  def weights_update_function(weigths,GradientLoss):
    return list(map(lambda w: w +LearningRate*(GradientLoss + WeightDecay*w),weigths))
  
  for i in Model.GetAllNeurons(): i.SetUpdateWeightsFunction(weights_update_function)
  
  for fold in range(cv.GetFoldsNum()):
    tr,vl = cv.GetKFold(fold)
    training=TrainPhase(Model,tr)
    evaluation=EvaluationPhase(Model,vl)
    for i in range(Steps):# fine training dopo un numero fisso di epoche / verrà cambiato con un criterio di fermata
      
      training.Work(len(tr))          #train model
      evaluation.Work(len(vl))        #evaluate after change of parameters

    TrainingPerformance.append(training.GetMetrics())
    ValidationPerformance.append(evaluation.GetMetrics())
  
  ModelSelectionPerformance["training"]=TrainingPerformance
  ModelSelectionPerformance["validation"]=ValidationPerformance

  ModelSelectionPerformance["hyperparameters"]=dict()
  return ModelSelectionPerformance

#Print a graph of KFold
def KFoldGraph(MetricsData: dict, Colors, LabelX, Title):
  
  XAxis= np.linspace(0, len(MetricsData["training"][0]),len(MetricsData["training"][0]))
  fig, axs= plt.subplots(len(MetricsData),1)

  fig.subtitle(f"Kfold cv with {len(MetricsData['training'])} folds")
  for i in len(MetricsData["training"]):
    axs[i].plot(XAxis,MetricsData["training"],linestyle='-', color = "red")
    axs[i].plot(XAxis,MetricsData["validation"],linestyle='-', color = "blue")
    axs[i].set_title(f'fold no. {i+1}')
  
  plt.show()

#instance desired model
Model=BuildTwoLevelFeedForward(10, 1, 3, lambda op, tp: (op-tp)**2, lambda x, y: 0)
Model.SetBeforeLossFunctionEvaluation(lambda op, tp: (((op[0]-tp[0])**2)+((op[1]-tp[1])**2)+((op[2]-tp[2])**2), 0))

#load dataset
Data=TakeDataset('FilesData/ML-CUP23-TR.csv')[:60]

#set hyperparamters values
LearningRate=np.linspace(0.1,0.9,10)
WeightDecay=np.linspace(0,2,10)# a.k.a. Lambda in Tikohonv regularization
FoldsNumber= [1]
threshold= np.linspace(0.1,0.9,10)

#grid creation
ParameterGrid = np.array(np.meshgrid(LearningRate, WeightDecay, FoldsNumber)).T.reshape(-1, 3)[:2]

print(f"performing grid search on {len(ParameterGrid[0])} hyperparamters with {len(ParameterGrid)} combinations")
print(f"using a data set with {len(Data)} examples, {len(Data[0][0])} input features and {len(Data[0][1])}")

#Choose best model
BestModelIndex=-1
BestModelError=float('inf')
BestModelParameters=None
BestModelPerformance=None
for hyperparameters in ParameterGrid:
  ModelPerformance=ModelSelection(Model,
                                  Data,
                                  LearningRate=hyperparameters[0],
                                  WeightDecay=hyperparameters[1],
                                  FoldsNumber=hyperparameters[2])
  
  ValidationError=0 
  FoldsPerformance=list(map(lambda fold: fold["Loss"][-1], ModelPerformance["validation"]))
  ValidationError=sum(FoldsPerformance)/len(FoldsPerformance)
  
  if ValidationError < BestModelError:
    BestModelError=ValidationError
    BestModelParameters=hyperparameters
    BestModelPerformance=ModelPerformance

assert BestModelError != float('inf') and BestModelParameters!=None and BestModelPerformance != None

print(f"the best model is no.{BestModelIndex} with a validation error of {BestModelError}")