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
import parameter_grid
from models import BuildTwoLevelFeedForward, BuildTwoLevelFeedForwardMonk, BuildTwoLevelFeedForwardMonk1
from model_selection import model_selection

#Get all metrics for all K-folds considering: "LearningRate", "WeightDecay", "FoldsNumber"
#Print a graph of KFold


#instance desired model
#load dataset
#Data=TakeDataset('FilesData/ML-CUP23-TR.csv')[:200]

import Dataset
import numpy as np
#loading monk dataset
monk1_tr=Dataset.TakeMonksDataSet("FilesData/monks-1.train")
#describe features and their values
monk_features={
  "a1":[1,2,3],
  "a2":[1,2,3],
  "a3":[1,2],
  "a4":[1,2,3],
  "a5":[1,2,3,4],
  "a6":[1,2],
  "class":[0,1]
}

#create the encoding for the one hot representation
monk_encoding= encode_dataset_to_one_hot(monk_features)
#apply encoding to the dataset
Data=convert_to_one_hot(["a1","a2","a3","a4","a5","a6"],["class"],monk_encoding,monk1_tr)

dataset=Dataset.DataSet(Data,17,2)


#Choose best model
BestModelIndex=-1
BestModelError=np.inf
BestModelParameters=None
BestModelPerformance=None


parameters={
  #"learning_rate":np.linspace(0.07,0.25,5),for stocastic
  "learning_rate":np.linspace(0.2,0.5,20),
  "weight_decay":np.array([0]),
  #"folds_number":np.array([2])
}
grid=parameter_grid.ParameterGrid(parameters)

print(f"performing grid search on {len(grid.get_parameters())} hyperparameters with {grid.get_size()} combinations")
print(f"using a data set with {dataset.size()} examples, {dataset.input_size()} input features and {dataset.output_size()} output features")

Model=BuildTwoLevelFeedForwardMonk1(17, 5, 2, lambda op, tp: (op-tp)**2, lambda x, y: 0)
Model.SetBeforeLossFunctionEvaluation(lambda op, tp: ( ((op[0]-tp[0])**2), 0))
InitialState=Model.ExtractLearningState()



tr_percent=0.8


#split dataset
#tr,vl = dataset.get_dataset()[:int(dataset.size()*tr_percent)],dataset.get_dataset()[int(dataset.size()*tr_percent):]
tr= DataSet(dataset.get_dataset()[:int(dataset.size()*tr_percent)],17,2,True)
vl= DataSet(dataset.get_dataset()[int(dataset.size()*tr_percent):],17,2)
print(f"dataset split {tr_percent}/{1-tr_percent}")
print(f"{tr.size()} samples for training")
print(f"{vl.size()} samples for validation")

best_performance={
  "index":-1,
  "training":{"Loss":[np.inf],"Precision":[np.inf]},
  "validation":{"Loss":[np.inf],"Precision":[np.inf]},
}

for i in range(grid.get_size()):
  #reset model to initial state
  Model.LoadLearningState(InitialState)
  hyperparameters=grid[i]
  print(f"hyperparameters:{hyperparameters}")

  training=TrainPhase(Model,tr,lambda x:np.array([ 1 if v==np.max(x) else 0 for v in x ]))
  evaluation=EvaluationPhase(Model,vl,lambda x:np.array([ 1 if v==np.max(x) else 0 for v in x ]))

  def weights_update_function(weights,GradientLoss):
    #print(GradientLoss)
    return list(map(lambda w: w[0] +hyperparameters["learning_rate"]*w[1] - hyperparameters["weight_decay"]*w[0],zip(weights,GradientLoss)))
    
  for n in Model.GetAllNeurons(): 
    n.SetUpdateWeightsFunction(weights_update_function)
  for epoch in range(1000):
    training.Work(tr.size(),True)
    print("tr precision ",training.GetMetrics()["Precision"][-1],"\tloss ",training.GetMetrics()["Loss"][-1],epoch+1)
    evaluation.Work(vl.size(),True)
  if evaluation.GetMetrics()["Loss"][-1] < best_performance["validation"]["Loss"][-1]:
    best_performance["index"]=i
    best_performance["training"]=training.GetMetrics()
    best_performance["validation"]=evaluation.GetMetrics()
print(f"grid search done, bestmodel: np.{best_performance["index"]} with hyperparameters {grid[best_performance["index"]]}")
plot_model_performance(best_performance["training"],"red","blue","epochs",f"best model with hyperparameters {grid[best_performance["index"]]}")
#retrain on whole dataset the best model
whole_tr= DataSet(dataset.get_dataset(),17,2,True)
final_training=TrainPhase(Model,whole_tr,lambda x:np.array([ 1 if v==np.max(x) else 0 for v in x ]))
for epoch in range(1000):
  final_training.Work(final_training,True)
plot_model_performance(final_training.GetMetrics(),"red","blue","epochs",f"final retraining performance with hyperparameters {grid[best_performance["index"]]}")



for i in range(grid.get_size()):
  print(f">> training model no.{i+1} hyperparameters: {grid[i]}")
  hyperparameters=grid[i]
  # reset the model to his initial state
  Model.LoadLearningState(InitialState)
  

  ModelPerformance=model_selection(Model,
                                  dataset.get_dataset(),
                                  LearningRate=hyperparameters["learning_rate"],
                                  WeightDecay=hyperparameters["weight_decay"],
                                  FoldsNumber=hyperparameters["folds_number"]
                                  )
  
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