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
Data=convert_to_one_hot(["a1","a2","a3","a4","a5","a6"],monk_encoding,monk1_tr)

dataset=Dataset.DataSet(Data,17,1)







#Choose best model
BestModelIndex=-1
BestModelError=np.inf
BestModelParameters=None
BestModelPerformance=None
parameters={
  "learning_rate":np.array([0.3,0.5]),
  "weight_decay":np.array([0]),
  #"folds_number":np.array([2])
}
grid=parameter_grid.ParameterGrid(parameters)


print(f"performing grid search on {len(grid.get_parameters())} hyperparameters with {grid.get_size()} combinations")
print(f"using a data set with {dataset.size()} examples, {dataset.input_size()} input features and {dataset.output_size()} output features")



Model=BuildTwoLevelFeedForwardMonk1(17, 5, 2, lambda op, tp: (op-tp)**2, lambda x, y,z: 0)
Model.SetBeforeLossFunctionEvaluation(lambda op, tp: ( ((op[0]-tp[0])**2), 0))
InitialState=Model.ExtractLearningState()



tr_percent=1



tr,vl = dataset.get_dataset()[:int(dataset.size()*tr_percent)],dataset.get_dataset()[int(dataset.size()*tr_percent):]
performance_metrics=dict()

best_performance={
  "validation":{
    "Loss":[np.inf]
  }
}

for i in range(grid.get_size()):
  #reset model to initial state
  Model.LoadLearningState(InitialState)
  hyperparameters=grid[i]
  print(f"hyperparameters:{hyperparameters}")
  training=TrainPhase(Model,tr)
  evaluation=EvaluationPhase(Model,vl)
  def weights_update_function(weigths,GradientLoss,NeuronsOutput):
    return list(map(lambda w: w[0] +hyperparameters["learning_rate"]*GradientLoss*w[1] - hyperparameters["weight_decay"]*w[0],zip(weigths,NeuronsOutput)))
    
  for i in Model.GetAllNeurons(): 
    i.SetUpdateWeightsFunction(weights_update_function)
  
  accuracy_arr=[]
  for epoch in range(500):
    grad=np.zeros(2)
    tr_precision=0
    for sample in tr:
      sample_in=sample[0]
      sample_target=sample[1]
      if sample_target[0]==0:
        sample_target=np.array([0,1])
      else:
        sample_target=np.array([1,0])

      sample_out=Model.Predict(sample_in)
      grad+=sample_target-sample_out

      if np.argmax(sample_out)==0:
        out_repr=np.array([1,0])
      else:
        out_repr=np.array([0,1])

      if np.array_equal(sample_target,out_repr):
        tr_precision+=1
    Model.Learn(grad)
    vl_precision=0
    for sample in vl:
      sample_in=sample[0]
      sample_target=sample[1]
      sample_out=Model.Predict(sample_in)
      if np.argmax(sample_out)==0:
        out_repr=np.array([1,0])
      else:
        out_repr=np.array([0,1])
      if np.array_equal(sample_target,out_repr):
        vl_precision+=1
    print(f"tr {tr_precision}/{len(tr)}\tvl {vl_precision}/{len(vl)}")
    #training.Work(len(tr),True)          #train model
    accuracy_arr.append(tr_precision/len(tr))
    tr_precision=0
    #evaluation.Work(len(vl),True)
  print(accuracy_arr)
  performance_metrics["training"]=training.GetMetrics()
  performance_metrics["validation"]=evaluation.GetMetrics()
  #print(performance_metrics)
  if performance_metrics["validation"]["Loss"][-1] < best_performance["validation"]["Loss"][-1]:
    best_performance=performance_metrics

Graph({"validation":best_performance["validation"]["Loss"],"training":best_performance["training"]["Loss"]},["red","blue"],"epochs",f"test on model no. {BestModelIndex} with hyperparameters {hyperparameters}")
  


""" for i in range(grid.get_size()):
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

Graph({"validation":FoldsGraphVal,"training":FoldsGraphTrain},["red","blue"],"epochs",f"test on model no. {BestModelIndex} with hyperparameters {hyperparameters}") """