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
from GridSearch import GridSearch
from Holdout import Holdout
from KFold import KFold
from Layer import *
from NeuralNetwork import *
from Neuron import *
from Phase import *
from Regularization import *
from Result import *
import parameter_grid
from models import BuildTwoLevelFeedForward, BuildTwoLevelFeedForwardMonk, build_CUP_MLP
from model_selection import model_selection
import Dataset
import numpy as np
#loading monk dataset
ml_cup_tr=Dataset.TakeCupDataset("FilesData/ML-CUP23-TR.csv")

dataset=Dataset.DataSet(ml_cup_tr,10,3)

grid=parameter_grid.ParameterGrid({
  "learning_rate":np.linspace(0.03,0.25,2),
  "weight_decay":np.array([0.0]),
})

print(f"performing grid search on {len(grid.get_parameters())} hyperparameters with {grid.get_size()} combinations")
print(f"using a data set with {dataset.size()} examples, {dataset.input_size()} input features and {dataset.output_size()} output features")

Model=build_CUP_MLP(10, 4, 3, lambda op, tp: (op-tp)**2, lambda x, y: 0)
Model.SetBeforeLossFunctionEvaluation(lambda op, tp: ( math.sqrt( (op[0]-tp[0])**2 + (op[1]-tp[1])**2 + (op[2]-tp[2])**2  )  , 0))


InitialState=Model.ExtractLearningState()

folds_number=4

#split dataset
print(f"folds number {folds_number}")
training_strategy=KFold(dataset.get_dataset(),
                        dataset.input_size(),
                        dataset.output_size(),
                        False,
                        folds_number
                        )

grid_search=GridSearch(Model,
                      1,
                      grid,
                      training_strategy
                      )

result=grid_search.search()

Model.LoadLearningState(result["model_state"])
final_holdout=Holdout(dataset.get_dataset(),
                      dataset.input_size(),
                      dataset.output_size(),
                      False,
                      dataset.size()
                      )
final_retrain=final_holdout.train(Model,1)
plot_model_performance(final_retrain,"red","blue","","final retraining performance of the best model")
exit()

#Choose best model
BestModelIndex=-1
BestModelError=np.inf
BestModelParameters=None
BestModelPerformance=None




best_performance={
  "index":-1,
  "training":{"Loss":[np.inf],"Precision":[np.inf]},
  "validation":{"Loss":[np.inf],"Precision":[np.inf]},
}
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