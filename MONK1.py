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
from Layer import *
from NeuralNetwork import *
from Neuron import *
from Phase import *
from Regularization import *
from Result import *
import parameter_grid
from models import build_monk1_MLP
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


grid=parameter_grid.ParameterGrid({
  "learning_rate":np.linspace(0.01,0.02,2),
  "weight_decay":np.array([0.0]),
})

print(f"performing grid search on {len(grid.get_parameters())} hyperparameters with {grid.get_size()} combinations")
print(f"using a data set with {dataset.size()} examples, {dataset.input_size()} input features and {dataset.output_size()} output features")

Model=build_monk1_MLP(17, 4, 2, lambda op, tp: (op-tp)**2, lambda x, y: 0)
Model.SetBeforeLossFunctionEvaluation(lambda op, tp: ( ((op[0]-tp[0])**2), 0))


InitialState=Model.ExtractLearningState()

tr_percent=0.8
tr_size=int(dataset.size()*tr_percent)
#split dataset
print(f"dataset split {tr_percent}/{1-tr_percent}")
monk_classification=lambda x:np.array([ 1 if v==np.max(x) else 0 for v in x ])
training_strategy=Holdout(dataset.get_dataset(),
                          dataset.input_size(),
                          dataset.output_size(),
                          False,
                          tr_size)

grid_search=GridSearch(Model,
                      1,
                      grid,
                      training_strategy
                      )

result=grid_search.search(monk_classification)
#Model.LoadLearningState(result["model_state"])

best_params=grid[result["index"]]


def weights_update_function(weights,GradientLoss):
  w=np.array(weights)
  g=np.array(GradientLoss)
  return w+(1/dataset.size())*g*best_params["learning_rate"]-best_params["weight_decay"]*w

Model.SetWeightsUpdateFunction(weights_update_function)
def bias_update_function(old_bias,gradient_value):
  return old_bias + (1/dataset.size())*gradient_value*best_params["learning_rate"]-best_params["weight_decay"]*old_bias

training=TrainPhase(Model,dataset,monk_classification)
for epoch in range(300):
    training.Work(1,True)

plot_model_performance({"training":training.GetMetrics()},"red","blue","","final retraining performance of the best model")

#loading monk dataset
monk1_ts=Dataset.TakeMonksDataSet("FilesData/monks-1.test")
#create the encoding for the one hot representation
monk_encoding= encode_dataset_to_one_hot(monk_features)
#apply encoding to the dataset
Data=convert_to_one_hot(["a1","a2","a3","a4","a5","a6"],["class"],monk_encoding,monk1_ts)

ts_dataset=Dataset.DataSet(Data,17,2)

def weights_update_function(weights,GradientLoss):
  w=np.array(weights)
  g=np.array(GradientLoss)
  return w+(1/ts_dataset.size())*g*best_params["learning_rate"]-best_params["weight_decay"]*w

Model.SetWeightsUpdateFunction(weights_update_function)
def bias_update_function(old_bias,gradient_value):
  return old_bias + (1/ts_dataset.size())*gradient_value*best_params["learning_rate"]-best_params["weight_decay"]*old_bias

eval_func=lambda t,o:np.sum((o-t)@(o-t))*(1/ts_dataset.size())

evaluation=EvaluationPhase(Model,ts_dataset,eval_func,monk_classification)
evaluation.Work(1,True)
print(f"final evaluation on the test set\naccuracy={evaluation.GetMetrics()["Precision"][-1]}\tloss value={evaluation.GetMetrics()["Loss"][-1]}")