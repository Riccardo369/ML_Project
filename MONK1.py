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
from Simulator import Simulator
import parameter_grid
from models import build_monk1_MLP
import Dataset
import numpy as np
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

#loading monk dataset
monk1_tr=Dataset.TakeMonksDataSet("FilesData/monks-1.train")
#describe features and their values

#apply encoding to the dataset
Data=convert_to_one_hot(["a1","a2","a3","a4","a5","a6"],["class"],monk_encoding,monk1_tr)

dataset=Dataset.DataSet(Data,17,1)

#loading monk dataset
monk1_ts=Dataset.TakeMonksDataSet("FilesData/monks-1.test")
#apply encoding to the dataset
Data=convert_to_one_hot(["a1","a2","a3","a4","a5","a6"],["class"],monk_encoding,monk1_ts)

ts_dataset=Dataset.DataSet(Data,17,1)
#print(f"performing grid search on {len(grid.get_parameters())} hyperparameters with {grid.get_size()} combinations")
print(f"using a data set with {dataset.size()} examples, {dataset.input_size()} input features and {dataset.output_size()} output features")

Model=build_monk1_MLP(17, 4, 1, lambda op, tp: (op-tp)**2, lambda x, y: 0)
for n in Model.GetAllNeurons():
  n.BiasValue=0
for n in Model.GetAllHiddenNeurons():
  hidden_neuron_net=n.GetSetEnterBridge()
  fan_in=len(hidden_neuron_net)
  for w in hidden_neuron_net:
    w.Weight= np.std(np.random.uniform(-(1/np.sqrt(fan_in)),(1/np.sqrt(fan_in))))
Model.SetBeforeLossFunctionEvaluation(lambda op, tp: ( ((op[0]-tp[0])**2), 0))
InitialState=Model.ExtractLearningState()
#split dataset
print(f"dataset split {dataset.size()} for tr /{ts_dataset.size()} for vl")
monk_classification1=lambda x:np.array([1] if x >=0.5 else [0])

simulator=Simulator(
  model=Model,
  tr_dataset=dataset,
  vl_dataset=ts_dataset,
  batch_size=1,
  grid=parameter_grid.ParameterGrid({
    "learning_rate":np.array([0.2]),
    "weight_decay":np.array([0]),
    "momentum":np.array([0.4]),
  }),
  training_strategy=Holdout(
                          tr_data=dataset.get_dataset(),
                          vl_data=ts_dataset.get_dataset(),
                          input_size=dataset.input_size(),
                          output_size=dataset.output_size(),
                          shuffle_dataset=True,
                          ),
  classification_function=monk_classification1                    
)
ts_loss,ts_prec,best_performance=simulator.run()

plot_model_performance(best_performance,"red","blue","","final retraining performance of the best model")

print(f"ts set performance: precision={ts_prec} loss={ts_loss}")