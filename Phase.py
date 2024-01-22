from Dataset import *

import numpy as np

class Phase:
  def __init__(self, Model, DataSet):
    self._Model = Model
    self._Dataset = DataSet

    self._Metrics = dict()

    self._Metrics["Loss"] = []
    self._Metrics["Precision"] = []

  def GetMetrics(self):
    return self._Metrics

  def Work(self):
    raise NotImplemented

class TrainPhase(Phase):
  def __init__(self, Model, TrainSet,classification_function=lambda x:x):
    super().__init__(Model, TrainSet)
    self._classification_function=classification_function

  def Work(self, BatchDimension,conf_matrix=False):
    if(BatchDimension < 1 or BatchDimension > self._Dataset.size()): raise ValueError("batch dimension must be above 0 and lower or equal to the dataset dimension")
    batch=self._Dataset.next_epoch(BatchDimension)
    epoch_size=sum(map(len,batch))
    self._Model.Learn(batch)
    precision=0
    outputs=np.empty(self._Dataset.size(),dtype=object)
    targets=np.empty(self._Dataset.size(),dtype=object)
    for i in range(self._Dataset.size()):
      example=self._Dataset[i]
      input_vector=example[0]
      target_vector=example[1]
      output_vector=self._Model.Predict(input_vector)
      if conf_matrix and np.array_equal(self._classification_function(output_vector),target_vector):
        precision+=1
      outputs[i]=output_vector
      targets[i]=target_vector
    
    loss_value= np.sum( (outputs-targets)@(outputs-targets)) * (1/epoch_size)
    self._Metrics["Loss"].append(loss_value)
    if conf_matrix:
      self._Metrics["Precision"].append(precision/epoch_size)

class EvaluationPhase(Phase): 
  def __init__(self, Model, DataSet,evaluation_function=lambda t,o: np.sum((t-o)**2),classification_function=lambda x:x):
    super().__init__(Model, DataSet)
    self._classification_function=classification_function
    self._evaluation_function=evaluation_function


  def Work(self, BatchDimension,conf_matrix=False):
    precision=0
    outputs=np.empty(self._Dataset.size(),dtype=object)
    targets=np.empty(self._Dataset.size(),dtype=object)
    for i in range(self._Dataset.size()):
      example=self._Dataset[i]
      input_vector=example[0]
      target_vector=example[1]
      output_vector=self._Model.Predict(input_vector)
      if conf_matrix and np.array_equal(self._classification_function(output_vector),target_vector):
        precision+=1
      outputs[i]=output_vector
      targets[i]=target_vector
    
    #loss_value= np.sum( (outputs-targets)@(outputs-targets) )*(1/BatchDimension)
    loss_value= self._evaluation_function(targets,outputs)
    self._Metrics["Loss"].append(loss_value)
    if conf_matrix and self._Dataset.size()!=0:
      self._Metrics["Precision"].append(precision/self._Dataset.size())
