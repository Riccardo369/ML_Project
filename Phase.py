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
    TotalLossValue = 0  
    gradient=np.zeros(self._Dataset.output_size(),dtype=np.longdouble)
    batch=self._Dataset.next_batch(BatchDimension)

    error_signals=[]
    output_values=[]

    for example in batch:
      input_vector=example[0]
      target_vector=example[1]
      output_vector=self._Model.Predict(input_vector)
      #gradient+=target_vector-output_vector
      neurons_loss,neurons_output=self._Model.Learn(target_vector-output_vector)
      error_signals.append(neurons_loss)
      output_values.append(neurons_output)
    
    
    for n in self._Model.GetAllOutputNeurons()+self._Model.GetAllHiddenNeurons():
      entering_neurons=map(lambda x:x.GetStartNeuron(),n.GetSetEnterBridge())
      grad=[]
      #error signal of the unit for each pattern
      errors=np.array([ e[n] for e in error_signals ])
      for en in entering_neurons:
        grad.append(np.array( [o[en] for o in output_values ])@errors )
      new_weights=n.CalculateUpdatedWeights(grad)
      for i,w in enumerate(n.GetSetEnterBridge()):
        w.Weight=new_weights[i]


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
    
    loss_value= np.sum( (outputs-targets)@(outputs-targets) )
    self._Metrics["Loss"].append(loss_value)
    if conf_matrix:
      self._Metrics["Precision"].append(precision/self._Dataset.size())

class EvaluationPhase(Phase): 
  def __init__(self, Model, DataSet,classification_function=lambda x:x):
    super().__init__(Model, DataSet)
    self._classification_function=classification_function


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
    
    loss_value= np.sum( (outputs-targets)@(outputs-targets) )
    self._Metrics["Loss"].append(loss_value)
    if conf_matrix:
      self._Metrics["Precision"].append(precision/self._Dataset.size())
