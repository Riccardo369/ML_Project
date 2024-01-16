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
  def __init__(self, Model, TrainSet):
    super().__init__(Model, TrainSet)

  def Work(self, BatchDimension,conf_matrix=False):
    if(BatchDimension < 1 or BatchDimension > len(self._Dataset)): raise ValueError("batch dimension must be above 0 and lower or equal to the dataset dimension")
    Batches = BatchesExtraction(self._Dataset, BatchDimension)
    precision=0
    TotalLossValue = 0  
    for Batch in Batches:
      DirectionLoss = np.array([0]*len(Batch[0][1]), dtype=np.longdouble)
      
      #print(f"Start direction loss {DirectionLoss}")
      
      OutputValueVector = []
      TargetValueVector = []
      
      for r in Batch:
        
        InputVector = r[0]
        TargetValue = r[1]


        OutputValue = self._Model.Predict(InputVector)
        
        #NewDirectionLoss = self._Model.GradientDirectionLoss(OutputValue, TargetValue)
        NewDirectionLoss= TargetValue-OutputValue
        DirectionLoss += -NewDirectionLoss
        if OutputValue == TargetValue and conf_matrix:
          precision+=1
        OutputValueVector.append(OutputValue)
        TargetValueVector.append(TargetValue)
      


      self._Model.Learn(DirectionLoss/len(Batch))


      TotalLossValue += self._Model.LossFunctionEvaluation(OutputValueVector, TargetValueVector)
      
      #print(self._Model.LossFunctionEvaluation(OutputValueVector, TargetValueVector))
      
    self._Metrics["Loss"].append(TotalLossValue/len(Batches))   
    if conf_matrix:
      self._Metrics["Precision"].append(precision/len(self._Dataset))   
    
class EvaluationPhase(Phase):
  def __init__(self, Model, DataSet):
    super().__init__(Model, DataSet)

  def Work(self, BatchDimension,conf_matrix=False):

    Batches = BatchesExtraction(self._Dataset, BatchDimension)
    precision=0
    for Batch in Batches:
      
      OutputValueVector = []
      TargetValueVector = []
      
      for r in Batch:
        Output=self._Model.Predict(r[0])
        Target=r[1]
        OutputValueVector.append(Output)
        TargetValueVector.append(Target)
        if Output == Target and conf_matrix:
          precision+=1
        
    self._Metrics["Loss"].append(self._Model.LossFunctionEvaluation(OutputValueVector, TargetValueVector))
    if conf_matrix:
      self._Metrics["Precision"].append(precision/len(self._Dataset))