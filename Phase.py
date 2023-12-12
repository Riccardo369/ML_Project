from Dataset import *

class Phase:
  def __init__(self, Model, DataSet):
    self._Model = Model
    self._Dataset = DataSet

    self._Metrics = dict()

    self._Metrics["Loss"] = []
    self._Metrics["Accuracy"] = []

  def GetMetrics(self):
    return self._Metrics

  def Work(self):
    raise NotImplemented

class TrainPhase(Phase):
  def __init__(self, Model, TrainSet):
    super().__init__(Model, TrainSet)

  def Work(self,BatchDimension,**Hyperparameters):
    if(BatchDimension < 1 or BatchDimension > len(self._Dataset)): raise ValueError("batch dimension must be above 0 and lower or equal to the dataset dimension")
    Batches = BatchesExtraction(self._Dataset, BatchDimension)

    TotalLossValue = 0

    for Batch in Batches:
      #Do mean of all loss values
      LossValue = 0
      for r in Batch:
        InputVector=r[0]
        TargetValue=r[1]
        LossValue += self._Model.GetLossLambdaFunctionEvaluation()(self._Model.Predict(InputVector), TargetValue)

      LossValue /= len(Batch)

      #Add loss value batch to total loss value
      TotalLossValue += LossValue

      #Calculate gradient
      #DirectionLoss = self._Model.GradientDirectionLoss(LossValue)

      #Learn model
      self._Model.Learn(TotalLossValue)

    #Apply mean of all loss values
    self._Metrics["Loss"].append(TotalLossValue / len(Batches))   
    
class EvaluationPhase(Phase):
  def __init__(self, Model, DataSet, n):
    super().__init__(Model, DataSet, n, lambda: True)

  def Work(self):

    Batches = BatchesExtraction(self._Dataset, 1)

    TotalLossValue = 0

    for Batch in Batches:

      #Do mean of all loss values
      LossValue = 0

      for r in Batch: LossValue += self._Model.GetLossLambdaFunctionEvaluation(self._Model.Predict(r[0]), r[1])
      LossValue /= len(r)

      #Add loss value batch to total loss value
      TotalLossValue += LossValue

    #Apply mean of all loss values
    self._Metrics["Loss"].append(TotalLossValue / len(Batches))