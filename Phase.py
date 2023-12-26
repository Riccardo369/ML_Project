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

  def Work(self,BatchDimension):
    if(BatchDimension < 1 or BatchDimension > len(self._Dataset)): raise ValueError("batch dimension must be above 0 and lower or equal to the dataset dimension")
    Batches = BatchesExtraction(self._Dataset, BatchDimension)

    TotalLossValue = 0

    for Batch in Batches:
      LossValue = 0
      for r in Batch:
        InputVector=r[0]
        TargetValue=r[1]
        OutputValue=self._Model.Predict(InputVector)
        DirectionLoss = self._Model.GradientDirectionLoss(OutputValue,TargetValue)
        self._Model.Learn(DirectionLoss)
        LossValue += self._Model.LossFunctionEvaluation(OutputValue, TargetValue)
      LossValue/=len(Batch)
      TotalLossValue+=LossValue
    self._Metrics["Loss"].append(TotalLossValue/len(Batches))   
    
class EvaluationPhase(Phase):
  def __init__(self, Model, DataSet):
    super().__init__(Model, DataSet)

  def Work(self,BatchDimension):

    Batches = BatchesExtraction(self._Dataset, BatchDimension)

    #TotalLossValue = 0

    for Batch in Batches:
      LossValue = 0
      for r in Batch: LossValue += self._Model.LossFunctionEvaluation(self._Model.Predict(r[0]), r[1])
    self._Metrics["Loss"].append(LossValue / len(Batches))

      #Add loss value batch to total loss value
      #TotalLossValue += LossValue

    #Apply mean of all loss values