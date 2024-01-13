import numpy as np
from Evaluation import CrossValidation
from Phase import EvaluationPhase, TrainPhase


def ModelSelection(Model, DataSet, LearningRate, WeightDecay, FoldsNumber, Threshold=0 ,Steps=50):
  print(f"beginning model selection with {Steps} steps and {FoldsNumber} folds")
  ModelSelectionPerformance=dict()
  
  TrainingPerformance=[]
  ValidationPerformance=[]
  
  cv=CrossValidation(DataSet,FoldsNumber)
  
  def weights_update_function(weigths,GradientLoss):
    print(GradientLoss)
    return list(map(lambda w: w +LearningRate*GradientLoss - WeightDecay*w,weigths))
  
  for i in Model.GetAllNeurons(): 
    i.SetUpdateWeightsFunction(weights_update_function)

  
  for fold in range(cv.GetFoldsNum()):
    tr,vl = cv.GetKFold(fold)
    training=TrainPhase(Model,tr)
    evaluation=EvaluationPhase(Model,vl)
    for i in range(Steps):# fine training dopo un numero fisso di epoche / verrà cambiato con un criterio di fermata
      training.Work(len(tr))          #train model
      evaluation.Work(len(vl))        #evaluate after change of parameters
    TrainingPerformance.append(training.GetMetrics())
    ValidationPerformance.append(evaluation.GetMetrics())
    print(f"fold no.{fold+1}")
    print(f"training : {TrainingPerformance[-1]}")
    print(f"validation : {ValidationPerformance[-1]}")
  
  ModelSelectionPerformance["training"]=TrainingPerformance
  ModelSelectionPerformance["validation"]=ValidationPerformance

  ModelSelectionPerformance["hyperparameters"]=dict()
  return ModelSelectionPerformance

def make_grid(*args):
  return np.array(np.meshgrid(*args)).T.reshape(-1, 3)