import numpy as np
from Dataset import DataSet
from Phase import EvaluationPhase, TrainPhase


class KFold:
    def __init__(self,data,input_size,output_size,shuffle_dataset,folds_number):
        self._folds_number=folds_number
        self._folds= np.array([DataSet.Dataset(data[fold],input_size,output_size,shuffle_dataset) for fold in np.array_split(data,folds_number)])
    def train(self,model,batch_size,classification_function=lambda x:x):
        best_fold_index=0
        best_fold_performance=dict()
        for i,fold in enumerate(self._folds):
            tr=fold
            vl=np.concatenate([self._folds[0:i+1],self._folds[i+2:]])
            training=TrainPhase(model,tr,classification_function)
            evaluation=EvaluationPhase(model,vl,classification_function)
            for epoch in range(1000):
                training.Work(batch_size,True)
                evaluation.Work(batch_size,True)
            if evaluation.GetMetrics()["Loss"][-1] < best_fold_performance["validation"]["Loss"][-1]:
                best_fold_performance=training.GetMetrics()
                best_fold_index=i
            print("epoch ",i+1,"tr precision ",training.GetMetrics()["Precision"][-1],"\tloss ",training.GetMetrics()["Loss"][-1],epoch+1)
        return {"training":self._folds[best_fold_index]["training"],"validation":evaluation.GetMetrics()}