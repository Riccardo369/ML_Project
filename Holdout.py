import numpy as np
from Dataset import DataSet
from Phase import EvaluationPhase, TrainPhase
from Result import plot_model_performance


class Holdout:
    def __init__(self,data,input_size,output_size,shuffle_dataset,split_index) -> None:
        self._tr_dataset=DataSet(data[0:split_index],input_size,output_size,shuffle_dataset)
        self._vl_dataset=DataSet(data[split_index:-1],input_size,output_size)
    def train(self,model,batch_size,classification_function=lambda x:x):
        training=TrainPhase(model,self._tr_dataset,classification_function)
        eval_func=lambda t,o:np.sum((o-t)@(o-t))*(1/self._vl_dataset.size())
        evaluation=EvaluationPhase(model,self._vl_dataset,eval_func,classification_function)
        for epoch in range(500):
            training.Work(batch_size,True)
            evaluation.Work(batch_size,True)
            print("tr loss",training.GetMetrics()["Loss"][-1],"tr precision",training.GetMetrics()["Precision"][-1],epoch+1)
        #plot_model_performance({"training":training.GetMetrics(),"validation":evaluation.GetMetrics()},"red","blue","","learning curve")
        return {"training":training.GetMetrics(),"validation":evaluation.GetMetrics()}