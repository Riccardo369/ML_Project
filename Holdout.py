from Dataset import DataSet
from Phase import EvaluationPhase, TrainPhase


class Holdout:
    def __init__(self,data,input_size,output_size,shuffle_dataset,split_index) -> None:
        self._tr_dataset=DataSet(data[0:split_index],input_size,output_size,shuffle_dataset)
        self._vl_dataset=DataSet(data[split_index:-1],input_size,output_size)
    def train(self,model,batch_size,classification_function=lambda x:x):
        training=TrainPhase(model,self._tr_dataset,classification_function)
        evaluation=EvaluationPhase(model,self._vl_dataset,classification_function)
        for epoch in range(50):
            training.Work(batch_size,True)
            print("tr precision ",training.GetMetrics()["Precision"][-1],"\tloss ",training.GetMetrics()["Loss"][-1],epoch+1)
            evaluation.Work(batch_size,True)
        return {"training":training.GetMetrics(),"validation":evaluation.GetMetrics()}