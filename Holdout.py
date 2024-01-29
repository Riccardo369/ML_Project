import numpy as np
from Dataset import DataSet
from Phase import EvaluationPhase, TrainPhase
from Result import plot_model_performance


class Holdout:
    def __init__(self, tr_data, vl_data, input_size, output_size, shuffle_dataset) -> None:
        self._tr_dataset = DataSet(tr_data, input_size, output_size, shuffle_dataset)
        self._vl_dataset = DataSet(vl_data, input_size, output_size)
        
    #Train model on train dataset and validate model on validation dataset
    def train(self, model, batch_size, classification_function=lambda x: x):
        print(f"beginning hold out {self._tr_dataset.size()} {self._vl_dataset.size()}")
        
        training = TrainPhase(model, self._tr_dataset, classification_function)

        def eval_func(t, o): return np.sum((o-t)@(o-t))*(1/self._vl_dataset.size())
        
        evaluation = EvaluationPhase(model, self._vl_dataset, eval_func, classification_function)
        
        for epoch in range(200):
            training.Work(batch_size, True)
            evaluation.Work(batch_size, True)
            
            print("tr loss", training.GetMetrics()["Loss"][-1], "tr precision", training.GetMetrics()["Precision"][-1],
                  "vl loss", evaluation.GetMetrics()["Loss"][-1], "vl precision", evaluation.GetMetrics()["Precision"][-1], epoch+1,)
            
        # plot_model_performance({"training":training.GetMetrics(),"validation":evaluation.GetMetrics()},"red","blue","","learning curve")
        return {"training": training.GetMetrics(), "validation": evaluation.GetMetrics()}
