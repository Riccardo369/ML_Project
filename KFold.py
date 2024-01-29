import numpy as np
from Dataset import DataSet
from Phase import EvaluationPhase, TrainPhase


class KFold:
    def __init__(self, data, input_size, output_size, shuffle_dataset, folds_number):
        self._folds_number = folds_number
        self._folds = np.array_split(data, folds_number)
        self._input_size = input_size

        self._output_size = output_size

    # Train the model considering 
    def train(self, model, batch_size, classification_function=lambda x: x):
        best_fold_index = 0
        best_fold_performance = dict()
        for i, fold in enumerate(self._folds):
            tr = DataSet(fold, self._input_size, self._output_size, False)
            vl = DataSet(np.concatenate(
                (*self._folds[:i+1], *self._folds[i+2:]), dtype=object), self._input_size, self._output_size, False)
            training = TrainPhase(model, tr, classification_function)
            def eval_func(t, o): return np.sum((o-t)**2)*(1/batch_size)
            evaluation = EvaluationPhase(
                model, vl, eval_func, classification_function)
            print(f"beginning epoch no.{i+1}")
            for epoch in range(1000):
                training.Work(batch_size, False)
                evaluation.Work(batch_size, False)

            if evaluation.GetMetrics()["Loss"][-1] < best_fold_performance["validation"]["Loss"][-1]:
                best_fold_performance = training.GetMetrics()
                best_fold_index = i

            print("epoch ", i+1, "tr precision ", training.GetMetrics()
                  ["Precision"][-1], "\tloss ", training.GetMetrics()["Loss"][-1], epoch+1)
        return {"training": self._folds[best_fold_index]["training"], "validation": evaluation.GetMetrics()}
