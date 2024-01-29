import numpy as np
from GridSearch import GridSearch
from Phase import EvaluationPhase
from Result import plot_model_performance

#Simulator class
class Simulator:
    def __init__(self, model, tr_dataset, vl_dataset, grid, batch_size, training_strategy, retraining_strategy=None, classification_function=lambda x: x) -> None:
        self._model = model
        self._tr_dataset = tr_dataset
        self._vl_dataset = vl_dataset
        self._batch_size = batch_size
        self._training_strategy = training_strategy
        self._grid = grid
        self._grid_search = GridSearch(
            self._model, self._batch_size, self._grid, self._training_strategy)
        self._classification_function = classification_function
        self._retraining_strategy = retraining_strategy
        
    #Run the simulator
    def run(self):
        print("simulator,running search")
        result = self._grid_search.search(self._classification_function)
        plot_model_performance(result["model_performance"], "red",
                               "blue", "", "final retraining performance of the best model")
        print("simulator,search done")
        best_params = self._grid[result["index"]]

        def weights_update_function(weights, GradientLoss):
            return weights+(1/self._tr_dataset.size())*GradientLoss*best_params["learning_rate"]-best_params["weight_decay"]*weights

        self._model.SetWeightsUpdateFunction(weights_update_function)

        def bias_update_function(old_bias, gradient_value):
            return old_bias + (1/self._tr_dataset.size())*gradient_value*best_params["learning_rate"]-best_params["weight_decay"]*old_bias
        for n in self._model.GetAllNeurons():
            n.SetUpdateBiasFunction(bias_update_function)
            
        # load the best model
        if self._retraining_strategy != None:
            retrain_result = self._retraining_strategy.train(
                self._model, self._batch_size, self._classification_function)
            self._model.LoadLearningState(retrain_result["model_state"])
        else:
            self._model.LoadLearningState(result["model_state"])
            
        # perform the last evaluation against the validation set
        evaluation = EvaluationPhase(self._model, self._vl_dataset, lambda t, o: np.sum(
            (o-t)@(o-t))*(1/self._vl_dataset.size()), self._classification_function)
        evaluation.Work(self._batch_size, True)
        return evaluation.GetMetrics()["Loss"], evaluation.GetMetrics()["Precision"], result["model_performance"]