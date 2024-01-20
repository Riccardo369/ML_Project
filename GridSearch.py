import numpy as np
from Result import plot_model_performance


class GridSearch:
    def __init__(self,model,batch_size,grid,training_strategy) -> None:
        self._model=model
        self._batch_size=batch_size
        self._grid=grid
        self._training_strategy=training_strategy
    def search(self,classification_function,
                    bias_update_function=lambda x:0):
        best_performance={
            "training":{"Loss":[np.inf],"Precision":[np.inf]},
            "validation":{"Loss":[np.inf],"Precision":[np.inf]},
        }
        best_performance_index=-1
        final_learning_state=None
        InitialState=self._model.ExtractLearningState()
        for i in range(self._grid.get_size()):
            #reset model to initial state
            self._model.LoadLearningState(InitialState)
            #current set of hyperparameters
            hyperparameters=self._grid[i]
            print(f"hyperparameters:{hyperparameters}")
            #set new hyperparameters
            def weights_update_function(weights,GradientLoss):
                w=np.array(weights)
                g=np.array(GradientLoss)
                return w+(1/self._batch_size)*g*hyperparameters["learning_rate"]-hyperparameters["weight_decay"]*w
            self._model.SetWeightsUpdateFunction(weights_update_function)
            def bias_update_function(old_bias,gradient_value):
                return old_bias + (1/self._batch_size)*gradient_value*hyperparameters["learning_rate"]-hyperparameters["weight_decay"]*old_bias
            for n in self._model.GetAllNeurons():
                n.SetUpdateBiasFunction(bias_update_function)
            #TODO aggiornamento della bias
            performance=self._training_strategy.train(self._model,self._batch_size,classification_function)
            #plot_model_performance(performance,"red","blue","epochs",f"best model with hyperparameters {self._grid[i]}")
            #evaluate performance
            if performance["validation"]["Loss"][-1] < best_performance["validation"]["Loss"][-1]:
                final_learning_state=self._model.ExtractLearningState()
                best_performance_index=i
                best_performance=performance
        print(f"grid search done, bestmodel: np.{best_performance_index} with hyperparameters {self._grid[best_performance_index]}")
        return {
            "model_state": final_learning_state,
            "model_performance":best_performance,
            "index":best_performance_index,
        }