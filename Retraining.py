from Dataset import DataSet
from Phase import TrainPhase

# Retraining phase
class Retraining:
    def __init__(self, data, input_size, output_size) -> None:
        self._tr_dataset = DataSet(data, input_size, output_size)
        
    #Re training the model with batch_size selected
    def train(self, model, batch_size, classification_function=lambda x: x):
        initial_state = model.ExtractLearningState()
        training = TrainPhase(model, self._tr_dataset, classification_function)
        for epoch in range(500):
            training.Work(batch_size, True)
        final_learning_state = model.ExtractLearningState()
        model.LoadLearningState(initial_state)
        return {
            "model_state": final_learning_state,
            "model_performance": training.GetMetrics()
        }
