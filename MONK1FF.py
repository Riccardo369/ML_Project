from FFNeuralNetwork import FFNeuralNetwork
from layers import InputLayer, Layer, OutputLayer

from matplotlib import pyplot as plt
import numpy as np
from Dataset import TakeMonksDataSet, convert_to_one_hot, encode_dataset_to_one_hot, DataSet, monk_features
from activation_functions import sigmoid, sigmoid_prime
from loss_functions import mee_loss, mse_loss
from parameter_grid import ParameterGrid
from utils import evaluate_performance, monk_classification

import time

monk_encoding= encode_dataset_to_one_hot(monk_features)

#loading monk dataset for training
monk1_tr=TakeMonksDataSet("FilesData/monks-1.train")
Data=convert_to_one_hot(["a1","a2","a3","a4","a5","a6"],["class"],monk_encoding,monk1_tr)
dataset=DataSet(Data,17,1)

#Loading monk for validation
monk1_ts=TakeMonksDataSet("FilesData/monks-1.test")
Data_ts=convert_to_one_hot(["a1","a2","a3","a4","a5","a6"],["class"],monk_encoding,monk1_ts)
ts_dataset=DataSet(Data_ts,17,1)

#Start training
print(f"beginning training with {dataset.size()} pattern, and {ts_dataset.size()} pattern in the vl set")

STOCHASTIC=1
BATCH=dataset.size()

best_model_state=None
best_model_performance=np.inf
best_model_index=-1

grid=ParameterGrid({
    "learning_rate": np.array([0.45]),
    "weight_decay": np.array([0]),
    "momentum": np.array([0.55]*10),
    "batch_size": np.array([BATCH]),
  })

TitlePlot = "MONK1 batch"

TrainFinalErrorOutputs = []
ValidationFinalErrorOutputs = []
TrainFinalAccuracyOutputs = []
ValidationFinalAccuracyOutputs = []

for i in range(grid.get_size()):
    hyperparameters=grid[i]

    training_performance=[]
    validation_performance=[]
    validation_performance_prec=[]
    training_performance_prec=[]
    
    #Create new model
    nn = FFNeuralNetwork(
        InputLayer(17,np.eye(17),lambda x:x,lambda x:1),
        [
            Layer(5,np.random.rand(5,17)*0.5),
        ],
        OutputLayer(1,np.random.rand(1,5),sigmoid, sigmoid_prime)
    )

    learning_rate = hyperparameters["learning_rate"]
    weight_decay = hyperparameters["weight_decay"]
    momentum = hyperparameters["momentum"]
    batch_size = hyperparameters["batch_size"]

    for epoch in range(400):
        
        nn.fit(dataset.get_dataset(),
            learning_rate,
            weight_decay,
            momentum,
            batch_size)
        
        tr_loss, tr_prec=evaluate_performance(nn, dataset, mse_loss,monk_classification)
        vl_loss, vl_prec=evaluate_performance(nn, ts_dataset,mse_loss,monk_classification)
        
        training_performance.append(tr_loss)
        training_performance_prec.append(tr_prec*100)
        validation_performance.append(vl_loss)
        validation_performance_prec.append(vl_prec*100)
        
    TrainFinalErrorOutputs.append(training_performance[-1])
    ValidationFinalErrorOutputs.append(validation_performance[-1])
    
    TrainFinalAccuracyOutputs.append(training_performance_prec[-1])
    ValidationFinalAccuracyOutputs.append(validation_performance_prec[-1])
    
    #Show plots
    
    plt.title(TitlePlot, fontsize=16)
    plt.plot(training_performance, label="training", color="red")
    plt.plot(validation_performance, label="validation", color="blue")
    plt.title(f"{TitlePlot} MSE")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    
    plt.savefig(f'Plot graphic/{TitlePlot} MSE {i+1}.png')
    #plt.show()
    plt.clf()
    
    plt.title(TitlePlot, fontsize=16)
    plt.plot(training_performance_prec, label="training", color="orange")
    plt.plot(validation_performance_prec, label="validation", color="purple")
    plt.title(f"{TitlePlot} Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("%")
    plt.legend()
    
    plt.savefig(f'Plot graphic/{TitlePlot} Accuracy {i+1}.png')
    #plt.show()
    plt.clf()
    
    print(f'{TitlePlot}, {grid[i]}')

print(f"Train loss = Mean: {np.mean(np.array(TrainFinalErrorOutputs))}, Variance = {np.var(np.array(TrainFinalErrorOutputs))}")
print(f"Validation loss = Mean: {np.mean(np.array(ValidationFinalErrorOutputs))}, Variance = {np.var(np.array(ValidationFinalErrorOutputs))}")

print(f"Train accuracy = Mean: {np.mean(np.array(TrainFinalAccuracyOutputs))}, Variance = {np.var(np.array(TrainFinalAccuracyOutputs))}")
print(f"Validation accuracy = Mean: {np.mean(np.array(ValidationFinalAccuracyOutputs))}, Variance = {np.var(np.array(ValidationFinalAccuracyOutputs))}")