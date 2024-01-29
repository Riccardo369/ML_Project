from FFNeuralNetwork import FFNeuralNetwork
from layers import InputLayer, Layer, OutputLayer

from matplotlib import pyplot as plt
import numpy as np
from Dataset import TakeMonksDataSet, convert_to_one_hot, encode_dataset_to_one_hot, DataSet, monk_features
from activation_functions import sigmoid, sigmoid_prime
from loss_functions import mee_loss, mse_loss
from parameter_grid import ParameterGrid
from utils import evaluate_performance, monk_classification

monk_encoding= encode_dataset_to_one_hot(monk_features)

#loading monk dataset for trainin
monk1_tr=TakeMonksDataSet("FilesData/monks-1.train")
Data=convert_to_one_hot(["a1","a2","a3","a4","a5","a6"],["class"],monk_encoding,monk1_tr)
dataset=DataSet(Data,17,1)

#Loading monk for validation
monk1_ts=TakeMonksDataSet("FilesData/monks-1.test")
Data_ts=convert_to_one_hot(["a1","a2","a3","a4","a5","a6"],["class"],monk_encoding,monk1_ts)
ts_dataset=DataSet(Data_ts,17,1)

#Create model
nn = FFNeuralNetwork(
    InputLayer(17,np.eye(17),lambda x:x,lambda x:1),
    [
        Layer(5,np.random.rand(5,17)*0.5),
    ],
    OutputLayer(1,np.random.rand(1,5),sigmoid, sigmoid_prime)
)

#Start training
print(f"beginning training with {dataset.size()} pattern, and {ts_dataset.size()} pattern in the vl set")

initial_state = nn.dump()


STOCHASTIC=1
BATCH=-1

best_model_state=None
best_model_performance=np.inf
best_model_index=-1
grid=ParameterGrid({
    "learning_rate":np.array([0.01,0.02,0.1]),
    "weight_decay":np.array([0]),
    "momentum":np.array([0.0]),
    "batch_size":np.array([STOCHASTIC]),
  })

for i in range(grid.get_size()):
    hyperparameters=grid[i]

    training_performance=[]
    validation_performance=[]
    validation_performance_prec=[]
    training_performance_prec=[]
    
    nn.load(initial_state)

    learning_rate = hyperparameters["learning_rate"]
    weight_decay = hyperparameters["weight_decay"]
    momentum = hyperparameters["momentum"]
    batch_size = dataset.size() if hyperparameters["batch_size"] < 0 else hyperparameters["batch_size"]

    for epoch in range(300):
        
        nn.fit(dataset.get_dataset(),
            learning_rate,
            weight_decay,
            momentum,
            batch_size)
        
        tr_loss, tr_prec=evaluate_performance(nn, dataset, mse_loss,monk_classification)
        vl_loss, vl_prec=evaluate_performance(nn, ts_dataset,mse_loss,monk_classification)
        
        training_performance.append(tr_loss)
        training_performance_prec.append(tr_prec)
        validation_performance.append(vl_loss)
        validation_performance_prec.append(vl_prec)

    #Show plots
    plt.plot(training_performance,label="training loss")
    plt.plot(validation_performance,label="validation loss")

    plt.plot(training_performance_prec,label="training precision")
    plt.plot(validation_performance_prec,label="validation precision")
    plt.legend()

    plt.title(f'MONK1,{ " ".join([ f"{k}={v} " for k,v in grid[i].items()])}')
    
    plt.savefig(f'Plot graphic/MONK1, {" ".join([ f"{k}={v} " for k,v in grid[i].items()])}.png')
    
    print(f'MONK1, {grid[i]}')

    #plt.show()
    
    plt.clf()