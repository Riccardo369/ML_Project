from FFNeuralNetwork import FFNeuralNetwork
from layers import InputLayer, Layer, OutputLayer

from matplotlib import pyplot as plt
import numpy as np
from Dataset import TakeMonksDataSet, convert_to_one_hot, encode_dataset_to_one_hot,DataSet,monk_features,TakeCupDataset
from activation_functions import sigmoid,sigmoid_prime
from loss_functions import mee_loss,mse_loss
from parameter_grid import ParameterGrid
from utils import evaluate_performance, k_fold, monk_classification
monk_encoding= encode_dataset_to_one_hot(monk_features)

#loading monk dataset
cup_tr=TakeCupDataset("FilesData/ML-CUP23-TR.csv")
dataset=DataSet(cup_tr,10,3)
cup_vl=TakeCupDataset("FilesData/ML-CUP23-TS.csv")
vl_dataset=DataSet(cup_tr,10,3)

nn=FFNeuralNetwork(
    InputLayer(10,np.eye(10),lambda x:x,lambda x:1),
    [
        Layer(5,np.random.rand(5,10)*0.5),
        Layer(5,np.random.rand(5,5)*0.5),
    ],
    OutputLayer(3,np.random.rand(3,5),lambda x:x,lambda x:1)
)

print(f"beginning training with {dataset.size()} pattern, and {vl_dataset.size()} pattern in the vl set")


#config parameters
learning_rate=0.6
weight_decay=0.0
momentum=0.3
batch_size=dataset.size()
folds_number=5

folds = k_fold(dataset.get_dataset(),folds_number)
initial_state=nn.dump()

STOCHASTIC=1
BATCH=-1

best_model_state=None
best_model_performance=np.inf
best_model_index=-1

grid=ParameterGrid({
    "learning_rate":np.array([0.1]),
    "weight_decay":np.array([0]),
    "momentum":np.array([0.02]),
    "batch_size":np.array([STOCHASTIC,BATCH]),
  })

for i in range(grid.get_size()):
    print("hyperparameters combination no.",i+1,"=",grid[i])
    hyperparameters=grid[i]
    model_performance=0
    for j,(tr,vl) in enumerate(folds):
        print("fold no.",j+1)
        nn.load(initial_state)
        training_mse=[]
        validation_mse=[]
        validation_mee=[]
        training_mee=[]
        for epoch in range(300):
            nn.fit(tr,
                learning_rate=hyperparameters["learning_rate"],
                weight_decay=hyperparameters["weight_decay"],
                momentum=hyperparameters["momentum"],
                batch_size= len(tr) if hyperparameters["batch_size"] < 0 
                                    else hyperparameters["batch_size"])
            tr_mse,tr_mee=evaluate_performance(nn,DataSet(tr,10,3),mse_loss,mee_loss)
            vl_mse,vl_mee=evaluate_performance(nn,DataSet(vl,10,3),mse_loss,mee_loss)
            training_mse.append(tr_mse)
            training_mee.append(tr_mee)
            validation_mse.append(vl_mse)
            validation_mee.append(vl_mee)
        model_performance+=vl_mse
    model_performance/=folds_number
    print("final performance",model_performance)
    if model_performance < best_model_performance:
        best_model_performance=model_performance
        best_model_index=i
    
training_mse=[]
validation_mse=[]
validation_mee=[]
training_mee=[]
best_hyperparameters=grid[best_model_index]
nn.load(initial_state)
for epoch in range(300):
            nn.fit(dataset.get_dataset(),
                learning_rate=best_hyperparameters["learning_rate"],
                weight_decay=best_hyperparameters["weight_decay"],
                momentum=best_hyperparameters["momentum"],
                batch_size= dataset.size() if best_hyperparameters["batch_size"] < 0 
                                    else best_hyperparameters["batch_size"])
            tr_mse,tr_mee=evaluate_performance(nn,dataset,mse_loss,mee_loss)
            vl_mse,vl_mee=evaluate_performance(nn,vl_dataset,mse_loss,mee_loss)
            training_mse.append(tr_mse)
            training_mee.append(tr_mee)
            validation_mse.append(vl_mse)
            validation_mee.append(vl_mee)
    

plt.plot(training_mse,label="training MSE")
plt.plot(validation_mse,label="test MSE")
plt.legend()
plt.show()

plt.plot(training_mee,label="training MEE")
plt.plot(validation_mee,label="test MEE")
plt.legend()
plt.show()