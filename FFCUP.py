from FFNeuralNetwork import FFNeuralNetwork
from layers import InputLayer, Layer, OutputLayer

from matplotlib import pyplot as plt
import numpy as np
from Dataset import TakeMonksDataSet, convert_to_one_hot, encode_dataset_to_one_hot,DataSet,TakeCupDataset
from activation_functions import sigmoid,sigmoid_prime
from loss_functions import mee_loss,mse_loss
from parameter_grid import ParameterGrid
from utils import evaluate_performance, k_fold, monk_classification

#loading monk dataset
cup_tr=TakeCupDataset("FilesData/ML-CUP23-TR.csv")
dataset=DataSet(cup_tr,10,3)
cup_vl=TakeCupDataset("FilesData/ML-CUP23-TS.csv")
vl_dataset=DataSet(cup_tr,10,3)

nn=FFNeuralNetwork(
    InputLayer(10,np.eye(10),lambda x:x,lambda x:1),
    [
        Layer(10,np.random.rand(10,10)*0.5),
        Layer(10,np.random.rand(10,10)*0.5),
        Layer(10,np.random.rand(10,10)*0.5),
    ],
    OutputLayer(3,np.random.rand(3,10),lambda x:x,lambda x:1)
)

print(f"beginning training with {dataset.size()} pattern, and {vl_dataset.size()} pattern in the vl set")


#config parameters
learning_rate=0.6
weight_decay=0.0
momentum=0.8
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
    "learning_rate":np.array([0.01,0.001,0.0001]),
    "weight_decay":np.array([0.0]),
    "momentum":np.array([0.0,0.1,0.4,0.8]),
    "batch_size":np.array([STOCHASTIC]),
  })

patience=100
max_epochs=10000
error_percent=1.1

for i in range(grid.get_size()):
    print("hyperparameters combination no.",i+1,"=",grid[i])
    hyperparameters=grid[i]
    mee_performance=0
    plt.figure(figsize=(12,6))
    for j,(tr,vl) in enumerate(folds):
        print("fold no.",j+1)
        nn.load(initial_state)
        training_mse=[]
        validation_mse=[]
        validation_mee=[]
        training_mee=[]

        old_vl_mee=np.inf
        plt.clf()
        plt.title(f"comb no.{i+1}, internal set {folds_number} fold with values "+" ".join([ f"{k}={v}" for k,v in grid[i].items()]))
        epoch=0
        stop_epochs=0
        while (True):
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
            plt.plot(training_mee,label="training MEE",color="blue")
            plt.plot(training_mse,label="training MSE",color="red")
            plt.plot(validation_mee,label="validation MEE",color="blue")
            plt.plot(validation_mse,label="training MSE",color="red")
            
            if (old_vl_mee/vl_mee)>error_percent:
                stop_epochs+=1
            else:
                stop_epochs=0
            if stop_epochs >= patience or epoch >= max_epochs:
                 break
            epoch+=1
                
                
        
        mee_performance+=vl_mee
    mee_performance/=folds_number
    print("final performance",mee_performance)
    if mee_performance < best_model_performance:
        best_model_performance=mee_performance
        best_model_index=i
    plt.savefig(f"Plot Graphic/CUP/{folds_number}-fold {" ".join([ f"{k}={v}" for k,v in grid[i].items()])}.png")
    
training_mse=[]
validation_mse=[]
validation_mee=[]
training_mee=[]
best_hyperparameters=grid[best_model_index]
nn.load(initial_state)
for epoch in range(1000):
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
"""     
plt.clf()
plt.plot(training_mse,label="training MSE")
plt.plot(validation_mse,label="test MSE")
plt.legend()
plt.show()

plt.clf()
plt.plot(training_mee,label="training MEE")
plt.plot(validation_mee,label="test MEE")
plt.legend()
plt.show() """