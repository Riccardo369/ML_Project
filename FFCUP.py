from FFNeuralNetwork import FFNeuralNetwork
from layers import InputLayer, Layer, OutputLayer

from matplotlib import pyplot as plt
import numpy as np
from Dataset import TakeMonksDataSet, convert_to_one_hot, encode_dataset_to_one_hot,DataSet,monk_features,TakeCupDataset
from activation_functions import sigmoid,sigmoid_prime
from loss_functions import mee_loss,mse_loss
from utils import evaluate_performance, k_fold, monk_classification
monk_encoding= encode_dataset_to_one_hot(monk_features)

#loading monk dataset
cup_tr=TakeCupDataset("FilesData/monks-2.train")
dataset=DataSet(cup_tr,10,3)
cup_vl=TakeCupDataset("FilesData/monks-2.train")
vl_dataset=DataSet(cup_tr,10,3)

nn=FFNeuralNetwork(
    InputLayer(10,np.eye(10),lambda x:x,lambda x:1),
    [
        Layer(5,np.random.rand(5,10)*0.5),
    ],
    OutputLayer(3,np.random.rand(3,5),sigmoid,sigmoid_prime)
)

print(f"beginning training with {dataset.size()} pattern, and {vl_dataset.size()} pattern in the vl set")

training_mse=[]
validation_mse=[]
validation_mee=[]
training_mee=[]
#config parameters
learning_rate=0.6
weight_decay=0.0
momentum=0.3
batch_size=dataset.size()
folds_number=5

folds = k_fold(dataset.get_dataset(),folds_number)
initial_state=nn.dump()
for tr,vl in folds:
    nn.load(initial_state)
    for epoch in range(300):
        nn.fit(dataset.get_dataset(),
            learning_rate,
            weight_decay,
            momentum,
            batch_size)
        tr_mse,tr_mee=evaluate_performance(nn,dataset,mse_loss,mee_loss)
        vl_mse,vl_mee=evaluate_performance(nn,vl_dataset,mse_loss,mee_loss)
    training_mse.append(tr_mse)
    training_mee.append(tr_mee)
    validation_mse.append(vl_mse)
    validation_mee.append(vl_mee)


plt.plot(training_mse,label="training MSE")
plt.plot(validation_mse,label="validation MSE")
plt.legend()
plt.show()

plt.plot(training_mee,label="training MEE")
plt.plot(validation_mee,label="validation MEE")
plt.legend()
plt.show()