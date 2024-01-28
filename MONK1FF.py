from FFNeuralNetwork import FFNeuralNetwork
from layers import InputLayer, Layer, OutputLayer

from matplotlib import pyplot as plt
import numpy as np
from Dataset import TakeMonksDataSet, convert_to_one_hot, encode_dataset_to_one_hot,DataSet
from activation_functions import sigmoid,sigmoid_prime
from loss_functions import mee_loss,mse_loss

def evaluate_performance(model,dataset,loss_function,classification_function):
    outputs=[]
    targets=[]
    for pattern in dataset.get_dataset():
        input_val=pattern[0]
        target_val=pattern[1]
        res=model.predict(input_val)
        targets.append(target_val)
        outputs.append(res)
    return loss_function(np.array(outputs),np.array(targets)),classification_function(np.array(outputs),np.array(targets))

def monk_classification(o,t):
    precision=0
    for target,value in zip(t,o):
        classification= np.array([1] if value[0] >= 0.5 else [0] ) 
        if np.array_equal(classification,target):
            precision+=1
    return precision/len(t)
monk_features={
  "a1":[1,2,3],
  "a2":[1,2,3],
  "a3":[1,2],
  "a4":[1,2,3],
  "a5":[1,2,3,4],
  "a6":[1,2],
  "class":[0,1]
}
monk_encoding= encode_dataset_to_one_hot(monk_features)

#loading monk dataset
monk1_tr=TakeMonksDataSet("FilesData/monks-1.train")
Data=convert_to_one_hot(["a1","a2","a3","a4","a5","a6"],["class"],monk_encoding,monk1_tr)
dataset=DataSet(Data,17,1)

monk1_ts=TakeMonksDataSet("FilesData/monks-1.test")
Data_ts=convert_to_one_hot(["a1","a2","a3","a4","a5","a6"],["class"],monk_encoding,monk1_ts)
ts_dataset=DataSet(Data_ts,17,1)

nn=FFNeuralNetwork(
    InputLayer(17,np.eye(17),lambda x:x,lambda x:1),
    [
        Layer(5,np.random.rand(5,17)*0.5),
    ],
    OutputLayer(1,np.random.rand(1,5),sigmoid,sigmoid_prime)
)


print(f"beginning training with {dataset.size()} pattern, and {ts_dataset.size()} pattern in the vl set")
training_performance=[]
validation_performance=[]
validation_performance_prec=[]
training_performance_prec=[]

for epoch in range(600):
    nn.fit(dataset.get_dataset(),0.5,0.0,0.8,dataset.size())

    tr_loss,tr_prec=evaluate_performance(nn,dataset,mse_loss,monk_classification)
    vl_loss,vl_prec=evaluate_performance(nn,ts_dataset,mse_loss,monk_classification)

    training_performance.append(tr_loss)
    training_performance_prec.append(tr_prec)
    validation_performance.append(vl_loss)
    validation_performance_prec.append(vl_prec)

plt.plot(training_performance,label="training loss")
plt.plot(validation_performance,label="validation loss")
plt.legend()
plt.show()

plt.plot(training_performance_prec,label="training precision")
plt.plot(validation_performance_prec,label="validation precision")
plt.legend()
plt.show()