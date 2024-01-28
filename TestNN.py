from matplotlib import pyplot as plt
import numpy as np
from Dataset import TakeMonksDataSet, convert_to_one_hot, encode_dataset_to_one_hot,DataSet

class AbstractLayer:
    def __init__(self,layer_dim:int,
                      weights:np.ndarray,
                      activation_function=lambda x: max(x,0),
                      activation_function_derivative=lambda x: 1 if x>=0 else 0,
                      biases=None
                      ) -> None:
        if biases is None:
            self._biases=np.zeros(layer_dim,dtype=np.longdouble)
        else:
            self._biases=biases
        self._layer_dim=layer_dim

        self._biases_momentum=np.zeros(layer_dim,dtype=np.longdouble)
        self._weights_momentum=np.zeros(weights.shape,dtype=np.longdouble)

        self._input_nets=np.empty(layer_dim,dtype=np.longdouble)
        self._neurons_output=np.empty(layer_dim,dtype=np.longdouble)
        self._error_signals=np.empty(layer_dim,dtype=np.longdouble)
        self._weights_mat=weights

        self._weights_grad=np.zeros(weights.shape,dtype=np.longdouble)
        self._bias_grad=np.zeros(layer_dim,dtype=np.longdouble)

        self._activation_function=activation_function 
        self._activation_function_derivative= activation_function_derivative
    def reset_grad(self):
        self._weights_grad.fill(0)
        self._bias_grad.fill(0)
    def get_output(self):
        return self._neurons_output
    def get_weights(self):
        return self._weights_mat
    def update(self,learning_rate,momentum,weight_decay,batch_size):
        self._weights_mat+= learning_rate*(1/batch_size)*self._weights_grad + momentum*self._weights_momentum - weight_decay*self._weights_mat
        self._biases+= learning_rate*(1/batch_size)*self._bias_grad + momentum*self._biases_momentum - weight_decay*self._biases

        self._weights_momentum= learning_rate*(1/batch_size)*self._weights_grad
        self._biases_momentum=learning_rate*(1/batch_size)*self._bias_grad

        self.reset_grad()
    def forward(self,values:np.ndarray):
        self._input_nets=np.dot(self._weights_mat,values) + self._biases
        self._neurons_output= np.fromiter(map(self._activation_function,self._input_nets),dtype=np.longdouble)
        return self._neurons_output
    def backward(self,signal_errors:np.ndarray,previous_outputs:np.ndarray,previous_weights:np.ndarray):
        raise NotImplemented

class Layer(AbstractLayer):
    def __init__(self,*args) -> None:
        return super().__init__(*args)
    def backward(self,signal_errors:np.ndarray,previous_outputs:np.ndarray,previous_weights:np.ndarray,learning_rate,momentum=0,weight_decay=0):
        self._error_signals=np.dot(signal_errors,previous_weights)
        self._error_signals*=np.fromiter(map(self._activation_function_derivative,self._input_nets),dtype=np.longdouble)
        #update weights
        self._weights_grad+=(np.matrix(self._error_signals).T@np.matrix(previous_outputs))
        self._bias_grad+=self._error_signals
        return self._error_signals
class InputLayer(AbstractLayer):
    def __init__(self,*args) -> None:
        return super().__init__(*args)
    def forward(self,values:np.ndarray):
        self._input_nets=values + self._biases
        self._neurons_output= np.fromiter(map(self._activation_function,self._input_nets),dtype=np.longdouble)
        return self._neurons_output
    def backward(self,signal_errors:np.ndarray,previous_weights:np.ndarray):
        self._error_signals=np.dot(signal_errors,previous_weights)
        self._error_signals*=np.array(list(map(self._activation_function_derivative,self._input_nets)))
        self._bias_grad+=self._error_signals
        return self._error_signals
    def update(self,learning_rate,momentum,weight_decay,batch_size):
        self._biases+= learning_rate*(1/batch_size)*self._bias_grad + momentum*self._biases_momentum - weight_decay*self._biases
        self._biases_momentum=learning_rate*(1/batch_size)*self._bias_grad
        self.reset_grad()
class OutputLayer(AbstractLayer):
    def __init__(self,*args) -> None:
        return super().__init__(*args)
    def backward(self,signal_errors:np.ndarray,previous_outputs:np.ndarray,learning_rate,momentum=0,weight_decay=0):
        self._error_signals=signal_errors*np.fromiter(map(self._activation_function_derivative,self._input_nets),dtype=np.longdouble)
        self._weights_grad+=(np.matrix(self._error_signals).T@np.matrix(previous_outputs))
        self._bias_grad+=self._error_signals
        return self._error_signals

sigmoid=lambda x:1/(1 + np.exp(-x)) 
sigmoid_prime=lambda x:sigmoid(x)*(1-sigmoid(x)) 
class FFNeuralNetwork:
    def __init__(self,input_layer:InputLayer,hidden_layers:list[Layer],output_layer:OutputLayer) -> None:
        self._input_layer=input_layer
        self._hidden_layers=hidden_layers
        self._output_layer=output_layer
        self._layers_number=2+len(hidden_layers)
    def update(self,learning_rate,momentum,weight_decay,batch_size):
        self._input_layer.update(learning_rate,momentum,weight_decay,batch_size)
        for layer in self._hidden_layers: 
            layer.update(learning_rate,momentum,weight_decay,batch_size)
        self._output_layer.update(learning_rate,momentum,weight_decay,batch_size)
        

    def predict(self,input_value):
        self.forward(input_value)
        return self._output_layer.get_output()
    def forward(self,input_value):
        output_value=self._input_layer.forward(input_value)
        for layer in self._hidden_layers:
            output_value=layer.forward(output_value)
        self._output_layer.forward(output_value)
    def backward(self,target,output,learning_rate,momentum,weight_decay):
        signal_errors=self._output_layer.backward(target-output,
                                                      self._hidden_layers[-1].get_output(),
                                                      learning_rate,
                                                      momentum,
                                                      weight_decay
                                                      )
        weights=self._output_layer.get_weights()
        for i,layer in reversed(self._hidden_layers[1:]):
            signal_errors=layer.backward(signal_errors,
                                             self._hidden_layers[i-1].get_output(),
                                             weights,
                                             learning_rate,
                                             momentum,
                                             weight_decay
                                             )
            weights=layer.get_weights()
        signal_errors=self._hidden_layers[0].backward(signal_errors,
                                                          self._input_layer.get_output(),
                                                          weights,
                                                          learning_rate,
                                                          momentum,
                                                          weight_decay
                                                          )
        weights=self._hidden_layers[0].get_weights()
        self._input_layer.backward(signal_errors,weights)
    def fit(self,epoch,learning_rate,weight_decay,momentum,batch_size):
        minibatches= np.array_split(np.array(epoch,dtype=object),int(len(epoch)/batch_size))
        for mb in minibatches:
            for pattern in mb:
                input_val=pattern[0]
                target_val=pattern[1]
                #forward phase
                result_val=self.predict(input_val)
                #backward phase
                self.backward(target_val,result_val,learning_rate,momentum,weight_decay)
            self.update(learning_rate,momentum,weight_decay,batch_size)


monk_features={
  "a1":[1,2,3],
  "a2":[1,2,3],
  "a3":[1,2],
  "a4":[1,2,3],
  "a5":[1,2,3,4],
  "a6":[1,2],
  "class":[0,1]
}
#create the encoding for the one hot representation
monk_encoding= encode_dataset_to_one_hot(monk_features)

#loading monk dataset
monk1_tr=TakeMonksDataSet("FilesData/monks-1.train")
#describe features and their values

#apply encoding to the dataset
Data=convert_to_one_hot(["a1","a2","a3","a4","a5","a6"],["class"],monk_encoding,monk1_tr)

dataset=DataSet(Data,17,1)
#loading monk dataset
monk1_ts=TakeMonksDataSet("FilesData/monks-1.test")
#describe features and their values

#apply encoding to the dataset
Data_ts=convert_to_one_hot(["a1","a2","a3","a4","a5","a6"],["class"],monk_encoding,monk1_ts)

ts_dataset=DataSet(Data_ts,17,1)
mse_loss= lambda o,t: 0.5*np.mean((o-t)**2)
nn=FFNeuralNetwork(
    InputLayer(17,np.eye(17),lambda x:x,lambda x:1),
    [
        Layer(5,np.random.rand(5,17)*0.5),
    ],
    OutputLayer(1,np.random.rand(1,5),sigmoid,sigmoid_prime)
)

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

training_performance=[]
validation_performance=[]
validation_performance_prec=[]
training_performance_prec=[]

def monk_classification(o,t):
    precision=0
    for target,value in zip(t,o):
        classification= np.array([1] if value[0] >= 0.5 else [0] ) 
        if np.array_equal(classification,target):
            precision+=1
    return precision/len(t)
print(f"beginning training with {dataset.size()} pattern, and {ts_dataset.size()} pattern in the vl set")
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