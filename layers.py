import numpy as np


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
    def dump(self):
        res=dict()
        for k,v in vars(self).items():
            if isinstance(v,np.ndarray):
                res[k]=np.copy(v)
            else:
                res[k]=v
        return res
    def load(self,state):

        raise NotImplemented


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