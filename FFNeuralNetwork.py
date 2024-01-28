import numpy as np
from Layer import InputLayer, Layer, OutputLayer


class FFNeuralNetwork:
    
    def __init__(self,input_layer: InputLayer, hidden_layers: list[Layer], output_layer: OutputLayer) -> None:
        
        self._input_layer = input_layer
        self._hidden_layers = hidden_layers
        self._output_layer = output_layer
        self._layers_number = 2+len(hidden_layers)
        
    def update(self,learning_rate,momentum,weight_decay,batch_size):
        
        self._input_layer.update(learning_rate, momentum, weight_decay, batch_size)
        
        for layer in self._hidden_layers: 
            layer.update(learning_rate, momentum,weight_decay, batch_size)
            
        self._output_layer.update(learning_rate, momentum, weight_decay, batch_size)

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
        
    def fit(self,epoch, learning_rate, weight_decay, momentum,batch_size):
        
        minibatches= np.array_split(np.array(epoch,dtype=object), int(len(epoch)/batch_size))
        
        for mb in minibatches:
            
            for pattern in mb:
                
                input_val=pattern[0]
                target_val=pattern[1]
                
                #forward phase
                result_val=self.predict(input_val)
                
                #backward phase
                self.backward(target_val,result_val,learning_rate,momentum,weight_decay)
                
            self.update(learning_rate,momentum,weight_decay,batch_size)