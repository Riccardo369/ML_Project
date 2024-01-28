import numpy as np


sigmoid=lambda x:1/(1 + np.exp(-x)) 
sigmoid_prime=lambda x:sigmoid(x)*(1-sigmoid(x)) 