import numpy as np


class ParameterGrid:
    def __init__(self,parameters) :
        self.__parameters_names=list(parameters.keys())
        self.__grid=np.array(np.meshgrid(*parameters.values())).T.reshape(-1, len(parameters))
    def get_size(self):
        size,_=self.__grid.shape
        return size
    def get_shape(self):
        return self.__grid.shape
    def get_parameters(self):
        return self.__parameters_names
    def __getitem__(self,index):
        item = self.__grid[index]
        return { self.__parameters_names[j]:value for j,value in enumerate(item)}
    def __str__(self):
        Result = f"hyperparmeters:{self.__parameters_names}, shape:{self.get_shape()}\n"
        for i in range(self.__grid): Result += f"{str(self.__grid[i])} {i}\n"
        return Result