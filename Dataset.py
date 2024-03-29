import numpy as np

monk_features={
  "a1":[1,2,3],
  "a2":[1,2,3],
  "a3":[1,2],
  "a4":[1,2,3],
  "a5":[1,2,3,4],
  "a6":[1,2],
  "class":[0,1]
}

# Take data from dataset file
def TakeCupDataset(path):
    Data = np.genfromtxt(path, delimiter=',', skip_header=7, dtype=np.float64)[:, 1:]
    Result = np.array([[i[:-3], i[-3:]] for i in Data], dtype=object)
    return Result

#Dataset class
class DataSet:
    def __init__(self, data, input_size, output_size, shuffle_dataset=False):
        for row in data:
            if row[0].size != input_size or row[1].size != output_size:
                raise ValueError(
                    "the size for the input and the output must be matched by every row in the dataset")
        self.__data = data
        self.__size = len(data)
        self.__input_size = input_size
        self.__output_size = output_size
        self.__batch_index = 0
        self.__shuffle_dataset = shuffle_dataset

    def __getitem__(self, index):
        if index > (self.__size - 1) or index < 0:
            raise ValueError("error index out of bound")
        return self.__data[index]
      
    #Number of data
    def size(self):
        return self.__size
      
    #Set dimension foreach batch
    def set_batch_index(self, index):
        if index < 0 or index >= self.__size:
            raise ValueError("index must be within dataset size")
        self.__batch_index = index
        
    #Get batches of next epcoh, where n is dimension of batches
    def next_epoch(self, n, take_rest=False):
        batches = []
        i = 0
        while i < self.__size:
            batches.append(self.__data[i:i+n])
            i += n
        if take_rest and i > self.__size:
            batches.append(self.__data[i-n:])
        return batches
    
    #Input size of data
    def input_size(self):
        return self.__input_size
      
    #Output size of data
    def output_size(self):
        return self.__output_size
      
    #Get data of dataset
    def get_dataset(self):
        return self.__data

    def __str__(self):
        Result = f"dataset size:{self.__size}, input size:{self.__input_size}, output size:{self.__output_size}\n"
        for i in range(self.__data):
            Result += f"{str(self.__data[i])} {i}\n"
        return Result

#Extract data form monk dataset
def TakeMonksDataSet(path):
    Data = np.genfromtxt(path, delimiter=' ', dtype=np.int32)
    Result = np.array([[row[:1], row[1:7]] for row in Data], dtype=object)
    return Result

# Extract batches from Training dataset
def BatchesExtraction(Dataset, n):
    l = int(len(Dataset)/n)
    print(Dataset[:l])
    result = np.reshape(np.array(Dataset[:l], dtype=object), (-1, n))
    return result
  
# Convert every value to own one-hot encoding value
def one_hot_encoding(values):
    """"given an array of values creates the corresponding one-hot-encoding"""
    n = len(values)
    encoding = dict()
    for i in range(n):
        encoding[values[i]] = np.zeros(n)
        encoding[values[i]][i] = 1
    return encoding

# Returns list of association of one hot - target values  (?)
def convert_to_one_hot(features_names, target_names, encoding, data):
    result = []
    for example in data:
        target = example[0]
        input_value = example[1]

        encodings = []
        for i, input_feature in enumerate(features_names):
            encodings.append(encoding[input_feature][input_value[i]])
        result.append([np.concatenate(encodings), np.array(target)])
    return result

#Convert all data in own one hot encoding values
def encode_dataset_to_one_hot(features):
    features_encoding = dict()

    for k, v in features.items():
        features_encoding[k] = one_hot_encoding(v)

    return features_encoding
