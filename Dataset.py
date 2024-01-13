import numpy as np

#Talke data from dataset file
def TakeDataset(path):
  Data = np.genfromtxt(path, delimiter=',', skip_header = 7,dtype=np.longdouble)[:, 1:]
  Result = [[i[:-3], i[-3:]] for i in Data]
  return Result

def TakeMonksDataSet(path):
  Data = np.genfromtxt(path, delimiter=' ',dtype=np.int32)
  Result = [[row[:1],row[1:7]] for row in Data]
  return Result

#Extract batches from Training dataset
def BatchesExtraction(TR, n, TakeResidue = True):
  Result = []
  for i in range(0, len(TR), n): Result.append(TR[i: i+n])
  if(len(TR)%n > 0 and not TakeResidue): del Result[-1]
  return Result

def one_hot_encoding(values):
  """"given an array of values creates the corresponding one-hot-encoding"""
  n=len(values)
  encoding=dict()
  for i in range(n):
    encoding[ values[i]]= np.zeros(n)
    encoding[ values[i]][i]=1
  return encoding

def convert_to_one_hot(features_names,encoding,data):
  result=[]
  for example in data:
    target=example[0]
    input_value=example[1]

    encodings=[]
    for i,feature in enumerate(features_names):
        encodings.append(encoding[feature][input_value[i]])
    result.append([np.concatenate(encodings),np.array(target)])
  return result
