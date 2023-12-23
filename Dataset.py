import numpy as np

def TakeDataset(path):
  Data = np.genfromtxt(path, delimiter=',', skip_header = 7,dtype=np.float64)[:, 1:]
  Result = [[i[:-3], i[-3:]] for i in Data]
  return Result

def BatchesExtraction(TR, n, TakeResidue = True):
  Result = []
  for i in range(0, len(TR), n): Result.append(TR[i: i+n])
  if(len(TR)%n > 0 and not TakeResidue): del Result[-1]
  return Result