from Evaluation import *

Dataset = [[[11, 21, 31, 41], [51, 61]],
           [[12, 22, 32, 42], [52, 62]],
           [[13, 23, 33, 43], [53, 63]],
           [[14, 24, 34, 44], [54, 64]],
           [[15, 25, 35, 45], [55, 65]],
           [[16, 26, 36, 45], [56, 66]],
           [[17, 27, 37, 47], [57, 67]],
           [[18, 28, 38, 48], [58, 68]]]

CV = CrossValidation(Dataset, 1)

TR, VL = CV.GetKFold(0)

print(TR)
print("")
print(VL)