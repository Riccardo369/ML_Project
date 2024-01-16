import Dataset

raw_data= Dataset.TakeMonksDataSet("FilesData/monks-1.train")

dataset=Dataset.DataSet(raw_data,6,1)
print(dataset)