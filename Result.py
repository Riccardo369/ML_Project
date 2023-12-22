import matplotlib.pyplot as plt
import numpy as np

def Graph(MetricsData: dict, Colors, LabelX, Title):
  DataList = list(MetricsData.values())
  if(sum(map(len, DataList)))/len(DataList) != len(DataList[0]): raise ValueError("All metrics must be on same size")

  Xdata = np.linspace(0, len(DataList[0]), len(DataList[0]))

  r = 0

  for i in MetricsData.keys():
    plt.plot(Xdata, MetricsData[i], linestyle='-', color = Colors[r], label=i)
    r += 1

  plt.title(Title)
  plt.xlabel(LabelX)
  plt.ylabel('Percentual')

  if(max(map(lambda i: max(MetricsData[i]), MetricsData.keys())) <= 100):
    plt.ylim([0, 100])

  plt.legend()
  plt.show()