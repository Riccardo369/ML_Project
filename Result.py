import matplotlib.pyplot as plt
import numpy as np

def plot_model_performance(MetricsData: dict, loss_color, precision_color, LabelX, Title):
  DataList = list(MetricsData.values())
  if(sum(map(len, DataList)))/len(DataList) != len(DataList[0]): raise ValueError("All metrics must be on same size")

  fig,(ax_loss,ax_prec)=plt.subplots(1,2)
  ax_loss.plot(np.arange(len(MetricsData["Loss"])), MetricsData["Loss"], linestyle='-', color = loss_color, label="Loss")
  ax_prec.plot(np.arange(len(MetricsData["Precision"])), MetricsData["Precision"], linestyle='-', color = precision_color, label="Precision")

  plt.title(Title)
  plt.xlabel(LabelX)
  plt.ylabel('Percentual')

  plt.legend()
  plt.show()

def Graph(MetricsData: dict, Colors, LabelX, Title):
  DataList = list(MetricsData.values())
  if(sum(map(len, DataList)))/len(DataList) != len(DataList[0]): raise ValueError("All metrics must be on same size")

  #Xdata = np.arange(len(DataList))

  r = 0

  for label,data,color in zip(MetricsData.keys(),MetricsData.values(),Colors):
    x_data=np.arange(len(data))
    plt.plot(x_data, data, linestyle='-', color = color, label=label)
    r += 1

  plt.title(Title)
  plt.xlabel(LabelX)
  plt.ylabel('Percentual')

  plt.legend()
  plt.show()

def KFoldGraph(MetricsData: dict, Colors, LabelX, Title):
  XAxis= np.linspace(0, len(MetricsData["training"][0]),len(MetricsData["training"][0]))
  fig, axs= plt.subplots(len(MetricsData),1)

  fig.subtitle(f"Kfold cv with {len(MetricsData['training'])} folds")
  for i in len(MetricsData["training"]):
    axs[i].plot(XAxis,MetricsData["training"],linestyle='-', color = "red")
    axs[i].plot(XAxis,MetricsData["validation"],linestyle='-', color = "blue")
    axs[i].set_title(f'fold no. {i+1}')
  
  plt.show()