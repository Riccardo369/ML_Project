import matplotlib.pyplot as plt
import numpy as np

def plot_model_performance(MetricsData: dict, training_color, validation_color, LabelX, Title):
  DataList = list(MetricsData.values())
  if(sum(map(len, DataList)))/len(DataList) != len(DataList[0]): raise ValueError("All metrics must be on same size")
  fig,(ax_loss,ax_prec)=plt.subplots(1,2)
  ax_loss.plot(np.arange(len(MetricsData["training"]["Loss"])), MetricsData["training"]["Loss"], linestyle='-', color = training_color,label="training loss")
  ax_loss.set_xlabel('loss | # of epochs')
  ax_loss.set_ylabel('loss value')
  
  ax_prec.plot(np.arange(len(MetricsData["training"]["Precision"])), MetricsData["training"]["Precision"], linestyle='-', color = training_color,label="training precision")
  ax_prec.set_xlabel('precision | # of epochs')
  ax_prec.set_ylabel('percentage')
  ax_prec.set_ylim([0,1.1])
  if "validation" in MetricsData:
    ax_loss.plot(np.arange(len(MetricsData["validation"]["Loss"])), MetricsData["validation"]["Loss"], linestyle='-', color = validation_color,label="validation loss")
    ax_prec.plot(np.arange(len(MetricsData["validation"]["Precision"])), MetricsData["validation"]["Precision"], linestyle='-', color = validation_color,label="validation precision")

  fig.suptitle(Title)
  ax_prec.legend()
  ax_loss.legend()
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