import matplotlib.pyplot as plt
import numpy as np

def plot_model_performance(MetricsData: dict, loss_color, precision_color, LabelX, Title):
  DataList = list(MetricsData.values())
  if(sum(map(len, DataList)))/len(DataList) != len(DataList[0]): raise ValueError("All metrics must be on same size")
  print(MetricsData)
  fig,((ax_tr_loss,ax_tr_prec),(ax_vl_loss,ax_vl_prec))=plt.subplots(2,2)
  ax_tr_loss.plot(np.arange(len(MetricsData["training"]["Loss"])), MetricsData["training"]["Loss"], linestyle='-', color = loss_color, label="Loss")
  ax_tr_loss.set_xlabel('training loss')
  
  ax_tr_prec.plot(np.arange(len(MetricsData["training"]["Precision"])), MetricsData["training"]["Precision"], linestyle='-', color = precision_color, label="Precision")
  ax_tr_prec.set_xlabel('training precision')
  ax_tr_prec.set_ylim([0,1])

  ax_vl_loss.plot(np.arange(len(MetricsData["validation"]["Loss"])), MetricsData["validation"]["Loss"], linestyle='-', color = loss_color, label="Loss")
  ax_vl_loss.set_xlabel('validation loss')

  ax_vl_prec.plot(np.arange(len(MetricsData["validation"]["Precision"])), MetricsData["validation"]["Precision"], linestyle='-', color = precision_color, label="Precision")
  ax_vl_prec.set_xlabel('validation precision')
  ax_vl_prec.set_ylim([0,1])
  fig.suptitle(Title)
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