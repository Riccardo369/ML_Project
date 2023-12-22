from Layer import *
from Neuron import *

L1 = Layer(OutputNeuron, 5, lambda x, y: x)
L2 = Layer(InputNeuron, 4)
L3 = Layer(ActivationNeuron, 3, lambda x: x, lambda x, y: x, lambda x, y: 0, lambda x, y: 0, 1)

try: L3 = Layer(ActivationNeuron, 3, lambda x: x, lambda x: x)
except: pass

L1.ApplyDropOut(0.4)
for i in L1.GetNeurons(): print(i.GetStateActived())

print("")

L1.TurnOnAllNeurons()
for i in L1: print(i.GetStateActived())

L1[0]

try: L1[5]
except: pass

assert len(L1) == 5

L1.ConnectTo(L2)

NeuronList1 = L1.GetNeurons()
NeuronList2 = L2.GetNeurons()

for i in NeuronList1:

  assert len(i.GetSetExitBridge()) == 4

  for r in NeuronList2:

    assert i.GetExitBridgeToNeuron(r) == r.GetEnterBridgeFromNeuron(i)
    assert len(r.GetSetEnterBridge()) == 5
    
    
#######################################################################################################################################
    
PL1 = PerceptronLayer(4, 10, lambda x, y: 0, lambda x, y: 0, lambda x, y: 1/2*(x-y)**2, 0)

#######################################################################################################################################

IL1 = InputLayer(5)

#######################################################################################################################################

OL1 = OutputLayer(2, lambda x, y: x)