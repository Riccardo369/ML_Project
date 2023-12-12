from Neuron import *
from Bridge import *

N1 = Neuron(lambda x, y: x, lambda x, y: x, lambda x, y: x, 0)
N2 = Neuron(lambda x, y: x, lambda x, y: x, lambda x, y: x, 0)

B1 = Bridge(N1, N2)
B2 = Bridge(N1, N2, 1000)
B3 = Bridge(N2, N1)

assert B1 == B2
assert B1 != B3
assert B2 != B3

assert B1.GetStartNeuron() == N1
assert B1.GetFinishNeuron() == N2

assert B1.GetUsedCount() == 0
B1.IncrementUsedCount()
assert B1.GetUsedCount() == 1
B1.ResetUsedCount()
assert B1.GetUsedCount() == 0

print(N1)
print(N2)
print(B1)
print(B2)
print(B3)

#################################################################################################################################

AN1 = ActivationNeuron(lambda x: x, lambda x, y: x, lambda x, y: x, lambda x, y: 0, 1)

N1 = Neuron(lambda x, y: x, lambda x, y: x+y, lambda x, y: x, 10)
N2 = Neuron(lambda x, y: x, lambda x, y: x+y, lambda x, y: x, 20)
N3 = Neuron(lambda x, y: x, lambda x, y: x+y, lambda x, y: x, 30)

N1.AddConnectionTo(AN1)
N2.AddConnectionTo(AN1)
N3.AddConnectionTo(AN1)

B1 = Bridge(N1, AN1)
B2 = Bridge(N2, AN1)
B3 = Bridge(N3, AN1)

W_N1_AN1 = N1.GetExitBridgeToNeuron(AN1).Weight
W_N2_AN1 = N2.GetExitBridgeToNeuron(AN1).Weight
W_N3_AN1 = N3.GetExitBridgeToNeuron(AN1).Weight

assert AN1.LoadInputFromBridge(B1, 3) == True
assert AN1.LoadInputFromBridge(B2, 5) == True
assert AN1.LoadInputFromBridge(B3, 8) == True

assert AN1.Calculate() == W_N1_AN1*3 + W_N2_AN1*5 + W_N3_AN1*8 + 1

AN1.ResetInputVector()

assert AN1.LoadInputFromBridge(N3.GetExitBridgeToNeuron(AN1), 8) == True
assert AN1.LoadInputFromBridge(N1.GetExitBridgeToNeuron(AN1), 3) == True
assert AN1.LoadInputFromBridge(N2.GetExitBridgeToNeuron(AN1), 5) == True

assert AN1.Calculate() == W_N1_AN1*3 + W_N2_AN1*5 + W_N3_AN1*8 + 1

#################################################################################################################################

I1 = InputNeuron()

N1 = Neuron(lambda x, y: x, lambda x, y: x+y, lambda x, y: x, 10)
N2 = Neuron(lambda x, y: x, lambda x, y: x+y, lambda x, y: x, 20)
N3 = Neuron(lambda x, y: x, lambda x, y: x+y, lambda x, y: x, 30)

N1.AddConnectionTo(I1)
N2.AddConnectionTo(I1)
N3.AddConnectionTo(I1)

W_N1_AN1 = N1.GetExitBridgeToNeuron(I1).Weight
W_N2_AN1 = N2.GetExitBridgeToNeuron(I1).Weight
W_N3_AN1 = N3.GetExitBridgeToNeuron(I1).Weight

assert I1.LoadInputFromBridge(N1.GetExitBridgeToNeuron(I1), 3) == True
assert I1.LoadInputFromBridge(N2.GetExitBridgeToNeuron(I1), 5) == True
assert I1.LoadInputFromBridge(N3.GetExitBridgeToNeuron(I1), 8) == True

assert I1.Calculate() == W_N1_AN1*3 + W_N2_AN1*5 + W_N3_AN1*8

I1.ResetInputVector()

assert I1.GetInputVector() == [None, None, None]

#################################################################################################################################

O1 = OutputNeuron(lambda x, y: x)

N1 = Neuron(lambda x, y: x, lambda x, y: x+y, lambda x, y: x, 10)
N2 = Neuron(lambda x, y: x, lambda x, y: x+y, lambda x, y: x, 20)
N3 = Neuron(lambda x, y: x, lambda x, y: x+y, lambda x, y: x, 30)

N1.AddConnectionTo(O1)
N2.AddConnectionTo(O1)
N3.AddConnectionTo(O1)

W_N1_AN1 = N1.GetExitBridgeToNeuron(O1).Weight
W_N2_AN1 = N2.GetExitBridgeToNeuron(O1).Weight
W_N3_AN1 = N3.GetExitBridgeToNeuron(O1).Weight

assert O1.LoadInputFromBridge(N1.GetExitBridgeToNeuron(O1), 3) == True
assert O1.LoadInputFromBridge(N2.GetExitBridgeToNeuron(O1), 5) == True
assert O1.LoadInputFromBridge(N3.GetExitBridgeToNeuron(O1), 8) == True

assert O1.Calculate() == W_N1_AN1*3 + W_N2_AN1*5 + W_N3_AN1*8

O1.ResetInputVector()

assert O1.GetInputVector() == [None, None, None]

#################################################################################################################################

P1 = Perceptron(10, lambda x, y: x, lambda x, y: 0, lambda x, y: 1/2*(x-y)**2, 0)

N1 = Neuron(lambda x, y: x, lambda x, y: x+y, lambda x, y: x, 10)
N2 = Neuron(lambda x, y: x, lambda x, y: x+y, lambda x, y: x, 20)
N3 = Neuron(lambda x, y: x, lambda x, y: x+y, lambda x, y: x, 30)

N1.AddConnectionTo(P1)
N2.AddConnectionTo(P1)
N3.AddConnectionTo(P1)

W_N1_AN1 = N1.GetExitBridgeToNeuron(P1).Weight
W_N2_AN1 = N2.GetExitBridgeToNeuron(P1).Weight
W_N3_AN1 = N3.GetExitBridgeToNeuron(P1).Weight

assert P1.LoadInputFromBridge(N1.GetExitBridgeToNeuron(P1), 0) == True
assert P1.LoadInputFromBridge(N2.GetExitBridgeToNeuron(P1), 0) == True
assert P1.LoadInputFromBridge(N3.GetExitBridgeToNeuron(P1), 0) == True

assert P1.Calculate() == 0

assert P1.GetThreshold() == 10

P1.SetThreshold(-1)

assert P1.Calculate() == 1

