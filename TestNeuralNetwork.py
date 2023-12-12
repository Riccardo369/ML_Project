from NeuralNetwork import *

NN1 = NeuralNetwork(2, 1, lambda yo, yt: 1/2 * (yo - yt)**2)

InputNeurons = NN1.GetAllInputNeurons()
OutputNeurons = NN1.GetAllOutputNeurons()

N1 = ActivationNeuron(lambda x: x, lambda x, y: x, lambda x, y: x, lambda yo, yt: 1/2 * (yo - yt)**2)

InputNeurons[0].AddConnectionTo(N1)
InputNeurons[1].AddConnectionTo(N1)
N1.AddConnectionTo(OutputNeurons[0])

NN1.CalculateAllStructure()

W_I0_N1 = InputNeurons[0].GetExitBridgeToNeuron(N1).Weight

W_I1_N1 = InputNeurons[1].GetExitBridgeToNeuron(N1).Weight
W_N1_O0 = N1.GetExitBridgeToNeuron(OutputNeurons[0]).Weight

Result = NN1.Predict(np.array([1, 2]))

assert Result == (1*W_I0_N1 + 2*W_I1_N1 + N1.BiasValue) * W_N1_O0

BridgeCounts = list(map(lambda i: i.GetUsedCount(), NN1.GetAllBridges()))

assert min(BridgeCounts) == max(BridgeCounts)
assert min(BridgeCounts) == 1

NN1.Learn(1)

BridgeCounts = list(map(lambda i: i.GetUsedCount(), NN1.GetAllBridges()))

assert min(BridgeCounts) == max(BridgeCounts)
assert min(BridgeCounts) == 0

####################################################################################################################################

MLP1 = MLP(2, 1, lambda yo, yt: 1/2 * (yo - yt)**2)