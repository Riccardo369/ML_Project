ListInputNeurons = [InputNeuron() for i in range(10)]
ApplyDropOut(ListInputNeurons, 0.2)

for i in ListInputNeurons: print(i.GetStateActived())