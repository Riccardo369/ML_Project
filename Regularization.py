def ApplyDropOut(Neurons, Percentual):
    for i in Neurons:
      if(random.choices([True, False], [1-Percentual, Percentual], k=1)[0]): i.TurnOff()    #Turn off neurons