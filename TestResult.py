from Result import *

Metrics = dict()

Metrics["Training loss"] = [1,2,3,4,5,6,7,8,9,10]
Metrics["Validation loss"] = [1,5,2,7,3,4,9,8,7,10]

Graph(Metrics, ["red", "blue"], "Cycle", "Title 1")