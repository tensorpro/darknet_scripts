import numpy as np
from darkflow.net.build import TFNet
from _darkflow.net.build import TFNet

options = {"model": "cfg/yolo.cfg", "load": "yolo.weights", "threshold": 0.1}

with net.graph.as_default() as g:
    l = net.states[-5].out
    loss  = l
    iloss = net.intermediate(np.zeros((400,400,3)), l, loss)
