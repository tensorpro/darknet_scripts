import numpy as np
from darkflow.net.build import TFNet

options = {"model": "cfg/yolo.cfg", "load": "yolo.weights", "threshold": 0.1}
net = TFNet(options)
with net.graph.as_default() as g:
    l = net.outputs[-5].out
    loss  = l
    iloss = net.inter_grad(np.zeros((400,400,3)), l, loss)
    print("Identity loss")
    print(iloss)
    loss  = net.out
    iloss = net.inter_grad(np.zeros((400,400,3)), l, loss)
    print("loss with respect to output")
    print(iloss)
