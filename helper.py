import torch
import numpy as np
class Helper():
    def __init__(self):
        return

    def smooth(self,x, N):
        cumsum = np.cumsum(np.insert(x, 0, 0))
        return (cumsum[N:] - cumsum[:-N]) / float(N)
