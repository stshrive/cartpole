import torch      as T
import torch.cuda as C

gpu_compatible = C.is_available()

FloatTensor = C.FloatTensor if gpu_compatible else T.FloatTensor
LongTensor  = C.LongTensor  if gpu_compatible else T.LongTensor
ByteTensor  = C.ByteTensor  if gpu_compatible else T.ByteTensor

_device = T.device("cuda" if gpu_compatible else "cpu")

def from_numpy(np):
    return T.from_numpy(np).float().to(_device)
