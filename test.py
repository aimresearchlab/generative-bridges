import torch.nn as nn

if isinstance(model, nn.DataParallel):
    print("Model is wrapped in nn.DataParallel")
else:
    print("Model is NOT wrapped in nn.DataParallel")