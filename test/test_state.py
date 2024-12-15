import keratorch as kt 

import torch, torch.nn as nn


state = kt.state.ktState()

state.train.update(batch = (torch.rand(1, 2), torch.rand(1, 2)))

print(state)




