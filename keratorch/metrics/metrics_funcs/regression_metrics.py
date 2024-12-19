from ..metric import Metric

import torch 
import torch.nn.functional as F


__all__ = ["MSEMetric", "mse"]


@torch.no_grad()
def _mse(state: "ktState"):

    return F.mse_loss(
        state.train.model_output.outputs, state.train.batch[1]
    )


class MSEMetric(Metric):

    def __init__(self):
        super(MSEMetric, self).__init__(
            name= "mse", metric_func= _mse
        )


def mse():
    return MSEMetric()
