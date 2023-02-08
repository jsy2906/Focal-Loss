import torch
from torch import Tensor
import torch.nn as nn
from typing import Union

class FocalLoss(nn.Module):
    def __init__(self,
                 device,
                 weights: Union[None, Tensor] = None,
                 reduction: str='mean',
                 label_smoothing: float = None,
                 gamma=2,
                #  eps=1e-20
                 ):
        super().__init__()
        if reduction not in ['mean', 'none', 'sum']:
            raise NotImplementedError(
                'Reduction {} not implemented.'.format(reduction)
                )

        assert weights is None or isinstance(weights, Tensor), \
            'weights should be of type Tensor or None, but {} given'.format(type(weights))

        assert label_smoothing is None or label_smoothing>0 and label_smoothing<=1, \
            ' 0 < label smoothing eps <=1 or None'

        self.weights = weights
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.gamma = gamma
        self.device = device
        # self.eps = eps

    def _get_w(self, target):
        if self.weights is not None:
            w = self.weights.gather(0, target.data.view(-1))
        else:
            w = torch.ones(target.shape[0])
        return w

    def forward(self, inputs: Tensor, target: Tensor):
        
        w = self._get_w(target.type(torch.long))
        if self.label_smoothing is None:
            # ce = nn.CrossEntropyLoss()(inputs, target)+self.eps
            ce = nn.CrossEntropyLoss()(inputs, target)
        else:
            ce = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)(inputs, target)
            # ce = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)(inputs, target)+self.eps
        pt = torch.exp(-ce)

        floss = w*(1-pt)**self.gamma*ce

        if self.reduction == 'mean':
            return floss.mean()
        elif self.reduction == 'sum':
            return floss.sum()
        else:
            return floss
