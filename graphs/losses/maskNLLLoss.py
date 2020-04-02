import torch
import torch.nn.functional as F
import torch.nn as nn

import numpy as np


class maskNLLLoss(nn.Module):
    def __init__(self, config=None):
        super(maskNLLLoss, self).__init__()
        self.config = config

    def maskNLLLoss(self, inp, target, mask):

        crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)))

        loss = crossEntropy.masked_select(mask).mean()
        loss = loss.to(self.config.device)
        return loss

    def forward(self, inputs, targets, mask):
        return self.maskNLLLoss(inputs, targets, mask)