import torch.nn as nn
import torch
import math


class MDHLoss(nn.Module):
    def __init__(self, eta1, eta2, eta3):
        super(MDHLoss, self).__init__()
        self.eta1 = eta1
        self.eta2 = eta2
        self.eta3 = eta3

    def forward(self, U_batch, U, S, C, N):
        theta = U.t() @ U_batch / 2
        # Prevent exp overflow
        theta = torch.clamp(theta, min=-100, max=50)
        metric_loss = (torch.log(1 + torch.exp(theta)) - S * theta).mean()

        multip = U_batch @ U_batch.t()
        fi1B = C * N - torch.trace(2 * multip)
        fi2B = U_batch.sum().pow(2).mean()
        fi3B = multip.pow(2).mean() - N * math.sqrt(C)
        quantization_loss = self.eta1 * fi1B + self.eta2 * fi2B + self.eta3 * fi3B

        loss = metric_loss + quantization_loss

        return loss
