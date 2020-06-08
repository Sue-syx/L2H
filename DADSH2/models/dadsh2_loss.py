import torch.nn as nn
import torch


class DADSH2_Loss(nn.Module):
    """
    Loss function of ADSH

    Args:
        code_length(int): Hashing code length.
        gamma(float): Hyper-parameter.
    """
    def __init__(self, code_length, eta, mu, gamma, varphi, device):
        super(DADSH2_Loss, self).__init__()
        self.device = device
        self.code_length = code_length
        self.eta = eta
        self.mu = mu
        self.gamma = gamma
        self.varphi = varphi


    def forward(self, F, B, S, index):
        m = index.shape[0]  # m为样本数
        I1 = torch.eye(m, device=self.device)
        I2 = torch.eye(self.code_length, device=self.device)
        discrete_loss1 = self.eta * ((F.t() @ I1) ** 2).sum()
        discrete_loss2 = self.mu * (((F.t()@F - 2*m/3 * I2)) ** 2).sum()
        discrete_loss3 = self.gamma * (abs(F.t()).mul((1 - F.t()**2))).sum()

        hash_loss = ((self.code_length * S - F @ B.t()) ** 2).sum()
        quantization_loss = self.varphi * ((F - B[index, :]) ** 2).sum()

        loss = (discrete_loss1 + discrete_loss2 + discrete_loss3
                + hash_loss + quantization_loss) / (F.shape[0] * B.shape[0])

        return loss
