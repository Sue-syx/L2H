import torch.nn as nn
import torch


class DADSH_Loss(nn.Module):
    """
    Loss function of ADSH

    Args:
        code_length(int): Hashing code length.
        gamma(float): Hyper-parameter.
    """
    def __init__(self, code_length, eta, mu, gamma, varphi, delta, device):
        super(DADSH_Loss, self).__init__()
        self.device = device
        self.code_length = code_length
        self.eta = eta
        self.mu = mu
        self.gamma = gamma
        self.varphi = varphi
        self.delta = delta


    def forward(self, Y, U, V, S, W1, W2, index):
        m = index.shape[0]  # m为样本数
        discrete_loss1 = self.eta*torch.norm(U @ torch.eye(m, device=self.device))**2
        discrete_loss2 = self.mu*torch.norm((U@U.t() - 2*m/3*torch.eye(self.code_length, device=self.device)))**2
        discrete_loss3 = 2*self.gamma*(abs(U).mul((1-U**2))).sum()

        quantization_loss1 = torch.norm(Y.t() - W1.t() @ U)**2
        quantization_loss2 = self.varphi*torch.norm(V[:, index] - U) ** 2
        hash_loss = self.delta*torch.norm(self.code_length*S - (W1.t()@U).t() @ (W2.t()@V))**2

        loss = (discrete_loss1 + discrete_loss2 + discrete_loss3 + hash_loss
                + quantization_loss1 + quantization_loss2) / (V.shape[1] * U.shape[1])

        return loss
