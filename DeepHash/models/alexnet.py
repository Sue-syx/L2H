# import torch
# import torch.nn as nn
# from torchvision import models
#
#
# def load_model(code_length):
#     """
#     Load CNN model.
#
#     Args
#         code_length (int): Hashing code length.
#
#     Returns
#         model (torch.nn.Module): CNN model.
#     """
#     model = AlexNet(code_length)
#     return model
#
#
# class AlexNet(nn.Module):
#
#     def __init__(self, code_length):
#         super(AlexNet, self).__init__()
#         alexnet_model = models.alexnet(pretrained=True)
#         cl1 = nn.Linear(256 * 6 * 6, 4096)
#         cl1.weight = alexnet_model.classifier[1].weight
#         cl1.bias = alexnet_model.classifier[1].bias
#         cl2 = nn.Linear(4096, 4096)
#         cl2.weight = alexnet_model.classifier[4].weight
#         cl2.bias = alexnet_model.classifier[4].bias
#
#         self.features = nn.Sequential(*list(alexnet_model.features.children()))
#         self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
#         self.classifier1 = nn.Sequential(nn.Dropout(), cl1, nn.ReLU(inplace=True))
#         self.classifier2 = nn.Sequential(nn.Dropout(), cl2, nn.ReLU(inplace=True))
#         self.hash_layer = nn.Sequential(nn.Linear(2*4096, code_length), nn.Tanh())
#
#     def forward(self, x):
#         x = self.features(x)
#         x = self.avgpool(x)
#         x = x.view(x.size(0), 256 * 6 * 6)
#         x1 = self.classifier1(x)
#         x2 = self.classifier2(x1)
#         x = torch.cat([x2, x1], dim=1)
#         x = self.hash_layer(x)
#         return x

import torch.nn as nn

from torch.hub import load_state_dict_from_url


def load_model(code_length):
    """
    Load CNN model.

    Args
        code_length (int): Hashing code length.

    Returns
        model (torch.nn.Module): CNN model.
    """
    model = AlexNet(code_length)
    state_dict = load_state_dict_from_url('https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth')
    model.load_state_dict(state_dict, strict=False)

    return model


class AlexNet(nn.Module):

    def __init__(self, code_length):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096 ,1000),
        )

        self.classifier = self.classifier[:-1]
        self.hash_layer = nn.Sequential(
            nn.Linear(4096, code_length),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        x = self.hash_layer(x)
        return x

