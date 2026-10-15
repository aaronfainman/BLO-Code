import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np

AUX_NET_INPUT_DIM = 24



class LearnableParameter(nn.Module):

    def __init__(self, init_value: float):
        super().__init__()
        self.value = nn.Linear(in_features=1, out_features=1, bias=False)
        self.value.weight.data.copy_(torch.tensor([init_value]))
        self.value.requires_grad_(True)

    def forward(self):
        return self.value.weight[0]

class CIFARSqueezeNet(nn.Module):
    '''Pretrained SqueezeNet features with a 24-dimensional CIFAR head.'''

    def __init__(self, pretrained=True, feature_dim=AUX_NET_INPUT_DIM):
        super().__init__()
        weights = torchvision.models.SqueezeNet1_1_Weights.DEFAULT if pretrained else None
        backbone = torchvision.models.squeezenet1_1(weights=weights)

        self.features = backbone.features
        self.last_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, feature_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(feature_dim, 10)

    def forward(self, x, return_features=False):
        feature_maps = self.features(x)
        last_input = self.last_layer(feature_maps)
        logits = self.classifier(last_input)

        if return_features:
            return logits, last_input

        return logits

class AuxiliaryNet(nn.Module):
    "Penultimate Features to Sample Weight"

    def __init__(self, feature_dim=AUX_NET_INPUT_DIM):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, 1),
            # nn.ReLU(),
            # nn.Linear(feature_dim//2, 1),
            # nn.Linear(feature_dim // 2, 1), 
        )
        # self.feature_dim = feature_dim + 1
        # self.Wk = nn.Linear(self.feature_dim, 1)
        # self.tau = nn.Parameter(data=torch.tensor([0.15]))
        # self.bn = nn.BatchNorm1d(1)
        # self.sigma = nn.Parameter(data=torch.tensor([0.1]))#0.15
        # self.Wq = nn.Linear(self.feature_dim,1)

    def forward(self, features:torch.Tensor):

        return torch.sigmoid(self.network(features.detach())).squeeze(-1)
        # s = features.shape
        # feats = torch.zeros([s[0], s[1] + 1], device=features.device)
        # feats[:, 0:s[1]] = features
        # feats[:, s[1]] = iter

        # return torch.sigmoid(self.network(feats)).squeeze(-1)
        # K = self.Wk(features)
        # Q = self.Wq(features)

        # # size is batch x batch (each sample queries all keys)
        # scores = Q@K.T/(self.feature_dim**0.5) 

        # #softmax across batch, size is batch x batch
        # weights = torch.softmax(scores,dim=-1)
        
        # return weights.sum(dim=0)


class WindowNet(nn.Module):
    "Penultimate Features to Sample Weight"

    def __init__(self, feature_dim=AUX_NET_INPUT_DIM):
        super().__init__()

        # self.a = nn.Parameter(data=torch.tensor([0.1]))
        # self.b = nn.Parameter(data=torch.tensor([0.2]))
        self.tau = nn.Parameter(data=torch.tensor([0.15]))
        self.log_s = 0.2#nn.Parameter(data=torch.tensor([-1.0]))


    def forward(self, losses:torch.Tensor):
        # with torch.no_grad():
        #     gap = torch.abs(self.a-self.b)
        #     self.a, self.b = [ min([self.a,self.b]), max([self.a,self.b]) ]
        #     if gap<1e-3:
        #         self.b += 1e-3
            
        p = torch.exp(-losses.detach()).clamp(1e-6, 1 - 1e-6)

        s = self.log_s #torch.sigmoid(self.log_s/0.5)
        weights = 1/np.sqrt(2*np.pi*s**2)*torch.exp(-( p - self.tau)**2/(2*s**2))
        # weights = torch.sigmoid((p - self.a) / s) * torch.sigmoid((self.b - p) / s)
        return weights


class AuxiliaryNet2(nn.Module):
    """Learns the optimal loss function: (logits, true label) -> scalar loss."""

    def __init__(self, num_classes: int = 10, hidden_dim: int = 64):
        super().__init__()
        self.num_classes = num_classes
        self.network = nn.Sequential(
            nn.Linear(num_classes * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),  # non-negative output
        )

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        logits : [B, num_classes]  — raw predictions from ll_model
        labels : [B]               — integer class indices
        returns: scalar mean loss over the batch
        """
        one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
        x = torch.cat([logits, one_hot], dim=-1)          # [B, 2*num_classes]
        per_sample_loss = self.network(x).squeeze(-1)     # [B]
        return per_sample_loss.mean()