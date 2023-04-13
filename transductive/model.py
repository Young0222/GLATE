import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import sys


class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret


class Encoder(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation,
                 base_model=GCNConv, k: int = 2):
        super(Encoder, self).__init__()
        self.base_model = base_model

        assert k >= 2
        self.k = k
        self.conv = [base_model(in_channels, 2 * out_channels)]
        for _ in range(1, k-1):
            self.conv.append(base_model(2 * out_channels, 2 * out_channels))
        self.conv.append(base_model(2 * out_channels, out_channels))
        self.conv = nn.ModuleList(self.conv)
        self.activation = activation

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        for i in range(self.k):
            x = self.activation(self.conv[i](x, edge_index))
        return x


class Model(torch.nn.Module):
    def __init__(self, encoder: Encoder, num_hidden: int, num_proj_hidden: int,
                 tau1: float = 0.8, tau2: float = 0.2):
        super(Model, self).__init__()
        self.encoder: Encoder = encoder
        self.tau1: float = tau1
        self.tau2: float = tau2
        self.pre_grad: float = 0.0

        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def uniform_loss(self, z: torch.Tensor, t: int = 2):
        return torch.pdist(z, p=2).pow(2).mul(-t).exp().mean().log()

    def decay(self, x_start: float, decay: float = 0.999):
         return x_start * decay

    def momentum(self, x_start: float, z: torch.Tensor, step: float = 0.001, discount: float = 0.7): 
        if x_start <= self.tau2:
            return x_start
        x = x_start
        grad = -self.uniform_loss(z).item()
        self.pre_grad = self.pre_grad * discount + 1 / grad
        x -= self.pre_grad * step

        # x -= grad * step
        return x

    def momentum_batch(self, x_start: float, z: torch.Tensor, step: float = 0.001, discount: float = 0.7): 
        if x_start <= self.tau2:
            return x_start
        x = x_start
        grad = -self.uniform_loss(z).item()
        self.pre_grad = self.pre_grad * discount + 1 / grad
        x -= self.pre_grad * step

        # x -= grad * step
        return x

    def torch_cov(self, input_vec: torch.Tensor):    
        x = input_vec- torch.mean(input_vec,axis=0)
        cov_matrix = torch.matmul(x.T, x) / (x.shape[0]-1)
        return torch.det(cov_matrix)

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, epoch: int):
        f = lambda x: torch.exp(x / self.tau1)
        # self.tau1 = self.decay(self.tau1)
        self.tau1 = self.momentum(self.tau1, z1)
    
        # print("self.tau1: ", self.tau1)
        # hard_negative1 = torch.mul(self.sim(z1, z1), self.sim(z1, z1)>0.8)
        # hard_negative2 = torch.mul(self.sim(z1, z2), self.sim(z1, z2)>0.8)
        # refl_sim = f(hard_negative1)
        # between_sim = f(hard_negative2)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(between_sim.diag() / (refl_sim.sum(1) - refl_sim.diag() + between_sim.sum(1)))    #sum(1)求行和

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, epoch: int,
                          batch_size: int):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau1)
        # self.tau1 = self.decay(self.tau1)
        # print("self.tau1: ", self.tau1)
        self.tau1 = self.momentum_batch(self.tau1, z1)
        print("self.tau1: ", self.tau1)
        
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            # hard_negative1 = torch.mul(self.sim(z1[mask], z1), self.sim(z1[mask], z1)>0.6)
            # hard_negative2 = torch.mul(self.sim(z1[mask], z2), self.sim(z1[mask], z2)>0.6)
            # refl_sim = f(hard_negative1)
            # between_sim = f(hard_negative2)
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]
            losses.append(-torch.log(
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses)

    def loss(self, z1: torch.Tensor, z2: torch.Tensor, epoch: int =0, mean: bool = True, batch_size: int = 0):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size == 0:
            l1 = self.semi_loss(h1, h2, epoch)
            l2 = self.semi_loss(h2, h1, epoch)
        else:
            l1 = self.batched_semi_loss(h1, h2, epoch, batch_size)
            l2 = self.batched_semi_loss(h2, h1, epoch, batch_size)

        ret = (l1 + l2) * 0.5
        
        ret = ret.mean() if mean else ret.sum()

        return ret, self.tau1

def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x