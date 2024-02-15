import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, Module, ModuleList
from torch_geometric.nn import MessagePassing
import math
from math import pi as PI


class ShiftedSoftplus(Module):
    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift


class CFConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_filters, edge_mlp, cutoff, smooth):
        super(CFConv, self).__init__(aggr="add")
        self.node_mlp1 = Linear(in_channels, num_filters, bias=False)
        self.node_mlp2 = Linear(num_filters, out_channels)
        self.edge_mlp = edge_mlp
        self.cutoff = cutoff
        self.smooth = smooth

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.node_mlp1.weight)
        torch.nn.init.xavier_uniform_(self.node_mlp2.weight)
        self.node_mlp2.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_length, edge_attr):
        if self.smooth:
            C = 0.5 * (torch.cos(edge_length * PI / self.cutoff) + 1.0)
            C = (
                C * (edge_length <= self.cutoff) * (edge_length >= 0.0)
            )  # Modification: cutoff
        else:
            C = (edge_length <= self.cutoff).float()

        W = self.edge_mlp(edge_attr) * C.view(-1, 1)
        x = self.node_mlp1(x)
        x = self.propagate(edge_index, x=x, W=W)
        x = self.node_mlp2(x)
        return x

    def message(self, x_j, W):
        return x_j * W


class TimeEmbedding(Module):
    def __init__(self, emb_dim, out_dim):
        super(TimeEmbedding, self).__init__()
        self.emb_dim = emb_dim
        self.out_dim = out_dim
        self.proj = Linear(emb_dim, out_dim)

    def act(self, xx):
        return xx * torch.sigmoid(xx)

    def forward(self, tt):
        assert len(tt.shape) == 2 and tt.shape[1] == 1
        half_dim = self.emb_dim // 2
        emb = math.log(1000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        emb = emb.to(device=tt.device)
        emb = tt.float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        if self.emb_dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))

        return self.proj(self.act(emb))


class InteractionBlock(Module):
    def __init__(self, hidden_channels, edge_channels, num_filters, cutoff, smooth, time_channels):
        super(InteractionBlock, self).__init__()
        edge_mlp = Sequential(
            Linear(edge_channels, num_filters),
            ShiftedSoftplus(),
            Linear(num_filters, num_filters),
        )
        self.conv1 = CFConv(
            hidden_channels, hidden_channels, num_filters, edge_mlp, cutoff, smooth
        )
        self.conv2 = CFConv(
            hidden_channels, hidden_channels, num_filters, edge_mlp, cutoff, smooth
        )
        self.act = ShiftedSoftplus()
        self.lin = Linear(hidden_channels, hidden_channels)
        self.time_emb = TimeEmbedding(time_channels, hidden_channels)

    def forward(self, tt, xx, edge_index, edge_length, edge_attr):
        xx = self.conv1(xx, edge_index, edge_length, edge_attr)
        tt = self.time_emb(tt)
        xx = self.act(xx + tt)

        xx = self.conv2(xx, edge_index, edge_length, edge_attr)
        xx = self.act(xx)

        xx = self.lin(xx)
        xx = self.act(xx)
        return xx


class SchNetEncoder(Module):
    def __init__(
        self,
        hidden_channels=128,
        num_filters=None,
        num_interactions=6,
        edge_channels=100,
        cutoff=10.0,
        smooth=False,
        node_types=4,
        time_channels=256,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_filters = hidden_channels if num_filters is None else num_filters
        self.num_interactions = num_interactions
        self.cutoff = cutoff

        self.interactions = ModuleList()
        for _ in range(num_interactions):
            block = InteractionBlock(
                hidden_channels, edge_channels, hidden_channels, cutoff, smooth, time_channels
            )
            self.interactions.append(block)

    @classmethod
    def from_config(cls, config):
        encoder = cls(
            hidden_channels=config.node_dim,
            edge_channels=config.edge_dim,
            num_filters=config.num_filters,
            num_interactions=config.num_convs,
            cutoff=config.cutoff,
            smooth=config.smooth_conv,
            time_channels=config.time_dim,
        )
        return encoder

    def forward(
        self, tt, zz, edge_index, edge_length, edge_attr, **kwargs
    ):
        for interaction in self.interactions:
            zz = zz + interaction(tt, zz, edge_index, edge_length, edge_attr)
        return zz
