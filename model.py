import torch
from torch.nn import Linear, Sequential
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, GCNConv, TAGConv


class GNN(torch.nn.Module):

    def __init__(self, Conv=SAGEConv, n_hidden=2):
        super(GNN, self).__init__()
        torch.manual_seed(42)
        self.layer1 = Conv(6, 16)
        self.layer2 = torch.nn.ModuleList()
        for _ in range(n_hidden-1):
            self.layer2.append(Conv(16,16))
        self.classifier = Linear(16, 8)

    def forward(self, x, edge_index):
        h = self.layer1(x, edge_index)
        h = F.relu(h)
        for layer in self.layer2:
            h = layer(h, edge_index)
            h = F.relu(h)
        out = self.classifier(h)
        return out

class NN(torch.nn.Module):

    def __init__(self, Conv=Linear, n_hidden=2):
        super(NN, self).__init__()
        torch.manual_seed(42)
        self.layer1 = Linear(6, 16)
        self.layer2 = torch.nn.ModuleList()
        for _ in range(n_hidden-1):
            self.layer2.append(Conv(16,16))
        self.classifier = Linear(16, 8)

    def forward(self, x, edge_index):
        h = self.layer1(x)
        h = F.relu(h)
        for layer in self.layer2:
            h = layer(h)
            h = F.relu(h)
        out = self.classifier(h)
        return out
