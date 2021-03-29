import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv

class GraphSAGE(torch.nn.Module):
    def __init__(self):
        super(GraphSAGE, self).__init__()
        torch.manual_seed(42)
        self.conv1 = SAGEConv(4, 10)
        self.conv2 = SAGEConv(10, 20)
        self.conv3 = SAGEConv(20, 20)
        self.classifier = Linear(20, 8)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = self.conv2(h, edge_index)
        h = F.relu(h)
        h = self.conv3(h, edge_index)
        h = F.relu(h) 
        out = self.classifier(h)
        return out


class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        torch.manual_seed(42)
        self.conv1 = GCNConv(4, 10)
        self.conv2 = GCNConv(10, 20)
        self.conv3 = GCNConv(20, 20)
        self.classifier = Linear(20, 8)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = self.conv2(h, edge_index)
        h = F.relu(h)
        h = self.conv3(h, edge_index)
        h = F.relu(h) 
        out = self.classifier(h)
        return out


class NN(torch.nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        torch.manual_seed(42)
        self.layer1 = Linear(4, 10)
        self.layer2 = Linear(10, 20)
        self.layer3 = Linear(20, 20)
        self.classifier = Linear(20, 8)

    def forward(self, x, edge_index):
        h = self.layer1(x)
        h = F.relu(h)
        h = self.layer2(h)
        h = F.relu(h)
        h = self.layer3(h)
        h = F.relu(h) 
        out = self.classifier(h)
        return out
