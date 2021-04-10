import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from torch_geometric.utils import to_networkx
import networkx as nx
from dataset import MOD_ROOM_CLASS

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def visualize(d, bbs=None):
    G = to_networkx(d, to_undirected=True)
    plt.figure(figsize=(7,7))
    plt.axis('off')
    labels = {i: MOD_ROOM_CLASS[int(d.y[i])] for i in range(len(d.y))}
    c = plt.get_cmap('Dark2').colors
    color = [c[i] for i in d.y]
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=True, labels=labels, node_color=color, cmap='Dark2')
    plt.show()
    if bbs is not None:
        plt.figure(figsize=(7,7))
        for i, (xmin, ymin, xmax, ymax) in enumerate(bbs):
            rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, edgecolor='k', facecolor=c[d.y[i]], alpha=0.9)
            plt.gca().add_patch(rect)
        plt.show()

def accuracy(model, dataloader):
    correct = 0
    num_nodes = 0
    model.to(device)
    model.eval()
    for data in dataloader:
        data = data.to(device)
        out = model(data.x, data.edge_index)
        pred = out.argmax(1)
        correct += sum(pred==data.y)
        num_nodes += data.num_nodes
    return (correct/num_nodes).item()
