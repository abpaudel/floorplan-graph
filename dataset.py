from collections import defaultdict, Counter
from torch_geometric.data import Data, DataLoader
from torch_geometric.data import Dataset
import torch
import numpy as np

# ACTUAL_ROOM_CLASS = {1: "living_room", 
#                     2: "kitchen",
#                     3: "bedroom",
#                     4: "bathroom",
#                     5: "missing",
#                     6: "closet",
#                     7: "balcony",
#                     8: "corridor",
#                     9: "dining_room",
#                     10: "laundry_room"}

MOD_ROOM_CLASS = {0: "living_room", 
                1: "kitchen",
                2: "bedroom",
                3: "bathroom",
                4: "closet",
                5: "balcony",
                6: "corridor",
                7: "dining_room"}

class FloorplanGraphDataset(Dataset):
    def __init__(self, path, split=None):
        super(FloorplanGraphDataset, self).__init__()
        self.path = path
        self.subgraphs = np.load('{}'.format(self.path), allow_pickle=True)
        self.subgraphs = self.filter_graphs(self.subgraphs)
        if split=='train':
            self.subgraphs = self.subgraphs[:120000]
        elif split=='test':
            self.subgraphs = self.subgraphs[120000:]    
        num_nodes = defaultdict(int)
        for g in self.subgraphs:
            labels = g[0] 
            if len(labels) > 0:
                num_nodes[len(labels)] += 1
        print(f'Number of graphs: {len(self.subgraphs)}')
        print(f'Number of graphs by rooms: {num_nodes}')
        
    def len(self):
        return len(self.subgraphs)

    def get(self, index, bbs=False):
        graph = self.subgraphs[index]
        labels = np.array(graph[0])
        rooms_bbs = np.array(graph[1])
        edge2node = [item for sublist in graph[3] for item in sublist]
        node_doors = np.array(edge2node)[graph[4]]
        doors_count = Counter(node_doors)
        features = []
        rooms_bbs_new = []
        for i, bb in enumerate(rooms_bbs):
            x0, y0 = bb[0], bb[1]
            x1, y1 = bb[2], bb[3]
            xmin, ymin = min(x0, x1), min(y0, y1)
            xmax, ymax = max(x0, x1), max(y0, y1)
            l, b = xmax - xmin, ymax - ymin
            area = l*b
            if l<b:
                l, b = b, l
            features.append([area, l, b, doors_count[i], 0, 0]) 
            rooms_bbs_new.append(np.array([xmin, ymin, xmax, ymax]))
        rooms_bbs = np.stack(rooms_bbs_new)
        intersect = self.intersect(rooms_bbs,rooms_bbs)
        for i in range(len(rooms_bbs)):
            is_child = []
            is_parent = []
            for j in range(i+1,len(rooms_bbs)):
                if intersect[i,j]>0.7*intersect[j,j]:
                    if intersect[i,i]>intersect[j,j]: #is i a parent
                        features[i][5] = 1
                        features[j][4] = 1
                    else:   # i is child
                        features[i][4] = 1
                        features[j][5] = 1
                if intersect[i,j]>0.7*intersect[i,i]:
                    if intersect[j,j]>intersect[i,i]: 
                        features[j][5] = 1
                        features[i][4] = 1
                    else:
                        features[j][4] = 1
                        features[i][5] = 1

        rooms_bbs = rooms_bbs/256.0

        tl = np.min(rooms_bbs[:, :2], 0)
        br = np.max(rooms_bbs[:, 2:], 0)
        shift = (tl+br)/2.0 - 0.5
        rooms_bbs[:, :2] -= shift
        rooms_bbs[:, 2:] -= shift
        tl -= shift
        br -= shift
        edges = self.build_graph(rooms_bbs) 
        labels = labels - 1
        labels[labels>=5] = labels[labels>=5] - 1
        x = torch.tensor(features, dtype=torch.float)
        edge_index = torch.tensor(edges.T, dtype=torch.long)
        y = torch.tensor(labels, dtype=torch.long)
        d = Data(x=x, edge_index=edge_index, y=y)
        if bbs:
            return d, rooms_bbs
        return d

    def build_graph(self, bbs):
        edges = []
        for k in range(len(bbs)):
            for l in range(len(bbs)):
                if l > k:
                    bb0 = bbs[k]
                    bb1 = bbs[l]
                    if self.is_adjacent(bb0, bb1):
                        edges.append([k, l])
                        edges.append([l, k])
        edges = np.array(edges)
        return edges

    def filter_graphs(self, graphs):
        new_graphs = []
        for g in graphs:       
            labels = g[0]
            rooms_bbs = g[1]
            # discard broken samples
            check_none = np.sum([bb is None for bb in rooms_bbs])
            check_node = np.sum([nd == 0 for nd in labels])
            if (len(labels) < 2) or (check_none > 0) or (check_node > 0):
                continue
            new_graphs.append(g)
        return new_graphs

    def is_adjacent(self, box_a, box_b, threshold=0.03):
        
        x0, y0, x1, y1 = box_a
        x2, y2, x3, y3 = box_b

        h1, h2 = x1-x0, x3-x2
        w1, w2 = y1-y0, y3-y2

        xc1, xc2 = (x0+x1)/2.0, (x2+x3)/2.0
        yc1, yc2 = (y0+y1)/2.0, (y2+y3)/2.0

        delta_x = np.abs(xc2-xc1) - (h1 + h2)/2.0
        delta_y = np.abs(yc2-yc1) - (w1 + w2)/2.0

        delta = max(delta_x, delta_y)

        return delta < threshold

    def intersect(self, A,B):
        A, B = A[:,None], B[None]
        low = np.s_[...,:2]
        high = np.s_[...,2:]
        A,B = A.copy(),B.copy()
        A[high] += 1; B[high] += 1
        intrs = (np.maximum(0,np.minimum(A[high],B[high])
                            -np.maximum(A[low],B[low]))).prod(-1)
        return intrs #/ ((A[high]-A[low]).prod(-1)+(B[high]-B[low]).prod(-1)-intrs)

