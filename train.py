import torch
import numpy as np
from dataset import FloorplanGraphDataset
from utils import  accuracy
from model import GNN, NN
from torch_geometric.data import DataLoader
from torch_geometric.nn import SAGEConv, GATConv, GCNConv, TAGConv


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)


outpath = '/scratch/apaudel4/gnn/results/'
# outpath = './'
torch.manual_seed(42)

dataset = FloorplanGraphDataset(path='housegan_clean_data.npy', split=None)
# train = FloorplanGraphDataset(path='housegan_clean_data.npy', split='train')
train = [dataset[i].to(device) for i in range(120000)]
trainloader = DataLoader(train, batch_size=128, shuffle=True)
trainloader2 = DataLoader(train, batch_size=120000)
# test = FloorplanGraphDataset(path='housegan_clean_data.npy', split='test')
test = [dataset[i].to(device) for i in range(120000,143184)]
testloader = DataLoader(test, batch_size=23184)

Conv = [SAGEConv, GCNConv, GATConv, TAGConv]
models = []

hid_r = 2,7

for n_h in range(*hid_r):
    for C in Conv:
        models.append(GNN(Conv=C, n_hidden=n_h))
for n_h in range(*hid_r):
    models.append(NN(n_hidden=n_h))


lr = 0.004
step_size = 10
gamma = 0.8

for model in models:
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    model = model.to(device)
    num_epochs = 100
    loss_ep = []
    te_acc_ep = []
    tr_acc_ep = []
    print(model)
    
    for epoch in range(num_epochs):
        model.train()
        loss = 0
        for data in trainloader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss_ = criterion(out, data.y)
            loss_.backward()
            optimizer.step()
            loss += loss_.item()
        exp_lr_scheduler.step()
        loss/=len(train)
        tr_acc = accuracy(model, trainloader2)
        te_acc = accuracy(model, testloader)
        loss_ep.append(loss)
        tr_acc_ep.append(tr_acc)
        te_acc_ep.append(te_acc)
        print(f'Epoch [{epoch+1}/{num_epochs}] Loss: {loss:.10f}, Train Acc: {tr_acc:.6f}, Test Acc: {te_acc:.6f}')
        
    result = np.array([loss_ep, tr_acc_ep, te_acc_ep]).T
    np.savetxt(f'{outpath}{type(model.layer1).__name__}{len(model.layer2)+1}_loss_tracc_teacc_{lr}_{num_epochs}_{step_size}_{gamma}.txt', result)
    max_idx = result[:,2].argmax()
    print(f'\nMax Test Accuracy at Epoch {max_idx+1}: {result[max_idx]}\n')
