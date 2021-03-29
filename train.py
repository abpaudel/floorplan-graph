import torch
import numpy as np
from dataset import FloorplanGraphDataset
from utils import visualize, accuracy
from model import GraphSAGE, GCN, NN
from torch_geometric.data import DataLoader

train = FloorplanGraphDataset(path='housegan_clean_data.npy', split='train')
trainloader = DataLoader(train, batch_size=128, shuffle=True)
trainloader2 = DataLoader(train, batch_size=60000)
test = FloorplanGraphDataset(path='housegan_clean_data.npy', split='test')
testloader = DataLoader(train, batch_size=23184)

model = GraphSAGE()
# model = GCN()
# model = NN()

lr = 0.001
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = model.to(device)

num_epochs = 100
loss_ep = []
te_acc_ep = []
tr_acc_ep = []
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
    loss/=len(train)
    tr_acc = accuracy(model, trainloader)
    te_acc = accuracy(model, testloader)
    loss_ep.append(loss)
    tr_acc_ep.append(tr_acc)
    te_acc_ep.append(te_acc)
    print(f'Epoch [{epoch+1}/{num_epochs}] Loss: {loss:.10f}, Train Acc: {tr_acc:.6f}, Test Acc: {te_acc:.6f}')

result = np.array([loss_ep, tr_acc_ep, te_acc_ep]).T
np.savetxt(f'./{type(model).__name__}_loss_tracc_teacc__{lr}_{num_epochs}.txt', result)
