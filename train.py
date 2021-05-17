import torch
import numpy as np
from dataset import FloorplanGraphDataset
from utils import  accuracy
from model import Model
from torch.nn import Linear
from torch_geometric.data import DataLoader
from torch_geometric.nn import SAGEConv, GATConv, GCNConv, TAGConv
import pathlib
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', choices=['mlp', 'gcn', 'gat', 'sage', 'tagcn'], default='sage', help='Type of model')
    parser.add_argument('--hidden', type=int, default=2, help='Number of hidden/messsage passing layers')
    parser.add_argument('--epoch', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.004, help='Learning rate')
    parser.add_argument('--step', type=int, default=10, help='Step size for exponential learning rate scheduling')
    parser.add_argument('--gamma', type=float, default=0.8, help='Decay rate for exponential learning rate scheduling')
    parser.add_argument('--bs', type=int, default=128, help='Batch size for training')
    parser.add_argument('--outpath', type=str, default='./results', help='Path to save results')
    parser.add_argument('--dataset_file', type=str, default='./data/housegan_clean_data.npy', help='House-GAN dataset .npy file path')
    args = parser.parse_args()
    models = {
                'mlp': Linear,
                'gcn': GCNConv,
                'gat': GATConv,
                'sage': SAGEConv,
                'tagcn': TAGConv,
            }
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    outpath = pathlib.Path(args.outpath)
    outpath.mkdir(parents=True, exist_ok=True) 

    torch.manual_seed(42)
    model = Model(layer_type=models[args.model], n_hidden=args.hidden)
    print(model)
    model = model.to(device)

    dataset = FloorplanGraphDataset(path=args.dataset_file, split=None)
    train = [dataset[i].to(device) for i in range(120000)]
    trainloader = DataLoader(train, batch_size=args.bs, shuffle=True)
    trainloader2 = DataLoader(train, batch_size=120000)
    test = [dataset[i].to(device) for i in range(120000,143184)]
    testloader = DataLoader(test, batch_size=23184)

    num_epochs = args.epoch
    lr = args.lr
    step_size = args.step
    gamma = args.gamma

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

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
        exp_lr_scheduler.step()
        loss/=len(train)
        tr_acc = accuracy(model, trainloader2)
        te_acc = accuracy(model, testloader)
        loss_ep.append(loss)
        tr_acc_ep.append(tr_acc)
        te_acc_ep.append(te_acc)
        print(f'Epoch [{epoch+1}/{num_epochs}] Loss: {loss:.10f}, Train Acc: {tr_acc:.6f}, Test Acc: {te_acc:.6f}')
        
    result = np.array([loss_ep, tr_acc_ep, te_acc_ep]).T
    np.savetxt(outpath/f'{type(model.layer1).__name__}{len(model.layer2)+1}_loss_tracc_teacc_{lr}_{num_epochs}_{step_size}_{gamma}.txt', result)
    max_idx = result[:,2].argmax()
    print(f'\nMax Test Accuracy at Epoch {max_idx+1}: {result[max_idx]}\n')
