## Room Classification on Floor Plan Graphs using Graph Neural Networks

[Paper](https://arxiv.org/abs/2108.05947) | 
[Dataset](https://www.dropbox.com/sh/p707nojabzf0nhi/AAB4UPwW0EgHhbQuHyq60tCKa?dl=0&preview=housegan_clean_data.npy) |
[Install PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric#installation)

### Usage
```
$ python train.py --help
usage: train.py [-h] [--model {mlp,gcn,gat,sage,tagcn}] [--hidden HIDDEN]  
                [--epoch EPOCH] [--lr LR] [--step STEP] [--gamma GAMMA]    
                [--bs BS] [--outpath OUTPATH] [--dataset_file DATASET_FILE]

optional arguments:
  -h, --help            show this help message and exit
  --model {mlp,gcn,gat,sage,tagcn}
                        Type of model (default: sage)
  --hidden HIDDEN       Number of hidden/messsage passing layers (default: 2)
  --epoch EPOCH         Number of epochs to train (default: 100)
  --lr LR               Learning rate (default: 0.004)
  --step STEP           Step size for exponential learning rate scheduling
                        (default: 10)
  --gamma GAMMA         Decay rate for exponential learning rate scheduling
                        (default: 0.8)
  --bs BS               Batch size for training (default: 128)
  --outpath OUTPATH     Path to save results (default: ./results)
  --dataset_file DATASET_FILE
                        House-GAN dataset .npy file path (default:
                        ./data/housegan_clean_data.npy)
```
