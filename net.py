import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class board_data(Dataset):
    def __init__(self, dataset): #np.array of (s, p, v)
        self.X = dataset[:,0]
        self.y_p, self.y_v = dataset[:,1], dataset[:,2]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_p[idx], self.y_v[idx]

class ConvBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 128, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)

    def forward(self, s):
        s = s.view(-1, 3, 6, 7)
        s = F.relu(self.bn1(self.conv1(s)))
        return s

class ResBlock(nn.Module):
    def __init__(self, inplanes=128, planes=128, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, s):
        residual = s
        out = F.relu(self.bn1(self.conv1(s)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class OutBlock(nn.Module):
    def __init__(self):
        super().__init__()
        #p
        self.conv1 = nn.Conv2d(128, 32, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.fc = nn.Linear(32*6*7, 7)

        #v
        self.conv = nn.Conv2d(128, 3, kernel_size=1)
        self.bn = nn.BatchNorm2d(3)
        self.fc1 = nn.Linear(3*6*7, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, s):
        p = F.relu(self.bn1(self.conv1(s)))
        p = p.view(-1, 6*7*32)
        p = self.fc(p)
        p = self.logsoftmax(p).exp()

        v = F.relu(self.bn(self.conv(s)))
        v = v.view(-1, 3*6*7)
        v = F.relu(self.fc1(v))
        v = torch.tanh(self.fc2(v))

        return p, v

class ConnectNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = ConvBlock()
        for i in range(19):
            setattr(self, f'self.res{i}', ResBlock())
        self.outblock = OutBlock()

    def forward(self, s):
        s = self.conv(s)
        for i in range(19):
            s = getattr(self, f'self.res{i}')(s)
        s = self.outblock(s)
        return s

class AlphaLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_value, value, y_policy, policy):
        value_error = (value - y_value) ** 2 
        policy_error = torch.sum((-policy* (1e-8 + y_policy.float()).float().log()), 1)
        total_error = (value_error.view(-1).float() + policy_error).mean()
        return total_error

def train(model, dataset, epoch_start=0, epoch_stop=50):
    model.train()
    loss_fn = AlphaLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_set = board_data(dataset)
    train_loader = DataLoader(train_set, batch_size=30, shuffle=True)

    for epoch in range(epoch_start, epoch_stop):
        for data in train_loader:
            state, policy, value = data
            y_policy, y_value = model(state)
            optimizer.zero_grad()
            l = loss_fn(y_value, value, y_policy, policy)
            l.backward()
            optimizer.step()

def pit():
    #TODO
    pass
#TODO do experimenting with torch.save() and torch.load() before actual trianing
#TODO pit, and mcts self-play that collects data set
#TODO i guess this all goes in train
"""
1. MCTS self-play to generate data sets
    -search for 777 simulations until game is over
    -collect (s, p) s = state, p = mcts distribution
    -at the end of game append value to all the states except for first
    -have dataset (s, pi, r)
2. The network will train with this dataset
    -The net outputs (p, v)
    -loss is computed to minimizei difference between (r and v) and (p and pi)
3. This is the training loop and this will continue to trian until some point

"""
