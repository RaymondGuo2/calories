import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, hidden_layer, dropout=0.2):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(7, hidden_layer)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_layer, hidden_layer//2)
        self.fc3 = nn.Linear(hidden_layer//2, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x