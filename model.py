import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self, i, h, o):
        super(NeuralNet, self).__init__()
        self.relu = nn.ReLU()
        self.l1 = nn.Linear(i, h) 
        self.l2 = nn.Linear(h, h) 
        self.l3 = nn.Linear(h, o)
        self.dropout = nn.Dropout(p=0.4)
    
    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.l3(x)
        return x