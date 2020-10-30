#定义模型
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self, hidden_size):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self,x):
        #x = input.view(-1,1)
        x = self.fc1(x)
        x = F.relu(x)
        out = self.fc2(x)
        return out

        