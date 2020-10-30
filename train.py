import torch 
import torch.nn as nn
import torch.optim as optim
import os 
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import MyDataset
from model import MyModel
epoch = 8000
hidden_size = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#实例化模型、损失函数、优化器
my_model = MyModel(hidden_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.SGD(my_model.parameters(), lr = 1e-5)
if os.path.exists("./result/model.pkl"):
    my_model.load_state_dict(torch.load("./result/model.pkl"))
   # optimizer.load_state_dict(torch.load("./result/optimizer.pkl"))
#开始训练
my_dataset = MyDataset(csv_file = "./data/data.csv")
train_dataloader = DataLoader(dataset = my_dataset, batch_size= 20, shuffle= True, num_workers= 0)


def train(epoch):
    loss_list =[]
    
    for i, (label, data) in enumerate(train_dataloader):
        optimizer.zero_grad()
        label = label.to(device)
        data = data.to(device)
        output = my_model(data)
        loss = criterion(label,output)
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
        print('Train Epoch:{} [{}/{}  ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,i*len(data),
            len(train_dataloader.dataset), 100*i/len(train_dataloader), loss.item()))
    torch.save(my_model.state_dict(),"./result/model.pkl")
    torch.save(optimizer.state_dict(),"./result/optimizer.pkl")
    loss_avg = np.mean(loss_list)
    x = epoch
    plt.scatter(x, loss_avg,color = 'r')
    plt.title('The loss in epoch')
    plt.ylabel('loss value')
    plt.xlabel('epoch number')
    plt.pause(0.0001)
    plt.savefig('loss.jpg')
for i in range(epoch):
    train(i)

