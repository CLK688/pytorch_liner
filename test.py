import torch
import torch.nn as nn
from model import MyModel
from dataset import MyDataset
import os
import  numpy
from torch.utils.data import DataLoader 
import pandas as pd 
hidden_size = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

my_model = MyModel(hidden_size).to(device)
criterion = nn.MSELoss()
my_model.load_state_dict(torch.load("./result/model.pkl"))

def test():
    y_list = []
    with open("predict.txt","w") as f:
        my_model.eval()
        my_dataset = MyDataset(csv_file = "./data/data.csv")
        test_dataloader = DataLoader(dataset = my_dataset, batch_size= 1, shuffle= True, num_workers= 0)
        for idx , (label, x) in enumerate(test_dataloader):
            with torch.no_grad():
                label, x = label.to(device), x.to(device)
                output = my_model(x) 
                y = output.cpu().data.numpy().tolist()
                y_list.append(y)
                loss = criterion(output,label)
                #f.write(str(idx))
                f.write( '{}| output:{} | label: {} |loss: {} '.format( idx,output.item(), label.item(),loss.item()))
                f.write("\n")

    y_numpy = numpy.array(y_list).reshape(-1,1)
    df = pd.DataFrame(y_numpy,columns=None, index=None)
    df.to_csv("./data/data2.csv")
    

test()