#创建数据集
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd 


class MyDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        

    # def __getitem__(self, index):
    #     label = self.data.iloc[index, 0]
    #     data = self.data.iloc[index,1:].values
    #     data = data.astype('float').reshape(-1, 1)
    def __getitem__(self, index):
        label = self.data.iloc[index, 1]
        label = label.astype('float32').reshape(-1)
        data = self.data.iloc[index,2:].values
        data = data.astype('float32').reshape(-1) 
        return label, data
    def __len__(self):
        return len(self.data)


# my_dataset = MyDataset("./data/data.csv")
# train_dataloader = DataLoader(dataset = my_dataset, batch_size=50, shuffle= True, num_workers= 0)
# for i, (label, data) in enumerate(train_dataloader):
#     print(label.size(),data.size())
    


