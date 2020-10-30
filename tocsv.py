import pandas as pd
# import torch
import numpy as np

x1 = np.random.rand(100,1).astype('float32')
x2 = np.random.rand(100,1).astype('float32')
y = x1 + x2

data = np.hstack((y,x1,x2))
df = pd.DataFrame(data,columns=["y","x1","x2"], index=None)
df.to_csv("./data/data.csv")

