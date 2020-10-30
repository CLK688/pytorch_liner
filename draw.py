#画图
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd


def Draw(data_file, result_file):
    data = pd.read_csv(data_file)
    data2 = pd.read_csv(result_file)
    x1 = data.iloc[:,2].values
    x2 = data.iloc[:,3].values
    Y_true = data.iloc[:,1].values
    Y_pre = data2.iloc[:,1].values
    fig = plt.figure()
    ax1 = fig.add_subplot(221,projection='3d')  #这种方法也可以画多个子图
    ax2 = fig.add_subplot(222,projection='3d')  #这种方法也可以画多个子图
    ax3 = fig.add_subplot(223,projection='3d')  #这种方法也可以画多个子图
    ax4 = fig.add_subplot(224,projection='3d')  #这种方法也可以画多个子图
    #X1, X2 = np.meshgrid(x1, x2)

    ax1.scatter(x1, x2, Y_true, c='skyblue', s=60)
    ax2.scatter(x1, x2, Y_pre,  c='skyblue', s=60)
    ax3.plot_trisurf(x1, x2, Y_true, cmap='rainbow', linewidth=0.01)
    ax4.plot_trisurf(x1, x2, Y_pre,  cmap='rainbow', linewidth=0.01)
    

    
    #ax1.plot_surface(X1, X2, Y_true, rstride=1, cstride=1, cmap='rainbow')
#    ax2.plot_surface(X1, X2, Y_pre, rstride=1, cstride=1, cmap='rainbow')
    # ax = Axes3D(fig)
    # X1, X2 = np.meshgrid(x1, x2)
    # ax.plot_surface(X1, X2, Y_true, rstride=1, cstride=1, cmap='rainbow')
    # ax.set_title('surface')
    plt.show()

Draw("./data/data.csv","./data/data2.csv")