import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE
def dataget():
    digits = datasets.load_digits(n_class=10)
    data = digits.data
    label = digits.target
    n_samples, n_features = data.shape  # 数据集的形状
    return data, label, n_samples, n_features

def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)  # 对数据进行归一化处理
    fig = plt.figure()  # 创建图形实例
    ax = plt.subplot(111)  # 创建子图
    # 遍历所有样本
    for i in range(data.shape[0]):
        print(label[i] / 10)
        # 在图中为每个数据点画出标签
        if label[i] == 0:
            plt.text(data[i, 0], data[i, 1], 'o', color=plt.cm.Set1((label[i] / 10)+0.3),
                     fontdict={'weight': 'bold', 'size': 7},size=7,label = 'normal')
            # plt.legend()
        if label[i] == 1:
            plt.text(data[i, 0], data[i, 1], '*', color=plt.cm.Set1((label[i] / 10)+0.1),
                     fontdict={'weight': 'bold', 'size': 7},size=9,label = 'Ponzi')
            # plt.legend()
    # plt.bar(x_2,b_2,width=2,label='第二次考试')
    plt.xticks(fontsize=20,family='Times New Roman')  # 指定坐标的刻度
    plt.yticks(fontsize=20,family='Times New Roman')

    # plt.legend((s1, s2), ('0', '1'))
    ax.grid(True)
    plt.title(title,fontsize=30,family='Times New Roman')
    # 返回值

    ax.grid(True)
    # 返回值
    return fig
import pandas as pd
from sklearn import preprocessing
def main():
    data = pd.read_csv("features2.csv")
    data = np.array(data)
    data = data[:500]
    a = data[:, 1:-1]

    zscore = preprocessing.StandardScaler()
    zscore = zscore.fit_transform(a)

    # print(data.shape)
    # a = np.load("fedproto_protos.npy")
    b = data[:,-1]
    print(b)
    in1 = a.shape[0]
    a = a.reshape(in1,-1)
    print(a.shape)
    print(b.shape)

    print('Starting compute t-SNE Embedding...')
    ts = TSNE(n_components=2, init='pca', random_state=0)
    print(ts,"!111")
    # t-SNE降维
    reslut = ts.fit_transform(zscore)
    print(reslut, "!111")
    # 调用函数，绘制图像
    fig = plot_embedding(reslut, b, '')
    # 显示图像
    plt.savefig('sdt2.pdf')
    plt.show()

main()

