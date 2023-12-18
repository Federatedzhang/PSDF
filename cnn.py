import keras.backend
import torch
import torch.nn as nn
from torch import dropout
from torchvision import models
from torchsummary import summary
import numpy as np

class CNN(nn.Module):
    def __init__(self, num_classes=2):#num_classes类别数为2
        super(CNN, self).__init__()#单继承，即只有一个父类
        self.features1 = nn.Sequential(nn.Conv1d(1, 1, kernel_size=3, stride=1, padding=1),
                                      nn.ReLU(),
                                      nn.Conv1d(1, 1, kernel_size=3, stride=1, padding=1),
                                      nn.ReLU()
                                      )

        self.fc1 = nn.Linear(25, 8)
        self.fc2 = nn.Linear(8, 2)

        self.fc4 = nn.Linear(2, 5)
        self.fc5 = nn.Linear(16, 20)


    def forward(self, x):
        # print(x.shape)
        x1 = x[:,:,:2]
        x2 = x[:,:,2:]
        x1 = self.fc4(x1)
        x2 = self.fc5(x2)
        x =  torch.cat((x1, x2), 2)
        # print(x1.shape)
        # print(x2.shape)
        x = self.features1(x)

        x = x.view(x.size(0), -1)
        # print(x.shape)
        # print(x.shape)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def get_feature(self, x):
        # print(x.shape)
        x1 = x[:,:,:2]
        x2 = x[:,:,2:]
        x1 = self.fc4(x1)
        x2 = self.fc5(x2)
        x =  torch.cat((x1, x2), 2)
        # print(x1.shape)
        # print(x2.shape)
        x = self.features1(x)

        x = x.view(x.size(0), -1)
        print(x.shape)
        return x

    # # dropout函数的实现
    # def dropout(x, level):
    #     if level < 0. or level >= 1:  # level是概率值，必须在0~1之间
    #         raise ValueError('Dropout level must be in interval [0, 1[.')
    #     retain_prob = 1. - level
    #
    #     # 我们通过binomial函数，生成与x一样的维数向量。binomial函数就像抛硬币一样，我们可以把每个神经元当做抛硬币一样
    #     # 硬币 正面的概率为p，n表示每个神经元试验的次数
    #     # 因为我们每个神经元只需要抛一次就可以了所以n=1，size参数是我们有多少个硬币。
    #     random_tensor = np.random.binomial(n=1, p=retain_prob,
    #                                        size=x.shape)  # 即将生成一个0、1分布的向量，0表示这个神经元被屏蔽，不工作了，也就是dropout了
    #     print(random_tensor)
    #
    #     x *= random_tensor
    #     print(x)
    #     x /= retain_prob
    #
    #     return x



if __name__ == '__main__':
    net = CNN(2)
    net.cuda()
    summary(net, (1, 16))
