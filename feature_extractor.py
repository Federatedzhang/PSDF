from __future__ import print_function, division

import os
import time
import random
import pickle
import time
from glob import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorflow.python.keras.losses import binary_crossentropy
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from models.cnn import CNN
from utils.dataset import get_normal, gen_dataset
from torch.utils.tensorboard import SummaryWriter

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# 训练参数设置
test_size = 0.4
batch_size = 64
learning_rate = 0.001
epochs = 60
import time
start = time.time()

#is_balance = False
is_balance = True

# 加载数据
if is_balance:
    with open('./dataset/cnn_balance_data_' + str(test_size) + '.pkl', 'rb') as f:
        data = pickle.load(f)
else:
    with open('./dataset/cnn_data_' + str(test_size) + '.pkl', 'rb') as f:
        data = pickle.load(f)

print(data)
x_train = data['x_train']
x_test = data['x_test']
y_train = data['y_train']
y_test = data['y_test']

# 创建模型保存路径
save_dir = './runs'
save_path = './runs/'+str(epochs)
os.makedirs(save_path, exist_ok=True)

writer = SummaryWriter(os.path.join(save_path, 'logs'))  # 记录loss

train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train))
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test))
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

# --------------------训练过程---------------------------------

model = CNN(2)
if torch.cuda.is_available():
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)#动态调整
loss_func = nn.CrossEntropyLoss()

Loss_list = []
train_loss_list = []
train_acc_list = []
test_loss_list = []
test_acc_list = []

# start = time.time()
for epoch in range(epochs):
    print('epoch {}'.format(epoch + 1))
    # training-----------------------------
    model.train()
    train_loss = 0.
    train_acc = 0.
    for batch_x, batch_y in train_dataloader:
        if torch.cuda.is_available():
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
        out = model(batch_x)
        loss = loss_func(out, batch_y)
        train_loss += loss.item()
        pred = torch.max(out, 1)[1]
        train_correct = (pred == batch_y).sum()
        train_acc += train_correct.item()
        optimizer.zero_grad()
        #通过计算梯度来调用优化器
        loss.backward()
        #进行参数权重的更新
        optimizer.step()
    print('Train Loss: {:.10f}, Acc: {:.6f}'.format(train_loss / (len(
        train_dataloader)), train_acc / (len(train_dataset))))
    writer.add_scalar('training_loss', train_loss / (len(train_dataloader)), epoch)
    writer.add_scalar('training_acc', train_acc / (len(train_dataset)), epoch)

    # evaluation--------------------------------
    model.eval()
    test_loss = 0.
    test_acc = 0.
    global_test_acc = 0.
    for batch_x, batch_y in test_dataloader:
        # batch_x, batch_y = Variable(batch_x, volatile=True).cuda(), Variable(batch_y, volatile=True).cuda()
        if torch.cuda.is_available():
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
        out = model(batch_x)
        loss = loss_func(out, batch_y)
        test_loss += loss.item()
        pred = torch.max(out, 1)[1]
        num_correct = (pred == batch_y).sum()
        test_acc += num_correct.item()
    print('Test Loss: {:.10f}, Acc: {:.6f}'.format(test_loss / (len(
        test_dataloader)), test_acc / (len(test_dataset))))
    writer.add_scalar('test_loss', test_loss / (len(test_dataloader)), epoch)
    writer.add_scalar('test_acc', test_acc / (len(test_dataset)))

    train_loss_list.append(train_loss / (len(train_dataloader)))
    train_acc_list.append(train_acc / (len(train_dataset)))

    test_loss_list.append(test_loss / (len(test_dataloader)))
    test_acc_list.append(test_acc / (len(test_dataset)))

    torch.save(model, os.path.join(save_path, 'last.pt'))
    if test_acc / (len(test_dataset)) > global_test_acc:
        torch.save(model, os.path.join(save_path, 'best.pt'))
        global_test_acc = test_acc / (len(test_dataset))
    scheduler.step(test_acc / (len(test_dataset)))
end = time.time()
print(end-start)


plt.figure()
plt.plot(list(range(len(train_loss_list))), train_loss_list)
plt.plot(list(range(len(test_loss_list))), test_loss_list)
plt.legend(['train loss', 'test loss'])
plt.xlabel('epochs')
plt.ylabel('Loss')
plt.title('Loss')
plt.savefig(os.path.join(save_path, 'loss.png'))

plt.figure()
plt.plot(list(range(len(train_acc_list))), train_acc_list)
plt.plot(list(range(len(test_acc_list))), test_acc_list)
plt.legend(['train acc', 'test acc'])
plt.xlabel('epochs')
plt.ylabel('Acc')
plt.title('Acc')
plt.savefig(os.path.join(save_path, 'acc.png'))


if os.path.exists('batch_time_acc.pkl'):
    with open('batch_time_acc.pkl', 'rb') as f:
        batch_time_acc = pickle.load(f)
    batch_time_acc[batch_size] = {'time':end-start, 'acc':test_acc / (len(test_dataset))}
else:
    batch_time_acc = {}
    batch_time_acc[batch_size] = {'time': end - start, 'acc': test_acc / (len(test_dataset))}
with open('batch_time_acc.pkl', 'wb') as f:
    pickle.dump(batch_time_acc, f)

if os.path.exists('test_size_acc.pkl'):
    with open('test_size_acc.pkl', 'rb') as f:
        test_size_acc = pickle.load(f)
    test_size_acc[test_size] = test_acc / (len(test_dataset))
else:
    test_size_acc = {}
    test_size_acc[test_size] = test_acc / (len(test_dataset))
with open('test_size_acc.pkl', 'wb') as f:
    pickle.dump(test_size_acc, f)




