import matplotlib.pyplot as plt
import numpy as np
plt.figure(figsize=(13, 10))
# 构建数据
x_data = ['Coarse-O',
'SD',
'Coarse-A',
'LT',
'CD',
'GT',
'Bal',
'Num_inv',
'Max_pay',
'GS',
'Num_pay',
'Paid_rate',
'EP',
'ME',
'Kr',
'CR',
'D_ind',
'MD'

]
# y_data2 = [0.169204738,	0.861252115,	0.507614213,	4.737732657,	0.507614213,	1.184433164,	0,	0.169204738,	1.199661591]
y_data = [0.512936223,
0.454765732,
0.43148672,
0.421305798,
0.351154974,
0.271202073,
0.250300098,
0.237303702,
0.236505652,
0.207709287,
0.199811555,
0.194407227,
0.191558825,
0.187021651,
0.17390247,
0.163791722,
0.102063301,
0.041981068
]
cmap = plt.get_cmap('coolwarm')
colors = [cmap(x) for x in np.linspace(0, 1, 18)]

cmap = plt.get_cmap('coolwarm')
colors_1 = [cmap(i) for i in np.linspace(0, 1, 18 )]
bar_width=0.8
# Y轴数据使用range(len(x_data), 就是0、1、2...
a = plt.barh(y=range(len(x_data)), width=y_data, label='normal contract',
color= colors, alpha=0.8, height=bar_width)
# Y轴数据使用np.arange(len(x_data))+bar_width,
# 就是bar_width、1+bar_width、2+bar_width...这样就和第一个柱状图并列了
# plt.barh(y=np.arange(len(x_data))+bar_width, width=y_data2,
# label='Ponzi contract', color='indianred', alpha=0.8, height=bar_width)
# 在柱状图上显示具体数值, ha参数控制水平对齐方式, va控制垂直对齐方式
for y, x in enumerate(y_data):
    plt.text(x+5000, y-bar_width/1, '%s' % x, ha='center', va='bottom')
# for y, x in enumerate(y_data2):
#     plt.text(x+5000, y+bar_width/1, '%s' % x, ha='center', va='bottom')
# 为Y轴设置刻度值



plt.yticks(np.arange(len(x_data))+bar_width/2, x_data,fontsize = 20,family='Times New Roman')
# 设置标题
# plt.title("Information Gain of Feature", fontsize = 20,family='Times New Roman')
# 为两条坐标轴设置名称
plt.xlabel("Information gain", fontsize = 18,family='Times New Roman')
plt.ylabel("Feature ", fontsize = 20,family='Times New Roman')
# plt.colorbar(a)
# 显示图例
# plt.legend()
plt.savefig('infobar.pdf')
plt.show()