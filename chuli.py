import pandas as pd

def getdata(filename):  # 导入原有txt文件中的数组
    data = pd.read_csv(filename)
    x= data['return_rio']
    # x= data['A_bal']
    # x = data['investments_num']
    # x = data['payments_num']
    # x= data['Pr']
    # x=data['maxpay']
    # x=data['D_ind']

    nums1 = []
    for data in x:
        nums1.append(data)
    nums = nums1[:131]
    return nums


def mean(numbers):  # 计算平均值
    s = 0.0
    for n in numbers:
        num = int(n)
        s = s + num
    return s / len(numbers)


def dev(numbers, mean):  # 计算标准差
    sdev = 0
    for num in numbers:
        sdev = sdev + (num - mean) ** 2
    return pow(sdev / (len(numbers) - 1), 0.5)


def median(numbers):  # 计算中位数
    news_numbers = sorted(numbers)
    size = len(news_numbers)
    if size % 2 == 0:
        med = (news_numbers[size // 2 - 1] + news_numbers[size // 2]) / 2
    else:
        med = news_numbers[size // 2]
    return med


filename = 'features2.csv'  # 打开已有文件
numbers = getdata(filename)
m = mean(numbers)
print("平均值为{},标准差为{:.2},中位数为{}.".format(m, dev(numbers, m), median(numbers)))

