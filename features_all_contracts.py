import os
import csv
import sys
import numpy as np
from feature_extraction import ContractFeature

class ExtractFeatureOfAllContract(object):  #定义类
    def __init__(self):#强制给定义类绑定相应的属性，初始化参数
        self.ponzi_file = './data/ponziContracts.csv'
        self.non_ponzi_file = './data/non_ponziContracts.csv'
        self.feature_save_file = './data/features.csv'
        self.feature2_save_file = './data/features3.csv'
        #定义文件表头fieldname+fieldnames2
        self.fieldnames = ['address', 'return_rio', 'A_bal', 'investments_num', 'payments_num', 'Pr', 'maxpay', 'D_ind']
        self.fieldnames2 = ['GASLIMIT', 'EXP', 'CALLDATALOAD', 'SLOAD', 'CALLER', 'LT', 'GAS', 'MOD', 'MSTORE', 'ponzi']

        self.initialize_files()#初始化文件

        self.extract_features()#提取特征

    def initialize_files(self):
        # extract feature of opcode
        #os.path模块主要用于文件的属性获取,exists是“存在”的意思，所以顾名思义，os.path.exists()就是判断括号里的文件是否存在的意思，括号内的可以是文件路径。存在返回 True 不存在返回 False
        if os.path.exists(self.feature2_save_file):
            os.remove(self.feature2_save_file)#os.remove()删除文件feature2_save_file
        if not os.path.exists(self.feature2_save_file):
            with open(self.feature2_save_file, 'w') as f2:#f2是打开文件的对象feature2_save_file
                writer = csv.DictWriter(f2, fieldnames=self.fieldnames + self.fieldnames2)#fieldnames用来设置文件的表头
                writer.writeheader()#调用writeheader将fieldnames写入csv文件的第一行

    def extract_features(self):
        with open(self.ponzi_file, 'r') as ponzi_csv:#读取ponziContracts.csv文件
            csv_reader = csv.reader(ponzi_csv, delimiter=',')#创建一个reader对象，分隔符为，
            line_count = 0 #行数从0开始
            for row in csv_reader:#按行遍历读取文件
                features2 = {}#创建一个字典，将ponzi的特征写入features2构建一个组合型特征数据集
                if line_count > 0:#如果行数大于0
                    contractAddress = row[1] #按行读取row[1]表示第一行第一个元素即合约地址
                    if os.path.exists('./data/contracts/ponzi/' + contractAddress + '.txt'):#在data/contarct/ponzi文件夹中是否存在合约地址的txt
                        contractFeature = ContractFeature(contractAddress, 'ponzi')#调用特征提取函数提取txt文件中的特征

                        features2['address'] = contractAddress
                        features2['return_rio'] = contractFeature.return_rio
                        features2['A_bal'] = contractFeature.A_bal
                        features2['investments_num'] = contractFeature.investments_num
                        features2['payments_num'] = contractFeature.payments_num
                        features2['Pr'] = contractFeature.Pr
                        features2['maxpay'] = contractFeature.maxpay
                        features2['D_ind'] = contractFeature.D_ind
                        features2['GASLIMIT'] = contractFeature.action_ratio['GASLIMIT']
                        features2['EXP'] = contractFeature.action_ratio['EXP']
                        features2['CALLDATALOAD'] = contractFeature.action_ratio['CALLDATALOAD']
                        features2['SLOAD'] = contractFeature.action_ratio['SLOAD']
                        features2['CALLER'] = contractFeature.action_ratio['CALLER']
                        features2['LT'] = contractFeature.action_ratio['LT']
                        features2['GAS'] = contractFeature.action_ratio['GAS']
                        features2['MOD'] = contractFeature.action_ratio['MOD']
                        features2['MSTORE'] = contractFeature.action_ratio['MSTORE']
                        features2['ponzi'] = 1

                        with open(self.feature2_save_file, 'a') as feat2_file_csv:
                            writer = csv.DictWriter(feat2_file_csv, fieldnames=self.fieldnames + self.fieldnames2)
                            writer.writerow(features2)
                features2 = {}
                line_count += 1
        #正常合约的特征提取
        with open(self.non_ponzi_file, 'r') as nonponzi_csv:
            csv_reader = csv.reader(nonponzi_csv, delimiter=',')
            line_count = 0
            for row in csv_reader:
                features2 = {}
                if line_count > 0:
                    contractAddress = row[1]
                    if os.path.exists('./data/contracts/nonponzi/' + contractAddress + '.txt'):
                        contractFeature = ContractFeature(contractAddress, 'nonponzi')

                        features2['address'] = contractAddress
                        features2['return_rio'] = contractFeature.return_rio
                        features2['A_bal'] = contractFeature.A_bal
                        features2['investments_num'] = contractFeature.investments_num
                        features2['payments_num'] = contractFeature.payments_num
                        features2['Pr'] = contractFeature.Pr
                        features2['maxpay'] = contractFeature.maxpay
                        features2['D_ind'] = contractFeature.D_ind
                        features2['GASLIMIT'] = contractFeature.action_ratio['GASLIMIT']
                        features2['EXP'] = contractFeature.action_ratio['EXP']
                        features2['CALLDATALOAD'] = contractFeature.action_ratio['CALLDATALOAD']
                        features2['SLOAD'] = contractFeature.action_ratio['SLOAD']
                        features2['CALLER'] = contractFeature.action_ratio['CALLER']
                        features2['LT'] = contractFeature.action_ratio['LT']
                        features2['GAS'] = contractFeature.action_ratio['GAS']
                        features2['MOD'] = contractFeature.action_ratio['MOD']
                        features2['MSTORE'] = contractFeature.action_ratio['MSTORE']
                        features2['ponzi'] = 0

                        with open(self.feature2_save_file, 'a') as feat2_file_csv:
                            writer = csv.DictWriter(feat2_file_csv, fieldnames=self.fieldnames + self.fieldnames2)
                            writer.writerow(features2)
                features2 = {}
                line_count += 1


if __name__ == '__main__':
    extract_feature_of_all_contract = ExtractFeatureOfAllContract()
