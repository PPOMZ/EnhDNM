import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
import time
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import argparse
import logging
import torch.nn.functional as F
import math
from scipy.io import loadmat
# pytorch 版本 移植：龚鸣 2021/2/8 原文: Decision-tree-initialized Dendritic Neuron Model
# for Fast and Accurate Data Classification
parser = argparse.ArgumentParser(description='Your description here')
parser.add_argument('--name', type=str, default='Echocardiogram_Data', help='Dataset name')
parser.add_argument('--iter', type=int, default=1000, help='Number of iterations')
parser.add_argument('--num_models', type=int, default=4, help='Number of models')
parser.add_argument('--device', type=str, default="0", help='device id')
parser.add_argument('--flag_confi', type=str, default=False, help='Number of models')
args = parser.parse_args()

name = args.name  
# path = r"Datasets/Datasets2/"+name+".mat"  # mat文件路径
path = r"Datasets/Datasets/real_data/MAT/"+name+".mat"  # mat文件路径

data_high = loadmat(path)

if "MAT" in path:
    train_data = data_high["Xtr"].T
    train_label = data_high["Ytr"].T-1
    test_data = data_high["Xtt"].T
    test_label = data_high["Ytt"].T-1
    x_data=np.concatenate((train_data,test_data))
    y_data=np.concatenate((train_label,test_label))
else:
    data1 = data_high['data']
    train_data = data1['train'][0][0]
    train_label = data1['trainLabel'][0][0][:,1]
    test_data = data1['test'][0][0]
    test_label = data1['testLabel'][0][0][:,1]
    x_data=np.concatenate((train_data,test_data))
    y_data=np.concatenate((train_label,test_label))
    y_data=np.expand_dims(y_data,axis=-1)

row=x_data.shape[1]+1

parameter = {}

parameter['German_data'] = [10, 10, 0.5,1]
parameter['SpectEW_data'] = [20, 20, 0.5,10]
parameter['Tic-tac-toe_data'] = [10, 10, 0.5,20]
parameter['Australia_data'] = [10, 10, 0.5,10]
parameter['KrVsKpEW_data'] = [10, 10, 0.5,10]
parameter['Vote_data'] = [5, 10, 0.5,2]
parameter['CongressEW_data'] = [5, 15, 0.5,2]
#----------------------------------------------------------------------------------------------------------
parameter['Breast_Cancer_Wisconsin_Diagnosis'] = [1, 10, 0.5,2]
parameter['Breast_Cancer_Wisconsin_Original'] = [10, 1, 0.5,20]
parameter['BUPA_Liver_Disorders'] = [1, 10, 0.5,10]
parameter['Climate_Model_Simulation_Crashes'] = [2, 10, 0.5,2]
parameter['Credit_Approval'] = [10, 10, 0.5,10]
parameter['Diabetic_Retinopathy_Debrecen'] = [5, 20, 0.5,10]
parameter['Echocardiogram_Data'] = [10, 10, 0.5,10]
parameter['Heart_Disease_Cleveland'] = [20, 1, 0.5,10]
parameter['Hepatitis_Domain'] = [20, 20, 0.5,2]
parameter['Indian_Liver_Patient_Dataset'] = [1, 20, 0.5,2]
parameter['Pima_Indians_Diabetes'] = [10, 1, 0.5,2]
parameter['SPECTF_Heart_Dataset'] = [10,10, 0.5,10]
parameter['US_Congressional_Voting_Records'] = [20, 1, 0.5,20]
parameter['Ionosphere'] = [10, 7, 0.5,8]
parameter['Parkinsons_Dataset'] = [8 ,8, 0.5,25]

k1 = parameter[name][0] #dnm
k2 = parameter[name][1]#dnm
qs = parameter[name][2]#dnm
M = parameter[name][3]#dnm
# M=9
repeat_k=6
num_splits = 5
iter = args.iter
test_recall_index=0
test_f1_index=0
p_value_index=0
acc_index=0
# device="cpu"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# row = data.shape[1]  # 读取矩阵长度c
# x_data = data[:, 0:row - 1]
# t = -1
# y_data = data[:, row-1:row]
learning_rate = 0.01
flag_confi=args.flag_confi
milestones=[]
mean=0
std=0.1
max_number=100000000
class Dnm_Net_old(torch.nn.Module):
    def __init__(self, M, row):
        """
        M ： 待训练的数据行数
        row ： 待训练的数据列数
        """
        super(Dnm_Net_old, self).__init__()
        self.W_orig = nn.Parameter(torch.randn(M, row - 1, device=device, dtype=torch.float32))
        self.q_orig = nn.Parameter(torch.randn(M, row - 1, device=device, dtype=torch.float32))
    def forward(self, x):
        """
        此方法中实现了DNM的前向传播，输入X为二维矩阵
        """
        X_orig = torch.tensor(x, dtype=torch.float32).to(self.W_orig.device)
        X_orig = X_orig.unsqueeze(dim=1)
        X_orig = X_orig.expand(X_orig.shape[0], M, X_orig.shape[2])  # 若输入X为多维矩阵，此处应修改
        y_temp_orig = torch.sigmoid(k1 * 1.0 * (torch.mul(X_orig, self.W_orig) - self.q_orig))
        y_orig = torch.prod(y_temp_orig, dim=2, keepdim=False)  # 将不同特征的维度相乘
        y1_orig = torch.sum(y_orig, dim=1, keepdim=True)  # 并相加
        y2_orig = torch.sigmoid(k2 * 1.0 * (y1_orig - qs))
        return y2_orig
class Dnm_Net(torch.nn.Module):
    def __init__(self, M, row):
        """
        M ： 待训练的数据行数
        row ： 待训练的数据列数
        """
        super(Dnm_Net, self).__init__()
        self.W_orig = nn.Parameter(torch.randn(M, row - 1, device=device, dtype=torch.float32))
        self.q_orig = nn.Parameter(torch.randn(M, row - 1, device=device, dtype=torch.float32))

        self.Conv1d_cross1=nn.Conv1d( row-1,1,1,padding=0)
        self.Conv1d_1 = nn.Conv1d(  1,row-1, 1,padding=0)
        self.W_orig3 = nn.Parameter(torch.randn(M, row - 1, device=device, dtype=torch.float32))
        self.q_orig3 = nn.Parameter(torch.randn(M, row - 1, device=device, dtype=torch.float32))
        self.init()
    def init(self):
        # nn.init.normal_(self.k2_orig, mean=0, std=0.5)
        # torch.nn.init.xavier_uniform_(self.k2_orig, gain=0.1)
        return
    def SFL(self,feat):
        feat_mean_config1=self.Conv1d_cross1(feat.permute(0,2,1)).permute(0,2,1)#Conv1d不能多，是Vertebral数据集的关键
        feat_mean_config4=torch.sigmoid(feat_mean_config1*feat)#sigmoid 是两个特殊数据集的关键，feat是全部的关键
        feat=feat_mean_config4
        return feat
    def forward(self, x):
        """
        此方法中实现了DNM的前向传播，输入X为二维矩阵
        """
        if flag_confi==True:
            pass
        else:
            X_orig = torch.tensor(x, dtype=torch.float32).to(self.W_orig.device)
            X_orig = X_orig.unsqueeze(dim=1)
            X_orig_old=X_orig
            X_orig = X_orig.expand(X_orig.shape[0], M, X_orig.shape[2])  # 若输入X为多维矩阵，此处应修改

            qs_adptive=torch.sigmoid(torch.mean(torch.mean(self.Conv1d_1(torch.mean(X_orig*self.W_orig3-self.q_orig3,dim=-1,keepdim=True).permute(0,2,1)).permute(0,2,1),dim=-1),dim=-1,keepdim=True))


            y_temp_orig = torch.sigmoid(k1 * 1.0 * (torch.mul(X_orig, self.W_orig) - self.q_orig))

            y_temp_orig=self.SFL(y_temp_orig)


            y_orig = torch.prod(y_temp_orig, dim=2, keepdim=False)  # 将不同特征的维度相乘
            y1_orig = torch.sum(y_orig, dim=1, keepdim=True)  # 并相加

            # y2_orig = torch.sigmoid(k2 * 1.0 * (y1_orig - qs))
            y2_orig = torch.sigmoid(k2 * 1.0 * (y1_orig - qs_adptive))
            if self.training:
                return y2_orig
            else:
                return y2_orig
class danjinzhi_Net(torch.nn.Module):
    def __init__(self, row ):
        """
        M ： 待训练的数据行数
        row ： 待训练的数据列数
        """
        super(danjinzhi_Net, self).__init__()
        self.W_orig = nn.Parameter(torch.randn( row-1, device=device, dtype=torch.float32))
        self.q_orig = nn.Parameter(torch.randn( row-1, device=device, dtype=torch.float32))
        self.input_size=row-1
        self.k2=5
        self.qs=0.5
        # self.init()
    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.input_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
    def forward(self, x):
        """
        此方法中实现了DNM的前向传播，输入X为二维矩阵
        """
        X_orig = torch.tensor(x, dtype=torch.float32).to(self.W_orig.device)

        y_temp_orig = (torch.mul(X_orig, self.W_orig) + self.q_orig)

        y_orig = torch.prod(y_temp_orig, dim=-1, keepdim=False)  # 将不同特征的维度相乘


        y2_orig = torch.sigmoid(self.k2 * 1.0 * (y_orig - self.qs)).unsqueeze(-1)
        if self.training:
            return y2_orig
        else:
            return y2_orig      
loss_fn = torch.nn.MSELoss(reduction='mean')
logging.basicConfig(filename=args.name+"_result_ours.log",
                    level=logging.INFO,
                    format='%(message)s',
                    filemode='w')
# logging.info("parameter list:lr={}".format(learning_rate))
logging.info("parameter list: M={}, k1={}, k2={}, qs={}, lr={}".format(M,k1,k2,qs,learning_rate))
for ii in range(repeat_k):
    print("This repeat is "+str(ii))
    kf = KFold(n_splits=num_splits, shuffle=True)
    for times,(train_index, test_index) in enumerate(kf.split(x_data)):

        X_train, X_test, y_train, y_test = x_data[train_index], x_data[test_index], y_data[train_index], y_data[test_index]
        standardScaler = MinMaxScaler()
        # standardScaler = StandardScaler()
        standardScaler.fit(x_data)

        X_train = standardScaler.transform(X_train)
        X_test = standardScaler.transform(X_test)
        y_train = torch.from_numpy(y_train).to(device, dtype=torch.float32)
        y_test = torch.from_numpy(y_test).to(device, dtype=torch.float32)

        model = Dnm_Net(M, row).to(device)
        # model = danjinzhi_Net(row).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler=lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)
        for epoch in range(iter):
            y2_orig = model(X_train)
            loss = loss_fn(y2_orig,y_train)
            dnm_accuracy_train_value = np.int64(y2_orig.cpu() > 0.5)
            acc = accuracy_score(y_train.cpu(), dnm_accuracy_train_value)
            start = time.time()
            # print(optimizer.state_dict()['param_groups'][0]['lr'])
            # print('Train : Epoch :{}, Loss:{:.4f}, Acc: {:.4f}'
            #     .format(epoch+1,loss, acc))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if milestones:
                scheduler.step()
        with torch.no_grad():
            model.eval()
            y_orig_test = model(X_test)
            loss = loss_fn(y_orig_test, y_test)
            dnm_accuracy_test_value = np.int64(y_orig_test.cpu() > 0.5)
            acc = accuracy_score(y_test.cpu(), dnm_accuracy_test_value)
            start = time.time()
            print('Test{} : Loss:{:.4f}, Acc: {:.4f}'
                .format(times,loss, acc))
            logging.info('Test{} : Loss:{:.4f}, Acc: {:.4f}'
                .format(times,loss, acc))
            test_fpr, test_tpr, test_threshold = roc_curve(y_test.cpu(), dnm_accuracy_test_value)  ###计算真正率和假正率
            test_auc = auc(test_fpr, test_tpr)

            test_recall = recall_score(y_test.cpu(), dnm_accuracy_test_value)

            test_f1 = f1_score(y_test.cpu(), dnm_accuracy_test_value)

            p_value = precision_score(y_test.cpu(), dnm_accuracy_test_value)
            print('Test_recall : {:.4f}  Test_f1 : {:.4f}  pvalue : {:.4f}'.format(test_recall,test_f1,p_value))
            logging.info('Test_recall : {:.4f}  Test_f1 : {:.4f}  pvalue : {:.4f}'.format(test_recall,test_f1,p_value))
            logging.info('====================================================================================================')
            test_recall_index=test_recall_index+test_recall
            test_f1_index=test_f1_index+test_f1
            p_value_index=p_value_index+p_value
            acc_index=acc_index+acc
test_recall_avg=test_recall_index/(num_splits*repeat_k)
test_f1_avg=test_f1_index/(num_splits*repeat_k)
p_value_avg=p_value_index/(num_splits*repeat_k)
acc_avg=acc_index/(num_splits*repeat_k)
print('acc_avg :{:.4f} Test_recall_avg : {:.4f}  Test_f1_avg : {:.4f}  pvalue_avg : {:.4f}'.format(acc_avg,test_recall_avg,test_f1_avg,p_value_avg))
logging.info('acc_avg :{:.4f} Test_recall_avg : {:.4f}  Test_f1_avg : {:.4f}  pvalue_avg : {:.4f}'.format(acc_avg,test_recall_avg,test_f1_avg,p_value_avg))