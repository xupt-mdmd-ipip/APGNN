from os import sep
from pickle import TRUE
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import scipy.io as sio
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.model_selection import train_test_split
import torch.optim as optim
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from operator import truediv
# from torchsummary import summary
# import pandas as pd
import logging
import matplotlib.pyplot as plt

from einops import rearrange

# from timm.models.layers import DropPath, to_2tuple, trunc_normal_
# from timm.models.registry import register_model
# from timm.models.vision_transformer import _cfg
import numpy as np
# from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint
# from models import p2t4
# from models import SSFTTnet
# from models import ViT2
# from models import ViT
# from models import GT
# from models import resnet12
import sys
import datetime
import os
# -------------------------------------------------------------------------------
# 开始记录终端内容并保存为文本文档
print("**************************************************")
print("Starts to record terminal contents and save them as text files ! ! !")
print("**************************************************")

day = datetime.datetime.now()
day_str = day.strftime('%m_%d_%H_%M')
save_loger_name = './terminal_log/' + '/' + day_str + 'loger.txt'
# save_matrices_name = './matrices/' + '/' + day_str + 'matrices.mat'


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


log_type = sys.getfilesystemencoding()
sys.stdout = Logger(save_loger_name)


def get_module():
    def main_module_name():
        mod = sys.modules['__main__']
        file = getattr(mod, '__file__', None)
        return file and os.path.splitext(os.path.basename(file))[0]

    def modname(fvars):

        file, name = fvars.get('__file__'), fvars.get('__name__')
        if file is None or name is None:
            return None

        if name == '__main__':
            name = main_module_name()
        return name

    module_name = modname(globals())
    print(globals())
    print("当前运行的模块为：", module_name)
    print("**************************************************")


get_module()


# -------------------------------------------------------------------------------

# 模型效果衡量函数
def AA_andEachClassAccuracy(confusion_matrix):
    counter = confusion_matrix.shape[0]
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return average_acc


# 保存训练日志
def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s]%(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # sh = logging.StreamHandler()
    # sh.setFormatter(formatter)
    # logger.addHandler(sh)

    return logger


# patch_size = 11  # 每个像素周围提取 patch 的尺寸
# pca_components = 30  # 使用 PCA 降维，得到主成分的数量

# 导入训练数据和测试数据
Xtrain = np.load('./dataset_split/Indian_pines_corrected/train_data_15.npy')
Ytrain = np.load('./dataset_split/Indian_pines_corrected/train_label_15.npy')
Xtest = np.load('./dataset_split/Indian_pines_corrected/test_data_15.npy')
Ytest = np.load('./dataset_split/Indian_pines_corrected/test_label_15.npy')
print('ytrain shape', Ytrain.shape)
print('Xtest shape: ', Xtest.shape)  # (10169, 15, 15, 30)
print('Ytest shape', Ytest.shape)
# 计算不同类别数量##
# ytrain_gb = Ytrain.flatten()
# # ytrain_gb = pd.Series(ytrain_gb)
# ytrain_gb = ytrain_gb.value_counts()
# print(ytrain_gb)
# 改变 Xtrain, Ytrain 的形状，以符合 keras 的要求
# Xtest = Xtest.reshape(-1, patch_size, patch_size, pca_components, 1)
# Xtrain = Xtrain.reshape(-1, patch_size, patch_size, pca_components, 1)
# # print('before transpose: Xtrain shape: ', Xtrain.shape)   #(80, 15, 15, 30, 1)
# # 为了适应 pytorch 结构，数据要做 transpose
Xtest = Xtest.transpose(0, 3, 2, 1)
Xtrain = Xtrain.transpose(0, 3, 2, 1)
# Xtest = rearrange(Xtest, 'a b c d -> (a c d) b')
# Xtrain = rearrange(Xtrain, 'a b c d -> (a c d) b')

# print('after transpose: Xtrain shape: ', Xtrain.shape)    #(80, 1, 30, 15, 15)

class TrainDS(torch.utils.data.Dataset):
    """ Train dataset"""

    def __init__(self):
        self.len = Xtrain.shape[0]
        self.x_data = torch.FloatTensor(Xtrain)
        self.y_data = torch.LongTensor(Ytrain)

    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        # 返回文件数据的数目
        return self.len


class TestDS(torch.utils.data.Dataset):
    """ Test dataset"""

    def __init__(self):
        self.len = Xtest.shape[0]
        self.x_data = torch.FloatTensor(Xtest)
        self.y_data = torch.LongTensor(Ytest)

    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        # 返回文件数据的数目
        return self.len


trainset = TrainDS()
testset = TestDS()
train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=64, shuffle=False)

# 使用GPU训练，可以在菜单 "代码执行工具" -> "更改运行时类型" 里进行设置,网络放到GPU上
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

# create model
# TODO

# Model = VAN().to(device)

# Model = create_model(
#     "p2t_base",
#     # pretrained=False,
#     # num_classes=16,
#     # drop_rate=0.1,
#     # drop_path_rate=0.1,
#     # drop_block_rate=None,
# ).to(device)

# Model = SSFTTnet().to(device)
# Model = ViT.Transformer(dim=30, depth=1, heads=1, dim_head=8, mlp_head=1, dropout=0.1, num_channel=30,
# mode="ViT").to(device)
# Model = ViT2.VisionTransformer().to(device)
# Model = resnet12.ResNet(16, 30, 1.0, TRUE)
# from zhenshi_OD23D import BasicBlock111
from zhenshi_OD23D import BasicBlock111_jingtai
Model = BasicBlock111_jingtai(30, 60)

# Model.train(mode=True)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(Model.parameters(), lr=0.0004)

# torch.cuda.empty_cache()
# print(torch.cuda.memory_summary(device="cuda:0", abbreviated=False))
# 开始训练
total_loss = 0
OA_list = []
AA_list = []
# logger = get_logger('Net_15Patch_30Channel_ten_shot_Proposed_log.txt')
# print("zheshi  1:")
for epoch in range(500):
    correct = 0
    count = 0
    # print("zheshi  2:")

    for i, (inputs, labels) in enumerate(train_loader):
        Model.train(mode=True)
        inputs = inputs.to(device)  # ([80, 1, 30, 15, 15])

        labels = labels.to(device)
        # 优化器梯度归零
        optimizer.zero_grad()
        # 正向传播 +　反向传播 + 优化
        outputs = Model(inputs)
        # print("****************",outputs.shape)    #[80, 16]
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        predicted = torch.max(outputs.data, 1)[1]
        correct += (predicted == labels).sum()
        # print("zheshi  3:")

    for inputs, _ in test_loader:
        # print("zheshi  4:")

        Model.eval()
        inputs = inputs.to(device)
        outputs = Model(inputs)
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        if count == 0:
            y_pred_test = outputs
            count = 1
        else:
            y_pred_test = np.concatenate((y_pred_test, outputs))
        # print("zheshi  5:")

    oa = accuracy_score(Ytest, y_pred_test)
    confusion = confusion_matrix(Ytest, y_pred_test)
    aa = AA_andEachClassAccuracy(confusion)
    print('Epoch :{}\t Loss:{:.6f}\t Accuracy:{:.3f} OA:{:.2f}\t AA:{:.2f}'.format(epoch + 1, loss.data.item(),
                                                                                   float(correct * 100) / float(16),
                                                                                   oa * 100, aa * 100))
    # print("zheshi  6:")

# -------------------------------------------------------------------------------
# 开始进行训练
# def train(train_loader, epochs):
#     # 使用GPU训练，可以在菜单 "代码执行工具" -> "更改运行时类型" 里进行设置
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     # 网络放到GPU上
#     model = VAN().to(device)
#     print(model)
#     # 交叉熵损失函数
#     criterion = nn.CrossEntropyLoss()
#     # 初始化优化器
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
#     # 开始训练
#     total_loss = 0
#     for epoch in range(epochs):
#         model.train()
#         for i, (data, target) in enumerate(train_loader, train_label_15):
#             data, target = data.to(device), target.to(device)
#             # 正向传播 +　反向传播 + 优化
#             # 通过输入得到预测的输出
#             outputs = model(data)
#             # 计算损失函数
#             loss = criterion(outputs, target)
#             # 优化器梯度归零
#             optimizer.zero_grad()
#             # 反向传播
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#         print('[Epoch: %d]   [loss avg: %f]   [current loss: %f]' % (epoch + 1,
#                                                                      total_loss / (epoch + 1),
#                                                                      loss.item()))
#
#     print('Finished Training')
#
#     # plot the accuracy
#     # plt.plot(total_loss)
#     # plt.ylabel('total_loss')
#     # plt.xlabel('total_loss / (epoch + 1)')
#     # plt.title("total_loss / (epoch + 1)")
#     # plt.show()
#     return model, device


# -------------------------------------------------------------------------------
