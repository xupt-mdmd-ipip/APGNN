import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.model_selection import train_test_split
import torch.optim as optim
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from operator import truediv
# from IP_Net_Proposed import *
# from torchsummary import summary
# import pandas as pd
import logging
# import matplotlib.pyplot as plt
import torch
# from A import *
import torch.nn as nn


# pip install scipy https://pypi.tuna.tsinghua.edu.cn/simple gevent
##模型效果衡量函数##
def AA_andEachClassAccuracy(confusion_matrix):
    counter = confusion_matrix.shape[0]
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return average_acc


##保存训练日志##
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


patch_size = 15  # 每个像素周围提取 patch 的尺寸
pca_components = 30  # 使用 PCA 降维，得到主成分的数量
##导入训练数据和测试数据##
Xtrain = np.load(r'/media/XUPT/0158571d-fa1a-4147-b721-b7993d1903b5/lx/master/dataset_split/Indian_pines_corrected/train_data_15.npy')
Ytrain = np.load(r'/media/XUPT/0158571d-fa1a-4147-b721-b7993d1903b5/lx/master/dataset_split/Indian_pines_corrected/train_label_15.npy')
Xtest = np.load(r'/media/XUPT/0158571d-fa1a-4147-b721-b7993d1903b5/lx/master/dataset_split/Indian_pines_corrected/test_data_15.npy')
Ytest = np.load(r'/media/XUPT/0158571d-fa1a-4147-b721-b7993d1903b5/lx/master/dataset_split/Indian_pines_corrected/test_label_15.npy')
print('Xtrain shape: ', Xtrain.shape)  # (80, 15, 15, 30)
print('ytrain shape', Ytrain.shape)
print('Xtest shape: ', Xtest.shape)  # (10169, 15, 15, 30)
print('Ytest shape', Ytest.shape)

# 3D
Xtrain = Xtrain.reshape(-1, patch_size, patch_size, pca_components, )
Xtest = Xtest.reshape(-1, patch_size, patch_size, pca_components, )

# 为了适应 pytorch 结构，数据要做 transpose
Xtest = Xtest.transpose(0, 3,  2, 1)
Xtrain = Xtrain.transpose(0, 3,  2, 1)

""" Train dataset"""


class TrainDS(torch.utils.data.Dataset):
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


""" Test dataset"""


class TestDS(torch.utils.data.Dataset):
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
train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=100, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=369, shuffle=False)

# 使用GPU训练，可以在菜单 "代码执行工具" -> "更改运行时类型" 里进行设置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 网络放到GPU上
from zhenshi_OD23D import BasicBlock111

Model = BasicBlock111(30, 60)
print(Model)
# Model.train(mode=True)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(Model.parameters(), lr=0.002)

# 开始训练
total_loss = 0
OA_list = []
AA_list = []
# logger = get_logger('Net_15Patch_30Channel_ten_shot_Proposed_log.txt')
for epoch in range(100):
    correct = 0
    count = 0
    for i, (inputs, labels) in enumerate(train_loader):
        Model.train(mode=True)
        inputs = inputs.to(device)  # ([80, 1, 30, 15, 15])

        labels = labels.to(device)  # [80]

        # 优化器梯度归零
        optimizer.zero_grad()
        # 正向传播 +　反向传播 + 优化
        outputs = Model(inputs)  # [80, 16]

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        predicted = torch.max(outputs.data, 1)[1]
        correct += (predicted == labels).sum()
    for inputs, _ in test_loader:
        Model.eval()
        inputs = inputs.to(device)
        outputs = Model(inputs)
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        if count == 0:
            y_pred_test = outputs
            count = 1
        else:
            y_pred_test = np.concatenate((y_pred_test, outputs))
    oa = accuracy_score(Ytest, y_pred_test)
    confusion = confusion_matrix(Ytest, y_pred_test)
    aa = AA_andEachClassAccuracy(confusion)
    print('Epoch :{}\t Loss:{:.6f}\t Accuracy:{:.3f} OA:{:.2f}\t AA:{:.2f}'.format(epoch + 1, loss.data.item(),
                                                                                   float(correct * 100) / float(16),
                                                                                   oa * 100, aa * 100))
    # logger.info(
    #     'Epoch:{}\t loss={:.5f}\t OA:{:.2f}\t AA:{:.2f}'.format(epoch + 1, loss.data.item(), oa * 100, aa * 100))
    # OA_list.append(oa * 100)
    # AA_list.append(aa * 100)
    # torch.save(Model.state_dict(), 'save_model\\IndianPines_15Patch_30Channel_ten_shot_Net_Proposed_' + str(
    #                epoch + 1) + '.pkl')
    # oa_test=0.99
    # aa_test=0.99
    # if oa>oa_test :
    #     if aa>aa_test:
    #         torch.save(Model.state_dict(), 'IndianPines_Net_Proposed_best.pkl')
# IP_Net_Proposed_OA_list = np.array(OA_list)
# IP_Net_Proposed_AA_list = np.array(AA_list)
# np.save(
#     '/home/LiuRunJi/径向基/Hyperspectral_Image_Classification/IndianPines_experiment/IP_Result/IP_Net_Proposed_15Patch_30Channel_OA_list',
#     IP_Net_Proposed_OA_list)
# np.save(
#     '/home/LiuRunJi/径向基/Hyperspectral_Image_Classification/IndianPines_experiment/IP_Result/IP_Net_Proposed_15Patch_30Channel_AA_list',
#     IP_Net_Proposed_AA_list)
# torch.save(Model.state_dict(), 'IndianPines_Net_Proposed.pkl')
# print('Finished Training')
