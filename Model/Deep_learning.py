import os
import torch
from torch import nn
from torch.utils.data import Dataset
import sklearn
import time
import numpy as np
import sklearn.metrics as metrics
import pickle
import matplotlib.pyplot as plt
import Config
from torchvision.transforms import Compose, Normalize, ToTensor
from torch.utils.data import DataLoader
from sklearn.preprocessing import normalize
from sklearn.metrics import classification_report
from torch.optim.optimizer import Optimizer
from Evaluation_metrics import Metrix_computing


class FNN_Model(nn.Module):
    def __init__(self, Num_classes):
        super(FNN_Model, self).__init__()

        self.feature = nn.Sequential(
            nn.Linear(102, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.output = nn.Sequential(
            nn.Linear(16, Num_classes)
        )

    def forward(self, x):
        embedding = self.feature(x)
        x = self.output(embedding)

        outputDir = {"embedding": embedding, "output": x}

        return outputDir



class CNN_Model(nn.Module):
    def __init__(self, Num_classes):
        super(CNN_Model, self).__init__()

        self.feature0 = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.adaptpool2 = nn.AdaptiveAvgPool1d(128)
        self.feature1 = nn.Sequential(
            nn.Dropout(0.2),

            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

        self.output = nn.Linear(32, Num_classes)

    def forward(self, x):
        x = self.feature0(x)
        x = torch.flatten(x, start_dim=1)
        x = self.adaptpool2(x)
        embedding = self.feature1(x)
        x = self.output(embedding)

        outputDir = {"embedding": embedding, "output": x}

        return outputDir



def Train_one_epoch(model, optimizer, data_loader, device, epoch):
    # 训练模式
    model.train()
    optimizer.zero_grad()

    task_ce_criterion = nn.CrossEntropyLoss().to(device)

    # metric
    acclosses = AverageMeter()
    train_loss_sum, train_acc_sum, n = 0.0, 0.0, 0
    all_prediction, all_labels = [], []
    for batch, (data, labels) in enumerate(data_loader):
        # 数据
        data, labels = data.float().to(device), labels.long().to(device)

        # 前向
        outputs = model(data)["output"]

        # loss
        loss = task_ce_criterion(outputs, labels)
        acclosses.update(loss.item(), data.shape[0])

        # 后向
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrix
        # train_loss_sum += loss.cpu().item()
        train_acc_sum += (outputs.argmax(dim=1) == labels).sum().cpu().item()
        n += len(labels)

        # 将cuda上的数据转移到cpu进行计算
        _, prediction = torch.max(outputs.data, dim=1)
        all_prediction.extend(prediction.to('cpu'))
        all_labels.extend(labels.to('cpu'))

    train_loss_sum = acclosses.avg

    confusion_matrix, sen, spe, ae, hs, report = Metrix_computing(all_labels, all_prediction)

    return (train_loss_sum, train_acc_sum / n, sen, spe, ae, hs, confusion_matrix, report)



@torch.no_grad()
def Evaluate(model, dataloader, device):
    # 训练模式
    model.eval()

    # Loss Function
    task_ce_criterion = nn.CrossEntropyLoss().to(device)

    # Metrix
    acclosses = AverageMeter()  # 损失累积
    test_loss_sum, test_acc_sum, n = 0, 0, 0
    all_prediction, all_labels = [], []


    for batch, (data, labels) in enumerate(dataloader):
        data, labels = data.float().to(device), labels.long().to(device)

        outputs = model(data)["output"]

        # 计算损失函数
        loss = task_ce_criterion(outputs, labels)
        acclosses.update(loss.item(), data.shape[0])

        _, prediction = torch.max(outputs.data, dim=1)
        all_prediction.extend(prediction.to('cpu'))
        all_labels.extend(labels.to('cpu'))

        test_acc_sum += (outputs.argmax(dim=1) == labels).sum().cpu().item()
        n += len(labels)

    test_loss_sum = acclosses.avg

    confusion_matrix, sen, spe, ae, hs, report = Metrix_computing(all_labels, all_prediction)

    return (test_loss_sum, test_acc_sum / n,  sen, spe, ae, hs, confusion_matrix, report)




def Normalized_confusion_matrix(confusion_matrix):
    return normalize(confusion_matrix, axis=1, norm='l1')




class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def Recod_and_Save_Train_Detial(count, dir, train_recorder, test_recorder, show_fucntion=False):
    # Loss, ACC, SEN, SPE
    train_loss, test_loss = [], []
    train_acc, test_acc = [], []
    train_sen, test_sen = [], []
    train_spe, test_spe = [], []
    for i in range(len(train_recorder)):
        train_loss.append(train_recorder[i][0])
        train_acc.append(train_recorder[i][1])
        train_sen.append(train_recorder[i][2])
        train_spe.append(train_recorder[i][3])

        test_loss.append(test_recorder[i][0])
        test_acc.append(test_recorder[i][1])
        test_sen.append(test_recorder[i][2])
        test_spe.append(test_recorder[i][3])


    fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(6, 8))

    for i, record in enumerate([("loss", train_loss, test_loss), ("acc", train_acc, test_acc),
                                ("sen", train_sen, test_sen), ("spe", train_spe, test_spe)]):

        ax[i].set_title(record[0])
        ax[i].plot(record[1], label="Train")
        ax[i].plot(record[2], color="red", label="Test")
        ax[i].legend()


    # 画图
    plt.subplots_adjust(hspace=0.5, top=0.95, bottom=0.05)
    plt.savefig(f"{dir}\\Record\\Fold_{count}_train_detail.jpg")
    if show_fucntion == True:
        plt.show()
    else:
        plt.clf()

    # 记录存储
    record = {
        "train_recorder": train_recorder,
        "test_recorder": test_recorder
    }

    with open(f"{dir}\\Record\\Fold_{count}_train_detail_record.dat", 'wb') as f:
        pickle.dump(record, f)

    return



class DatasetLoad(Dataset):
    def __init__(self, data_dir, feature_name, input_transform=None):
        self.data = data_dir
        self.feature_name = feature_name
        self.input_transform = input_transform

    def __getitem__(self, index):
        samples = self.Data_Acq(self.data[index])
        data = samples[self.feature_name]
        label = samples["label"]

        # 转化
        if self.input_transform is not None:
            if type(self.input_transform) is list:
                data = (data - self.input_transform[0]) / self.input_transform[1]
            else:
                data = self.input_transform(data)

        return data, label

    def Data_Acq(self, dir):
        file = open(dir, 'rb')
        sample = pickle.load(file, encoding='latin1')
        file.close()
        return sample

    def __len__(self):
        return len(self.data)



# Compute the mean and std value of train dataset.
def Get_mean_and_std(dataset):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)

    # # For 语谱图
    if "spectrogram" in dataset.feature_name:
        mean = torch.zeros(1)
        std = torch.zeros(1)
        print("Spectrogram based")
        print('==> Computing mean and std..')
        for inputs, targets in dataloader:
            mean += inputs.mean()
            std += inputs.std()

        mean.div_(len(dataset))
        std.div_(len(dataset))

        return mean, std
    else:
        print("Statistics_feature")
        print('==> Computing mean and std..')

        value = torch.zeros((1, 102))
        for inputs, targets in dataloader:
            value = torch.concat((value, inputs))

        value = value[1:]
        mean = value.mean(dim=0)
        std = value.std(dim=0)

        return mean.cpu().numpy() , std.cpu().numpy()



def Load_data(dir):
    data_dir = []
    with open(dir, 'r') as file:
        line = file.readline()
        while line:
            data_dir.append(line.strip())
            line = file.readline()

    return data_dir



if __name__ == '__main__':

    count = 0
    train_data_dir_list = Load_data(f"{Config.savedir_train_and_test}\\Fold_{count}\\train_list.txt")
    print(len(train_data_dir_list))

    nor, crk, wheeze, both = 0, 0, 0, 0
    dataloader = torch.utils.data.DataLoader(
        DatasetLoad(train_data_dir_list, "signal"),
        batch_size=1,
        shuffle=True,
        num_workers=0
    )
    for data, label in dataloader:
        if label == 0:
            nor += 1
        elif label == 1:
            crk += 1
        elif label == 2:
            wheeze += 1
        elif label == 3:
            both += 1

    print(nor, crk, wheeze, both)



    test_data_dir_list = Load_data(f"{Config.savedir_train_and_test}\\Fold_{count}\\test_list.txt")
    print(len(test_data_dir_list))
    nor, crk, wheeze, both = 0, 0, 0, 0
    dataloader = torch.utils.data.DataLoader(
        DatasetLoad(test_data_dir_list, "signal"),
        batch_size=1,
        shuffle=True,
        num_workers=0
    )
    for data, label in dataloader:
        if label == 0:
            nor += 1
        elif label == 1:
            crk += 1
        elif label == 2:
            wheeze += 1
        elif label == 3:
            both += 1

    print(nor, crk, wheeze, both)






