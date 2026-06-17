import copy
import time

import torch
from sympy.integrals.intpoly import best_origin
from torch import nn
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as Data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model import LeNet


def train_val_data_process():
    train_data = FashionMNIST(root='./data',
        train=True,
        transform=transforms.ToTensor(),
        download=True)

    train_data, val_data=Data.random_split(train_data,[round(len(train_data)*0.8),round(len(train_data)*0.2)])

    train_dataloader = Data.DataLoader(dataset=train_data,
        batch_size=32,
        shuffle=True,
        num_workers=4)

    val_dataloader = Data.DataLoader(dataset=train_data,
        batch_size=32,
        shuffle=True,
        num_workers=4)

    return train_dataloader,val_dataloader

def train_model_process(model,train_dataloader,val_dataloader,num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    criterion = nn.CrossEntropyLoss()
    model=model.to(device)
    best_model_wts = copy.deepcopy(model.state_dict())


    best_acc = 0.0
    train_loss_all=[]

    train_acc_all=[]
    val_loss_all=[]
    val_acc_all=[]
    since = time.time()

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch,num_epochs-1))
        print("-" * 10)
        #初始化参数
        train_loss = 0.0
        train_acc = 0#train_correct
        val_loss = 0.0
        val_acc = 0
        #样本数量
        train_num =0
        val_num =0
        #对每一个batch计算
        for step, (b_x, b_y) in enumerate(train_dataloader):
            #放入设备
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            model.train()
            #前向传播
            output = model(b_x)
            pre_label = torch.max(output, 1)[1]
            loss = criterion(output, b_y)

            #梯度
            optimizer.zero_grad()
            #反向传播
            loss.backward()
            #参数更新
            optimizer.step()

            train_loss += loss.item()*b_x.size(0)

            train_acc += (pre_label == b_y).sum().item()


            train_num += b_x.size(0)


        for step, (b_x, b_y) in enumerate(val_dataloader):
            #放入设备
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            #设置模型为评估模型
            model.eval()
            output = model(b_x)
            pre_label = torch.max(output, 1)[1]
            loss = criterion(output, b_y)
            #损失函数累加
            val_loss += loss.item()*b_x.size(0)
            #预测正确加1
            val_acc += (pre_label == b_y).sum().item()
            #用于验证的样品数量
            val_num += b_x.size(0)

        #计算保存每一次的loss与acc
        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_acc / train_num)
        val_loss_all.append(val_loss / val_num)
        val_acc_all.append(val_acc / val_num)

        print("train loss: {:.4f}  train acc: {:.4f}".format(train_loss_all[-1], train_acc_all[-1]))
        print("val loss:   {:.4f}  val acc:   {:.4f}".format(val_loss_all[-1], val_acc_all[-1]))


        #寻找最高准确度
        if val_acc_all[-1]>best_acc:
            best_acc = max(val_acc_all[-1],best_acc)
            #最高准确度
            best_model_wts = copy.deepcopy(model.state_dict())

    time_use = time.time() - since

    print("训练和验证时间{:.0f}".format(time_use))

    #选择最优参数和模型
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(),"./best_model.pth")

    train_process=pd.DataFrame(data={"eppoch":range(num_epochs),
                                     "train_loss":train_loss_all,
                                     "train_acc":train_acc_all,
                                     "val_loss": val_loss_all,
                                      "val_acc":val_acc_all})

    return train_process


def matplot_acc_loss(train_process):
    plt.figure(figsize=(12, 4))

    # 子图1：Loss 曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_process["eppoch"], train_process["train_loss"], "b-o", label="train_loss", linewidth=2)
    plt.plot(train_process["eppoch"], train_process["val_loss"], "r-s", label="val_loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True)

    # 子图2：Accuracy 曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_process["eppoch"], train_process["train_acc"], "b-o", label="train_acc", linewidth=2)
    plt.plot(train_process["eppoch"], train_process["val_acc"], "r-s", label="val_acc", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training & Validation Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    LeNet=LeNet()


    train_dataloader,val_dataloader=train_val_data_process()

    train_process = train_model_process(LeNet,train_dataloader,val_dataloader,20)

    matplot_acc_loss(train_process)
