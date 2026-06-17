import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST
import torch.utils.data as Data

import data
from model import LeNet


def test_data_process():
    test_data = FashionMNIST(root='./data',
        train=True,
        transform=transforms.Compose([transforms.Resize(28), transforms.ToTensor(), ]),
        download=True)





    test_dataloader = Data.DataLoader(dataset=test_data,
        batch_size=32,
        shuffle=True,
        num_workers=4)

    return test_dataloader
#测试
#test_dataloader = test_val_data_process()

def test_model_process(model,test_dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=model.to(device)
    #初始化参数
    test_corrects=0.0
    test_num=0
    #只进行前向传播
    with torch.no_grad():
        for test_data_x,test_data_y in test_dataloader:
            test_data_x=test_data_x.to(device)
            test_data_y=test_data_y.to(device)
            model.eval()
            #前向传播
            output=model(test_data_x)
            pre_label=output.max(dim=1)[1]

            test_corrects=test_corrects + torch.sum(pre_label==test_data_y)

            test_num=test_num+test_data_x.size(0)

    test_acc=test_corrects/test_num
    print("准确率为{}".format(test_acc))

if __name__ == '__main__':
    model=LeNet()
    model.load_state_dict(torch.load('best_model.pth',weights_only=True))

    test_dataloader=test_data_process()
    test_model_process(model,test_dataloader)


    #推理
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=model.to(device)
    model.eval()
    with torch.no_grad():
        for b_x,b_y in test_dataloader:
            b_x=b_x.to(device)
            b_y=b_y.to(device)


            output=model(b_x)
            pre_label=torch.argmax(output,dim=1)
            for i in range(len(b_y)):
                result = pre_label[i].item()
                label = b_y[i].item()
                print("预测值{}".format(result), "真实值{}".format(label))

