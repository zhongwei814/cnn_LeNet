from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt


train_data=FashionMNIST(root='./data',
                        train=True,
                        transform=transforms.Compose([transforms.Resize(224),transforms.ToTensor(),]),
                        download=True)

train_loader=Data.DataLoader(dataset=train_data,
                             batch_size=64,
                             shuffle=True,
                             num_workers=0)

for step, (b_x, b_y) in enumerate(train_loader):
    if step>0:
        break
batch_x=b_x.squeeze().numpy()
batch_y=b_y.numpy()
class_lable=train_data.classes
print(class_lable)

# plt,figure(figsize=(10,10))
# plt.imshow(batch_x[0])
# plt.colorbar()
# plt.grid(False)