import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from three_dataset import MtcnnDataset
import torchvision.models as models
from model import *
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from matplotlib import pyplot as plt
import time

# 全局变量
batch_size = 64   # 每次喂入的数据量
writer = SummaryWriter("/result_logs/3_D")

# num_print=int(50000//batch_size//4)
num_print = 100

epoch_num = 21  # 总迭代次数

lr = 0.005
step_size = 30  # 每n次epoch更新一次学习率

# 数据获取(数据增强,归一化)
def transforms_fun():
    # transforms.RandomResizedCrop(size=224),
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.226, 0.224, 0.225))])

    train_dataset = MtcnnDataset(img_path="/home/yelu/PycharmProjects/ml_minilab/3D_dataset/train", label_path="/home/yelu/PycharmProjects/ml_minilab/3D_dataset/train.txt", transform=None)
    test_dataset = MtcnnDataset(img_path="/home/yelu/PycharmProjects/ml_minilab/3D_dataset/valid", label_path="/home/yelu/PycharmProjects/ml_minilab/3D_dataset/valid.txt", transform=transform)

    return train_dataset, test_dataset

train_dataset, test_dataset = transforms_fun()

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 模型,优化器
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = models.resnet34(pretrained=False).to(device)
# 添加新的全连接层作为分类器
num_classes = 3
model.fc = nn.Linear(model.fc.in_features, num_classes).to(device)

# 在多分类情况下一般使用交叉熵
criterion = nn.MSELoss(size_average=None, reduce=None, reduction='mean')

# optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.8, weight_decay=0.001)
optimizer = optim.Adam(model.parameters(), lr=lr)
# schedule = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5, last_epoch=-1)

# 训练
loss_list = []  # 为了后续画出损失图
start = time.time()

# train
for epoch in range(epoch_num):
    count_train = 0
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader, 0):

        count_train += 1
        labels = labels.type(torch.FloatTensor)
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels).to(device)

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        loss_list.append(loss.item())

        if (i+1) % 10 == 0:
            print('[%d epoch,%d]  loss:%.6f' % (epoch+1, i+1, running_loss/i))

    writer.add_scalar("train loss:", running_loss/count_train, epoch + 1)
    lr_1 = optimizer.param_groups[0]['lr']
    print("learn_rate:%.15f" % lr_1)
    # schedule.step()

    # 测试
    # model.eval()
    valid_loss = 0.0
    count_valid = 0
    with torch.no_grad():  # 训练集不需要反向传播
        print("=======================test=======================")
        for inputs, labels in test_loader:

            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels).to(device)
            valid_loss += loss
            count_valid += 1

            pred = outputs.argmax(dim=1)  # 返回每一行中最大值元素索引

    writer.add_scalar("valid loss:", (valid_loss / count_valid), epoch + 1)
    print("valid loss:%.2f" % valid_loss)
    print("===============================================")

    if epoch % 10 == 0:
        file_path = '/home/yelu/PycharmProjects/ml_minilab/model_weight/3D_model/' + str(epoch) + "_3D.pth"
        torch.save(model, file_path)
