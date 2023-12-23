import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from classification_dataset import ClassDataset
import torch.nn as nn
import torchvision.models as models
import time
from torch.utils.tensorboard import SummaryWriter
from model import *

writer = SummaryWriter("/result_logs/classification")

# 全局变量
batch_size = 16   # 每次喂入的数据量

epoch_num = 100  # 总迭代次数

lr = 0.01
step_size = 10  # 每n次epoch更新一次学习率


# 数据获取(数据增强,归一化)
def transforms_RandomHorizontalFlip():
    # transforms.RandomHorizontalFlip(),
    # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    # transforms.Resize((224, 224)),
    transform_train = transforms.Compose([
                                         transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
                                         transforms.RandomRotation(20),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                         transforms.RandomPerspective(distortion_scale=0.5, p=0.5)
                                         ])

    # transforms.Normalize((0.485, 0.456, 0.406), (0.226, 0.224, 0.225))
    transform = transforms.Compose([
                                  transforms.Resize((224, 224)),
                                  transforms.ToTensor()
                                  ])

    train_dataset = ClassDataset(path="/home/yelu/PycharmProjects/ml_minilab/classification_dataset/train", transform=transform_train)
    test_dataset = ClassDataset(path="/home/yelu/PycharmProjects/ml_minilab/classification_dataset/valid", transform=transform)
    valid_dataset = train_dataset

    return train_dataset, test_dataset, valid_dataset

#数据增强:随机翻转
train_dataset,test_dataset, valid_dataset = transforms_RandomHorizontalFlip()

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)


# 模型,优化器
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = models.resnet18(pretrained=False).to(device)
# for param in model.parameters():
#     param.requires_grad = False

# 添加新的全连接层作为分类器
num_classes = 2
model.fc = nn.Linear(model.fc.in_features, num_classes).to(device)
# model = VGG16(num_class=2).to(device)

# 在多分类情况下一般使用交叉熵
criterion = nn.CrossEntropyLoss()

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
    acc = 0
    tol = 0
    for i, (inputs, labels) in enumerate(train_loader, 0):

        count_train += 1
        labels = labels.type(torch.LongTensor)
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        pred = outputs.argmax(dim=1)
        acc += pred.eq(labels).float().sum().item()
        tol += inputs.size(0)
        loss = criterion(outputs, labels).to(device)

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        loss_list.append(loss.item())

        if (i+1) % 10 == 0:
            print('[%d epoch,%d]  loss:%.6f' % (epoch+1, i+1, running_loss/i))

    writer.add_scalar("train loss:", running_loss/count_train, epoch + 1)
    writer.add_scalars("train & valid acc:", {'Train': (100 * acc / tol)}, epoch + 1)
    lr_1 = optimizer.param_groups[0]['lr']
    print("learn_rate:%.15f" % lr_1)
    # schedule.step()

    # 测试
    # model.eval()
    correct = 0.0
    total = 0
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
            total += inputs.size(0)
            correct += pred.eq(labels).float().sum().item()

    writer.add_scalars("train & valid acc:", {'Valid': (100 * correct / total)}, epoch + 1)
    # writer.add_scalar("valid acc:", (100 * correct / total), epoch + 1)
    # writer.add_scalar("valid loss:", (valid_loss / count_valid), epoch + 1)
    print("Accuracy of the network on the 1000 test images:%.2f %%" % (100 * correct / total))
    print("===============================================")

    if epoch % 20 == 0:
        file_path = '/home/yelu/PycharmProjects/ml_minilab/model_weight/class_model/' + str(epoch) + "_smile.pth"
        torch.save(model, file_path)

end = time.time()
print("time:{}".format(end-start))
