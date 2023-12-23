from sklearn.metrics import confusion_matrix
import torchvision.models as models
from class_dataset import ClassDataset
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont

# 以下代码为笑脸检测测试代码，外加生成混淆矩阵
transform = transforms.Compose([
                                  transforms.Resize((224, 224)),
                                  transforms.ToTensor()
                                  ])
test_dataset = ClassDataset(path="/dataset/classification_dataset/valid", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load('/model_weight/class_model/80_smile.pth').to(device)
gt, per = torch.tensor([]), torch.tensor([])
for i, (inputs, labels) in enumerate(test_loader, 0):
    inputs = inputs.to(device)
    labels = labels.to(device)
    outputs = model(inputs)
    pred = outputs.argmax(dim=1)
    gt = torch.cat((gt, labels.cpu()), 0)
    per = torch.cat((per, pred.cpu()), 0)
cm = confusion_matrix(gt, per)
classes = ['smile', 'non-smile']
classNamber = 2  # 类别数量
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)  # 按照像素显示出矩阵
plt.title('confusion_matrix_smile')  # 改图名
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=-45)
plt.yticks(tick_marks, classes)
plt.ylabel('Ture')
plt.xlabel('Prediction')
plt.tight_layout()
plt.show()

# 以下为图片测试代码
img_path = '/dataset/classification_dataset/train/smile/file0005.jpg'
img = Image.open(img_path)
draw_1 = ImageDraw.Draw(img)
transform_ten = transforms.Compose([
    transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
    transforms.ToTensor(),  # 将PIL Image或ndarray转为tensor，并且将像素值从[0, 255]变为[0.0, 1.0]
])
input = transform_ten(img).unsqueeze(0).to(device)
outputs = model(input)
pred = outputs.argmax(dim=1)
text = ""
if pred == 0:
    text = "smile"
else:
    text = "non-smile"
font_style = ImageFont.truetype('arial.ttf', 15)
text_color = "black"
width, height = img.size
text_position = (int(width/3), 0)
draw_1.text(text_position, text, font=font_style, fill=(0, 0, 0))
img.save("/result/result_images/task1_1/output.jpg")



# 以下代码为三维弧度值测试代码
from facenet_pytorch import MTCNN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(image_size=160, margin=0)
img_path = '/dataset/3D_dataset/train/file0003.jpg'
img = Image.open(img_path)
img_new = mtcnn(img).unsqueeze(0).to(device)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load('/model_weight/3D_model/20_3D.pth')
result = (model(img_new) / 5).cpu().detach().numpy().flatten().tolist()
value = []
for i in range(3):
    value.append(str(result[i])[:6])
draw_2 = ImageDraw.Draw(img)
text = "predict value: " + str(value) + "\n" + "true value: " + "[0.095 0.028 0.065]"
font_style = ImageFont.truetype('arial.ttf', 15)
text_color = "black"
width, height = img.size
text_position = (0, 0)
draw_2.text(text_position, text, font=font_style, fill=(255, 0, 0))
img.save("/result/result_images/task1_2/output.jpg")


