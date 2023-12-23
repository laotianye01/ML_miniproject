from torch.utils.data import Dataset
from facenet_pytorch import MTCNN
import torch
import numpy as np
import cv2
import os
from natsort import ns, natsorted
import re
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import time

class MtcnnDataset(Dataset):
    def __init__(self, img_path, label_path, transform=None):
        self.filename = img_path   # 文件路径
        self.path = label_path
        self.transform = transform # 是否对图片进行变化
        self.image_list, self.label_image = self.operate_file()
        self.tmp_box = None

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        # 由路径打开图片
        image = self.image_list[idx]
        # 获取标签值
        label = self.label_image[idx]
        # 是否需要处理
        # if self.transform:
        #     image = self.transform(image)

        # 转为tensor对象
        label = torch.from_numpy(np.array(label))
        return image, label

    def operate_file(self):
        img_list = []
        mtcnn = MTCNN(image_size=160, margin=0)
        dir_list = os.listdir(self.filename)
        dir_list = natsorted(dir_list, alg=ns.PATH)
        # 拼凑出图片完整路径 '../data/net_train_images' + '/' + 'xxx.jpg'
        img_path = [self.filename + '/' + name for name in dir_list]
        for path in img_path:
            img = Image.open(path)
            img_cropped = mtcnn(img)
            if img_cropped is None:
                print(path)
            img_list.append(img_cropped)

        label = []
        with open(self.path, 'r') as file:
            lines = file.readlines()
        for line in lines:
            matches = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", line)
            # 分割出第一个数字和其他三个数字
            first_num, *other_nums = matches
            num_list = [5 * float(num) for num in other_nums]
            label.append(num_list)

        # 获取所有的文件夹路径 '../data/net_train_images'的文件夹

        print(len(img_list), len(label))

        return img_list, label

if __name__ == "__main__":
    #Set = My_Dataset()
    print()