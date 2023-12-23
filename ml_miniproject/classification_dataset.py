from torch.utils.data import Dataset
import torch
import numpy as np
import os
import re
from torchvision import transforms
from PIL import Image

class ClassDataset(Dataset):
    def __init__(self, path, transform=None):
        self.transform = transform  # 是否对图片进行变化
        self.image_name, self.label_image = self.operate_file(path)
        print("dataset samples number: ", self.__len__())

    def __len__(self):
        return len(self.image_name)

    def __getitem__(self, idx):
        # 由路径打开图片
        image = Image.open(self.image_name[idx])
        label = self.label_image[idx]
        # 是否需要处理
        if self.transform:
            image = self.transform(image)
        # 转为tensor对象
        label = torch.from_numpy(np.array(label))
        return image,label

    def operate_file(self, path):
        label = []
        image = []
        smile_path = path + "/smile/"
        non_smile_path = path + "/non_smile/"
        # smile set
        for filename in os.listdir(smile_path):
            tmp = smile_path + filename
            image.append(tmp)
            label.append(0)
        # non_smile set
        for filename in os.listdir(non_smile_path):
            tmp = non_smile_path + filename
            image.append(tmp)
            label.append(1)
        return image, label

if __name__ == "__main__":
    #Set_2 = My_Dataset("D:\program\py_program\SimMIM-main\classification_dataset\\valid")
    #Set_1 = My_Dataset("D:\program\py_program\SimMIM-main\classification_dataset\\train")
    print()