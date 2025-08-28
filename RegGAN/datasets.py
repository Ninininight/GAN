# datasets.py

import glob
import random
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torch


class ImageDataset(Dataset):
    def __init__(self, root,noise_level,count = None,transforms_1=None,transforms_2=None, unaligned=False):
        self.transform1 = transforms.Compose(transforms_1)
        self.transform2 = transforms.Compose(transforms_2)
        # 修正 glob 模式以递归查找所有 .png 文件
        self.files_A = sorted(glob.glob(os.path.join(root, 'trainA/**'), recursive=True))
        self.files_A = [f for f in self.files_A if os.path.isfile(f) and f.endswith('.png')]
        
        self.files_B = sorted(glob.glob(os.path.join(root, 'trainB/**'), recursive=True))
        self.files_B = [f for f in self.files_B if os.path.isfile(f) and f.endswith('.png')]
        
        self.unaligned = unaligned
        self.noise_level =noise_level
        
    def __getitem__(self, index):
        # 使用 Image.open 打开 png 文件
        img_A = Image.open(self.files_A[index % len(self.files_A)]).convert('RGB')
        img_B = Image.open(self.files_B[index % len(self.files_B)]).convert('RGB')
        
        if self.noise_level == 0:
            seed = np.random.randint(2147483647) 
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            item_A = self.transform2(img_A)

            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            item_B = self.transform2(img_B)
        else:
            item_A = self.transform1(img_A)
            
            seed = np.random.randint(2147483647) # generate another seed for the second transform
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            item_B = self.transform2(img_B)
            
        return {'A': item_A, 'B': item_B}
        
    def __len__(self):
        # 如果任何一个列表为空，则返回 0
        if not self.files_A or not self.files_B:
            return 0
        return max(len(self.files_A), len(self.files_B))


class ValDataset(Dataset):
    def __init__(self, root,count = None,transforms_=None, unaligned=False):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        # 修正 glob 模式以递归查找所有 .png 文件
        self.files_A = sorted(glob.glob(os.path.join(root, 'testA/**'), recursive=True))
        self.files_A = [f for f in self.files_A if os.path.isfile(f) and f.endswith('.png')]
        
        self.files_B = sorted(glob.glob(os.path.join(root, 'testB/**'), recursive=True))
        self.files_B = [f for f in self.files_B if os.path.isfile(f) and f.endswith('.png')]
        
    def __getitem__(self, index):
        # 使用 Image.open 打开 png 文件
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]).convert('RGB'))
        # If unaligned, B is chosen randomly
        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]).convert('RGB'))
        else:
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]).convert('RGB'))
        
        return {'A': item_A, 'B': item_B}

    def __len__(self):
        if not self.files_A or not self.files_B:
            return 0
        return max(len(self.files_A), len(self.files_B))