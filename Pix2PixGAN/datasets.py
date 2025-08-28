import glob
import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import random


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode='train'): # 移除 unaligned 参数
        self.transform = transforms.Compose(transforms_)
        self.files_A = sorted(glob.glob(os.path.join(root, '%sA' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%sB' % mode) + '/*.*'))
        
        # 确保数据集A和B的数量相同，因为是成对数据
        assert len(self.files_A) == len(self.files_B), "Dataset A and B must have the same number of images for paired data."
         

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index]).convert('RGB'))
        item_B = self.transform(Image.open(self.files_B[index]).convert('RGB')) # 始终读取对应的B图像

        return {'A': item_A, 'B': item_B, 'A_paths': self.files_A[index],
                'B_paths': self.files_B[index]}

    def __len__(self):  
        return len(self.files_A)