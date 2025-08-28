import torchvision.transforms as transforms
from datasets import ImageDataset
from PIL import Image
from torch.utils.data import DataLoader


def create_dataset(opt):
    transform_list = [transforms.Resize(opt.fineSize + 30, Image.BICUBIC),  # optional: 大一点再裁剪
                      transforms.RandomCrop(opt.fineSize)]
    
    transforms_ = [transforms.ToTensor(),
                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    transform_list.extend(transforms_)

    if opt.isTrain:
        dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transform_list, mode='train'),
                                batch_size=opt.batchSize, shuffle=True, num_workers=opt.nThreads, drop_last=True)
    else:
        transform_list = [transforms.Resize(opt.fineSize, Image.BICUBIC),
                          transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transform_list, mode='test'),
                                batch_size=opt.batchSize, shuffle=False, num_workers=opt.nThreads)
    return dataloader
