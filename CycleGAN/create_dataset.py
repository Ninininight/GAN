import torchvision.transforms as transforms
from datasets import ImageDataset
from PIL import Image
from torch.utils.data import DataLoader


def create_dataset(opt):
    transform_list = [transforms.Resize(256, Image.BICUBIC),
                      transforms.RandomCrop(256),
                      transforms.RandomHorizontalFlip()]
    if opt.isTrain:  # Use opt.isTrain to determine if it's training or testing
        transforms_ = [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        transform_list.extend(transforms_)

        dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transform_list, mode='train',
                                             unaligned=(opt.dataset_mode == 'unaligned')),
                                batch_size=opt.batchSize, shuffle=True, num_workers=opt.nThreads, drop_last=True)
    else:  # Test mode
        transforms_ = [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        transform_list = [transforms.Resize(256, Image.BICUBIC), transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transform_list, mode='test',
                                             unaligned=(opt.dataset_mode == 'unaligned')),
                                batch_size=opt.batchSize, shuffle=False, num_workers=opt.nThreads, drop_last=True)

    return dataloader