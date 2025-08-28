import argparse
from options import TestOptions
from create_dataset import create_dataset
from models import Pix2PixModel  # 改为 Pix2PixModel
from tqdm import tqdm
import torch
import os
import util
from torchvision.utils import save_image


def set_device(gpu_ids):
    if len(gpu_ids) > 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_ids[0]}")
    else:
        device = torch.device("cpu")
    return device


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    options = TestOptions()
    opt = options.parse()

    device = set_device(opt.gpu_ids)

    dataset = create_dataset(opt)

    model = Pix2PixModel()
    model.initialize(opt)
    model.eval()

    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    img_dir = os.path.join(web_dir, 'images')
    print('creating directory %s...' % img_dir)
    os.makedirs(img_dir, exist_ok=True)


    for i, data in tqdm(enumerate(dataset), total=min(len(dataset), opt.how_many)):
        if i >= opt.how_many:
            break
        model.set_input(data)
        model.forward()

        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        short_path = os.path.basename(img_path[0])
        name = os.path.splitext(short_path)[0]

        for label, image_numpy in visuals.items():
            image_name = '%s_%s.png' % (name, label)
            save_path = os.path.join(img_dir, image_name)
            util.save_image(image_numpy, save_path)
            
    print('测试完成。结果保存在 %s' % web_dir)