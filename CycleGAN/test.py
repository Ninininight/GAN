import argparse
from options import TestOptions
from create_dataset import create_dataset
from torchvision.utils import save_image
from models import CycleGANModel  # Changed from GcGANShareModel
from tqdm import tqdm
import torch
import os
import util  # Import util for tensor2im


# Set device based on available GPUs and opt.gpu_ids
def set_device(gpu_ids):
    if len(gpu_ids) > 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_ids[0]}")
    else:
        device = torch.device("cpu")
    return device


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    options = TestOptions()
    opt = options.parse()  # Parse options

    device = set_device(opt.gpu_ids)  # Set device based on parsed options

    dataset = create_dataset(opt)  # Create dataset with parsed options

    model = CycleGANModel()  # Instantiate CycleGANModel
    model.initialize(opt)  # Initialize model with parsed options (will load pre-trained weights)
    model.eval()  # Set model to evaluation mode

    # Create output directories if they don't exist
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    img_dir = os.path.join(web_dir, 'images')
    print('creating directory %s...' % img_dir)
    util.mkdirs(img_dir)

    for i, data in tqdm(enumerate(dataset), total=min(len(dataset), opt.how_many)):
        if i >= opt.how_many:
            break
        model.set_input(data)  # Set input data for testing
        model.forward()  # Perform forward pass

        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        short_path = os.path.basename(img_path[0])
        name = os.path.splitext(short_path)[0]

        for label, image_numpy in visuals.items():
            image_name = '%s_%s.png' % (name, label)
            save_path = os.path.join(img_dir, image_name)
            util.save_image(image_numpy, save_path)