import argparse
from options import TrainOptions
from create_dataset import create_dataset
from models import Pix2PixModel
from tqdm import tqdm
import torch
import time
import os
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

def set_device(gpu_ids):
    if len(gpu_ids) > 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_ids[0]}")
        torch.cuda.set_device(gpu_ids[0])
    else:
        device = torch.device("cpu")
    return device

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    options = TrainOptions()
    opt = options.parse()

    device = set_device(opt.gpu_ids)

    dataset = create_dataset(opt)

    model = Pix2PixModel()
    model.initialize(opt)
    model.device = device
    if hasattr(model, 'netG') and model.netG is not None:
        model.netG.to(device)
    if model.isTrain and hasattr(model, 'netD') and model.netD is not None:
        model.netD.to(device)

    total_steps = 0

    # TensorBoard writer
    save_dir = os.path.join(opt.checkpoints_dir, opt.name)
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=save_dir)

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        epoch_iter = 0

        # 用于累积每个 batch 的 loss
        epoch_G_GAN_losses = []
        epoch_G_L1_losses = []
        epoch_D_losses = []

        for i, data in tqdm(enumerate(dataset), total=len(dataset), leave=True):
            total_steps += opt.batchSize 
            epoch_iter += opt.batchSize

            # --------------------
            # 前向 + 反向传播
            # --------------------
            model.set_input(data)
            model.optimize_parameters()

            # --------------------
            # 累积 loss
            # --------------------
            errors = model.get_current_errors()
            epoch_G_GAN_losses.append(errors['G_GAN'])
            epoch_G_L1_losses.append(errors['G_L1'])
            epoch_D_losses.append((errors['D_Real'] + errors['D_Fake']) / 2)

            # --------------------
            # 每 display_freq 个 batch 保存生成图像
            # --------------------
            if total_steps % opt.display_freq == 0:
                visuals = model.get_current_visuals()
                for key, image in visuals.items():
                    img_tensor = torch.from_numpy(image.transpose((2,0,1))).float() / 255.0
                    if img_tensor.ndim == 3:
                        img_tensor = img_tensor.unsqueeze(0)
                    grid = make_grid(img_tensor, normalize=True, scale_each=True)
                    writer.add_image(key, grid, total_steps)

        # --------------------
        # 每个 epoch 结束时写入平均 loss
        # --------------------
        writer.add_scalar('Loss/Generator_GAN', sum(epoch_G_GAN_losses)/len(epoch_G_GAN_losses), epoch)
        writer.add_scalar('Loss/Generator_L1', sum(epoch_G_L1_losses)/len(epoch_G_L1_losses), epoch)
        writer.add_scalar('Loss/Discriminator', sum(epoch_D_losses)/len(epoch_D_losses), epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
            (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

        # --------------------
        # 保存模型
        # --------------------
        if epoch % opt.save_epoch_freq == 0:
            print('Saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
            model.save('latest')
            model.save(epoch)

        # --------------------
        # 更新学习率
        # --------------------
        model.update_learning_rate()


    writer.close()
