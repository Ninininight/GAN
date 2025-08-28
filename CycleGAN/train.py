import argparse
from options import TrainOptions
from create_dataset import create_dataset
from models import CycleGANModel  # Changed from GcGANShareModel
from tqdm import tqdm
import torch
import time  # Import time for elapsed time tracking
import os


# Set device based on available GPUs and opt.gpu_ids
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
    opt = options.parse()  # Parse options

    device = set_device(opt.gpu_ids)  # Set device based on parsed options

    dataset = create_dataset(opt)  # Create dataset with parsed options

    model = CycleGANModel()  # Instantiate CycleGANModel
    model.initialize(opt)  # Initialize model with parsed options

    total_steps = 0

    # 新增：用于存储每个 epoch 的平均损失
    G_losses = []
    D_losses = []
    cycle_A_losses = []
    cycle_B_losses = []
    idt_A_losses = []
    idt_B_losses = []
    epochs_list = []  # 记录epoch，用于X轴

    for epoch in range(opt.epoch_count, opt.epoch + opt.decay_epoch + 1):  # Loop through total epochs
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0


        for i, data in tqdm(enumerate(dataset), total=len(dataset), leave=True):
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            model.set_input(data)  # Set input data
            model.optimize_parameters()  # Optimize parameters

            if total_steps % opt.print_freq == 0:
                errors = model.get_current_errors()
                t = (time.time() - iter_start_time) / opt.batchSize
                # print current errors to console
                message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, epoch_iter, t, t_data)
                for k, v in errors.items():
                    message += '%s: %.3f ' % (k, v)
                print(message)

               
            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save('latest')

            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save(epoch)

          

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.epoch + opt.decay_epoch, time.time() - epoch_start_time))

        model.update_learning_rate()  # Update learning rate at the end of each epoch
