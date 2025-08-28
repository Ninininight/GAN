import os
import torch
import torch.nn as nn

class BaseModel():
    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        # 确保 device 在 initialize 中设置，以便加载模型时使用
        self.device = torch.device(f"cuda:{self.gpu_ids[0]}") if self.gpu_ids and torch.cuda.is_available() else torch.device("cpu")

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    # used in test time, no backprop
    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, gpu_ids):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        # 将模型移动到CPU保存，避免GPU内存问题
        torch.save(network.cpu().state_dict(), save_path)
        # 如果有GPU，保存后移回GPU
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda(gpu_ids[0])

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        # 确保加载时指定 map_location 到正确的设备
        state_dict = torch.load(save_path, map_location=self.device)
        network.load_state_dict(state_dict)
        # 确保模型在正确的设备上
        network.to(self.device)


    # update learning rate (called once every epoch)
    def update_learning_rate(self):
        # 确保 self.schedulers 存在且不为空
        if hasattr(self, 'schedulers') and self.schedulers:
            for scheduler in self.schedulers:
                scheduler.step()
            lr = self.optimizers[0].param_groups[0]['lr']
            print('learning rate = %.7f' % lr)
        else:
            print("No learning rate schedulers defined.")


    def eval(self):
        if hasattr(self, 'model_names'):
            for name in self.model_names:
                # 检查属性是否存在，并确保是 nn.Module 的实例
                if isinstance(name, str) and hasattr(self, 'net' + name) and isinstance(getattr(self, 'net' + name), nn.Module):
                    net = getattr(self, 'net' + name)
                    net.eval()
                else:
                    print(f"Warning: Could not set {name} to eval mode. It might not be a network module or not named correctly.")
        else:
            print("Warning: 'model_names' attribute not found in the model. Cannot set networks to eval mode.")