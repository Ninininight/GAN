class Option:
    def __init__(self):
        self.name = 'RegGan'
        self.bidirect = False  # Unidirectional or bidirectional
        self.regist = True     # With or without registration network
        self.noise_level = 1   # noise level
        self.port = 6019       # port parameters
        self.save_root = './output/Cyc/NC+R/'
        self.image_save = './output/Cyc/NC+R/img/'

        # lamda weight
        self.Adv_lamda = 1
        self.Cyc_lamda = 10
        self.Corr_lamda = 20
        self.Smooth_lamda = 10

        self.epoch = 0         # starting epoch
        self.n_epochs = 80     # How often do you want to display output images during training
        self.batchSize = 1     # size of the batches
        # 修正数据根目录以适应您的数据集结构
        self.dataroot = './datasets/bras/'  # 现在直接指向 bras 目录
        self.val_dataroot = './datasets/bras/'
        self.lr = 0.0001       # initial learning rate
        self.decay_epoch = 35  # epoch to start linearly decaying the learning rate to 0
        self.size = 256        # size of the data crop
        self.input_nc = 3      # 输入图像通道数: 1表示灰度图，3表示RGB图
        self.output_nc = 3     # 输出图像通道数: 1表示灰度图，3表示RGB图
        self.cuda = True
        self.n_cpu = 1
        self.pool_size = 50    # Image pool size for CycleGAN


# 辅助函数，用于获取配置，类似于原始的get_config
def get_config():
    return Option().__dict__