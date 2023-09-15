import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import logging
def init_logger(log_file=None, log_file_level=logging.NOTSET):
    log_format = logging.Formatter(fmt='%(asctime)s-%(levelname)s: %(message)s', #-%(name)s
                                   datefmt='%m/%d/%Y %H:%M:%S')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_file_level)
        logger.addHandler(file_handler)

    return logger

class CosineScheduler:
    def __init__(self, max_update: int=10000, base_lr: float=1e-3, 
                 final_lr: float=0, warmup_steps: int=500, warmup_begin_lr: float=0):
        self.base_lr_orig = base_lr
        self.max_update = max_update
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.warmup_begin_lr = warmup_begin_lr
        self.max_steps = self.max_update - self.warmup_steps

    def get_warmup_lr(self, steps):
        increase = (self.base_lr_orig - self.warmup_begin_lr) * float(steps-1) / float(self.warmup_steps)
        return self.warmup_begin_lr + increase

    def __call__(self, steps):
        if steps < self.warmup_steps:
            return self.get_warmup_lr(steps)
        if steps <= self.max_update:
            self.base_lr = self.final_lr + (self.base_lr_orig - self.final_lr) * \
                           (1 + math.cos(math.pi * (steps-1 - self.warmup_steps) / self.max_steps)) / 2
        return self.base_lr

epsilon = 0.00000001
def squash(x):
    # not concern batch_size, maybe rewrite
    s_squared_norm = torch.sum(x*x,1,keepdim=True) + epsilon
    scale = torch.sqrt(s_squared_norm)/(1. + s_squared_norm)
    # out = (batch_size,1,10)*(batch_size,16,10) = (batch_size,16,10)
    out = scale * x
    return out


class ResNet(nn.Module):
    def __init__(self,input_cha):
        super(ResNet,self).__init__()
        self.conv1 = nn.Conv2d(input_cha,input_cha,3,padding=1)
        self.conv2 = nn.Conv2d(input_cha,input_cha,5,padding=2)
        self.conv3 = nn.Conv2d(input_cha,input_cha,7,padding=3)

        self.cbamBlock = CBAMBlock(input_cha) 

        self.bn1 = nn.BatchNorm2d(input_cha)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.LeakyReLU()

    def forward(self,x):
        init_x = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu2(out)

        out = self.conv1(out)
        out = self.bn1(out)
        out += init_x
        out = self.relu2(out)

        return out

class CBAMBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CBAMBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.channel_excitation = nn.Sequential(nn.Linear(channel,int(channel//reduction),bias=False), 
                                                nn.ReLU(inplace=True),
                                                nn.Linear(int(channel//reduction),channel,bias=False),
                                                )
        self.sigmoid = nn.Sigmoid()

        self.spatial_excitation = nn.Sequential(nn.Conv2d(2, 1, kernel_size=7,
                                                 stride=1, padding=3, bias=False),
                                               )


    def forward(self, x):
        bahs, chs, _, _ = x.size()     #16 16 24 42

        # Returns a new tensor with the same data as the self tensor but of a different size.
        chn_avg = self.avg_pool(x).view(bahs, chs)                       
        chn_avg = self.channel_excitation(chn_avg).view(bahs, chs, 1, 1)  
        chn_max = self.max_pool(x).view(bahs, chs)
        chn_max = self.channel_excitation(chn_max).view(bahs, chs, 1, 1)
        chn_add=chn_avg+chn_max
        chn_add=self.sigmoid(chn_add)

        chn_cbam = torch.mul(x, chn_add)

        avg_out = torch.mean(chn_cbam, dim=1, keepdim=True)
        max_out, _ = torch.max(chn_cbam, dim=1, keepdim=True)
        cat = torch.cat([avg_out, max_out], dim=1)

        spa_add = self.spatial_excitation(cat)
        spa_add = self.sigmoid(spa_add)
        spa_cbam = torch.mul(chn_cbam, spa_add)

        return spa_cbam

class Capsule(nn.Module):

    def __init__(self, in_units,in_channels, num_capsule, dim_capsule, routings=3, **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.in_units = in_units
        self.in_channels = in_channels
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        # (in_units,10,128,16)
        self.W = nn.Parameter((torch.randn(self.in_units,self.num_capsule,self.in_channels, self.dim_capsule)))  #1860,1,16,32

    def forward(self, u_vecs):
        u_vecs = u_vecs.permute(0,2,1)  # 每一块的行与列进行交换，即每一块做转置,permute(0,1,2),0表示维度（块）标识,1表示行标识，2表示列标识
        u_vecs = u_vecs.unsqueeze(2)  #指定位置N[此处在列维度(第三维度)]加上一个维数为1的维度
        u_vecs = u_vecs.unsqueeze(2)

        
        # (batch_size,in_units,1,1,in_channels)*(in_units,10,in_channels,16) = (batch_size,in_units,10,1,16)
        u_hat_vecs = torch.matmul(u_vecs,self.W)
        # (batch_size,in_units,10,16)
        u_hat_vecs = u_hat_vecs.permute(0,1,2,4,3).squeeze(4)  #squeeze()对数据的维度进行压缩，去掉维数为1的的维度
        
        # (batch_size,10,in_units,16)
        u_hat_vecs2 = u_hat_vecs.permute(0,2,1,3)
    
        # (batch_size,10,1,in_units)
        b = torch.zeros(u_hat_vecs.size(0),self.num_capsule,1,self.in_units,device=device)

        for i in range(self.routings):
            # (batch_size,10,1,in_units)
            c = F.softmax(b,-1)
            # s = (batch_size,10,1,in_units)*(batch_size,10,in_units,16) = (batch_size,10,1,16)
            s = torch.matmul(c,u_hat_vecs2)
            # (batch_size,16,10)
            s = s.permute(0,3,1,2).squeeze(3)
            # (batch_size,16,10)
            v = squash(s)
            # here
            # (batch_size,10,16,1)
            v = v.permute(0,2,1).unsqueeze(3)
            # (batch_size,10,in_units,16)*(batch_size,10,16,1) = (batch_size,10,in_units,1)
            sim = torch.matmul(u_hat_vecs2,v)
            # (batch_size,10,1,in_units)
            sim = sim.permute(0,1,3,2)
            b = b+sim
        # (batch_size,16,10)
        return v.permute(0,2,1,3).squeeze(3)