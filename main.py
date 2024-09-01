import numpy as np
import os
import torch
import torch.nn as nn
import torch.utils.data as Data
import time
from torch.cuda.amp import autocast, GradScaler
from utils import init_logger,CosineScheduler
from Data_preprocessing import Seqs2EqlTensor
from Model import MyModel,DeepPD
from LossFunction import *
from config import ArgsConfig
from torch.optim import lr_scheduler
from evaluation import evaluating_indicator
# from plot import plot
from torch.utils.tensorboard import SummaryWriter

args = ArgsConfig()
args.epochs = 50
args.max_len = 40
args.batch_size = 192
args.dropout = 0.6
args.info_bottleneck = False
args.exp_nums = 0.1
args.random_seed = 10
args.kflod = 2
args.info = f'CosLR,celoss,DeepPD_C,featA1=encoding_x+esm_x,NOIB,dp=0.6'
args.aa_dict = 'esm'
logger = init_logger(os.path.join(args.log_dir,f'{args.model_name}.log'))
# 将当前配置打印到日志文件中
logger.info(f"{'*'*40}当前模型训练的配置参数{'*'*40}")
for key, value in args.__dict__.items():
    logger.info(f" {key} = {value}")
logger.info(f"{'*'*100}")

## 记录当前时间，并格式化当前时间戳为年-月-日 时:分:秒的形式
logger.info(f'训练开始时间:{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))}')

np.random.seed(args.random_seed)
torch.manual_seed(args.random_seed) # 10

data,label = Seqs2EqlTensor(args.data_c1_dir,args.max_len,AminoAcid_vocab=args.aa_dict)
logger.info('data shape:%s,label shape:%s',data.shape,label.shape)

dataset = Data.TensorDataset(data,label)
train_sample_nums = int(args.split_size * len(dataset))
test_sample_nums = len(dataset) - train_sample_nums

train_dataset, test_dataset = Data.random_split(dataset, [train_sample_nums, test_sample_nums])
train_iter = Data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,drop_last = True,pin_memory=True)
test_iter = Data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,drop_last = True,pin_memory=True)

# 计算学习率调度器的最大更新次数
CoslrS_max_update = train_sample_nums/args.batch_size*args.epochs
logger.info(f"CosineScheduler's max_update:{CoslrS_max_update}")
CoslrS_max_update = CoslrS_max_update*1.1
logger.info(f"Actual CosineScheduler's max_update:{CoslrS_max_update}")

############################################################################################################################################## 
KF_Acc,KF_AUC,KF_sn,KF_sp,KF_f1,KF_mcc,KF_epoch = 0,0,0,0,0,0,0
for Kflod_num in range(args.kflod):
    Version_num = Kflod_num+1
    scaler = GradScaler()
    # 创建SummaryWriter对象
    writer = SummaryWriter(log_dir=f"{args.tensorboard_log_dir}/{args.model_name}")
    scheduler = CosineScheduler(max_update=CoslrS_max_update, base_lr=args.lr, warmup_steps=500)

    # net = MyModel().to(args.device)
    net = DeepPD(vocab_size=21,embedding_size=args.embedding_size,esm_path=args.ems_path,layer_idx=args.esm_layer_idx,seq_len=args.max_len,dropout=args.dropout,
               fan_layer_num=1,num_heads=8,encoder_layer_num=1,Contrastive_Learning=False,info_bottleneck=args.info_bottleneck).to(args.device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr,weight_decay=args.weight_decay)

    criterion_model = nn.CrossEntropyLoss(reduction='sum')
    # criterion_model = Poly_BCE(logits=True)
    # criterion_model = Poly_cross_entropy()
    # criterion_model = FocalLoss(alpha=args.fl_alpha,gamma=args.fl_gamma,logits=False,reduction='sum')
    def calc_loss(y_pred, labels, enc_mean, enc_std, beta=args.IB_beta,act=None):
        """    
        y_pred : [B,2]
        label : [B,1]    
        enc_mean,enc_std : [B,z_dim]  
        """   
        
        if act == 'sigmoid':
            y_pred = torch.sigmoid(y_pred)
        elif act == 'softmax':
            y_pred = torch.softmax(y_pred,dim=1)
        ce = criterion_model(y_pred,labels)
        KL = 0.5 * torch.sum(enc_mean.pow(2) + enc_std.pow(2) - 2*enc_std.log() - 1)
        return (ce + beta * KL) #/y_pred.shape[0]

    global_step = 0
    ETA,ETL,EVA,EVL = [],[],[],[]
    best_acc,AUC,sn,sp,f1,mcc,best_epoch = 0,0,0,0,0,0,0
    logger.info(f"Start training K-flod:{Kflod_num+1}")
    logger.info(f"{'-'*150}")
    logger.info(f"{'-'*150}")
    for epoch in range(args.epochs):
        loss_ls = []
        correct = 0
        total = len(train_dataset)
        t0 = time.time()
        net.train()

        for seq,label in train_iter:
            seq,label = seq.to(args.device),label.to(args.device)
            if args.is_autocast:
                with autocast():
                    output = net(seq)
                    loss = criterion_model(output,label)
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            
            else:
                if args.info_bottleneck:
                    y_pred,enc_means,enc_stds = net(seq)
                    loss = calc_loss(y_pred, label, enc_means, enc_stds,act='sigmoid')
                else:
                    y_pred,_,_ = net(seq)
                    loss = criterion_model(y_pred,label)
                optimizer.zero_grad() 
                loss.backward()
                optimizer.step()

            loss_ls.append(loss.item())
            correct += y_pred.argmax(dim=1).eq(label).sum()
            train_acc = correct/total

            # 更新学习率
            if scheduler:
                if scheduler.__module__ == lr_scheduler.__name__:
                    scheduler.step()
                else:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = scheduler(global_step)

            global_step += 1

        net.eval() 
        with torch.no_grad(): 
            Sn,Sp,precision,test_acc,F1,MCC,auc,valloss = evaluating_indicator(test_iter,net,criterion_model)

        if test_acc > best_acc:
            best_acc,AUC,sn,sp,f1,mcc,best_epoch = test_acc,auc,Sn,Sp,F1,MCC,epoch+1
            torch.save(net.state_dict(),f'{args.save_para_dir}/{args.exp_nums}_{Version_num}.pth')
                
        results = f"EPOCH:[{best_epoch}|{epoch+1}/{args.epochs}],loss: {np.mean(loss_ls)/args.batch_size:.4f},Train_ACC: {train_acc:.4f}"
        results += f' vloss: {np.mean(valloss)/args.batch_size:.4f},Auc:{auc:.4f},Test_ACC:{test_acc:.4f},Sn:{Sn:.4f},Sp:{Sp:.4f}'
        results += f' F1:{F1:.4f},MCC:{MCC:.4f}, time:{time.time()-t0:.2f}'
        logger.info(results)

        ETA.append(train_acc.cpu().detach().numpy()),ETL.append(np.mean(loss_ls)/args.batch_size)
        EVA.append(test_acc),EVL.append(np.mean(valloss)/args.batch_size)
        
    if best_acc > KF_Acc:
        K,KF_Acc,KF_AUC,KF_sn,KF_sp,KF_f1,KF_mcc,KF_epoch = Kflod_num+1,best_acc,AUC,sn,sp,f1,mcc,best_epoch

# plotsavepath = f'./plot/DeepPD/loss/DeepPD_{args.max_len}_noIB_loss_acc曲线.png'      
# plot(args.epochs,ETL,ETA,EVL,EVA,plotsavepath,title='(B)',const=0.8224)
logger.info(f'************************************************************{args.kflod}-flod Best Performance**************************************************************')
logger.info('第{}Flod,epoch:{},ACC:{:.4f},AUC:{:.4f},SN:{:.4f},SP:{:.4f},F1:{:.4f},MCC:{:.4f}'.format(K,KF_epoch,KF_Acc,KF_AUC,KF_sn,KF_sp,KF_f1,KF_mcc))