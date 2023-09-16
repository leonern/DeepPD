import numpy as np
import os
from sklearn import metrics
import torch
import torch.nn as nn
import torch.utils.data as Data
import time
import pickle
import torch.nn.functional as F
from termcolor import colored
from Biometric_extraction import featureGenera
from utils import CBAMBlock
from utils import Res_Net,Capsule,squash
from Data_preprocessing import genData2EqlTensor,Numseq2OneHot,genData

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_len = 40

##### 孪生网络数据集
path1 = r"D:\软件安装\vscode\Code\PD_model\data\PepFormerData\Homo_0.9.csv"
path2 = r"D:\软件安装\vscode\Code\PD_model\data\PepFormerData\Mus_0.9.csv"
path3 = r"D:\软件安装\vscode\Code\PD_model\data\GPMDB_Homo_sapiens_20190115\sorted_GPMDB_Homo_0.025.csv"

data,label = genData2EqlTensor(path3,max_len)
print(data.shape,label.shape)

dataset = Data.TensorDataset(data,label)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
#torch.manual_seed(10)
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

batch_size = 32
train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,drop_last = True)
test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,drop_last = True)


#########################################################################################################################################
class CapsuleNet(nn.Module):
    def __init__(self):
        super().__init__()
        embed_size = 20
        len_vocab = 21
        self.embedding = nn.Embedding(len_vocab,embed_size)
        self.conv1=nn.Conv2d(1,256,9)
        self.conv3=nn.Conv2d(1,256,7)
        self.cbamBlock = CBAMBlock(256) 
        self.conv2=nn.Conv2d(256,32*8,9,2)
        self.conv4=nn.Conv2d(256,32*8,8,2) 
        self.capsule = Capsule(2688,16,2,32)
   
    def forward(self,x):
        batch_size = x.size(0)
        y = featureGenera(x) #*,84,20/*,123,20
        out = self.embedding(x) #*,40,20 /*,79,20
        y = y.type_as(out)

        out = out.unsqueeze(1)
        y = y.unsqueeze(1)
        out = self.conv1(out) #*,256,32,12 /*,256,71,12
        out = self.cbamBlock(out)
        y = self.conv3(y) #*,256,78,14 /117,14
        y = self.cbamBlock(y)
        out = F.relu(out)
        y = F.relu(y)

        out = self.conv2(out) #256,256,12,2 /*,256,32,2
        y = self.conv4(y) #256,256,36,4 /55,4
        out = F.relu(out)
        y = F.relu(y)

        out = out.view(batch_size,16,-1) #*,16,384 /1024
        y = y.view(batch_size,16,-1) #*,16,2304 /3520
        out = torch.cat((out,y),2) #*,16,2688 /4544
        out = squash(out)
        out = out.view(out.size(0),out.size(1),-1)
        out = self.capsule(out)
        out = out.permute(0,2,1) #*,2,32

        output = torch.sqrt(torch.sum(out*out, 2)) #*,2

        return output

############################################################################################################################################## 
def evaluating_indicator(data_iter, net):
    all_true = []
    all_pred = []
    for x, y in data_iter:
        x,y = x.to(device),y.to(device)
        outputs = net(x)
        y_pre = outputs.argmax(dim=1)
        all_true.extend(y.cpu().detach().numpy())
        all_pred.extend(y_pre.cpu().detach().numpy())
    
    TN,FP,FN,TP = metrics.confusion_matrix(all_true,all_pred).ravel()
    P = FP + TP
    precision = TP / P
    Sn = TP / (TP + FN) #sn=recall
    Sp = TN / (TN + FP)
    ACC = metrics.accuracy_score(all_true,all_pred)
    MCC = metrics.matthews_corrcoef(all_true,all_pred)
    auc = metrics.roc_auc_score(all_true,all_pred)
    F1_score =  metrics.f1_score(all_true,all_pred)

    return Sn,Sp,precision,ACC,F1_score,MCC,auc


KF_Acc,KF_AUC,KF_sn,KF_sp,KF_f1,KF_mcc,KF_epoch = 0,0,0,0,0,0,0
for Kflod_num in range(2):
    Version_num =Kflod_num+1
    def To_log(log):
        with open("D:/软件安装/vscode/Code/PD_model/result/PD预测模型/CapsNet/Dataset_C/modellog"+str(Version_num)+".log","a+") as f:
            f.write(log+'\n')

    net = CapsuleNet().to(device)
    lr = 0.001
    optimizer = torch.optim.Adam(net.parameters(), lr=lr,weight_decay=5e-4)
    criterion_model = nn.CrossEntropyLoss(reduction='sum')
    best_acc,AUC,sn,sp,f1,mcc,best_epoch = 0,0,0,0,0,0,0
    EPOCH = 25 #60
    print(f'***************************************************Start training K-flod:{Kflod_num+1}******************************************************')

    for epoch in range(EPOCH):
        loss_ls = []
        correct = 0
        total = len(train_dataset)
        t0 = time.time()
        net.train()

        for seq,label in train_iter:
            seq = seq.to(device)
            label = label.to(device)
            output = net(seq)
            loss = criterion_model(output,label)
            optimizer.zero_grad() 
            loss.backward()
            optimizer.step()
            loss_ls.append(loss.item())

            correct += output.argmax(dim=1).eq(label).sum()
            train_acc = 100*correct/total


        net.eval() 
        with torch.no_grad(): 
            Sn,Sp,precision,test_acc,F1,MCC,auc = evaluating_indicator(test_iter,net)

        if test_acc>best_acc:
            best_acc,AUC,sn,sp,f1,mcc,best_epoch = test_acc,auc,Sn,Sp,F1,MCC,epoch+1
            torch.save(net.state_dict(),fr'D:\软件安装\vscode\Code\PD_model\result\PD预测模型\CapsNet\Dataset_C/{Version_num}.pth')   

        results = f"EPOCH:[{best_epoch}|{epoch+1}/{EPOCH}],loss: {np.mean(loss_ls)/batch_size:.4f},Train_ACC: {train_acc:.4f}%"
        results += f'\t Auc:{auc:.4f},Test_ACC:{test_acc:.4f}%,Sn:{Sn:.4f},Sp:{Sp:.4f}'
        results += f'\t F1:{F1:.4f},MCC:{MCC:.4f}, time:{time.time()-t0:.2f}'
        print(results)
        To_log(results)


    if best_acc > KF_Acc:
        K,KF_Acc,KF_AUC,KF_sn,KF_sp,KF_f1,KF_mcc,KF_epoch = Kflod_num+1,best_acc,AUC,sn,sp,f1,mcc,best_epoch

print('************************************************************5-flod Best Performance**************************************************************')
print('第{}Flod,epoch:{},ACC:{:.4f},AUC:{:.4f},SN:{:.4f},SP:{:.4f},F1:{:.4f},MCC:{:.4f}'.format(K,KF_epoch,KF_Acc,KF_AUC,KF_sn,KF_sp,KF_f1,KF_mcc))