import numpy as np
import os
from sklearn import metrics
import torch
import torch.nn as nn
import torch.utils.data as Data
import time
import pickle
from termcolor import colored
from Biometric_extraction import featureGenera
from Sub_unit import CBAMBlock
from Sub_unit import Res_Net,Capsule,squash
from Data_preprocessing import genData2EqlTensor,Numseq2OneHot
from transformers import BertTokenizer,BertModel
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cuda",0)
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
torch.manual_seed(10)
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

batch_size = 64
train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,drop_last = True)
test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,drop_last = True)


#########################################################################################################################################
bert_wight = BertModel.from_pretrained(r"C:\Users\xiaoleo\Anaconda3\envs\env3.8\Lib\site-packages\bert_pytorch128")
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        vocab_size = 21
        self.hidden_dim = 25
        self.gru_emb = 128
        self.emb_dim = 108

        self.model = bert_wight
        self.gru = nn.GRU(self.gru_emb, self.hidden_dim, num_layers=2, 
                               bidirectional=True,dropout=0.1)
        self.embedding = nn.Embedding(vocab_size, self.emb_dim, padding_idx=0)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)

        self.resnet = Res_Net(batch_size)
        self.cbamBlock = CBAMBlock(batch_size)
        self.dropout = nn.Dropout(0.2)

        self.convblock1 = nn.Sequential(
            nn.Conv2d(1,batch_size,1),
            nn.BatchNorm2d(batch_size),
            nn.LeakyReLU()
            )
        self.convblock2 = nn.Sequential(
            nn.Conv2d(batch_size,1,1),
            nn.BatchNorm2d(1),
            nn.LeakyReLU()
            )

        self.fc = nn.Sequential(    nn.Linear(4200,512),
                                    nn.BatchNorm1d(512),
                                    nn.LeakyReLU(),
                                    nn.Linear(512,32),
                                    nn.BatchNorm1d(32),
                                    nn.LeakyReLU(),
                                    nn.Linear(32,2))

    def forward(self, x):
        xx = self.embedding(x)  #* 40 128  #* 40 108
        z = Numseq2OneHot(x) #* 40 20
        z = z.type_as(xx)
        out = torch.cat([xx,z],2)
        out = self.transformer_encoder(out)

        out = out.unsqueeze(1)
        out = self.convblock1(out) #*,32,40,128
        out = self.resnet(out)
        out = self.resnet(out)
        out = self.cbamBlock(out)
        out = self.convblock2(out) #*,1,40,128
        out = out.squeeze(1)
        out = out.permute(1,0,2) #40,*,128
        out,hn = self.gru(out)
        out = out.permute(1,0,2) #*,40,50
        hn = hn.permute(1,0,2) #*,4,25 
        out = out.reshape(out.shape[0],-1) #* 2000
        hn = hn.reshape(hn.shape[0],-1) #* 100
        out = torch.cat([out,hn],1) #* 2100  /2600 /1600/4050

        out1 = self.model(x)[0] #*,40,128
        out1 = out1.permute(1,0,2) #40,*,128
        out1,hn1 = self.gru(out1)
        out1 = out1.permute(1,0,2) #*,40,50
        hn1= hn1.permute(1,0,2) #*,4,25 
        out1 = out1.reshape(out1.shape[0],-1) #* 2000
        hn1 = hn1.reshape(hn1.shape[0],-1) #* 100
        out1 = torch.cat([out1,hn1],1) #* 2100 /2600 /1600/4050

        out = torch.cat([out1,out],1) #* 4200
        out = self.fc(out)
        
        return out

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
for Kflod_num in range(5):
    Version_num = Kflod_num+1
    def To_log(log):
        with open("D:/软件安装/vscode/Code/PD_model/result/PD预测模型/DeepPD/Dataset_C/modellog"+str(Version_num)+".log","a+") as f:
            f.write(log+'\n')

    net = MyModel().to(device)
    lr = 0.001
    optimizer = torch.optim.Adam(net.parameters(), lr=lr,weight_decay=5e-4)
    criterion_model = nn.CrossEntropyLoss(reduction='sum')
    best_acc,AUC,sn,sp,f1,mcc,best_epoch = 0,0,0,0,0,0,0
    EPOCH = 25 #50
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
            torch.save(net.state_dict(),fr'D:\软件安装\vscode\Code\PD_model\result\PD预测模型\DeepPD\Dataset_C/{Version_num}.pth')
                
        results = f"EPOCH:[{best_epoch}|{epoch+1}/{EPOCH}],loss: {np.mean(loss_ls)/batch_size:.4f},Train_ACC: {train_acc:.4f}%"
        results += f'\t Auc:{auc:.4f},Test_ACC:{test_acc:.4f}%,Sn:{Sn:.4f},Sp:{Sp:.4f}'
        results += f'\t F1:{F1:.4f},MCC:{MCC:.4f}, time:{time.time()-t0:.2f}'
        print(results)
        To_log(results)
        
    if best_acc > KF_Acc:
        K,KF_Acc,KF_AUC,KF_sn,KF_sp,KF_f1,KF_mcc,KF_epoch = Kflod_num+1,best_acc,AUC,sn,sp,f1,mcc,best_epoch

print('************************************************************5-flod Best Performance**************************************************************')
print('第{}Flod,epoch:{},ACC:{:.4f},AUC:{:.4f},SN:{:.4f},SP:{:.4f},F1:{:.4f},MCC:{:.4f}'.format(K,KF_epoch,KF_Acc,KF_AUC,KF_sn,KF_sp,KF_f1,KF_mcc))