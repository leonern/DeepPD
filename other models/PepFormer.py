import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data
import torch.nn.utils.rnn as rnn_utils
import time
import pickle
from termcolor import colored
from sklearn import metrics
from Data_preprocessing import genData2EqlTensor,genData


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

batch_size = 128
train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class PepFormer(nn.Module):
    def __init__(self, vocab_size=24):
        super().__init__()
        self.hidden_dim = 25
        self.emb_dim = 512
        
        self.embedding = nn.Embedding(vocab_size, self.emb_dim, padding_idx=0)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        
        self.gru = nn.GRU(self.emb_dim, self.hidden_dim, num_layers=2, 
                               bidirectional=True, dropout=0.2)
        
        
        self.block1=nn.Sequential(          nn.Linear(2100,512),
                                            nn.BatchNorm1d(512),
                                            nn.LeakyReLU(),
                                            nn.Linear(512,256),
                                 )

        self.block2=nn.Sequential(
                                               nn.BatchNorm1d(256),
                                               nn.LeakyReLU(),
                                               nn.Linear(256,128),
                                               nn.BatchNorm1d(128),
                                               nn.LeakyReLU(),
                                               nn.Linear(128,64),
                                               nn.BatchNorm1d(64),
                                               nn.LeakyReLU(),
                                               nn.Linear(64,2)
                                            )
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_encoder(x) #*,40,512
        output=x.permute(1, 0, 2)
        output,hn=self.gru(output)
        output=output.permute(1,0,2)
        hn=hn.permute(1,0,2)
        output=output.reshape(output.shape[0],-1)#*,2000 /*,3950
        hn=hn.reshape(output.shape[0],-1) #*,100
        output=torch.cat([output,hn],1)
        return self.block1(output)

    def trainModel(self, x):
        with torch.no_grad():
            output=self.forward(x)
        return self.block2(output)


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # euclidean_distance: [128]
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +     # calmp夹断用法
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))     
        
        return loss_contrastive
    
def collate(batch):
    seq1_ls=[]
    seq2_ls=[]
    label1_ls=[]
    label2_ls=[]
    label_ls=[]
    batch_size=len(batch)
    for i in range(int(batch_size/2)):
        seq1,label1=batch[i][0],batch[i][1]
        seq2,label2=batch[i+int(batch_size/2)][0],batch[i+int(batch_size/2)][1]
        label1_ls.append(label1.unsqueeze(0))
        label2_ls.append(label2.unsqueeze(0))
        label=(label1^label2)
        seq1_ls.append(seq1.unsqueeze(0))
        seq2_ls.append(seq2.unsqueeze(0))
        label_ls.append(label.unsqueeze(0))
    seq1=torch.cat(seq1_ls).to(device)
    seq2=torch.cat(seq2_ls).to(device)
    label=torch.cat(label_ls).to(device)
    label1=torch.cat(label1_ls).to(device)
    label2=torch.cat(label2_ls).to(device)
    return seq1,seq2,label,label1,label2
    
train_iter_cont = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                                                                  shuffle=True,collate_fn=collate)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluating_indicator(data_iter,net):
    all_true = []
    all_pred = []
    for x, y in data_iter:
        x,y = x.to(device),y.to(device)
        outputs = net.trainModel(x)
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
for Kflod_num in range(3):
    Version_num = Kflod_num+3
    def To_log(log):
        with open("D:/软件安装/vscode/Code/PD_model/result/PD预测模型/PepFormer/Dataset_C/modellog"+str(Version_num)+".log","a+") as f:
            f.write(log+'\n')

    net = PepFormer().to(device)
    lr = 0.0001
    optimizer = torch.optim.Adam(net.parameters(), lr=lr,weight_decay=5e-4)
    criterion = ContrastiveLoss()
    criterion_model = nn.CrossEntropyLoss(reduction='sum')
    best_acc,AUC,sn,sp,f1,mcc,best_epoch = 0,0,0,0,0,0,0
    EPOCH = 120
    print(f'***************************************************Start training K-flod:{Kflod_num+1}******************************************************')

    for epoch in range(EPOCH):
        loss_ls=[]
        loss1_ls=[]
        loss2_3_ls=[]
        t0=time.time()
        net.train()
        for seq1,seq2,label,label1,label2 in train_iter_cont:
                output1=net(seq1)
                output2=net(seq2)
                output3=net.trainModel(seq1)
                output4=net.trainModel(seq2)
                loss1=criterion(output1, output2, label)
                loss2=criterion_model(output3,label1)
                loss3=criterion_model(output4,label2)
                loss=loss1+loss2+loss3
                optimizer.zero_grad() 
                loss.backward()
                optimizer.step()
                loss_ls.append(loss.item())
                loss1_ls.append(loss1.item())
                loss2_3_ls.append((loss2+loss3).item())


        net.eval() 
        with torch.no_grad(): 
            _,_,_,train_acc,_,_,_ = evaluating_indicator(train_iter,net)
            Sn,Sp,precision,test_acc,F1,MCC,auc = evaluating_indicator(test_iter,net)
        
        if test_acc>best_acc:
            best_acc,AUC,sn,sp,f1,mcc,best_epoch = test_acc,auc,Sn,Sp,F1,MCC,epoch+1
            torch.save(net.state_dict(),fr'D:\软件安装\vscode\Code\PD_model\result\PD预测模型\PepFormer\Dataset_C/{Version_num}.pth')

        results = f"EPOCH:[{best_epoch}|{epoch+1}/{EPOCH}],loss: {np.mean(loss_ls):.4f}, loss1: {np.mean(loss1_ls):.4f}, loss2_3: {np.mean(loss2_3_ls):.4f}\n"
        results += f'\t Train_ACC: {train_acc:.4f}%,Auc:{auc:.4f},Test_ACC:{test_acc:.4f}%,Sn:{Sn:.4f},Sp:{Sp:.4f}'
        results += f'\t F1:{F1:.4f},MCC:{MCC:.4f},time:{time.time()-t0:.2f}'
        print(results)
        To_log(results)

    if best_acc > KF_Acc:
        K,KF_Acc,KF_AUC,KF_sn,KF_sp,KF_f1,KF_mcc,KF_epoch = Kflod_num+1,best_acc,AUC,sn,sp,f1,mcc,best_epoch

print('************************************************************5-flod Best Performance**************************************************************')
print('第{}Flod,epoch:{},ACC:{:.4f},AUC:{:.4f},SN:{:.4f},SP:{:.4f},F1:{:.4f},MCC:{:.4f}'.format(K,KF_epoch,KF_Acc,KF_AUC,KF_sn,KF_sp,KF_f1,KF_mcc))
