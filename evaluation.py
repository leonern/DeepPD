from sklearn import metrics
from config import ArgsConfig
import torch

args = ArgsConfig()
def evaluating_indicator(data_iter, net,criterion):
    all_true = []
    all_pred = []
    loss_ls = []
    for x, y in data_iter:
        x,y = x.to(args.device),y.to(args.device)
        outputs,_,_ = net(x)
        loss = criterion(outputs,y)
        loss_ls.append(torch.tensor(loss.item()).cpu().detach().numpy())
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

    return Sn,Sp,precision,ACC,F1_score,MCC,auc,loss_ls