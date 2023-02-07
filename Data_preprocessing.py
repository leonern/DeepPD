from turtle import forward
import numpy as np
import os
import torch
import torch.optim as optim
import torch.nn.utils.rnn as rnn_utils

def genData2EqlTensor(file,max_len):
    aa_dict = {'A':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'K':9,'L':10,'M':11,'N':12,'P':13,'Q':14,'R':15,'S':16,'T':17,'V':18,'W':19,'Y':20,'U':0,'X':0}

    with open(file, 'r') as inf:
        lines = inf.read().splitlines()
        
    long_pep_counter=0
    pep_codes=[]
    labels=[]
    pos_cunt = 0
    neg_cunt = 0
    for pep in lines:
        pep,label = pep.split(",")
        labels.append(int(label))
        x = len(pep)
        if int(label) == int(1):
            pos_cunt+=1
        else:
            neg_cunt+=1
        
        if  x < max_len:
            current_pep=[]
            for aa in pep:
                current_pep.append(aa_dict[aa])
            pep_codes.append(torch.tensor(current_pep)) #torch.tensor(current_pep)
        else:
            pep_head = pep[0:int(max_len/2)]
            pep_tail = pep[int(x-int(max_len/2)):int(x)]
            new_pep = pep_head+pep_tail
            current_pep=[]
            for aa in new_pep:
                current_pep.append(aa_dict[aa])
            pep_codes.append(torch.tensor(current_pep))
            long_pep_counter += 1

    print("length>"+str(max_len)+':',long_pep_counter,'postive sample:',pos_cunt,'negative sample:',neg_cunt)
    data = rnn_utils.pad_sequence(pep_codes,batch_first=True)
    return data,torch.tensor(labels)

def genData(file,max_len):
    aa_dict = {'A':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'K':9,'L':10,'M':11,'N':12,'P':13,'Q':14,'R':15,'S':16,'T':17,'V':18,'W':19,'Y':20,'U':0,'X':0}
    with open(file, 'r') as inf:
        lines = inf.read().splitlines()
        
    long_pep_counter=0
    pep_codes=[]
    labels=[]
    pos_cunt = 0
    neg_cunt = 0
    for pep in lines:
        pep,label=pep.split(",")
        labels.append(int(label))

        if int(label) == int(1):
            pos_cunt+=1
        else:
            neg_cunt+=1

        if not len(pep) > max_len:
            current_pep=[]
            for aa in pep:
                current_pep.append(aa_dict[aa])
            pep_codes.append(torch.tensor(current_pep))
        else:
            long_pep_counter += 1
            
    print(f"length > {max_len}:",long_pep_counter,'postive sample:',pos_cunt,'negative sample:',neg_cunt)
    data = rnn_utils.pad_sequence(pep_codes,batch_first=True)
    return data,torch.tensor(labels)

def Numseq2OneHot(numseq):
    OneHot = []
    for seq in numseq:
        len_seq = len(seq)
        seq = seq.cpu().numpy()
        x = torch.zeros(len_seq,20)
        for i in range(len_seq):
            x[i][seq[i]-1] = 1
        OneHot.append(np.array(x))
    
    return torch.tensor(np.array(OneHot))