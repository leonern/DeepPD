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

def Seqs2EqlTensor(file_path:str,max_len:int,AminoAcid_vocab=None):
    '''
    Args:
        flie:文件路径 \n
        max_len:设定转换后的氨基酸序列最大长度 \n
        vocab_dict:esm / protbert / default ,默认为按顺序映射的词典
    '''

    # 只保留20种氨基酸和填充数,其余几种非常规氨基酸均用填充数代替
    # 使用 esm和portbert字典时，nn.embedding()的vocab_size = 25
    if AminoAcid_vocab =='esm':
        aa_dict = {'[PAD]': 1, 'L': 4, 'A': 5, 'G': 6, 'V': 7, 'S': 8, 'E': 9, 'R': 10, 'T': 11, 'I': 12, 'D': 13, 'P': 14, 'K': 15, 'Q': 16, 
                   'N': 17, 'F': 18, 'Y': 19, 'M': 20, 'H': 21, 'W': 22, 'C': 23, 'X': 1, 'B': 1, 'U': 1, 'Z': 1, 'O': 1}
    elif AminoAcid_vocab == 'protbert':
        aa_dict = {'[PAD]':0,'L': 5, 'A': 6, 'G': 7, 'V': 8, 'E': 9, 'S': 10, 'I': 11, 'K': 12, 'R': 13, 'D': 14, 'T': 15, 
               'P': 16, 'N': 17, 'Q': 18, 'F': 19, 'Y': 20, 'M': 21, 'H': 22, 'C': 23, 'W': 24, 'X': 0, 'U': 0, 'B': 0, 'Z': 0, 'O': 0}
    else:
        aa_dict = {'[PAD]':0,'A':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'K':9,'L':10,'M':11,'N':12,'P':13,'Q':14,'R':15,
               'S':16,'T':17,'V':18,'W':19,'Y':20,'U':0,'X':0,'J':0}
        # aa_dict = {'A':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'K':9,'L':10,'M':11,'N':12,'P':13,'Q':14,'R':15,'S':16,'T':17,'V':18,'W':19,'Y':20,'U':0,'X':0}
    ## Esm vocab
    ## protbert vocab
    
    padding_key = '[PAD]'
    default_padding_value = 0
    if padding_key in aa_dict:
        dict_padding_value = aa_dict.get('[PAD]')
    else:
        dict_padding_value = default_padding_value
        print(f"No padding value in the implicit dictionary, set to {default_padding_value} by default")

    with open(file_path, 'r') as inf:
        lines = inf.read().splitlines()
    # assert len(lines) % 2 == 0, "Invalid file format. Number of lines should be even."
    
    long_pep_counter=0
    pep_codes=[]
    labels=[]
    pos_count = 0
    neg_count = 0
    for line in lines:
        pep,label = line.split(",")
        labels.append(int(label))
        if int(label) == int(1):
            pos_count+=1
        else:
            neg_count+=1
    
        seq_len = len(pep)
        if  seq_len <= max_len:
            current_pep=[]
            for aa in pep:
                if aa.upper() in aa_dict.keys():
                    current_pep.append(aa_dict[aa.upper()])
            pep_codes.append(torch.tensor(current_pep)) #torch.tensor(current_pep)
        else:
            pep_head = pep[0:int(max_len/2)]
            pep_tail = pep[int(seq_len-int(max_len/2)):int(seq_len)]
            new_pep = pep_head+pep_tail
            current_pep=[]
            for aa in new_pep:
                current_pep.append(aa_dict[aa])
            pep_codes.append(torch.tensor(current_pep))
            long_pep_counter += 1
  
    print("length > {}:{},postive sample:{},negative sample:{}".format(max_len,long_pep_counter,pos_count,neg_count))
    data = rnn_utils.pad_sequence(pep_codes,batch_first=True,padding_value=dict_padding_value)
    return data,torch.tensor(labels)

def index_alignment(batch,condition_num=0,subtraction_num1=4,subtraction_num2=1):
    '''将其他蛋白质语言模型的字典索引和默认字典索引进行对齐，保持氨基酸索引只有20个数构成，且范围在[1,20]，[PAD]=0或者1 \n
    "esm"模型，condition_num=1,subtraction_num1=3，subtraction_num2=1； \n
    "protbert"模型，condition_num=0,subtraction_num1=4

    Args:               
        batch:形状为[batch_size,seq_len]的二维张量 \n
        condition_num:字典中的[PAD]值 \n 
        subtraction_num1:对齐非[PAD]元素所需减掉的差值 \n
        subtraction_num2:对齐[PAD]元素所需减掉的差值
    
    return:
        shape:[batch_size,seq_len],dtype=tensor.
    '''
    condition = batch == condition_num
    # 创建一个张量，形状和batch相同，表示非[PAD]元素要减去的值
    subtraction = torch.full_like(batch, subtraction_num1)
    if condition_num==0:
        # 使用torch.where()函数来选择batch中为0的元素或者batch减去subtraction中的元素
        output = torch.where(condition, batch, batch - subtraction)
    elif condition_num==1:
        # 创建一个张量，形状和batch相同，表示[PAD]元素要减去的值
        subtraction_2 = torch.full_like(batch, subtraction_num2)
        output = torch.where(condition, batch-subtraction_2, batch - subtraction)
    
    return output

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

def Data2EqlTensor(lines,max_len):
    aa_dict = {'A':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'K':9,'L':10,'M':11,'N':12,'P':13,'Q':14,'R':15,'S':16,'T':17,'V':18,'W':19,'Y':20,'U':0,'X':0}

    long_pep_counter=0
    pep_codes=[]
    ids = []
    for id,pep in lines:
        ids.append(id)
        x = len(pep)
        # 将第一个长度<max_len的序列填充到40，确保当输入序列均<max_len时，所有序列仍然能够填充到max_len
        pad_flag = 1
        if  x < max_len:
            current_pep=[]
            for aa in pep:
                current_pep.append(aa_dict[aa])
            if pad_flag:
                current_pep.extend([0] * (max_len - len(current_pep)))
                pad_flag = 0
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

    print("length>"+str(max_len)+':',long_pep_counter)
    data = rnn_utils.pad_sequence(pep_codes,batch_first=True)

    return data,ids

blosum62 = {
        '1': [4, -1, -2, -2, 0,  -1, -1, 0, -2,  -1, -1, -1, -1, -2, -1, 1,  0,  -3, -2, 0],  # A
        '15': [-1, 5,  0, -2, -3, 1,  0,  -2, 0,  -3, -2, 2,  -1, -3, -2, -1, -1, -3, -2, -3], # R
        '12': [-2, 0,  6,  1,  -3, 0,  0,  0,  1,  -3, -3, 0,  -2, -3, -2, 1,  0,  -4, -2, -3], # N
        '3': [-2, -2, 1,  6,  -3, 0,  2,  -1, -1, -3, -4, -1, -3, -3, -1, 0,  -1, -4, -3, -3], # D
        '2': [0,  -3, -3, -3, 9,  -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1], # C
        '14': [-1, 1,  0,  0,  -3, 5,  2,  -2, 0,  -3, -2, 1,  0,  -3, -1, 0,  -1, -2, -1, -2], # Q
        '4': [-1, 0,  0,  2,  -4, 2,  5,  -2, 0,  -3, -3, 1,  -2, -3, -1, 0,  -1, -3, -2, -2], # E
        '6': [0,  -2, 0,  -1, -3, -2, -2, 6,  -2, -4, -4, -2, -3, -3, -2, 0,  -2, -2, -3, -3], # G
        '7': [-2, 0,  1,  -1, -3, 0,  0,  -2, 8,  -3, -3, -1, -2, -1, -2, -1, -2, -2, 2,  -3], # H
        '8': [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4,  2,  -3, 1,  0,  -3, -2, -1, -3, -1, 3],  # I
        '10': [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2,  4,  -2, 2,  0,  -3, -2, -1, -2, -1, 1],  # L
        '9': [-1, 2,  0,  -1, -3, 1,  1,  -2, -1, -3, -2, 5,  -1, -3, -1, 0,  -1, -3, -2, -2], # K
        '11': [-1, -1, -2, -3, -1, 0,  -2, -3, -2, 1,  2,  -1, 5,  0,  -2, -1, -1, -1, -1, 1],  # M
        '5': [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0,  0,  -3, 0,  6,  -4, -2, -2, 1,  3,  -1], # F
        '13': [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7,  -1, -1, -4, -3, -2], # P
        '16': [1,  -1, 1,  0,  -1, 0,  0,  0,  -1, -2, -2, 0,  -1, -2, -1, 4,  1,  -3, -2, -2], # S
        '17': [0,  -1, 0,  -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1,  5,  -2, -2, 0],  # T
        '19': [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1,  -4, -3, -2, 11, 2,  -3], # W
        '20': [-2, -2, -2, -3, -2, -1, -2, -3, 2,  -1, -1, -2, -1, 3,  -3, -2, -2, 2,  7,  -1], # Y
        '18': [0,  -3, -3, -3, -1, -2, -2, -3, -3, 3,  1,  -2, 1,  -1, -2, -2, 0,  -3, -1, 4],  # V
        '0': [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # -
    }

def get_blosum62(seq):
    # 使用列表推导式和字典get方法代替循环
    seq = seq.tolist()
    seq2b62 = np.array([blosum62.get(str(i)) for i in seq])
    return seq2b62

def seqs2blosum62(sequences):
   
    evolution = np.array([get_blosum62(seq) for seq in sequences],dtype=float)

    return torch.from_numpy(evolution)