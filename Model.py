import torch
import torch.nn as nn
import torch.nn.functional as F
from Biometric_extraction import featureGenera
from Sub_unit import CBAMBlock
from Sub_unit import Res_Net,Capsule,squash
from Data_preprocessing import Numseq2OneHot
from transformers import BertModel
from Biometric_extraction import featureGenera,AP3_PhyAndChem

class DeepMS_1DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        vocab_size = 21
        self.emb_dim = 32

        self.embedding = nn.Embedding(vocab_size, self.emb_dim, padding_idx=0)
        self.dropout = nn.Dropout(0.2)

        self.Conv1d = nn.Sequential(
            nn.Conv1d(40,64,3),
            nn.BatchNorm1d(64),
            nn.ReLU()
            )

        self.Conv1d_1 = nn.Sequential(
            nn.Conv1d(64,64,3),
            nn.BatchNorm1d(64),
            nn.ReLU()
            )

        self.fc = nn.Sequential(    nn.Linear(3840,512),
                                    nn.BatchNorm1d(512),
                                    nn.LeakyReLU(),
                                    nn.Linear(512,32),
                                    nn.BatchNorm1d(32),
                                    nn.LeakyReLU(),
                                    nn.Linear(32,2))

    def forward(self, x):
        out = self.embedding(x)  #*,40,64 /79 128 /40,128/40,32
        out = self.dropout(out)
        out = self.Conv1d(out)     
        out = self.Conv1d_1(out)
        out = out.reshape(out.shape[0],-1) 
        out = self.fc(out)
        
        return out

class DeepMS_2DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        vocab_size = 21
        self.emb_dim = 32

        self.embedding = nn.Embedding(vocab_size, self.emb_dim, padding_idx=0)
        self.dropout = nn.Dropout(0.2)

        self.Conv1d = nn.Sequential(
            nn.Conv2d(64,64,5),
            nn.BatchNorm2d(64),
            nn.ReLU()
            )

        self.convblock1 = nn.Sequential(
            nn.Conv2d(1,64,3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
            )
        self.convblock2 = nn.Sequential(
            nn.Conv2d(64,1,1),
            nn.BatchNorm2d(1),
            nn.LeakyReLU()
            )

        self.fc = nn.Sequential(    nn.Linear(884,512),
                                    nn.BatchNorm1d(512),
                                    nn.LeakyReLU(),
                                    nn.Linear(512,32),
                                    nn.BatchNorm1d(32),
                                    nn.LeakyReLU(),
                                    nn.Linear(32,2))

    def forward(self, x):
        out = self.embedding(x)  #*,40,64 /79 128 /40,128/40,32
        out = self.dropout(out)
        out = out.unsqueeze(1) 
        out = self.convblock1(out) #*,64,38,62 /*,64,77,126/38,126/38,30
        out = self.dropout(out)
        out = self.Conv1d(out) #*,64,34,58 /*,64,38,62/34,122/34,26
        out = self.convblock2(out) 
        out = out.reshape(out.shape[0],-1) 
        out = self.fc(out)
        
        return out
        

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

        out = self.conv2(out) #*,256,12,2 /*,256,32,2
        y = self.conv4(y) #*,256,36,4 /55,4
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

bert_wight = BertModel.from_pretrained(r"C:\Users\xiaoleo\Anaconda3\envs\env3.8\Lib\site-packages\bert_pytorch128")
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        batch_size = 64
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
        out = out.reshape(out.shape[0],-1) #* 900
        hn = hn.reshape(hn.shape[0],-1) #* 100
        out = torch.cat([out,hn],1) #* 1000

        out1 = self.model(x)[0] #*,40,128
        out1 = out1.permute(1,0,2) #40,*,128
        out1,hn1 = self.gru(out1)
        out1 = out1.permute(1,0,2) #*,40,50
        hn1= hn1.permute(1,0,2) #*,4,25 
        out1 = out1.reshape(out1.shape[0],-1) #* 2000
        hn1 = hn1.reshape(hn1.shape[0],-1) #* 100
        out1 = torch.cat([out1,hn1],1) #* 2100

        out = torch.cat([out1,out],1) #* 4200
        out = self.fc(out)
        
        return out
