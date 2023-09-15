import torch
import torch.nn as nn
import torch.nn.functional as F
from Biometric_extraction import featureGenera
from utils import CBAMBlock
from utils import ResNet,Capsule,squash
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

bert_wight = BertModel.from_pretrained(r"D:\PretrainModel\bert_pytorch128")
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        vocab_size = 21
        batch_size = 64
        self.hidden_dim = 25
        self.gru_emb = 128
        self.emb_dim = 108

        self.model = bert_wight
        self.gru = nn.GRU(self.gru_emb, self.hidden_dim, num_layers=2, 
                               bidirectional=True,dropout=0.1)
        self.embedding = nn.Embedding(vocab_size, self.emb_dim, padding_idx=0)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)

        self.resnet = ResNet(batch_size)
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

from utils_etfc import *
import torch,esm
import torch.nn as nn
from Data_preprocessing import index_alignment,seqs2blosum62
import torch.nn.functional as f
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class DeepPD(nn.Module):
    def __init__(self, vocab_size:int, embedding_size:int, fan_layer_num:int, num_heads:int,encoder_layer_num:int=1,seq_len: int=40,
                 output_size:int=2, layer_idx=None,esm_path=None,dropout:float=0.6, max_pool: int=4,Contrastive_Learning=False,info_bottleneck=False):
        super(DeepPD, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.output_size = output_size
        self.seq_len = seq_len
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(self.dropout)
        self.encoder_layer_num = encoder_layer_num
        self.fan_layer_num = fan_layer_num
        self.num_heads = num_heads
        self.max_pool = max_pool
        self.ctl = Contrastive_Learning
        self.info_bottleneck = info_bottleneck

        self.ESMmodel,_ = esm.pretrained.load_model_and_alphabet_local(esm_path)
        self.ESMmodel.eval()
        self.layer_idx = layer_idx

        self.out_chs = 64
        self.kernel_sizes = [3,7]
        self.all_conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(self.embedding_size+20,out_channels=self.out_chs,kernel_size=self.kernel_sizes[i],padding=(self.kernel_sizes[i]-1)//2), #padding=(self.kernel_sizes[i]-1)//2,
                nn.BatchNorm1d(self.out_chs),
                nn.LeakyReLU()
            )
            for i in range(len(self.kernel_sizes))
        ])
       
        self.hidden_dim = 64
        self.gru = nn.GRU(self.out_chs*2, self.hidden_dim, num_layers=2, batch_first=True,
                               bidirectional=True,dropout=0.25)
        
        self.embed = nn.Embedding(self.vocab_size, self.embedding_size)
        # self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.embedding_size,nhead=self.num_heads,dropout=self.dropout)
        # self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        # self.MaxPool1d = nn.MaxPool1d(kernel_size=self.max_pool) # stride的默认值=kernel_size

        self.pos_encoding = PositionalEncoding(num_hiddens=self.embedding_size,dropout=self.dropout)
        self.attention_encode = AttentionEncode(self.dropout, self.embedding_size, self.num_heads,seq_len=self.seq_len,ffn=False)

        shape = int(40*(64*2+64)) # +64
        # self.fan = FAN_encode(self.dropout, shape)

        z_dim = 1024
        self.enc_mean = nn.Linear(shape,z_dim)
        self.enc_std = nn.Linear(shape,z_dim)
        self.dec = nn.Sequential(
                                    nn.Linear(z_dim,128),
                                    nn.BatchNorm1d(128),
                                    nn.LeakyReLU(),
                                    nn.Linear(128,self.output_size)
        )

        self.proj_layer = nn.Linear(self.embedding_size,self.out_chs)
        self.fc = nn.Sequential(    
                                    nn.Linear(shape,z_dim),
                                    nn.BatchNorm1d(z_dim),
                                    nn.LeakyReLU(),
                                    nn.Linear(z_dim,128),
                                    nn.BatchNorm1d(128),
                                    nn.LeakyReLU(),
                                    nn.Linear(128,self.output_size)
                                    )

    def CNN1DNet(self,x):

        for i in range(len(self.kernel_sizes)):
            conv = self.all_conv[i]
            conv_x = conv(x)
            # conv_x = self.MaxPool1d(conv_x)
            if i == 0:
                all_feats = conv_x
            else:
                all_feats = torch.cat([all_feats,conv_x],dim=1)
        return all_feats

    def forward(self, x):
        # x : [B,S=40]
        # get esm embedding
        with torch.no_grad():
            results = self.ESMmodel(x, repr_layers=[self.layer_idx], return_contacts=False)
        esm_x = results["representations"][self.layer_idx] #* 50 480 /640 /1280 # [B,S,480]

        x = index_alignment(x,condition_num=1,subtraction_num1=3,subtraction_num2=1)
        # feature A
        embed_x = self.embed(x) # [batch_size,seq_len,embedding_size] c
        pos_x = self.pos_encoding(embed_x * math.sqrt(self.embedding_size)) # [batch_size,seq_len,embedding_size]
        encoding_x = pos_x # [B,S,480]
        
        for _ in range(self.encoder_layer_num):
            encoding_x = self.attention_encode(encoding_x)
            encoding_x += embed_x
        featA = encoding_x + esm_x

        # feature B
        pssm = seqs2blosum62(x).to(device) # B,S,20
        featB = pssm.type_as(embed_x)
        featAB = torch.cat([featA,featB],dim=2) # B,S,480+20

        cnn_input = featAB.permute(0, 2, 1) # B,H,S
        cnn_output = self.CNN1DNet(cnn_input) # B,out_chs*2,S
        out = self.dropout_layer(cnn_output)
        # out = self.dropout_layer(featA)
        out = out.permute(0,2,1) # B,S,H:out_chs*2
        out,_ = self.gru(out)
        
        out = self.dropout_layer(out)
        final_featAB = out.reshape(x.size(0),-1) # B,S*H:40*hidden_dim(64)*2
        
        # feature C
        featC = self.proj_layer(esm_x)
        featC = self.dropout_layer(featC)
        featC = featC.reshape(featC.shape[0],-1)

        feat = torch.cat([final_featAB,featC],1) # B
        final_feat = self.dropout_layer(feat) # B,S*(64*2+64)
        # final_feat = final_featAB
        # final_feat = featC

        if self.info_bottleneck:
            # ToxIBTL prediction head
            enc_mean, enc_std = self.enc_mean(final_feat), f.softplus(self.enc_std(final_feat)-5)
            eps = torch.randn_like(enc_std)
            IB_out = enc_mean + enc_std*eps
            logits = self.dec(IB_out)
            return logits,enc_mean,enc_std
            # return featA,featB,featAB,final_featAB,featC,enc_mean
        else:
            # 全连接层
            logits = self.fc(final_feat)
            return logits,logits,logits
            # return featA,featB,featAB,final_featAB,featC,logits