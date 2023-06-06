from Model import MyModel
import torch
import torch.nn as nn
from Bio import SeqIO
from Data_preprocessing import Data2EqlTensor
device = "cuda" if torch.cuda.is_available() else "cpu"

file_path = './test.fa'
data = []
for record in SeqIO.parse(file_path, 'fasta'):
    data.append((record.id, str(record.seq)))

seqs,ids = Data2EqlTensor(data,40)

softmax = nn.Softmax(1)
def predict(seqs,data,model_path,threshold=0.5, device=device):
    with torch.no_grad():
        model = MyModel()
        model.eval()
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict,strict=False)
        model.to(device)
        seqs = seqs.to(device)
        out = model(seqs)
        prob = softmax(out)[:,1]

        final_out = []
        for i, j in zip(data, prob):
            temp = [i[0], i[1], f"{j:.3f}", 'Peptide' if j >threshold else 'Non Peptide']
            final_out.append(temp)
            
    return final_out