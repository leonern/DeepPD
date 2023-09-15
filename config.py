import os
import torch

class ArgsConfig:
    def __init__(self) -> None:
        self.batch_size = 128
        self.embedding_size = 480
        self.epochs = 50
        self.kflod = 2
        self.max_len = 40
        self.lr = 1.5e-3
        self.weight_decay = 0
        self.is_autocast = False
        self.info_bottleneck = False
        self.dropout = 0.5
        self.margin = 2.8
        self.scale_factor = 1
        self.IB_beta = 1e-3
        self.model_name = 'DeepPD_C' #
        self.exp_nums = 0.0
        self.aa_dict = None # 'protbert' /'esm'/ None
        self.fl_alpha = 0.2
        self.fl_gamma = 2.0
        self.info = f""  #对当前训练做的补充说明
        
        self.data_c_dir = './data/GPMDB_Homo_sapiens_20190115/sorted_GPMDB_Homo_0.025_0.9.csv'
        self.data_c1_dir = './data/GPMDB_Homo_sapiens_20190115/sorted_GPMDB_Homo_0.025.csv'
        self.data_homo_dir = './data/PepFormer/Homo_0.9.csv'
        self.data_mus_dir = './data/PepFormer/Mus_0.9.csv'
        # self.ebv_dir = './eq_21_21.pkl'
        # self.use_ebv = True

        self.log_dir = './result/logs'
        self.save_dir = './result/model_para'
        self.tensorboard_log_dir = './tensorboard'
        self.ems_path = './ESM2/esm2_t12_35M_UR50D.pt'
        self.esm_layer_idx = 12
        self.save_para_dir = os.path.join(self.save_dir,self.model_name)
        self.random_seed = 2023
        self.num_classes = 21
        self.split_size = 0.8
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.continue_training = False
        # self.checkpoint_path = r'result\model_para\CNN_BIGRU_test1\1.pth'
        
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        if not os.path.exists(self.save_para_dir):
            os.mkdir(self.save_para_dir)
        if not os.path.exists(self.tensorboard_log_dir):
            os.mkdir(self.tensorboard_log_dir)