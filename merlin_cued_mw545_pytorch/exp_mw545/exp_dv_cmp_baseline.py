# exp_dv_cmp_baseline.py

# d-vector style model
# https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/41939.pdf

# For each window, network input is a vector of stacked vocoder feature vectors

import os
from exp_mw545.exp_dv_config import dv_y_configuration

class dv_y_cmp_configuration(dv_y_configuration):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.finetune_model = False
        # self.learning_rate  = 0.0001
        # self.prev_nnets_file_name = ''
        self.python_script_name = os.path.realpath(__file__)
        self.data_dir_mode = 'scratch' # Use scratch for speed up

        # cmp input configuration
        self.y_feat_name   = 'cmp'
        self.init_cmp_data()
        self.out_feat_list = ['mgc', 'lf0', 'bap']
        self.update_cmp_dim()

        # self.dv_dim = 8
        # self.input_data_dim['S'] = 1 # For computing GPU requirement
        self.nn_layer_config_list = [
            {'type':'LReLU', 'size':256, 'dropout_p':0, 'batch_norm':True},
            {'type':'LReLU', 'size':256, 'dropout_p':0, 'batch_norm':True},
            {'type':'Linear', 'size':self.dv_dim, 'dropout_p':0, 'batch_norm':True}
        ]

        # self.gpu_id = 'cpu'
        self.gpu_id = 0

        self.auto_complete(cfg)


def train_model(cfg, dv_y_cfg=None):
    if dv_y_cfg is None: dv_y_cfg = dv_y_cmp_configuration(cfg)
    from exp_mw545.exp_dv_y import Build_DV_Y_Model_Trainer
    dv_y_model_trainer = Build_DV_Y_Model_Trainer(cfg, dv_y_cfg)
    dv_y_model_trainer.train()