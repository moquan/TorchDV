# exp_dv_cmp_baseline.py

# d-vector style model
# https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/41939.pdf

# For each window, network input is a vector of stacked vocoder feature vectors

import os
from exp_mw545.exp_dv_config import dv_y_configuration

class dv_y_cmp_configuration(dv_y_configuration):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.retrain_model = False
        self.learning_rate  = 0.0001
        # self.prev_nnets_file_name = ''
        self.python_script_name = os.path.realpath(__file__)
        self.data_dir_mode = 'data' # Use scratch for speed up

        # cmp input configuration
        self.y_feat_name   = 'cmp'
        self.init_cmp_data()
        self.out_feat_list = ['mgc', 'lf0', 'bap']
        self.update_cmp_dim()

        self.input_data_dim['S'] = 1
        self.feed_per_update = 40
        # self.learning_rate = self.learning_rate / self.feed_per_update
        S_per_update = self.input_data_dim['S'] * self.feed_per_update
        self.epoch_num_batch  = {'train': int(52000/S_per_update), 'valid': int(8000/self.input_data_dim['S'])}

        self.dv_dim = 512
        # self.input_data_dim['S'] = 1 # For computing GPU requirement
        self.nn_layer_config_list = [
            {'type':'LReLU', 'size':256*2, 'dropout_p':0, 'layer_norm':True},
            {'type':'LReLU', 'size':256*2, 'dropout_p':0, 'layer_norm':True},
            {'type':'Linear', 'size':self.dv_dim, 'dropout_p':0, 'layer_norm':True}
        ]

        # self.gpu_id = 'cpu'
        self.gpu_id = 0
        self.auto_complete(cfg)

def train_model(cfg, dv_y_cfg=None):
    if dv_y_cfg is None: dv_y_cfg = dv_y_cmp_configuration(cfg)
    from exp_mw545.exp_dv_y import Build_DV_Y_Model_Trainer
    dv_y_model_trainer = Build_DV_Y_Model_Trainer(cfg, dv_y_cfg)
    dv_y_model_trainer.train()

def test_model(cfg, dv_y_cfg=None):
    if dv_y_cfg is None: dv_y_cfg = dv_y_cmp_configuration(cfg)

    from exp_mw545.exp_dv_y import Build_DV_Y_Testing
    dv_y_model_test = Build_DV_Y_Testing(cfg, dv_y_cfg)
    
    # fig_file_name = '/home/dawna/tts/mw545/Export_Temp/PNG_out/vuv_loss_cmp.png'
    # dv_y_model_test.vuv_loss_test(fig_file_name)
    fig_file_name = '/home/dawna/tts/mw545/Export_Temp/PNG_out/positional_cmp.png'
    dv_y_model_test.positional_test(fig_file_name)

    # Additional output dir; also output to the exp dir
    # output_dir = '/home/dawna/tts/mw545/Export_Temp/PNG_out'
    # dv_y_model_test.gen_dv(output_dir)

