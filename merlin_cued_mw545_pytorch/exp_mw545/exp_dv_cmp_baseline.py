# exp_dv_cmp_baseline.py

# d-vector style model
# https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/41939.pdf

# For each window, network input is a vector of stacked vocoder feature vectors

import os
from exp_mw545.exp_dv_config import dv_y_configuration

class dv_y_cmp_configuration(dv_y_configuration):
    def __init__(self, cfg, cache_files=True):
        super().__init__(cfg)

        self.retrain_model = False
        self.learning_rate  = 0.0001
        # self.prev_nnets_file_name = '/home/dawna/tts/mw545/TorchDV/dv_cmp_baseline/dvy_cmp_lr1E-04_fpu40_LRe512L_LRe512L_Lin512L_DV512S10T40D3440_nTW5s_seed1/Model'
        self.python_script_name = os.path.realpath(__file__)
        self.data_dir_mode = 'scratch' # Use scratch for speed up
        self.train_by_window = True
        self.train_num_seconds = 5
        self.data_loader_random_seed = 0

        # cmp input configuration
        self.init_cmp_data()
        self.y_feat_name   = 'cmp'
        self.out_feat_list = ['mgc', 'lf0', 'bap']
        self.update_cmp_dim()

        self.input_data_dim['S'] = 10
        self.feed_per_update = 40
        # self.learning_rate = self.learning_rate / self.feed_per_update
        S_per_update = self.input_data_dim['S'] * self.feed_per_update
        self.epoch_num_batch  = {'train': int(52000/S_per_update), 'valid': int(8000/self.input_data_dim['S'])}
        # self.feed_per_update = 2
        # self.epoch_num_batch  = {'train': 10, 'valid': 10}

        self.dv_dim = 512
        self.nn_layer_config_list = [
            {'type':'LReLU', 'size':256*2, 'dropout_p':0, 'layer_norm':True},
            {'type':'LReLU', 'size':256*2, 'dropout_p':0., 'layer_norm':True},
            {'type':'Linear', 'size':self.dv_dim, 'dropout_p':0., 'layer_norm':True}
        ]

        # self.gpu_id = 'cpu'
        self.gpu_id = 0
        self.auto_complete(cfg, cache_files)

def train_model(cfg, dv_y_cfg=None):
    if dv_y_cfg is None: dv_y_cfg = dv_y_cmp_configuration(cfg)
    from exp_mw545.exp_dv_y import Build_DV_Y_Model_Trainer
    dv_y_model_trainer = Build_DV_Y_Model_Trainer(cfg, dv_y_cfg)
    dv_y_model_trainer.train()

def test_model(cfg, dv_y_cfg=None):
    if dv_y_cfg is None: dv_y_cfg = dv_y_cmp_configuration(cfg)

    from exp_mw545.exp_dv_y import Build_DV_Y_Testing
    dv_y_model_test = Build_DV_Y_Testing(cfg, dv_y_cfg)
    fig_file_name = '/home/dawna/tts/mw545/Export_Temp/PNG_out/vuv_loss_cmp.png'
    dv_y_model_test.vuv_loss_test(fig_file_name)
    fig_file_name = '/home/dawna/tts/mw545/Export_Temp/PNG_out/positional_cmp.png'
    dv_y_model_test.positional_test(fig_file_name)
    
    # Additional output dir; also output to the exp dir
    output_dir = '/home/dawna/tts/mw545/Export_Temp/PNG_out'
    dv_y_model_test.gen_dv(output_dir)
    dv_y_model_test.cross_entropy_accuracy_test()
    dv_y_model_test.number_secs_accu_test()
    dv_y_model_test.positional_test(output_dir, distance_type='cosine')
    


