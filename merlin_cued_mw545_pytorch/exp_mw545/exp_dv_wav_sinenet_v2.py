# exp_dv_wav_sinenet_v2.py

# d-vector style model
# https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/41939.pdf

# For each window, network input is a vector of stacked waveforms
# Then within each window, split into smaller windows (M)
# Reaper predicted lf0 and pitch values
# Stack with lf0, tau and vuv
import os
from exp_mw545.exp_dv_config import dv_y_configuration

# from frontend_mw545.data_loader import Build_dv_y_train_data_loader
class dv_y_wav_sinenet_configuration(dv_y_configuration):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.use_voiced_only = False   # Use voiced regions only
        self.use_voiced_threshold = 1. # Percentage of voiced required
        self.finetune_model = False
        # self.learning_rate  = 0.0001
        # self.prev_nnets_file_name = ''
        self.python_script_name = os.path.realpath(__file__)
        self.data_dir_mode = 'scratch' # Use scratch for speed up

        # Waveform-level input configuration
        self.y_feat_name   = 'wav'
        self.init_wav_data()
        self.out_feat_list = ['wav_ST', 'f_SBM', 'tau_SBM', 'vuv_SBM']

        # self.input_data_dim['S'] = 10
        # self.input_data_dim['T_S'] = 6000   # Number of frames at 16kHz; 32000 for 2s
        # self.input_data_dim['B_shift'] = 80
        # self.input_data_dim['T_B'] = 3200
        # self.input_data_dim['M_shift'] = 80
        # self.input_data_dim['T_M'] = 640

        # self.dv_dim = 8
        self.nn_layer_config_list = [
            {'type': 'Tensor_Reshape', 'io_name': 'wav_ST_2_wav_SBMT', 'win_len_shift_list':[[self.input_data_dim['T_B'], self.input_data_dim['B_shift']], [self.input_data_dim['T_M'], self.input_data_dim['M_shift']]]},
            {'type': 'Sinenet_V1_Residual', 'size':86, 'num_freq':16, 'dropout_p':0, 'batch_norm':False}, # S_B_D output; D <- M*(D+1)
            {'type': 'Tensor_Reshape', 'io_name': 'h_SBMD_2_h_SBD'},     # h_SBD
            {'type': 'LReLU', 'size':256, 'dropout_p':0, 'batch_norm':True},
            {'type': 'LReLU', 'size':256, 'dropout_p':0, 'batch_norm':True},
            {'type': 'Linear', 'size':self.dv_dim, 'dropout_p':0, 'batch_norm':True}
        ]

        # self.gpu_id = 'cpu'
        self.gpu_id = 0

        # if self.use_voiced_only:
        #     self.make_feed_dict_method_train = make_feed_dict_y_wav_sinenet_train_voiced_only
        #     self.make_feed_dict_method_test  = make_feed_dict_y_wav_sinenet_test_voiced_only
        # else:
        #     self.make_feed_dict_method_train = make_feed_dict_y_wav_sinenet_train
        #     self.make_feed_dict_method_test  = make_feed_dict_y_wav_sinenet_test
        #     self.make_feed_dict_method_distance  = make_feed_dict_y_wav_sinenet_distance
        # self.make_feed_dict_method_vuv_test = make_feed_dict_y_wav_sinenet_train
        self.auto_complete(cfg)

def train_model(cfg, dv_y_cfg=None):
    if dv_y_cfg is None: dv_y_cfg = dv_y_wav_sinenet_configuration(cfg)

    from exp_mw545.exp_dv_y import Build_DV_Y_Model_Trainer
    dv_y_model_trainer = Build_DV_Y_Model_Trainer(cfg, dv_y_cfg)
    dv_y_model_trainer.train()
    # dv_y_model_trainer.train_no_validation()

def test_model(cfg, dv_y_cfg=None):
    if dv_y_cfg is None: dv_y_cfg = dv_y_wav_sinenet_configuration(cfg)

    from exp_mw545.exp_dv_y import Build_DV_Y_Testing
    dv_y_model_test = Build_DV_Y_Testing(cfg, dv_y_cfg)
    # dv_y_model_test.vuv_loss_test()
    dv_y_model_test.sinenet_weight_test()