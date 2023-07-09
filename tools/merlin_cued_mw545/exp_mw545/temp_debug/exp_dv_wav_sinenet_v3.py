# exp_dv_wav_sinenet_v3.py

# d-vector style model
# https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/41939.pdf

# For each window, network input is a vector of stacked waveforms
# Then within each window, split into smaller windows (M)
# Reaper predicted lf0 and pitch values
import os

from exp_mw545.exp_dv_config import dv_y_configuration
from frontend_mw545.data_loader import Build_dv_y_train_data_loader

class dv_y_wav_sinenet_configuration(dv_y_configuration):
    
    def __init__(self, cfg):
        super().__init__(cfg)

        self.use_voiced_only = False   # Use voiced regions only
        self.use_voiced_threshold = 1. # Percentage of voiced required
        self.finetune_model = False
        # self.learning_rate  = 0.0001
        # self.prev_nnets_file_name = '/home/dawna/tts/mw545/TorchDV/dv_wav_sinenet_v3/dv_y_wav_lr_0.000100_Sin80f10_ReL256BN_ReL256BN_ReL8DR_DV8S100B10T3200D1/Model'
        self.python_script_name = os.path.realpath(__file__)
        # self.nn_data_dir = cfg.data_dir                 # Use when scratch is unavailable
        self.nn_data_dir = cfg.nn_feat_scratch_dir_root # Use scratch for speed up

        # Waveform-level input configuration
        self.y_feat_name   = 'wav'
        self.out_feat_list = ['wav_ST', 'f_SBM', 'tau_SBM', 'vuv_SBM']
        self.batch_seq_total_len = 6400 # Number of frames at 16kHz; 32000 for 2s
        self.batch_seq_len   = 3200 # T
        self.batch_seq_shift = 80
        self.seq_win_len   = 640
        self.seq_win_shift = 80

        # self.batch_num_spk = 100
        # self.dv_dim = 8
        self.nn_layer_config_list = [
            # Must contain: type, size; num_channels, dropout_p are optional, default 0, 1
            {'type': 'Tensor_Reshape', 'io_name': 'wav_ST_2_wav_SBMT', 'win_len_shift_list':[[self.batch_seq_len, self.batch_seq_shift], [self.seq_win_len, self.seq_win_shift]]},
            {'type':'SinenetV1', 'size':81, 'sine_size':80, 'num_freq':16, 'dropout_p':0, 'batch_norm':False},
            {'type':'LReLU', 'size':256, 'dropout_p':0, 'batch_norm':True},
            {'type':'LReLU', 'size':256, 'dropout_p':0, 'batch_norm':True},
            {'type':'Linear', 'size':self.dv_dim, 'dropout_p':0.2, 'batch_norm':True}
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