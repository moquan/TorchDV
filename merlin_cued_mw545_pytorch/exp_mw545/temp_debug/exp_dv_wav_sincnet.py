# exp_dv_wav_sincnet.py

# d-vector style model
# https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/41939.pdf

# For each window, network input is a vector of stacked waveforms
# Then within each window, split into smaller windows (M)
# Both slicings are performed in pytorch, not numpy

import os, sys, pickle, time, shutil, logging, copy
import math, numpy, scipy
numpy.random.seed(545)
from modules import make_logger, read_file_list, prepare_file_path, prepare_file_path_list, make_held_out_file_number, copy_to_scratch
from modules import keep_by_speaker, remove_by_speaker, keep_by_file_number, remove_by_file_number, keep_by_min_max_file_number, check_and_change_to_list
from modules_2 import compute_feat_dim, log_class_attri, resil_nn_file_list, norm_nn_file_list, get_utters_from_binary_dict, get_one_utter_by_name, count_male_female_class_errors
from modules_torch import torch_initialisation

from io_funcs.binary_io import BinaryIOCollection
io_fun = BinaryIOCollection()

from exp_mw545.exp_dv_cmp_pytorch import list_random_loader, dv_y_configuration, make_dv_y_exp_dir_name, make_dv_file_list, train_dv_y_model, class_test_dv_y_model, distance_test_dv_y_model, vuv_test_dv_y_model, ce_vs_var_nlf_test
from exp_mw545.exp_dv_cmp_pytorch import make_feed_dict_y_wav_subwin_train, make_feed_dict_y_wav_subwin_test


class dv_y_wav_sincnet_configuration(dv_y_configuration):
    
    def __init__(self, cfg):
        super().__init__(cfg)
        self.use_voiced_only = False # Use voiced regions only for training; weighted CE criterion
        self.use_voiced_threshold = 1. # Percentage of voiced required
        self.train_by_window = True # Optimise lambda_w; False: optimise speaker level lambda
        self.classify_in_training = True # Compute classification accuracy after validation errors during training
        self.batch_output_form = 'mean' # Method to convert from SBD to SD
        self.finetune_model = False
        # self.learning_rate  = 0.0001
        # self.prev_nnets_file_name = '/home/dawna/tts/mw545/TorchDV/dv_wav_subwin/dv_y_wav_lr_0.000001_ReL80_ReL256BN_ReL256BN_ReL8DR_DV8S100B23T3200D1/Model'
        self.python_script_name = os.path.realpath(__file__)

        # Waveform-level input configuration
        self.y_feat_name   = 'wav'
        self.out_feat_list = ['wav']
        self.batch_seq_total_len = int(0.25*16000) # Number of frames at 16kHz; 32000 for 2s
        self.batch_seq_len   = 3200 # T
        self.batch_seq_shift = 80
        self.seq_win_len   = 251
        self.seq_win_shift = 1
        self.seq_num_win   = int((self.batch_seq_len - self.seq_win_len) / self.seq_win_shift) + 1

        self.batch_num_spk = 100
        self.dv_dim = 8
        self.nn_layer_config_list = [
            # Must contain: type, size; num_channels, dropout_p are optional, default 0, 1
            # {'type':'LReLUSubWin', 'size':80, 'win_len':self.seq_win_len, 'num_win':self.seq_num_win, 'dropout_p':0, 'batch_norm':False},
            {'type':'SCNet_ST', 'size':80, 'win_len_shift_list':[[self.batch_seq_len, self.batch_seq_shift], [self.seq_win_len, self.seq_win_shift]], 'total_length':self.batch_seq_total_len, 'dropout_p':0, 'batch_norm':False},
            {'type':'LReLUDV', 'size':256, 'dropout_p':0, 'batch_norm':True},
            {'type':'LReLUDV', 'size':256, 'dropout_p':0, 'batch_norm':True},
            {'type':'LReLUDV', 'size':self.dv_dim, 'dropout_p':0.2, 'batch_norm':False}
        ]

        # self.gpu_id = 'cpu'
        self.gpu_id = 0

        from modules_torch import DV_Y_ST_model
        self.dv_y_model_class = DV_Y_ST_model

        if self.use_voiced_only:
            self.out_feat_list = ['wav', 'vuv']
        self.make_feed_dict_method_train = make_feed_dict_y_wav_subwin_train
        self.make_feed_dict_method_test  = make_feed_dict_y_wav_subwin_test
        # self.make_feed_dict_method_distance = make_feed_dict_y_wav_subwin_distance
        self.make_feed_dict_method_vuv_test = make_feed_dict_y_wav_subwin_train
        self.auto_complete(cfg)

        # self.dv_y_data_loader = dv_y_wav_subwin_data_loader

    def reload_model_param(self):
        self.nn_layer_config_list[0]['win_len_shift_list'] = [[self.batch_seq_len, self.batch_seq_shift], [self.seq_win_len, self.seq_win_shift]]
        self.nn_layer_config_list[0]['total_length'] = self.batch_seq_total_len

def train_dv_y_wav_model(cfg, dv_y_cfg=None):
    if dv_y_cfg is None: dv_y_cfg = dv_y_wav_sincnet_configuration(cfg)
    train_dv_y_model(cfg, dv_y_cfg)

def test_dv_y_wav_model(cfg, dv_y_cfg=None):
    if dv_y_cfg is None: dv_y_cfg = dv_y_wav_sincnet_configuration(cfg)
    class_test_dv_y_model(cfg, dv_y_cfg)
    # distance_test_dv_y_model(cfg, dv_y_cfg)
    # vuv_test_dv_y_model(cfg, dv_y_cfg)
    # ce_vs_var_nlf_test(cfg, dv_y_cfg)