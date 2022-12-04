# exp_dv_wav_sinenet_v4.py

# d-vector style model
# https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/41939.pdf

# For each window, network input is a vector of stacked waveforms
# Then within each window, split into smaller windows (M)
# Reaper predicted lf0 and pitch values
# In addition to Sinenet, add a DNN component

import os, sys, pickle, time, shutil, logging, copy
import math, numpy, scipy
numpy.random.seed(545)
from modules import make_logger, read_file_list, prepare_file_path, prepare_file_path_list, make_held_out_file_number, copy_to_scratch
from modules import keep_by_speaker, remove_by_speaker, keep_by_file_number, remove_by_file_number, keep_by_min_max_file_number, check_and_change_to_list
from modules_2 import compute_feat_dim, log_class_attri, resil_nn_file_list, norm_nn_file_list, get_utters_from_binary_dict, get_one_utter_by_name, count_male_female_class_errors
from modules_torch import torch_initialisation

from io_funcs.binary_io import BinaryIOCollection
io_fun = BinaryIOCollection()

from exp_mw545.exp_dv_cmp_pytorch import list_random_loader, dv_y_configuration, make_dv_y_exp_dir_name, make_dv_file_list, train_dv_y_model, class_test_dv_y_model, distance_test_dv_y_model #, plot_sinenet
from exp_mw545.exp_dv_wav_sinenet_v3 import make_feed_dict_y_wav_sinenet_train, make_feed_dict_y_wav_sinenet_test, make_feed_dict_y_wav_sinenet_distance

class dv_y_wav_sinenet_configuration(dv_y_configuration):
    
    def __init__(self, cfg):
        super().__init__(cfg)
        self.train_by_window = True # Optimise lambda_w; False: optimise speaker level lambda
        self.classify_in_training = True # Compute classification accuracy after validation errors during training
        self.batch_output_form = 'mean' # Method to convert from SBD to SD
        self.finetune_model = False
        # self.learning_rate  = 0.0000001
        # self.prev_nnets_file_name = '/home/dawna/tts/mw545/TorchDV/debug_nausicaa/dv_y_wav_lr_0.000100_Sin80_ReL256BN_ReL256BNDR_LRe16DR_DV16S100B10T3200D1_smallbatch/Model'
        self.python_script_name = os.path.realpath(__file__)

        # Waveform-level input configuration
        self.y_feat_name   = 'wav'
        self.out_feat_list = ['wav']
        self.batch_seq_total_len = 32000 # Number of frames at 16kHz; 32000 for 2s
        self.batch_seq_len   = 3200 # T
        self.batch_seq_shift = 3200
        self.seq_win_len   = 640
        self.seq_win_shift = 80
        self.seq_num_win   = int((self.batch_seq_len - self.seq_win_len) / self.seq_win_shift) + 1

        self.learning_rate = 0.0001
        self.batch_num_spk = 100
        self.dv_dim = 8
        self.nn_layer_config_list = [
            # Must contain: type, size; num_channels, dropout_p are optional, default 0, 1
            {'type':'SinenetV4', 'size':101, 'sine_size':80, 'num_freq':10, 'relu_size':20, 'win_len':self.seq_win_len, 'num_win':self.seq_num_win, 'dropout_p':0, 'batch_norm':False},
            {'type':'LReLUDV', 'size':256, 'dropout_p':0, 'batch_norm':True},
            {'type':'LReLUDV', 'size':256, 'dropout_p':0.2, 'batch_norm':True},
            {'type':'LReLUDV', 'size':self.dv_dim, 'dropout_p':0.2, 'batch_norm':False}
        ]

        # self.gpu_id = 'cpu'
        self.gpu_id = 1

        from modules_torch import DV_Y_Wav_SubWin_model
        self.dv_y_model_class = DV_Y_Wav_SubWin_model
        # from exp_mw545.exp_dv_wav_baseline import make_feed_dict_y_wav_cmp_test
        self.make_feed_dict_method_train = make_feed_dict_y_wav_sinenet_train
        self.make_feed_dict_method_test  = make_feed_dict_y_wav_sinenet_test
        self.make_feed_dict_method_distance  = make_feed_dict_y_wav_sinenet_distance
        self.auto_complete(cfg)


    def additional_action_epoch(self, logger, dv_y_model):
        pass

def train_dv_y_wav_model(cfg, dv_y_cfg=None):
    if dv_y_cfg is None: dv_y_cfg = dv_y_wav_sinenet_configuration(cfg)
    train_dv_y_model(cfg, dv_y_cfg)

def test_dv_y_wav_model(cfg, dv_y_cfg=None):
    if dv_y_cfg is None: dv_y_cfg = dv_y_wav_sinenet_configuration(cfg)
    # class_test_dv_y_model(cfg, dv_y_cfg)
    distance_test_dv_y_model(cfg, dv_y_cfg)
    # plot_sinenet(cfg, dv_y_cfg)
