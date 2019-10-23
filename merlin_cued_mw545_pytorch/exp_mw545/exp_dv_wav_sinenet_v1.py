# exp_dv_wav_sinenet_v1.py

# d-vector style model
# https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/41939.pdf

# For each window, network input is a vector of stacked waveforms

import os, sys, pickle, time, shutil, logging, copy
import math, numpy, scipy
numpy.random.seed(545)
from modules import make_logger, read_file_list, prepare_file_path, prepare_file_path_list, make_held_out_file_number, copy_to_scratch
from modules import keep_by_speaker, remove_by_speaker, keep_by_file_number, remove_by_file_number, keep_by_min_max_file_number, check_and_change_to_list
from modules_2 import compute_feat_dim, log_class_attri, resil_nn_file_list, norm_nn_file_list, get_utters_from_binary_dict, get_one_utter_by_name, count_male_female_class_errors
from modules_torch import torch_initialisation

from io_funcs.binary_io import BinaryIOCollection
io_fun = BinaryIOCollection()

from exp_mw545.exp_dv_cmp_pytorch import list_random_loader, dv_y_configuration, make_dv_y_exp_dir_name, make_dv_file_list, train_dv_y_model, class_test_dv_y_model


class dv_y_wav_cmp_configuration(dv_y_configuration):
    """docstring for ClassName"""
    def __init__(self, cfg):
        super().__init__(cfg)
        self.train_by_window = True # Optimise lambda_w; False: optimise speaker level lambda
        self.classify_in_training = True # Compute classification accuracy after validation errors during training
        self.batch_output_form = 'mean' # Method to convert from SBD to SD
        self.retrain_model = False
        self.previous_model_name = ''
        # self.python_script_name = '/home/dawna/tts/mw545/tools/merlin/merlin_cued_mw545_pytorch/exp_mw545/exp_dv_cmp_pytorch.py'
        self.python_script_name = os.path.realpath(__file__)

        # Waveform-level input configuration
        self.y_feat_name   = 'wav'
        self.out_feat_list = ['wav']
        self.batch_seq_total_len = 32000 # Number of frames at 16kHz; 32000 for 2s
        self.batch_seq_len   = 3200 # T
        self.batch_seq_shift = 5*80
        self.nn_layer_config_list = [
            # Must contain: type, size; num_channels, dropout_p are optional, default 0, 1
            # {'type':'SineAttenCNN', 'size':512, 'num_channels':1, 'dropout_p':1, 'CNN_filter_size':5, 'Sine_filter_size':200,'lf0_mean':5.04976, 'lf0_var':0.361811},
            # {'type':'CNNAttenCNNWav', 'size':1024, 'num_channels':1, 'dropout_p':1, 'CNN_kernel_size':[1,3200], 'CNN_stride':[1,80], 'CNN_activation':'ReLU'},
            {'type':'Sinenet', 'size':256, 'num_channels':2, 'channel_combi':'stack', 'dropout_p':0, 'batch_norm':False},
            {'type':'ReLUDVMax', 'size':256, 'num_channels':2, 'channel_combi':'maxout', 'dropout_p':0, 'batch_norm':False},
            {'type':'ReLUDVMax', 'size':256, 'num_channels':2, 'channel_combi':'maxout', 'dropout_p':0.5, 'batch_norm':False},
            # {'type':'ReLUDVMax', 'size':self.dv_dim, 'num_channels':2, 'channel_combi':'maxout', 'dropout_p':0.5, 'batch_norm':False}
            {'type':'LinDV', 'size':self.dv_dim, 'num_channels':1, 'dropout_p':0.5}
        ]

        self.gpu_id = 1

        from modules_torch import DV_Y_CMP_model
        self.dv_y_model_class = DV_Y_CMP_model
        from exp_dv_wav_baseline import make_feed_dict_y_wav_cmp_train, make_feed_dict_y_wav_cmp_test
        self.make_feed_dict_method_train = make_feed_dict_y_wav_cmp_train
        self.make_feed_dict_method_test  = make_feed_dict_y_wav_cmp_test
        self.auto_complete(cfg)

def train_dv_y_wav_model(cfg, dv_y_cfg=None):
    if dv_y_cfg is None: dv_y_cfg = dv_y_wav_cmp_configuration(cfg)
    train_dv_y_model(cfg, dv_y_cfg)

def test_dv_y_wav_model(cfg, dv_y_cfg=None):
    if dv_y_cfg is None: dv_y_cfg = dv_y_wav_cmp_configuration(cfg)
    class_test_dv_y_model(cfg, dv_y_cfg)

