# exp_dv_cmp_baseline.py

# d-vector style model
# https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/41939.pdf

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


def make_feed_dict_y_cmp_train(dv_y_cfg, file_list_dict, file_dir_dict, batch_speaker_list, utter_tvt, return_dv=False, return_y=False, return_frame_index=False, return_file_name=False):
    feat_name = dv_y_cfg.y_feat_name # Hard-coded here for now
    # Make i/o shape arrays
    # This is numpy shape, not Tensor shape!
    y  = numpy.zeros((dv_y_cfg.batch_num_spk, dv_y_cfg.spk_num_seq, dv_y_cfg.batch_seq_len, dv_y_cfg.feat_dim))
    dv = numpy.zeros((dv_y_cfg.batch_num_spk))

    # Do not use silence frames at the beginning or the end
    total_sil_one_side = dv_y_cfg.frames_silence_to_keep+dv_y_cfg.sil_pad
    min_file_len = dv_y_cfg.batch_seq_total_len + 2 * total_sil_one_side

    file_name_list = []
    start_frame_index_list = []
    for speaker_idx in range(dv_y_cfg.batch_num_spk):
        speaker_id = batch_speaker_list[speaker_idx]

        # Make classification targets, index sequence
        true_speaker_index = dv_y_cfg.speaker_id_list_dict['train'].index(speaker_id)
        dv[speaker_idx] = true_speaker_index

        # Draw multiple utterances per speaker: dv_y_cfg.spk_num_utter
        # Draw multiple windows per utterance:  dv_y_cfg.utter_num_seq
        # Stack them along B
        speaker_file_name_list, speaker_utter_len_list, speaker_utter_list = get_utters_from_binary_dict(dv_y_cfg.spk_num_utter, file_list_dict[(speaker_id, utter_tvt)], file_dir_dict, feat_name_list=[feat_name], feat_dim_list=[dv_y_cfg.feat_dim], min_file_len=min_file_len, random_seed=None)
        file_name_list.append(speaker_file_name_list)

        speaker_start_frame_index_list = []
        for utter_idx in range(dv_y_cfg.spk_num_utter):
            y_stack = speaker_utter_list[feat_name][utter_idx][:,dv_y_cfg.feat_index]
            frame_number   = speaker_utter_len_list[utter_idx]
            extra_file_len = frame_number - (min_file_len)
            start_frame_index = numpy.random.choice(range(total_sil_one_side, total_sil_one_side+extra_file_len+1))
            speaker_start_frame_index_list.append(start_frame_index)
            for seq_idx in range(dv_y_cfg.utter_num_seq):
                y[speaker_idx, utter_idx*dv_y_cfg.utter_num_seq+seq_idx, :, :] = y_stack[start_frame_index:start_frame_index+dv_y_cfg.batch_seq_len, :]
                start_frame_index = start_frame_index + dv_y_cfg.batch_seq_shift
        start_frame_index_list.append(speaker_start_frame_index_list)


    # S,B,T,D --> S,B,T*D
    x_val = numpy.reshape(y, (dv_y_cfg.batch_num_spk, dv_y_cfg.spk_num_seq, dv_y_cfg.batch_seq_len*dv_y_cfg.feat_dim))
    if dv_y_cfg.train_by_window:
        # S --> S*B
        y_val = numpy.repeat(dv, dv_y_cfg.spk_num_seq)
        batch_size = dv_y_cfg.batch_num_spk * dv_y_cfg.spk_num_seq
    else:
        y_val = dv
        batch_size = dv_y_cfg.batch_num_spk

    feed_dict = {'x':x_val, 'y':y_val}
    return_list = [feed_dict, batch_size]
    
    if return_dv:
        return_list.append(dv)
    if return_y:
        return_list.append(y)
    if return_frame_index:
        return_list.append(start_frame_index_list)
    if return_file_name:
        return_list.append(file_name_list)
    return return_list

def make_feed_dict_y_cmp_test(dv_y_cfg, file_dir_dict, speaker_id, file_name, start_frame_index, BTD_feat_remain):
    feat_name = dv_y_cfg.y_feat_name # Hard-coded here for now
    assert dv_y_cfg.batch_num_spk == 1
    # Make i/o shape arrays
    # This is numpy shape, not Tensor shape!
    # No speaker index here! Will add it to Tensor later
    y  = numpy.zeros((dv_y_cfg.spk_num_seq, dv_y_cfg.batch_seq_len, dv_y_cfg.feat_dim))
    dv = numpy.zeros((dv_y_cfg.batch_num_spk))

    # Do not use silence frames at the beginning or the end
    total_sil_one_side = dv_y_cfg.frames_silence_to_keep+dv_y_cfg.sil_pad

    # Make classification targets, index sequence
    try: true_speaker_index = dv_y_cfg.speaker_id_list_dict['train'].index(speaker_id)
    except ValueError: true_speaker_index = 0 # At generation time, since dv is not used, a non-train speaker is given an arbituary speaker index
    dv[0] = true_speaker_index

    if BTD_feat_remain is None:
        # Get new file, make BTD
        _min_len, features = get_one_utter_by_name(file_name, file_dir_dict, feat_name_list=[feat_name], feat_dim_list=[dv_y_cfg.feat_dim])
        y_features = features[feat_name]
        l = y_features.shape[0]
        l_no_sil = l - total_sil_one_side * 2
        features_no_sil = y_features[total_sil_one_side:total_sil_one_side+l_no_sil]
        B_total  = int((l_no_sil - dv_y_cfg.batch_seq_len) / dv_y_cfg.batch_seq_shift) + 1
        BTD_features = numpy.zeros((B_total, dv_y_cfg.batch_seq_len, dv_y_cfg.feat_dim))
        for b in range(B_total):
            start_i = dv_y_cfg.batch_seq_shift * b
            BTD_features[b] = features_no_sil[start_i:start_i+dv_y_cfg.batch_seq_len]
    else:
        BTD_features = BTD_feat_remain
        B_total = BTD_features.shape[0]

    if B_total > dv_y_cfg.spk_num_seq:
        B_actual = dv_y_cfg.spk_num_seq
        B_remain = B_total - B_actual
        gen_finish = False
    else:
        B_actual = B_total
        B_remain = 0
        gen_finish = True

    for b in range(B_actual):
        y[b] = BTD_features[b]

    if B_remain > 0:
        BTD_feat_remain = numpy.zeros((B_remain, dv_y_cfg.batch_seq_len, dv_y_cfg.feat_dim))
        for b in range(B_remain):
            BTD_feat_remain[b] = BTD_features[b + B_actual]
    else:
        BTD_feat_remain = None

    batch_size = B_actual

    # B,T,D --> S(1),B,T*D
    x_val = numpy.reshape(y, (dv_y_cfg.batch_num_spk, dv_y_cfg.spk_num_seq, dv_y_cfg.batch_seq_len*dv_y_cfg.feat_dim))
    if dv_y_cfg.train_by_window:
        # S --> S*B
        y_val = numpy.repeat(dv, dv_y_cfg.spk_num_seq)
    else:
        y_val = dv

    feed_dict = {'x':x_val, 'y':y_val}
    return_list = [feed_dict, gen_finish, batch_size, BTD_feat_remain]
    return return_list

class dv_y_cmp_configuration(dv_y_configuration):
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

        # Vocoder-level input configuration
        self.y_feat_name   = 'cmp'
        self.out_feat_list = ['mgc', 'lf0', 'bap']
        self.batch_seq_total_len = 400 # Number of frames at 200Hz; 400 for 2s
        self.batch_seq_len   = 40 # T
        self.batch_seq_shift = 5
        self.dv_dim = 64
        self.nn_layer_config_list = [
            # Must contain: type, size; num_channels, dropout_p are optional, default 0, 1
            # {'type':'SineAttenCNN', 'size':512, 'num_channels':1, 'dropout_p':1, 'CNN_filter_size':5, 'Sine_filter_size':200,'lf0_mean':5.04976, 'lf0_var':0.361811},
            # {'type':'CNNAttenCNNWav', 'size':1024, 'num_channels':1, 'dropout_p':1, 'CNN_kernel_size':[1,3200], 'CNN_stride':[1,80], 'CNN_activation':'ReLU'},
            {'type':'ReLUDVMax', 'size':256, 'num_channels':2, 'channel_combi':'maxout', 'dropout_p':0, 'batch_norm':False},
            {'type':'ReLUDVMax', 'size':256, 'num_channels':2, 'channel_combi':'maxout', 'dropout_p':0, 'batch_norm':False},
            # {'type':'ReLUDVMax', 'size':256, 'num_channels':2, 'channel_combi':'maxout', 'dropout_p':0.5, 'batch_norm':False},
            # {'type':'ReLUDVMax', 'size':self.dv_dim, 'num_channels':2, 'channel_combi':'maxout', 'dropout_p':0.5, 'batch_norm':False}
            {'type':'LinDV', 'size':self.dv_dim, 'num_channels':1, 'dropout_p':0.5}
        ]

        from modules_torch import DV_Y_CMP_model
        self.dv_y_model_class = DV_Y_CMP_model
        self.make_feed_dict_method_train = make_feed_dict_y_cmp_train
        self.make_feed_dict_method_test  = make_feed_dict_y_cmp_test
        self.auto_complete(cfg)

def train_dv_y_cmp_model(cfg, dv_y_cfg=None):
    if dv_y_cfg is None: dv_y_cfg = dv_y_cmp_configuration(cfg)
    train_dv_y_model(cfg, dv_y_cfg)

def test_dv_y_cmp_model(cfg, dv_y_cfg=None):
    if dv_y_cfg is None: dv_y_cfg = dv_y_cmp_configuration(cfg)
    # for s in [545,54,5]:
        # numpy.random.seed(s)
    class_test_dv_y_model(cfg, dv_y_cfg)

