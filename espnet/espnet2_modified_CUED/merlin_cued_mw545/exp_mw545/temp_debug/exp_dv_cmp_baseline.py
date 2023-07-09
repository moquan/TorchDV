# exp_dv_cmp_baseline.py

# d-vector style model
# https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/41939.pdf

# For each window, network input is a vector of stacked vocoder feature vectors

import os, sys, pickle, time, shutil, logging, copy
import math, numpy, scipy
numpy.random.seed(545)
from modules import make_logger, read_file_list, prepare_file_path, prepare_file_path_list, make_held_out_file_number, copy_to_scratch
from modules import keep_by_speaker, remove_by_speaker, keep_by_file_number, remove_by_file_number, keep_by_min_max_file_number, check_and_change_to_list
from modules_2 import compute_feat_dim, log_class_attri, resil_nn_file_list, norm_nn_file_list, get_utters_from_binary_dict, get_one_utter_by_name, count_male_female_class_errors
from modules_torch import torch_initialisation

from io_funcs.binary_io import BinaryIOCollection
io_fun = BinaryIOCollection()

from exp_mw545.exp_dv_cmp_pytorch import list_random_loader, dv_y_configuration, make_dv_y_exp_dir_name, make_dv_file_list, train_dv_y_model, class_test_dv_y_model, distance_test_dv_y_model

def make_feed_dict_y_cmp_train(dv_y_cfg, file_list_dict, file_dir_dict, batch_speaker_list, utter_tvt, all_utt_start_frame_index=None, return_one_hot=False, return_y=False, return_frame_index=False, return_file_name=False):
    feat_name = dv_y_cfg.y_feat_name # Hard-coded here for now
    # Make i/o shape arrays
    # This is numpy shape, not Tensor shape!
    y = numpy.zeros((dv_y_cfg.batch_num_spk, dv_y_cfg.spk_num_seq, dv_y_cfg.batch_seq_len, dv_y_cfg.feat_dim))
    one_hot = numpy.zeros((dv_y_cfg.batch_num_spk))
    
    # Do not use silence frames at the beginning or the end
    total_sil_one_side_cmp = dv_y_cfg.frames_silence_to_keep + dv_y_cfg.sil_pad
    min_file_len = dv_y_cfg.batch_seq_total_len + 2 * total_sil_one_side_cmp

    file_name_list = []
    start_frame_index_list = []
    for speaker_idx in range(dv_y_cfg.batch_num_spk):
        speaker_id = batch_speaker_list[speaker_idx]

        # Make classification targets, index sequence
        true_speaker_index = dv_y_cfg.speaker_id_list_dict['train'].index(speaker_id)
        one_hot[speaker_idx] = true_speaker_index

        # Draw 1 utterance per speaker
        # Draw multiple windows per utterance: dv_y_cfg.spk_num_seq
        # Stack them along B
        speaker_file_name_list, speaker_utter_len_list, speaker_utter_list = get_utters_from_binary_dict(1, file_list_dict[(speaker_id, utter_tvt)], file_dir_dict, feat_name_list=[feat_name], feat_dim_list=[dv_y_cfg.feat_dim], min_file_len=min_file_len, random_seed=None)
        file_name_list.append(speaker_file_name_list)

        speaker_start_frame_index_list = []

        file_name = speaker_file_name_list[0]
        cmp_file = speaker_utter_list[feat_name][0][:,dv_y_cfg.feat_index]
        cmp_file_len = speaker_utter_len_list[0]

        if all_utt_start_frame_index is None:
            # Use random starting frame index
            extra_file_len = cmp_file_len - min_file_len
            start_frame_index = numpy.random.randint(low=total_sil_one_side_cmp, high=total_sil_one_side_cmp+extra_file_len+1)
        else:
            start_frame_index = total_sil_one_side_cmp + all_utt_start_frame_index
        speaker_start_frame_index_list.append(start_frame_index)
        for seq_idx in range(dv_y_cfg.spk_num_seq):
            y[speaker_idx, seq_idx, :, :] = cmp_file[start_frame_index:start_frame_index+dv_y_cfg.batch_seq_len, :]
            start_frame_index = start_frame_index + dv_y_cfg.batch_seq_shift
                
        start_frame_index_list.append(speaker_start_frame_index_list)

    # S,B,T,D --> S,B,T*D
    x_val = numpy.reshape(y, (dv_y_cfg.batch_num_spk, dv_y_cfg.spk_num_seq, dv_y_cfg.batch_seq_len*dv_y_cfg.feat_dim))
    if dv_y_cfg.train_by_window:
        # S --> S*B
        y_val = numpy.repeat(one_hot, dv_y_cfg.spk_num_seq)
        batch_size = dv_y_cfg.batch_num_spk * dv_y_cfg.spk_num_seq
    else:
        y_val = one_hot
        batch_size = dv_y_cfg.batch_num_spk

    feed_dict = {'x':x_val, 'y':y_val}
    return_list = [feed_dict, batch_size]
    
    if return_one_hot:
        return_list.append(one_hot)
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
    y  = numpy.zeros((dv_y_cfg.batch_num_spk, dv_y_cfg.spk_num_seq, dv_y_cfg.batch_seq_len, dv_y_cfg.feat_dim))
    one_hot = numpy.zeros((dv_y_cfg.batch_num_spk))

    # Make classification targets, index sequence
    try: true_speaker_index = dv_y_cfg.speaker_id_list_dict['train'].index(speaker_id)
    except ValueError: true_speaker_index = 0 # At generation time, since one_hot is not used, a non-train speaker is given an arbituary speaker index
    one_hot[0] = true_speaker_index

    if BTD_feat_remain is not None:
        # There is only one feature, cmp, so no need to make a dict or something complicated
        # Also it is already in the shape of BTD; may change this if necessary
        cmp_feat_current = BTD_feat_remain
        B_total = cmp_feat_current.shape[0]
    else:
        # Get new file, make BTD
        file_min_len, features = get_one_utter_by_name(file_name, file_dir_dict, feat_name_list=[feat_name], feat_dim_list=[dv_y_cfg.feat_dim])
        cmp_file = features[feat_name]
        cmp_file_len = file_min_len
        # Do not use silence frames at the beginning or the end
        total_sil_one_side_cmp = dv_y_cfg.frames_silence_to_keep + dv_y_cfg.sil_pad
        len_no_sil = cmp_file_len - 2 * total_sil_one_side_cmp

        cmp_features_no_sil = cmp_file[total_sil_one_side_cmp:total_sil_one_side_cmp+len_no_sil]

        B_total = int((len_no_sil - dv_y_cfg.batch_seq_len) / dv_y_cfg.batch_seq_shift) + 1
        cmp_feat_current = numpy.zeros((B_total, dv_y_cfg.batch_seq_len, dv_y_cfg.feat_dim))
        for b in range(B_total):
            n_start = dv_y_cfg.batch_seq_shift * b
            cmp_feat_current[b] = cmp_features_no_sil[n_start:n_start+dv_y_cfg.batch_seq_len]

    if B_total > dv_y_cfg.spk_num_seq:
        B_actual = dv_y_cfg.spk_num_seq
        B_remain = B_total - B_actual
        gen_finish = False
        BTD_feat_remain = cmp_feat_current[B_actual:]
    else:
        B_actual = B_total
        B_remain = 0
        gen_finish = True
        BTD_feat_remain = None

    y[0,:B_actual] = cmp_feat_current[:B_actual]
    batch_size = B_actual

    # B,T,D --> S(1),B,T*D
    x_val = numpy.reshape(y, (dv_y_cfg.batch_num_spk, dv_y_cfg.spk_num_seq, dv_y_cfg.batch_seq_len*dv_y_cfg.feat_dim))
    if dv_y_cfg.train_by_window:
        # S --> S*B
        y_val = numpy.repeat(one_hot, dv_y_cfg.spk_num_seq)
    else:
        y_val = one_hot

    feed_dict = {'x':x_val, 'y':y_val}
    return_list = [feed_dict, gen_finish, batch_size, BTD_feat_remain]
    return return_list

def make_feed_dict_y_cmp_distance(dv_y_cfg, file_list_dict, file_dir_dict, batch_speaker_list, utter_tvt, all_utt_start_frame_index=None, return_y=False, return_frame_index=False, return_file_name=False):
    feat_name = dv_y_cfg.y_feat_name # Hard-coded here for now
    # Make i/o shape arrays
    # This is numpy shape, not Tensor shape!
    y_list = []
    for plot_idx in range(dv_y_cfg.num_to_plot + 1):
        y = numpy.zeros((dv_y_cfg.batch_num_spk, dv_y_cfg.spk_num_seq, dv_y_cfg.batch_seq_len, dv_y_cfg.feat_dim))
        y_list.append(y)
    
    # Do not use silence frames at the beginning or the end
    total_sil_one_side_cmp = dv_y_cfg.frames_silence_to_keep + dv_y_cfg.sil_pad
    min_file_len = dv_y_cfg.batch_seq_total_len + 2 * total_sil_one_side_cmp 
    # Add extra for shift distance test
    min_file_len = min_file_len + dv_y_cfg.max_len_to_plot

    file_name_list = []
    start_frame_index_list = []
    for speaker_idx in range(dv_y_cfg.batch_num_spk):
        speaker_id = batch_speaker_list[speaker_idx]

        # Draw 1 utterance per speaker
        # Draw multiple windows per utterance: dv_y_cfg.spk_num_seq
        # Stack them along B
        speaker_file_name_list, speaker_utter_len_list, speaker_utter_list = get_utters_from_binary_dict(1, file_list_dict[(speaker_id, utter_tvt)], file_dir_dict, feat_name_list=[feat_name], feat_dim_list=[dv_y_cfg.feat_dim], min_file_len=min_file_len, random_seed=None)
        file_name_list.append(speaker_file_name_list)

        speaker_start_frame_index_list = []

        file_name = speaker_file_name_list[0]
        cmp_file = speaker_utter_list[feat_name][0][:,dv_y_cfg.feat_index]
        cmp_file_len = speaker_utter_len_list[0]

        if all_utt_start_frame_index is None:
            # Use random starting frame index
            extra_file_len = cmp_file_len - min_file_len
            start_frame_index = numpy.random.randint(low=total_sil_one_side_cmp, high=total_sil_one_side_cmp+extra_file_len+1)
        else:
            start_frame_index = total_sil_one_side_cmp + all_utt_start_frame_index
        speaker_start_frame_index_list.append(start_frame_index)

        for plot_idx in range(dv_y_cfg.num_to_plot+1):
            plot_start_frame_index = start_frame_index + plot_idx * dv_y_cfg.gap_len_to_plot
            for seq_idx in range(dv_y_cfg.spk_num_seq):
                y_list[plot_idx][speaker_idx, seq_idx, :, :] = cmp_file[plot_start_frame_index:plot_start_frame_index+dv_y_cfg.batch_seq_len, :]
                plot_start_frame_index = plot_start_frame_index + dv_y_cfg.batch_seq_shift
        start_frame_index_list.append(speaker_start_frame_index_list)

    feed_dict_list = [{} for i in range(dv_y_cfg.num_to_plot+1)]
    for plot_idx in range(dv_y_cfg.num_to_plot+1):
        # S,B,T,D --> S,B,T*D
        x_val = numpy.reshape(y_list[plot_idx], (dv_y_cfg.batch_num_spk, dv_y_cfg.spk_num_seq, dv_y_cfg.batch_seq_len*dv_y_cfg.feat_dim))
        feed_dict_list[plot_idx] = {'x':x_val}
    batch_size  = dv_y_cfg.batch_num_spk * dv_y_cfg.spk_num_seq
    return_list = [feed_dict_list, batch_size]
    
    if return_y:
        return_list.append(y)
    if return_frame_index:
        return_list.append(start_frame_index_list)
    if return_file_name:
        return_list.append(file_name_list)
    return return_list

class dv_y_cmp_configuration(dv_y_configuration):
    
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
        self.batch_seq_total_len = 80 # Number of frames at 200Hz
        self.batch_seq_len   = 40 # T; Number of frames at 200Hz
        self.batch_seq_shift = 1
        self.dv_dim = 8
        self.nn_layer_config_list = [
            # Must contain: type, size; num_channels, dropout_p are optional, default 0, 1
            # {'type':'ReLUDVMax', 'size':256, 'num_channels':2, 'channel_combi':'maxout', 'dropout_p':0, 'batch_norm':False},
            # {'type':'LinDV', 'size':self.dv_dim, 'num_channels':1, 'dropout_p':0.5}
            {'type':'LReLUDV', 'size':256, 'dropout_p':0, 'batch_norm':True},
            {'type':'LReLUDV', 'size':256, 'dropout_p':0, 'batch_norm':True},
            {'type':'LReLUDV', 'size':self.dv_dim, 'dropout_p':0.2, 'batch_norm':False}
        ]

        self.gpu_id = 0

        from modules_torch import DV_Y_CMP_model
        self.dv_y_model_class = DV_Y_CMP_model
        self.make_feed_dict_method_train = make_feed_dict_y_cmp_train
        self.make_feed_dict_method_test  = make_feed_dict_y_cmp_test
        self.make_feed_dict_method_distance  = make_feed_dict_y_cmp_distance
        self.auto_complete(cfg)

def train_dv_y_cmp_model(cfg, dv_y_cfg=None):
    if dv_y_cfg is None: dv_y_cfg = dv_y_cmp_configuration(cfg)
    train_dv_y_model(cfg, dv_y_cfg)

def test_dv_y_cmp_model(cfg, dv_y_cfg=None):
    if dv_y_cfg is None: dv_y_cfg = dv_y_cmp_configuration(cfg)
    class_test_dv_y_model(cfg, dv_y_cfg)
    # distance_test_dv_y_model(cfg, dv_y_cfg)
