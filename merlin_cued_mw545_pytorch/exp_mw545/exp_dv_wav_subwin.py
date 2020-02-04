# exp_dv_wav_subwin.py

# d-vector style model
# https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/41939.pdf

# For each window, network input is a vector of stacked waveforms
# Then within each window, split into smaller windows (M)

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

def make_feed_dict_y_wav_subwin_train(dv_y_cfg, file_list_dict, file_dir_dict, batch_speaker_list, utter_tvt, all_utt_start_frame_index=None, return_dv=False, return_y=False, return_frame_index=False, return_file_name=False):
    logger = make_logger("make_dict")

    '''
    Draw Utterances; Load Data
    Draw starting frame; Slice; Fit into numpy holders
    '''
    feat_name_list = ['wav'] # Load wav
    feat_dim_list  = [1]
    # Make i/o shape arrays
    # This is numpy shape, not Tensor shape!
    wav = numpy.zeros((dv_y_cfg.batch_num_spk, dv_y_cfg.spk_num_seq, dv_y_cfg.seq_num_win, dv_y_cfg.seq_win_len))
    dv  = numpy.zeros((dv_y_cfg.batch_num_spk))

    wav_sr  = dv_y_cfg.cfg.wav_sr
    cmp_sr  = dv_y_cfg.cfg.frame_sr
    wav_cmp_ratio = int(wav_sr / cmp_sr)
    # Do not use silence frames at the beginning or the end
    total_sil_one_side_cmp = dv_y_cfg.frames_silence_to_keep + dv_y_cfg.sil_pad  # This is at 200Hz
    total_sil_one_side_wav = total_sil_one_side_cmp * wav_cmp_ratio              # This is at 16kHz
    min_file_len = dv_y_cfg.batch_seq_total_len + 2 * total_sil_one_side_wav # This is at 16kHz

    file_name_list = []
    start_frame_index_list = [[]]*dv_y_cfg.batch_num_spk
    
    for speaker_idx in range(dv_y_cfg.batch_num_spk):

        speaker_id = batch_speaker_list[speaker_idx]
        # Make classification targets, index sequence
        true_speaker_index = dv_y_cfg.speaker_id_list_dict['train'].index(speaker_id)
        dv[speaker_idx] = true_speaker_index

        # Draw multiple utterances per speaker: dv_y_cfg.spk_num_utter
        # Draw multiple windows per utterance:  dv_y_cfg.utter_num_seq
        # Stack them along B
        speaker_file_name_list, speaker_utter_len_list, speaker_utter_list = get_utters_from_binary_dict(dv_y_cfg.spk_num_utter, file_list_dict[(speaker_id, utter_tvt)], file_dir_dict, feat_name_list=feat_name_list, feat_dim_list=feat_dim_list, min_file_len=min_file_len, random_seed=None)
        file_name_list.append(speaker_file_name_list)

        for utter_idx in range(dv_y_cfg.spk_num_utter):
            file_name = speaker_file_name_list[utter_idx]
            wav_file  = speaker_utter_list['wav'][utter_idx] # T * 1; 16kHz
            wav_file  = numpy.squeeze(wav_file, axis=1)      # T*1 -> T
            wav_file_len = speaker_utter_len_list[utter_idx]

            # Find start frame index, random if None
            if all_utt_start_frame_index is None:
                extra_file_len = wav_file_len - min_file_len
                utter_start_frame_index = numpy.random.randint(low=total_sil_one_side_wav, high=total_sil_one_side_wav+extra_file_len+1)
            else:
                utter_start_frame_index = total_sil_one_side_wav + all_utt_start_frame_index
            start_frame_index_list[speaker_idx].append(utter_start_frame_index)

            # Slice data into seq and win
            for seq_idx in range(dv_y_cfg.utter_num_seq):
                spk_seq_index = utter_idx * dv_y_cfg.utter_num_seq + seq_idx
                seq_start = utter_start_frame_index + seq_idx * dv_y_cfg.batch_seq_shift
                for win_idx in range(dv_y_cfg.seq_num_win):
                    win_start = seq_start + win_idx * dv_y_cfg.seq_win_shift
                    win_end   = win_start + dv_y_cfg.seq_win_len - 1 # Inclusive index
                    wav[speaker_idx, spk_seq_index, win_idx, :] = wav_file[win_start:win_end+1]

    # S,B,M,D
    x_val = wav
    if dv_y_cfg.train_by_window:
        # S --> S*B
        y_val = numpy.repeat(dv, dv_y_cfg.spk_num_seq)
        batch_size = dv_y_cfg.batch_num_spk * dv_y_cfg.spk_num_seq
    else:
        y_val = one_hot
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

def make_feed_dict_y_wav_subwin_test(dv_y_cfg, file_dir_dict, speaker_id, file_name, start_frame_index, BTD_feat_remain):
    logger = make_logger("make_dict")

    '''Load Data; load starting frame; Slice; Fit into numpy holders
    '''
    # BTD_feat_remain is a tuple now,
    # BTD_feat_remain = (y_feat_remain, nlf_feat_remain, tau_feat_remain)
    feat_name_list = ['wav'] # Load wav
    feat_dim_list  = [1]
    assert dv_y_cfg.batch_num_spk == 1
    # Make i/o shape arrays
    # This is numpy shape, not Tensor shape!
    wav = numpy.zeros((dv_y_cfg.batch_num_spk, dv_y_cfg.spk_num_seq, dv_y_cfg.seq_num_win, dv_y_cfg.seq_win_len))
    dv  = numpy.zeros((dv_y_cfg.batch_num_spk))

    # Make classification targets, index sequence
    try: true_speaker_index = dv_y_cfg.speaker_id_list_dict['train'].index(speaker_id)
    except ValueError: true_speaker_index = 0 # At generation time, since dv is not used, a non-train speaker is given an arbituary speaker index
    dv[0] = true_speaker_index

    if BTD_feat_remain is not None:
        wav_feat_current = BTD_feat_remain
        B_total = wav_feat_current.shape[0]
    else:
        # Get new file, make BTD
        file_min_len, features = get_one_utter_by_name(file_name, file_dir_dict, feat_name_list=feat_name_list, feat_dim_list=feat_dim_list)
        wav_file = features['wav'] # T * 1; 16kHz
        wav_file = numpy.squeeze(wav_file, axis=1)      # T*1 -> T
        wav_file_len = file_min_len

        wav_sr = dv_y_cfg.cfg.wav_sr
        cmp_sr = dv_y_cfg.cfg.frame_sr
        wav_cmp_ratio = int(wav_sr / cmp_sr)

        # Do not use silence frames at the beginning or the end
        total_sil_one_side_cmp = dv_y_cfg.frames_silence_to_keep + dv_y_cfg.sil_pad
        total_sil_one_side_wav = total_sil_one_side_cmp * wav_cmp_ratio
        len_no_sil_wav = wav_file_len - 2 * total_sil_one_side_wav

        # Make numpy holders for no_sil data
        wav_features_no_sil = wav_file[total_sil_one_side_wav:total_sil_one_side_wav+len_no_sil_wav]
        B_total = int((len_no_sil_wav - dv_y_cfg.batch_seq_len) / dv_y_cfg.batch_seq_shift) + 1
        wav_feat_current = numpy.zeros((B_total, dv_y_cfg.seq_num_win, dv_y_cfg.seq_win_len))

        # Slice data into seq and win
        for seq_idx in range(B_total):
            spk_seq_index = seq_idx
            seq_start = 0 + seq_idx * dv_y_cfg.batch_seq_shift
            for win_idx in range(dv_y_cfg.seq_num_win):
                win_start = seq_start + win_idx * dv_y_cfg.seq_win_shift
                win_end   = win_start + dv_y_cfg.seq_win_len - 1 # Inclusive index
                wav_feat_current[spk_seq_index, win_idx, :] = wav_file[win_start:win_end+1]

    if B_total > dv_y_cfg.spk_num_seq:
        B_actual = dv_y_cfg.spk_num_seq
        B_remain = B_total - B_actual
        gen_finish = False
        wav_feat_remain = wav_feat_current[B_actual:]
        BTD_feat_remain = wav_feat_remain
    else:
        B_actual = B_total
        B_remain = 0
        gen_finish = True
        BTD_feat_remain = None

    wav[0,:B_actual] = wav_feat_current[:B_actual]
    batch_size = B_actual

    # B,T,D --> S(1),B,T*D
    x_val = wav
    if dv_y_cfg.train_by_window:
        # S --> S*B
        y_val = numpy.repeat(dv, dv_y_cfg.spk_num_seq)
    else:
        y_val = dv

    feed_dict = {'x':x_val, 'y':y_val}
    return_list = [feed_dict, gen_finish, batch_size, BTD_feat_remain]
    return return_list

def make_feed_dict_y_wav_subwin_distance(dv_y_cfg, file_list_dict, file_dir_dict, batch_speaker_list, utter_tvt, all_utt_start_frame_index=None, return_dv=False, return_y=False, return_frame_index=False, return_file_name=False):
    logger = make_logger("make_dict")

    '''
    Draw Utterances; Load Data
    Draw starting frame; Slice; Fit into numpy holders
    '''
    feat_name_list = ['wav'] # Load wav
    feat_dim_list  = [1]
    # Make i/o shape arrays
    # This is numpy shape, not Tensor shape!
    wav_list = []
    for plot_idx in range(dv_y_cfg.num_to_plot + 1):
        wav = numpy.zeros((dv_y_cfg.batch_num_spk, dv_y_cfg.spk_num_seq, dv_y_cfg.seq_num_win, dv_y_cfg.seq_win_len))
        wav_list.append(wav)

    wav_sr  = dv_y_cfg.cfg.wav_sr
    cmp_sr  = dv_y_cfg.cfg.frame_sr
    wav_cmp_ratio = int(wav_sr / cmp_sr)
    # Do not use silence frames at the beginning or the end
    total_sil_one_side_cmp = dv_y_cfg.frames_silence_to_keep + dv_y_cfg.sil_pad  # This is at 200Hz
    total_sil_one_side_wav = total_sil_one_side_cmp * wav_cmp_ratio              # This is at 16kHz
    min_file_len = dv_y_cfg.batch_seq_total_len + 2 * total_sil_one_side_wav # This is at 16kHz
    # Add extra for shift distance test
    min_file_len = min_file_len + dv_y_cfg.max_len_to_plot

    file_name_list = []
    start_frame_index_list = [[]]*dv_y_cfg.batch_num_spk
    
    for speaker_idx in range(dv_y_cfg.batch_num_spk):

        speaker_id = batch_speaker_list[speaker_idx]

        # Draw multiple utterances per speaker: dv_y_cfg.spk_num_utter
        # Draw multiple windows per utterance:  dv_y_cfg.utter_num_seq
        # Stack them along B
        speaker_file_name_list, speaker_utter_len_list, speaker_utter_list = get_utters_from_binary_dict(dv_y_cfg.spk_num_utter, file_list_dict[(speaker_id, utter_tvt)], file_dir_dict, feat_name_list=feat_name_list, feat_dim_list=feat_dim_list, min_file_len=min_file_len, random_seed=None)
        file_name_list.append(speaker_file_name_list)

        for utter_idx in range(dv_y_cfg.spk_num_utter):
            file_name = speaker_file_name_list[utter_idx]
            wav_file  = speaker_utter_list['wav'][utter_idx] # T * 1; 16kHz
            wav_file  = numpy.squeeze(wav_file, axis=1)      # T*1 -> T
            wav_file_len = speaker_utter_len_list[utter_idx]

            # Find start frame index, random if None
            if all_utt_start_frame_index is None:
                extra_file_len = wav_file_len - min_file_len
                utter_start_frame_index = numpy.random.randint(low=total_sil_one_side_wav, high=total_sil_one_side_wav+extra_file_len+1)
            else:
                utter_start_frame_index = total_sil_one_side_wav + all_utt_start_frame_index
            start_frame_index_list[speaker_idx].append(utter_start_frame_index)

            for plot_idx in range(dv_y_cfg.num_to_plot+1):
                plot_start_frame_index = utter_start_frame_index + plot_idx * dv_y_cfg.gap_len_to_plot
                # Slice data into seq and win
                for seq_idx in range(dv_y_cfg.utter_num_seq):
                    spk_seq_index = utter_idx * dv_y_cfg.utter_num_seq + seq_idx
                    seq_start = plot_start_frame_index + seq_idx * dv_y_cfg.batch_seq_shift
                    for win_idx in range(dv_y_cfg.seq_num_win):
                        win_start = seq_start + win_idx * dv_y_cfg.seq_win_shift
                        win_end   = win_start + dv_y_cfg.seq_win_len - 1 # Inclusive index
                        wav_list[plot_idx][speaker_idx, spk_seq_index, win_idx, :] = wav_file[win_start:win_end+1]

    # S,B,M,D
    feed_dict_list = [{}] * (dv_y_cfg.num_to_plot+1)
    for plot_idx in range(dv_y_cfg.num_to_plot+1):
        x_val = wav_list[plot_idx]
        feed_dict_list[plot_idx] = {'x':x_val}
    batch_size = dv_y_cfg.batch_num_spk * dv_y_cfg.spk_num_seq

    return_list = [feed_dict_list, batch_size]
    
    if return_dv:
        return_list.append(dv)
    if return_y:
        return_list.append(y)
    if return_frame_index:
        return_list.append(start_frame_index_list)
    if return_file_name:
        return_list.append(file_name_list)
    return return_list

class dv_y_wav_subwin_configuration(dv_y_configuration):
    
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
        self.batch_seq_total_len = 12000 # Number of frames at 16kHz; 12000 for 0.75s
        self.batch_seq_len   = 3200 # T
        self.batch_seq_shift = 5*80
        self.seq_win_len   = 640         # Number of frames at 16kHz; 640 for 40ms
        self.seq_win_shift = 80
        self.seq_num_win   = int((self.batch_seq_len - self.seq_win_len) / self.seq_win_shift) + 1

        self.learning_rate = 0.0001
        self.batch_num_spk = 100
        self.dv_dim = 16
        self.nn_layer_config_list = [
            # Must contain: type, size; num_channels, dropout_p are optional, default 0, 1
            {'type':'ReLUSubWin', 'size':80, 'win_len':self.seq_win_len, 'num_win':self.seq_num_win, 'dropout_p':0, 'batch_norm':False},
            {'type':'ReLUDV', 'size':256, 'dropout_p':0, 'batch_norm':True},
            {'type':'ReLUDV', 'size':256, 'dropout_p':0.2, 'batch_norm':True},
            {'type':'LReLUDV', 'size':self.dv_dim, 'dropout_p':0.2, 'batch_norm':False}
        ]

        # self.gpu_id = 'cpu'
        self.gpu_id = 1

        from modules_torch import DV_Y_Wav_SubWin_model
        self.dv_y_model_class = DV_Y_Wav_SubWin_model
        # from exp_mw545.exp_dv_wav_baseline import make_feed_dict_y_wav_cmp_test
        self.make_feed_dict_method_train = make_feed_dict_y_wav_subwin_train
        self.make_feed_dict_method_test  = make_feed_dict_y_wav_subwin_test
        self.make_feed_dict_method_distance  = make_feed_dict_y_wav_subwin_distance
        self.auto_complete(cfg)


    def additional_action_epoch(self, logger, dv_y_model):
        pass

def train_dv_y_wav_model(cfg, dv_y_cfg=None):
    if dv_y_cfg is None: dv_y_cfg = dv_y_wav_subwin_configuration(cfg)
    train_dv_y_model(cfg, dv_y_cfg)

def test_dv_y_wav_model(cfg, dv_y_cfg=None):
    if dv_y_cfg is None: dv_y_cfg = dv_y_wav_subwin_configuration(cfg)
    class_test_dv_y_model(cfg, dv_y_cfg)
    distance_test_dv_y_model(cfg, dv_y_cfg)
