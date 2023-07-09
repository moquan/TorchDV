# exp_dv_wav_sinenet_v3.py

# d-vector style model
# https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/41939.pdf

# For each window, network input is a vector of stacked waveforms
# Then within each window, split into smaller windows (M)
# Reaper predicted lf0 and pitch values

import os, sys, pickle, time, shutil, logging, copy
import math, numpy, scipy
numpy.random.seed(545)
from modules import make_logger, read_file_list, prepare_file_path, prepare_file_path_list, make_held_out_file_number, copy_to_scratch
from modules import keep_by_speaker, remove_by_speaker, keep_by_file_number, remove_by_file_number, keep_by_min_max_file_number, check_and_change_to_list
from modules_2 import compute_feat_dim, log_class_attri, resil_nn_file_list, norm_nn_file_list, get_utters_from_binary_dict, get_one_utter_by_name, count_male_female_class_errors
from modules_torch import torch_initialisation

from io_funcs.binary_io import BinaryIOCollection
io_fun = BinaryIOCollection()

from exp_mw545.exp_dv_cmp_pytorch import list_random_loader, dv_y_configuration, make_dv_y_exp_dir_name, make_dv_file_list, train_dv_y_model, class_test_dv_y_model, distance_test_dv_y_model, plot_sinenet, vuv_test_sinenet

def load_cmp_file(cmp_file_name, cmp_dim, feat_dim_index=None):
    # Load cmp file, and extract one dimension if provided
    from io_funcs.binary_io import BinaryIOCollection
    BIC = BinaryIOCollection()
    cmp_data = BIC.load_binary_file(cmp_file_name, cmp_dim) # T*D
    if feat_dim_index is None:
        return cmp_data
    else:
        return cmp_data[:,feat_dim_index]

def read_pitch_file(pitch_file_name):
    # Return a list of time where vuv=1
    pitch_t_list = []
    with open(pitch_file_name, 'r') as f:
        file_lines = f.readlines()
    for l in file_lines:
        x = l.strip().split(' ')
        # Content lines should have 3 values
        # Time stamp, vuv, F0 value
        if len(x) == 3:
            if int(x[1]) == 1:
                t = float(x[0])
                pitch_t_list.append(t)
    return numpy.array(pitch_t_list)

def cal_seq_win_tau_vuv_old(pitch_loc_data, utter_start_frame_index, dv_y_cfg, wav_sr):
    tau = numpy.zeros((dv_y_cfg.spk_num_seq, dv_y_cfg.seq_num_win))
    vuv = numpy.zeros((dv_y_cfg.spk_num_seq, dv_y_cfg.seq_num_win))
    # Slice data into seq and win
    for seq_idx in range(dv_y_cfg.spk_num_seq):
        seq_start = utter_start_frame_index + seq_idx * dv_y_cfg.batch_seq_shift
        for win_idx in range(dv_y_cfg.seq_num_win):
            win_start = seq_start + win_idx * dv_y_cfg.seq_win_shift
            win_end   = win_start + dv_y_cfg.seq_win_len - 1 # Inclusive index
            t_start = (win_start) / float(wav_sr)
            t_end   = (win_start + dv_y_cfg.seq_win_len) / float(wav_sr)

            # No pitch, return 0
            win_pitch_loc = 0.
            vuv_temp = 0
            for t in pitch_loc_data:
                if t > (t_start):
                    if t < (t_end):
                        t_r = t - t_start
                        win_pitch_loc = t_r
                        vuv_temp = 1
                        break
                elif t > t_end:
                    # No pitch found in interval
                    win_pitch_loc = 0.
                    vuv_temp = 0
                    break

            tau[seq_idx, win_idx] = win_pitch_loc
            vuv[seq_idx, win_idx] = vuv_temp
    return tau, vuv

def cal_seq_win_lf0_mid_old(lf0_norm_data, utter_start_frame_index, dv_y_cfg, wav_cmp_ratio):
    wav_sr  = dv_y_cfg.cfg.wav_sr
    cmp_sr  = dv_y_cfg.cfg.frame_sr
    
    nlf = numpy.zeros((dv_y_cfg.spk_num_seq, dv_y_cfg.seq_num_win))
    # Slice data into seq and win
    for seq_idx in range(dv_y_cfg.spk_num_seq):
        spk_seq_index = seq_idx
        seq_start = utter_start_frame_index + seq_idx * dv_y_cfg.batch_seq_shift
        for win_idx in range(dv_y_cfg.seq_num_win):
            win_start = seq_start + win_idx * dv_y_cfg.seq_win_shift
            t_start = (win_start) / float(wav_sr)
            t_end   = (win_start+dv_y_cfg.seq_win_len) / float(wav_sr)

            t_mid = (t_start + t_end) / 2.
            n_mid = t_mid * float(cmp_sr)
            # e.g. 1.3 is between 0.5, 1.5; n_l=0, n_r=1
            n_l = int(n_mid-0.5)
            n_r = n_l + 1
            l = lf0_norm_data.shape[0]
            if n_r >= l:
                lf0_mid = lf0_norm_data[-1]
            else:
                lf0_l = lf0_norm_data[n_l]
                lf0_r = lf0_norm_data[n_r]
                r = n_mid - ( n_l + 0.5 )
                lf0_mid = (r * lf0_r) + ((1-r) * lf0_l)

            nlf[spk_seq_index, win_idx] = lf0_mid

    return nlf

def make_n_mid_0_matrix(dv_y_cfg):
    ''' 
    Return mid index value of each sub-window, before adding time/index shift
    Type is float, for interpolation
    '''
    wav_sr  = dv_y_cfg.cfg.wav_sr
    cmp_sr  = dv_y_cfg.cfg.frame_sr
    wav_cmp_ratio = int(wav_sr / cmp_sr)
    n_mid_0 = numpy.zeros([dv_y_cfg.spk_num_seq, dv_y_cfg.seq_num_win])
    for seq_idx in range(dv_y_cfg.spk_num_seq):
        for win_idx in range(dv_y_cfg.seq_num_win):
            seq_start = seq_idx * dv_y_cfg.batch_seq_shift
            win_start = seq_start + win_idx * dv_y_cfg.seq_win_shift
            t_start = (win_start) / float(wav_sr)
            t_end   = (win_start+dv_y_cfg.seq_win_len) / float(wav_sr)

            t_mid = (t_start + t_end) / 2.
            n_mid = t_mid * float(cmp_sr)

            n_mid_0[seq_idx, win_idx] = n_mid
    return n_mid_0

def make_win_start_0_matrix(dv_y_cfg):
    '''  
    Return start index of each sub-window, before adding index shift
    '''
    wav_sr  = dv_y_cfg.cfg.wav_sr
    cmp_sr  = dv_y_cfg.cfg.frame_sr
    wav_cmp_ratio = int(wav_sr / cmp_sr)
    win_start_matrix = numpy.zeros([dv_y_cfg.spk_num_seq, dv_y_cfg.seq_num_win, 1])
    for seq_idx in range(dv_y_cfg.spk_num_seq):
        seq_start = seq_idx * dv_y_cfg.batch_seq_shift
        for win_idx in range(dv_y_cfg.seq_num_win):
            win_start_matrix[seq_idx, win_idx, 0] = seq_start + win_idx * dv_y_cfg.seq_win_shift
    return win_start_matrix

def cal_seq_win_lf0_mid(lf0_norm_data, utter_start_frame_index, n_mid_0, wav_cmp_ratio):
    ''' Derive position using utter_start_frame_index and n_mid_0 matrix, then extract from lf0_norm_data '''
    l = lf0_norm_data.shape[0]
    n_mid = n_mid_0 + (utter_start_frame_index / float(wav_cmp_ratio))
    n_l = (n_mid-0.5).astype(int)
    n_r = n_l + 1
    r = n_mid - (n_l + 0.5)
    n_l[n_r>= l] = -1
    n_r[n_r>= l] = -1
    lf0_l = lf0_norm_data[n_l]
    lf0_r = lf0_norm_data[n_r]
    lf0_mid = r * lf0_r + (1-r) * lf0_l
    return lf0_mid

def cal_seq_win_tau_vuv(pitch_loc_data, utter_start_frame_index, dv_y_cfg, win_start_0, wav_sr):
    ''' Calculate pitch location per window; if not found then vuv=0 '''
    spk_num_seq = dv_y_cfg.spk_num_seq
    seq_num_win = dv_y_cfg.seq_num_win
    seq_win_len = dv_y_cfg.seq_win_len

    l = pitch_loc_data.shape[0]
    tau = numpy.zeros((spk_num_seq, seq_num_win))
    vuv = numpy.zeros((spk_num_seq, seq_num_win))
    
    pitch_max = seq_win_len / float(wav_sr)

    win_start = win_start_0 + utter_start_frame_index
    t_start = win_start / float(wav_sr)

    t_start = numpy.repeat(t_start, l, axis=2)

    pitch_start = pitch_loc_data - t_start
    pitch_start[pitch_start <= 0.] = pitch_max
    pitch_start_min = numpy.amin(pitch_start, axis=2)

    vuv[pitch_start_min < pitch_max] = 1
    pitch_start_min[pitch_start_min >= pitch_max] = 0.
    # pitch_start_min = pitch_start_min * vuv
    tau = pitch_start_min

    return tau, vuv

def make_feed_dict_y_wav_sinenet_train(dv_y_cfg, file_list_dict, file_dir_dict, batch_speaker_list, utter_tvt, all_utt_start_frame_index=None, return_one_hot=False, return_y=False, return_frame_index=False, return_file_name=False, return_vuv=False):
    logger = make_logger("make_dict")

    '''
    Draw Utterances; Load Data
    Draw starting frame; Slice; Fit into numpy holders
    '''
    feat_name_list = ['wav'] # Load wav
    feat_dim_list  = [1]
    # Make i/o shape arrays
    # This is numpy shape, not Tensor shape!
    one_hot = numpy.zeros((dv_y_cfg.batch_num_spk))
    wav = numpy.zeros((dv_y_cfg.batch_num_spk, dv_y_cfg.batch_seq_total_len))
    nlf = numpy.zeros((dv_y_cfg.batch_num_spk, dv_y_cfg.spk_num_seq, dv_y_cfg.seq_num_win))
    tau = numpy.zeros((dv_y_cfg.batch_num_spk, dv_y_cfg.spk_num_seq, dv_y_cfg.seq_num_win))
    vuv = numpy.zeros((dv_y_cfg.batch_num_spk, dv_y_cfg.spk_num_seq, dv_y_cfg.seq_num_win))

    wav_sr  = dv_y_cfg.cfg.wav_sr
    cmp_sr  = dv_y_cfg.cfg.frame_sr
    wav_cmp_ratio = int(wav_sr / cmp_sr)
    # Do not use silence frames at the beginning or the end
    total_sil_one_side_cmp = dv_y_cfg.frames_silence_to_keep + dv_y_cfg.sil_pad  # This is at 200Hz
    total_sil_one_side_wav = total_sil_one_side_cmp * wav_cmp_ratio              # This is at 16kHz
    min_file_len = dv_y_cfg.batch_seq_total_len + 2 * total_sil_one_side_wav # This is at 16kHz

    file_name_list = [[] for i in range(dv_y_cfg.batch_num_spk)]
    start_frame_index_list = [[] for i in range(dv_y_cfg.batch_num_spk)]
    
    load_time = 0.
    lf0_time  = 0.
    tau_time  = 0.

    # one_hot part
    for speaker_idx in range(dv_y_cfg.batch_num_spk):
        speaker_id = batch_speaker_list[speaker_idx]
        # Make classification targets, index sequence
        true_speaker_index = dv_y_cfg.speaker_id_list_dict['train'].index(speaker_id)
        one_hot[speaker_idx] = true_speaker_index

    # waveform part
    for speaker_idx in range(dv_y_cfg.batch_num_spk):
        speaker_id = batch_speaker_list[speaker_idx]
        # Draw 1 utterance per speaker
        # Draw multiple windows per utterance:  dv_y_cfg.spk_num_seq
        # Stack them along B
        speaker_file_name_list, speaker_utter_len_list, speaker_utter_list = get_utters_from_binary_dict(1, file_list_dict[(speaker_id, utter_tvt)], file_dir_dict, feat_name_list=feat_name_list, feat_dim_list=feat_dim_list, min_file_len=min_file_len, random_seed=None)
        file_name_list[speaker_idx].extend(speaker_file_name_list)

        file_name = speaker_file_name_list[0]
        wav_file  = speaker_utter_list['wav'][0] # T * 1; 16kHz
        wav_file  = numpy.squeeze(wav_file, axis=1)      # T*1 -> T
        wav_file_len = speaker_utter_len_list[0]

        # Find start frame index, random if None
        if all_utt_start_frame_index is None:
            extra_file_len = wav_file_len - min_file_len
            utter_start_frame_index = numpy.random.randint(low=total_sil_one_side_wav, high=total_sil_one_side_wav+extra_file_len+1)
        else:
            utter_start_frame_index = total_sil_one_side_wav + all_utt_start_frame_index
        start_frame_index_list[speaker_idx].append(utter_start_frame_index)
        wav[speaker_idx, :] = wav_file[utter_start_frame_index:utter_start_frame_index+dv_y_cfg.batch_seq_total_len]

    # lf0 part
    for speaker_idx in range(dv_y_cfg.batch_num_spk):
        speaker_id = batch_speaker_list[speaker_idx]
        file_name = file_name_list[speaker_idx][0]
        utter_start_frame_index = start_frame_index_list[speaker_idx][0]
        # Load cmp and pitch data
        cmp_file_name = os.path.join(file_dir_dict['cmp'], file_name+'.cmp')
        lf0_index     = dv_y_cfg.cfg.acoustic_start_index['lf0']
        cmp_dim       = dv_y_cfg.cfg.nn_feature_dims['cmp']
        lf0_norm_data = load_cmp_file(cmp_file_name, cmp_dim=cmp_dim, feat_dim_index=lf0_index)

        # Get lf0_mid data in forms of numpy array operations, faster than for loops
        n_mid_0 = dv_y_cfg.return_n_mid_0_matrix()
        # nlf[speaker_idx] = cal_seq_win_lf0_mid(lf0_norm_data, utter_start_frame_index, n_mid_0, wav_cmp_ratio)
        nlf[speaker_idx] = cal_seq_win_lf0_mid_old(lf0_norm_data, utter_start_frame_index, dv_y_cfg, wav_cmp_ratio)

    # tau and vuv part
    for speaker_idx in range(dv_y_cfg.batch_num_spk):
        speaker_id = batch_speaker_list[speaker_idx]
        file_name = file_name_list[speaker_idx][0]
        utter_start_frame_index = start_frame_index_list[speaker_idx][0]
        # Load cmp and pitch data
        pitch_file_name = os.path.join(file_dir_dict['pitch'], file_name+'.pm')
        pitch_loc_data = read_pitch_file(pitch_file_name)

        # Get lf0_mid data in forms of numpy array operations, faster than for loops
        win_start_0 = dv_y_cfg.return_win_start_0_matrix()
        # tau_spk, vuv_spk = cal_seq_win_tau_vuv(pitch_loc_data, utter_start_frame_index, dv_y_cfg, win_start_0, wav_sr)
        tau_spk, vuv_spk = cal_seq_win_tau_vuv_old(pitch_loc_data, utter_start_frame_index, dv_y_cfg, wav_sr)
                           
        tau[speaker_idx] = tau_spk
        vuv[speaker_idx] = vuv_spk

    # S,B,M,D
    x_val = wav
    if dv_y_cfg.train_by_window:
        # S --> S*B
        y_val = numpy.repeat(one_hot, dv_y_cfg.spk_num_seq)
        batch_size = dv_y_cfg.batch_num_spk * dv_y_cfg.spk_num_seq
    else:
        y_val = one_hot
        batch_size = dv_y_cfg.batch_num_spk

    feed_dict = {'x':x_val, 'y':y_val}
    feed_dict['nlf'] = nlf
    feed_dict['tau'] = tau
    if dv_y_cfg.use_voiced_only:
        # Current method: Some b in B are voiced, use vuv as error mask
        assert dv_y_cfg.train_by_window
        # Make binary S * B matrix
        vuv_S_B = (vuv>0).all(axis=2)
        # Reshape to SB for pytorch cross-entropy function
        vuv_SB = numpy.reshape(vuv_S_B, (dv_y_cfg.batch_num_spk * dv_y_cfg.spk_num_seq))
        feed_dict['vuv'] = vuv_SB
    return_list = [feed_dict, batch_size]
    
    if return_one_hot:
        return_list.append(one_hot)
    if return_y:
        return_list.append(wav)
    if return_frame_index:
        return_list.append(start_frame_index_list)
    if return_file_name:
        return_list.append(file_name_list)
    if return_vuv:
        return_list.append(vuv)
    return return_list

class dv_y_wav_sinenet_configuration(dv_y_configuration):
    
    def __init__(self, cfg):
        super().__init__(cfg)
        self.use_voiced_only = False # Use voiced regions only
        self.use_voiced_threshold = 1. # Percentage of voiced required
        self.finetune_model = False
        # self.learning_rate  = 0.0001
        # self.prev_nnets_file_name = '/home/dawna/tts/mw545/TorchDV/dv_wav_sinenet_v3/dv_y_wav_lr_0.000100_Sin80f10_ReL256BN_ReL256BN_ReL8DR_DV8S100B10T3200D1/Model'
        self.python_script_name = os.path.realpath(__file__)

        # Waveform-level input configuration
        self.y_feat_name   = 'wav'
        self.out_feat_list = ['wav']
        self.batch_seq_total_len = 12000 # Number of frames at 16kHz; 32000 for 2s
        self.batch_seq_len   = 3200 # T
        self.batch_seq_shift = 10*80
        self.seq_win_len   = 640
        self.seq_win_shift = 80
        self.seq_num_win   = int((self.batch_seq_len - self.seq_win_len) / self.seq_win_shift) + 1

        self.batch_num_spk = 100
        self.dv_dim = 8
        self.nn_layer_config_list = [
            # Must contain: type, size; num_channels, dropout_p are optional, default 0, 1
            {'type':'SinenetV3_ST', 'size':81, 'sine_size':80, 'num_freq':16, 'win_len_shift_list':[[self.batch_seq_len, self.batch_seq_shift], [self.seq_win_len, self.seq_win_shift]], 'total_length':self.batch_seq_total_len, 'dropout_p':0, 'batch_norm':False},
            {'type':'LReLUDV', 'size':256, 'dropout_p':0, 'batch_norm':True},
            {'type':'LReLUDV', 'size':256, 'dropout_p':0, 'batch_norm':True},
            {'type':'LReLUDV', 'size':self.dv_dim, 'dropout_p':0.2, 'batch_norm':False}
        ]

        # self.gpu_id = 'cpu'
        self.gpu_id = 2

        from modules_torch import DV_Y_ST_model
        self.dv_y_model_class = DV_Y_ST_model

        self.make_feed_dict_method_train = make_feed_dict_y_wav_sinenet_train
        # if self.use_voiced_only:
        #     self.make_feed_dict_method_train = make_feed_dict_y_wav_sinenet_train_voiced_only
        #     self.make_feed_dict_method_test  = make_feed_dict_y_wav_sinenet_test_voiced_only
        # else:
        #     self.make_feed_dict_method_train = make_feed_dict_y_wav_sinenet_train
        #     self.make_feed_dict_method_test  = make_feed_dict_y_wav_sinenet_test
        #     self.make_feed_dict_method_distance  = make_feed_dict_y_wav_sinenet_distance
        # self.make_feed_dict_method_vuv_test = make_feed_dict_y_wav_sinenet_train
        self.auto_complete(cfg)

    def reload_model_param(self):
        self.nn_layer_config_list[0]['win_len_shift_list'] = [[self.batch_seq_len, self.batch_seq_shift], [self.seq_win_len, self.seq_win_shift]]
        self.nn_layer_config_list[0]['total_length'] = self.batch_seq_total_len

    def return_n_mid_0_matrix(self):
        try:
            return self.n_mid_0
        except AttributeError:
            self.n_mid_0 = make_n_mid_0_matrix(self)
            return self.n_mid_0

    def return_win_start_0_matrix(self):
        try:
            return self.win_start_matrix
        except AttributeError:
            self.win_start_matrix = make_win_start_0_matrix(self)
            return self.win_start_matrix

def train_dv_y_wav_model(cfg, dv_y_cfg=None):
    if dv_y_cfg is None: dv_y_cfg = dv_y_wav_sinenet_configuration(cfg)
    # tau_test(dv_y_cfg)
    train_dv_y_model(cfg, dv_y_cfg)
    
def test_dv_y_wav_model(cfg, dv_y_cfg=None):
    if dv_y_cfg is None: dv_y_cfg = dv_y_wav_sinenet_configuration(cfg)
    class_test_dv_y_model(cfg, dv_y_cfg)
    # distance_test_dv_y_model(cfg, dv_y_cfg)
    # plot_sinenet(cfg, dv_y_cfg)
    # vuv_test_sinenet(cfg, dv_y_cfg)
