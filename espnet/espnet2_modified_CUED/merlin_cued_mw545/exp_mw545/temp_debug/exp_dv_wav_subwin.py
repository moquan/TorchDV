# exp_dv_wav_subwin.py

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

def make_feed_dict_y_wav_subwin_train_old(dv_y_cfg, file_list_dict, file_dir_dict, batch_speaker_list, utter_tvt, all_utt_start_frame_index=None, return_one_hot=False, return_y=False, return_frame_index=False, return_file_name=False):
    logger = make_logger("make_dict")

    '''
    Draw Utterances; Load Data
    Draw starting frame; Slice; Fit into numpy holders
    '''
    feat_name_list = ['wav'] # Load wav
    feat_dim_list  = [1]
    # Make i/o shape arrays
    # This is numpy shape, not Tensor shape!
    wav = numpy.zeros((dv_y_cfg.batch_num_spk, dv_y_cfg.batch_seq_total_len))
    one_hot = numpy.zeros((dv_y_cfg.batch_num_spk))

    wav_sr  = dv_y_cfg.cfg.wav_sr
    cmp_sr  = dv_y_cfg.cfg.frame_sr
    wav_cmp_ratio = int(wav_sr / cmp_sr)
    # Do not use silence frames at the beginning or the end
    total_sil_one_side_cmp = dv_y_cfg.frames_silence_to_keep + dv_y_cfg.sil_pad  # This is at 200Hz
    total_sil_one_side_wav = total_sil_one_side_cmp * wav_cmp_ratio              # This is at 16kHz
    min_file_len = dv_y_cfg.batch_seq_total_len + 2 * total_sil_one_side_wav # This is at 16kHz

    file_name_list = []
    start_frame_index_list = [[] for i in range(dv_y_cfg.batch_num_spk)]
    
    for speaker_idx in range(dv_y_cfg.batch_num_spk):

        speaker_id = batch_speaker_list[speaker_idx]
        # Make classification targets, index sequence
        true_speaker_index = dv_y_cfg.speaker_id_list_dict['train'].index(speaker_id)
        one_hot[speaker_idx] = true_speaker_index

        # Draw 1 utterance per speaker
        # Draw multiple windows per utterance; multiple sub-windows per window; in pytorch, not here

        speaker_file_name_list, speaker_utter_len_list, speaker_utter_list = get_utters_from_binary_dict(1, file_list_dict[(speaker_id, utter_tvt)], file_dir_dict, feat_name_list=feat_name_list, feat_dim_list=feat_dim_list, min_file_len=min_file_len, random_seed=None)
        file_name_list.append(speaker_file_name_list)

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

    x_val = wav
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

def make_feed_dict_y_wav_subwin_test_old(dv_y_cfg, file_dir_dict, speaker_id, file_name, start_frame_index, BTD_feat_remain):
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
    wav = numpy.zeros((dv_y_cfg.batch_num_spk, dv_y_cfg.batch_seq_total_len))
    one_hot = numpy.zeros((dv_y_cfg.batch_num_spk))

    # Make classification targets, index sequence
    try: true_speaker_index = dv_y_cfg.speaker_id_list_dict['train'].index(speaker_id)
    except ValueError: true_speaker_index = 0 # At generation time, since one_hot is not used, a non-train speaker is given an arbituary speaker index
    one_hot[0] = true_speaker_index

    if BTD_feat_remain is not None:
        wav_feat_current = BTD_feat_remain
        T_total = wav_feat_current.shape[0]
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
        wav_feat_current = wav_features_no_sil
        T_total = len_no_sil_wav
        
    B_total = int((T_total - dv_y_cfg.batch_seq_len) / dv_y_cfg.batch_seq_shift) + 1

    if B_total > dv_y_cfg.spk_num_seq:
        B_actual = dv_y_cfg.spk_num_seq
        T_actual = dv_y_cfg.batch_seq_total_len
        B_remain = B_total - B_actual
        gen_finish = False
        wav_feat_remain = wav_feat_current[B_actual*dv_y_cfg.batch_seq_shift:]
        BTD_feat_remain = wav_feat_remain
    else:
        B_actual = B_total
        T_actual = T_total
        B_remain = 0
        gen_finish = True
        BTD_feat_remain = None

    wav[0,:T_actual] = wav_feat_current[:T_actual]
    batch_size = B_actual

    x_val = wav
    if dv_y_cfg.train_by_window:
        # S --> S*B
        y_val = numpy.repeat(one_hot, dv_y_cfg.spk_num_seq)
    else:
        y_val = one_hot

    feed_dict = {'x':x_val, 'y':y_val}
    return_list = [feed_dict, gen_finish, batch_size, BTD_feat_remain]
    return return_list

def make_feed_dict_y_wav_subwin_distance(dv_y_cfg, file_list_dict, file_dir_dict, batch_speaker_list, utter_tvt, all_utt_start_frame_index=None, return_y=False, return_frame_index=False, return_file_name=False):
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
        wav = numpy.zeros((dv_y_cfg.batch_num_spk, dv_y_cfg.batch_seq_total_len))
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
    start_frame_index_list = [[] for i in range(dv_y_cfg.batch_num_spk)]
    
    for speaker_idx in range(dv_y_cfg.batch_num_spk):

        speaker_id = batch_speaker_list[speaker_idx]

        # Draw 1 utterance per speaker
        # Draw multiple windows per utterance; multiple sub-windows per window; in pytorch, not here

        speaker_file_name_list, speaker_utter_len_list, speaker_utter_list = get_utters_from_binary_dict(1, file_list_dict[(speaker_id, utter_tvt)], file_dir_dict, feat_name_list=feat_name_list, feat_dim_list=feat_dim_list, min_file_len=min_file_len, random_seed=None)
        file_name_list.append(speaker_file_name_list)

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

        for plot_idx in range(dv_y_cfg.num_to_plot+1):
            plot_start_frame_index = utter_start_frame_index + plot_idx * dv_y_cfg.gap_len_to_plot
            wav_list[plot_idx][speaker_idx] = wav_file[plot_start_frame_index:plot_start_frame_index+dv_y_cfg.batch_seq_total_len]

    feed_dict_list = [{} for i in range(dv_y_cfg.num_to_plot+1)]
    for plot_idx in range(dv_y_cfg.num_to_plot+1):
        x_val = wav_list[plot_idx]
        feed_dict_list[plot_idx] = {'x':x_val}
    batch_size = dv_y_cfg.batch_num_spk * dv_y_cfg.spk_num_seq

    return_list = [feed_dict_list, batch_size]
    
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
        self.batch_seq_total_len = 8000 # Number of frames at 16kHz; 32000 for 2s
        self.batch_seq_len   = 3200 # T
        self.batch_seq_shift = 80
        self.seq_win_len   = 640
        self.seq_win_shift = 80
        self.seq_num_win   = int((self.batch_seq_len - self.seq_win_len) / self.seq_win_shift) + 1

        self.batch_num_spk = 100
        self.dv_dim = 8
        self.nn_layer_config_list = [
            # Must contain: type, size; num_channels, dropout_p are optional, default 0, 1
            # {'type':'LReLUSubWin', 'size':80, 'win_len':self.seq_win_len, 'num_win':self.seq_num_win, 'dropout_p':0, 'batch_norm':False},
            {'type':'ReLUSubWin_ST', 'size':80, 'win_len_shift_list':[[self.batch_seq_len, self.batch_seq_shift], [self.seq_win_len, self.seq_win_shift]], 'total_length':self.batch_seq_total_len, 'dropout_p':0, 'batch_norm':False},
            {'type':'LReLUDV', 'size':256, 'dropout_p':0, 'batch_norm':True},
            {'type':'LReLUDV', 'size':256, 'dropout_p':0, 'batch_norm':True},
            {'type':'LReLUDV', 'size':self.dv_dim, 'dropout_p':0.2, 'batch_norm':False}
        ]

        # self.gpu_id = 'cpu'
        self.gpu_id = 2

        from modules_torch import DV_Y_ST_model
        self.dv_y_model_class = DV_Y_ST_model

        if self.use_voiced_only:
            self.out_feat_list = ['wav', 'vuv']
        self.make_feed_dict_method_train = make_feed_dict_y_wav_subwin_train
        self.make_feed_dict_method_test  = make_feed_dict_y_wav_subwin_test
        self.make_feed_dict_method_distance = make_feed_dict_y_wav_subwin_distance
        self.make_feed_dict_method_vuv_test = make_feed_dict_y_wav_subwin_train
        self.auto_complete(cfg)

        # self.dv_y_data_loader = dv_y_wav_subwin_data_loader

    def reload_model_param(self):
        self.nn_layer_config_list[0]['win_len_shift_list'] = [[self.batch_seq_len, self.batch_seq_shift], [self.seq_win_len, self.seq_win_shift]]
        self.nn_layer_config_list[0]['total_length'] = self.batch_seq_total_len

def train_dv_y_wav_model(cfg, dv_y_cfg=None):
    if dv_y_cfg is None: dv_y_cfg = dv_y_wav_subwin_configuration(cfg)
    train_dv_y_model(cfg, dv_y_cfg)

def test_dv_y_wav_model(cfg, dv_y_cfg=None):
    if dv_y_cfg is None: dv_y_cfg = dv_y_wav_subwin_configuration(cfg)
    class_test_dv_y_model(cfg, dv_y_cfg)
    # distance_test_dv_y_model(cfg, dv_y_cfg)
    # vuv_test_dv_y_model(cfg, dv_y_cfg)
    # ce_vs_var_nlf_test(cfg, dv_y_cfg)

class Build_dv_y_wav_data_loader_Single_File_v1(object):
    """
    This data loader can load only for one speaker
    Pitch: 0 <= tau < win_T
    The f0 method extract f0 in the middle (round down) of the window
    start_sample_no_sil >= 0; sil_pad included
    """
    def __init__(self, cfg=None, dv_y_cfg=None):
        super().__init__()
        self.logger = make_logger("Data_Loader Single")

        self.cfg = cfg
        self.dv_y_cfg = dv_y_cfg
        self.DIO = Data_File_IO(cfg)

        self.load_seq_win_config(dv_y_cfg)

        self.init_n_matrix()

    def load_seq_win_config(self, dv_y_cfg):
        '''
        Load from dv_y_cfg, or hard-code here
        '''
        self.input_data_dim = {}
        if dv_y_cfg is None:
            self.input_data_dim['T_S'] = 6000 # Number of frames at 16kHz
            self.input_data_dim['T_B']   = 3200 # T
            self.input_data_dim['B_shift'] = 80
            self.input_data_dim['T_M']   = 640
            self.input_data_dim['M_shift'] = 80
            self.out_feat_list = ['wav_ST', 'f_SBM', 'tau_SBM', 'vuv_SBM']
        else:
            self.input_data_dim['T_S'] = dv_y_cfg.input_data_dim['T_S']
            self.input_data_dim['T_B']   = dv_y_cfg.input_data_dim['T_B']
            self.input_data_dim['B_shift'] = dv_y_cfg.input_data_dim['B_shift']
            self.input_data_dim['T_M']   = dv_y_cfg.input_data_dim['T_M']
            self.input_data_dim['M_shift'] = dv_y_cfg.input_data_dim['M_shift']
            self.out_feat_list = dv_y_cfg.out_feat_list
        
        # Parameters
        self.input_data_dim['B'] = int((self.input_data_dim['T_S'] - self.input_data_dim['T_B']) / self.input_data_dim['B_shift']) + 1
        self.input_data_dim['M'] = int((self.input_data_dim['T_B'] - self.input_data_dim['T_M']) / self.input_data_dim['M_shift']) + 1

        self.win_T = float(self.input_data_dim['T_M']) / float(self.cfg.wav_sr)

        wav_sr  = self.cfg.wav_sr
        cmp_sr  = self.cfg.frame_sr
        wav_cmp_ratio = int(wav_sr / cmp_sr)
        # Do not use silence frames at the beginning or the end
        total_sil_one_side_cmp = self.cfg.frames_silence_to_keep + self.cfg.sil_pad   # This is at 200Hz
        self.total_sil_one_side_wav = total_sil_one_side_cmp * wav_cmp_ratio          # This is at 16kHz

    def init_n_matrix(self):
        '''
        Make matrices of index, B*M
        '''
        B_M_grid = numpy.mgrid[0:self.input_data_dim['B'],0:self.input_data_dim['M']]
        self.start_n_matrix = self.input_data_dim['B_shift'] * B_M_grid[0] + self.input_data_dim['M_shift'] * B_M_grid[1]
        self.start_n_matrix = self.start_n_matrix + self.total_sil_one_side_wav
        self.mid_n_matrix   = self.start_n_matrix + self.input_data_dim['T_M'] / 2

    def make_feed_dict(self, file_id, start_sample_no_sil):
        '''
        1. Load x 
        2. Load pitch and vuv
        3. Load f0 
        '''
        dv_y_cfg = self.dv_y_cfg
        feed_dict = {}

        wav_resil_norm_file_name = os.path.join(self.cfg.nn_feat_resil_norm_dirs['wav'], file_id+'.wav')
        feed_dict['wav_T'] = self.make_wav_T(wav_resil_norm_file_name, start_sample_no_sil)

        if ('tau_SBM' in dv_y_cfg.out_feat_list) or ('vuv_SBM' in dv_y_cfg.out_feat_list):
            pitch_resil_norm_file_name = os.path.join(self.cfg.nn_feat_resil_dirs['pitch'], file_id+'.pitch')
            tau_BM, vuv_BM = self.make_tau_BM(pitch_resil_norm_file_name, start_sample_no_sil)
            feed_dict['tau_BM'] = tau_BM
            feed_dict['vuv_BM'] = vuv_BM

        if 'f_SBM' in dv_y_cfg.out_feat_list:
            f0_16k_file_name = os.path.join(self.cfg.nn_feat_resil_dirs['f016k'], file_id+'.f016k')
            f_BM = self.make_f_BM(f0_16k_file_name, start_sample_no_sil)
            feed_dict['f_BM'] = f_BM

        return feed_dict

    def make_wav_T(self, wav_resil_norm_file_name, start_sample_no_sil):
        '''
        Waveform data is S*T (1*T); window slicing is done in Pytorch
        '''
        wav_data, sample_number = self.DIO.load_data_file_frame(wav_resil_norm_file_name, 1)
        wav_data = numpy.squeeze(wav_data)

        start_sample_include_sil = start_sample_no_sil+self.total_sil_one_side_wav
        wav_T = wav_data[start_sample_include_sil:start_sample_include_sil+self.input_data_dim['T_S']]

        return wav_T

    def make_tau_BM(self, pitch_resil_norm_file_name, start_sample_no_sil):
        '''
        B * M
        vuv=1. if pitch found inside the window; vuv=0. otherwise
        pitch found: 0 <= tau < win_T
        '''
        vuv_BM = numpy.ones((self.input_data_dim['B'], self.input_data_dim['M']))

        pitch_16k_data, sample_number = self.DIO.load_data_file_frame(pitch_resil_norm_file_name, 1)
        pitch_16k_data = numpy.squeeze(pitch_16k_data)

        start_n_matrix = self.start_n_matrix + start_sample_no_sil
        tau_BM = pitch_16k_data[start_n_matrix.astype(int)]
        vuv_BM[tau_BM<0] = 0.
        vuv_BM[tau_BM>=self.win_T] = 0.

        tau_BM[vuv_BM==0] = 0.

        return tau_BM, vuv_BM

    def make_f_BM(self, f0_16k_file_name, start_sample_no_sil):
        '''
        B * M
        '''
        f0_16k_data, sample_number = self.DIO.load_data_file_frame(f0_16k_file_name, 1)
        f0_16k_data = numpy.squeeze(f0_16k_data)

        mid_n_matrix = self.mid_n_matrix + start_sample_no_sil
        f_BM = f0_16k_data[mid_n_matrix.astype(int)]
        return f_BM
