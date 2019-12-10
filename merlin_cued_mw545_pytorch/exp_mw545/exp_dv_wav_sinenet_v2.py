# exp_dv_wav_sinenet_v2.py

# d-vector style model
# https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/41939.pdf

# For each window, network input is a vector of stacked waveforms

# Use f0 and tau information from REAPER outputs

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
from exp_mw545.exp_dv_wav_baseline import make_feed_dict_y_wav_cmp_train, make_feed_dict_y_wav_cmp_test
from frontend.silence_reducer_keep_sil import SilenceReducer

def read_pitch_file(pitch_file_name):
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
    return pitch_t_list

def cal_win_pitch_loc(pitch_loc_data, t_start, t_end, t_removed=0):
    # print(t_start)
    # print(t_end)
    # print(t_removed)
    # print(pitch_loc_data)

    for t in pitch_loc_data:
        if t > (t_start+t_removed):
            if t < (t_end+t_removed):
                return (t - t_start - t_removed)
        elif t > t_end+t_removed:
            # No pitch found in interval
            return 0
    # No pitch, return 0
    return 0

def load_cmp_file(cmp_file_name, cmp_dim, feat_dim_index):
    from io_funcs.binary_io import BinaryIOCollection
    BIC = BinaryIOCollection()
    cmp_data = BIC.load_binary_file(cmp_file_name, cmp_dim)
    return cmp_data[:,feat_dim_index]

def cal_win_lf0_mid(lf0_norm_data, cmp_sr, t_start, t_end):
    # 1. Find central time t_mid
    # 2. Find 2 frames left and right of t_mid
    # 3. Find interpolated lf0 value at t_mid
    t_mid = (t_start + t_end) / 2
    n_mid = t_mid * cmp_sr
    # e.g. 1.3 is between 0.5, 1.5; n_l=0, n_r=1
    n_l = int(n_mid-0.5)
    n_r = n_l + 1
    l = lf0_norm_data.shape[0]
    if n_r >= l:
        return lf0_norm_data[-1]
    else:
        lf0_l = lf0_norm_data[n_l]
        lf0_r = lf0_norm_data[n_r]
        r = n_mid - n_l
        lf0_mid = r * lf0_r + (1-r) * lf0_l
        return lf0_mid

def make_feed_dict_y_wav_f0_tau_train(dv_y_cfg, file_list_dict, file_dir_dict, batch_speaker_list, utter_tvt, all_utt_start_frame_index=None, return_dv=False, return_y=False, return_frame_index=False, return_file_name=False):
    feat_name = dv_y_cfg.y_feat_name # Hard-coded here for now
    # Make i/o shape arrays
    # This is numpy shape, not Tensor shape!
    y  = numpy.zeros((dv_y_cfg.batch_num_spk, dv_y_cfg.spk_num_seq, dv_y_cfg.batch_seq_len, dv_y_cfg.feat_dim))
    dv = numpy.zeros((dv_y_cfg.batch_num_spk))
    nlf = numpy.zeros((dv_y_cfg.batch_num_spk, dv_y_cfg.spk_num_seq, 1, 1))
    tau = numpy.zeros((dv_y_cfg.batch_num_spk, dv_y_cfg.spk_num_seq, 1, 1))

    wav_sr  = dv_y_cfg.cfg.wav_sr
    cmp_sr  = dv_y_cfg.cfg.frame_sr
    wav_cmp_ratio = int(wav_sr / cmp_sr)
    # Do not use silence frames at the beginning or the end
    total_sil_one_side_cmp = dv_y_cfg.frames_silence_to_keep + dv_y_cfg.sil_pad # This is at 200Hz
    total_sil_one_side_wav = total_sil_one_side_cmp * wav_cmp_ratio             # This is at 16kHz
    min_file_len = dv_y_cfg.batch_seq_total_len + 2 * total_sil_one_side_wav    # This is at 16kHz
    sil_index_dict = dv_y_cfg.sil_index_dict

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
            file_name = speaker_file_name_list[utter_idx]
            no_sil_start_cmp = sil_index_dict[file_name][0]
            no_sil_end_cmp   = sil_index_dict[file_name][1]
            len_no_sil_cmp   = no_sil_end_cmp - no_sil_start_cmp + 1 # Inclusive
            len_no_sil_wav   = len_no_sil_cmp * wav_cmp_ratio
            sil_pad_first_idx_cmp = max(0, no_sil_start_cmp - total_sil_one_side_cmp)
            t_removed = float(sil_pad_first_idx_cmp) / cmp_sr
            remain_sil_before_cmp = no_sil_start_cmp - sil_pad_first_idx_cmp
            remain_sil_before_wav = remain_sil_before_cmp * wav_cmp_ratio
            if all_utt_start_frame_index is None:
                # Use random starting frame index
                extra_file_len = len_no_sil_wav - dv_y_cfg.batch_seq_total_len
                start_frame_index = numpy.random.randint(low=remain_sil_before_wav, high=remain_sil_before_wav+extra_file_len+1)
            else:
                start_frame_index = remain_sil_before_wav + all_utt_start_frame_index
            speaker_start_frame_index_list.append(start_frame_index)

            y_stack = speaker_utter_list[feat_name][utter_idx]
            # Load cmp and pitch data
            cmp_file_name = os.path.join(file_dir_dict['cmp'], file_name+'.cmp')
            lf0_index     = dv_y_cfg.cfg.acoustic_start_index['lf0']
            cmp_dim       = dv_y_cfg.cfg.nn_feature_dims['cmp']
            lf0_norm_data = load_cmp_file(cmp_file_name, cmp_dim, lf0_index)  # Extract lf0 from cmp
            pitch_file_name = os.path.join(file_dir_dict['pitch'], file_name+'.used.pm')
            pitch_loc_data = read_pitch_file(pitch_file_name)

            for utter_seq_idx in range(dv_y_cfg.utter_num_seq):
                n_start = start_frame_index + utter_seq_idx * dv_y_cfg.batch_seq_shift
                n_end   = n_start + dv_y_cfg.batch_seq_len - 1 # Inclusive index
                t_start = (n_start) / wav_sr
                t_end   = (n_end+1) / wav_sr

                lf0_mid = cal_win_lf0_mid(lf0_norm_data, cmp_sr, t_start, t_end) # lf0_norm_data should have same length as y_features
                win_pitch_loc = cal_win_pitch_loc(pitch_loc_data, t_start, t_end, t_removed)

                spk_seq_index = utter_idx * dv_y_cfg.utter_num_seq + utter_seq_idx
                y[speaker_idx, spk_seq_index, :, :]   = y_stack[n_start:n_end+1, :]
                nlf[speaker_idx, spk_seq_index, 0, 0] = lf0_mid
                tau[speaker_idx, spk_seq_index, 0, 0] = win_pitch_loc

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
    feed_dict['nlf'] = nlf
    feed_dict['tau'] = tau
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
  
def make_feed_dict_y_wav_f0_tau_test(dv_y_cfg, file_dir_dict, speaker_id, file_name, start_frame_index, BTD_feat_remain):
    feat_name = dv_y_cfg.y_feat_name # Hard-coded here for now
    assert dv_y_cfg.batch_num_spk == 1
    # Make i/o shape arrays
    # This is numpy shape, not Tensor shape!
    # No speaker index here! Will add it to Tensor later
    y   = numpy.zeros((dv_y_cfg.spk_num_seq, dv_y_cfg.batch_seq_len, dv_y_cfg.feat_dim))
    dv  = numpy.zeros((dv_y_cfg.batch_num_spk))
    nlf = numpy.zeros((dv_y_cfg.spk_num_seq, 1, 1))
    tau = numpy.zeros((dv_y_cfg.spk_num_seq, 1, 1))

    # Make classification targets, index sequence
    try: true_speaker_index = dv_y_cfg.speaker_id_list_dict['train'].index(speaker_id)
    except ValueError: true_speaker_index = 0 # At generation time, since dv is not used, a non-train speaker is given an arbituary speaker index
    dv[0] = true_speaker_index

    if BTD_feat_remain is None:
        # Get new file, make BTD
        _min_len, features = get_one_utter_by_name(file_name, file_dir_dict, feat_name_list=[feat_name], feat_dim_list=[dv_y_cfg.feat_dim])
        y_stack = features[feat_name]

        wav_sr  = dv_y_cfg.cfg.wav_sr
        cmp_sr  = dv_y_cfg.cfg.frame_sr
        wav_cmp_ratio = int(wav_sr / cmp_sr)
        # Do not use silence frames at the beginning or the end
        total_sil_one_side_cmp = dv_y_cfg.frames_silence_to_keep + dv_y_cfg.sil_pad # This is at 200Hz
        total_sil_one_side_wav = total_sil_one_side_cmp * wav_cmp_ratio             # This is at 16kHz
        min_file_len = dv_y_cfg.batch_seq_total_len + 2 * total_sil_one_side_wav    # This is at 16kHz
        sil_index_dict = dv_y_cfg.sil_index_dict

        no_sil_start_cmp = sil_index_dict[file_name][0]
        no_sil_end_cmp   = sil_index_dict[file_name][1]
        len_no_sil_cmp   = no_sil_end_cmp - no_sil_start_cmp + 1
        len_no_sil_wav   = len_no_sil_cmp * wav_cmp_ratio
        sil_pad_first_idx_cmp = max(0, no_sil_start_cmp - total_sil_one_side_cmp)
        t_removed = float(sil_pad_first_idx_cmp) / cmp_sr
        remain_sil_before_cmp = no_sil_start_cmp - sil_pad_first_idx_cmp
        remain_sil_before_wav = remain_sil_before_cmp * wav_cmp_ratio

        B_total  = int((len_no_sil_wav - dv_y_cfg.batch_seq_len) / dv_y_cfg.batch_seq_shift) + 1
        features_no_sil = y_stack[remain_sil_before_wav:remain_sil_before_wav+len_no_sil_wav]
        # Load cmp and pitch data
        cmp_file_name = os.path.join(file_dir_dict['cmp'], file_name+'.cmp')
        lf0_index     = dv_y_cfg.cfg.acoustic_start_index['lf0']
        cmp_dim       = dv_y_cfg.cfg.nn_feature_dims['cmp']
        lf0_norm_data = load_cmp_file(cmp_file_name, cmp_dim, lf0_index)  # Extract lf0 from cmp
        pitch_file_name = os.path.join(file_dir_dict['pitch'], file_name+'.used.pm')
        pitch_loc_data = read_pitch_file(pitch_file_name)

        y_features = numpy.zeros((B_total, dv_y_cfg.batch_seq_len, dv_y_cfg.feat_dim))
        nlf_features = numpy.zeros((B_total, 1, 1))
        tau_features = numpy.zeros((B_total, 1, 1))
        for utter_seq_idx in range(B_total):
            n_start = utter_seq_idx * dv_y_cfg.batch_seq_shift
            n_end   = n_start + dv_y_cfg.batch_seq_len - 1 # Inclusive index
            t_start = (n_start) / wav_sr
            t_end   = (n_end+1) / wav_sr

            lf0_mid = cal_win_lf0_mid(lf0_norm_data, cmp_sr, t_start, t_end) # lf0_norm_data should have same length as y_features
            win_pitch_loc = cal_win_pitch_loc(pitch_loc_data, t_start, t_end, t_removed)

            y_features[utter_seq_idx, :, :]   = features_no_sil[n_start:n_end+1, :]
            nlf_features[utter_seq_idx, 0, 0] = lf0_mid
            tau_features[utter_seq_idx, 0, 0] = win_pitch_loc
    else:
        y_features, nlf_features, tau_features = BTD_feat_remain
        B_total = y_features.shape[0]


    if B_total > dv_y_cfg.spk_num_seq:
        B_actual = dv_y_cfg.spk_num_seq
        B_remain = B_total - B_actual
        gen_finish = False
    else:
        B_actual = B_total
        B_remain = 0
        gen_finish = True

    for b in range(B_actual):
        y[b]   = y_features[b]
        nlf[b] = nlf_features[b]
        tau[b] = tau_features[b]


    if B_remain > 0:
        y_feat_remain   = numpy.zeros((B_remain, dv_y_cfg.batch_seq_len, dv_y_cfg.feat_dim))
        nlf_feat_remain = numpy.zeros((B_remain, 1, 1))
        tau_feat_remain = numpy.zeros((B_remain, 1, 1))
        for b in range(B_remain):
            y_feat_remain[b]   = y_features[b + B_actual]
            nlf_feat_remain[b] = nlf_features[b + B_actual]
            tau_feat_remain[b] = tau_features[b + B_actual]
        BTD_feat_remain = (y_feat_remain, nlf_feat_remain, tau_feat_remain)
    else:
        BTD_feat_remain = None

    batch_size = B_actual

    # B,T,D --> S(1),B,T*D
    x_val   = numpy.reshape(y, (dv_y_cfg.batch_num_spk, dv_y_cfg.spk_num_seq, dv_y_cfg.batch_seq_len*dv_y_cfg.feat_dim))
    # B,1,1 --> S(1),B,1,1
    nlf_val = numpy.reshape(nlf, (dv_y_cfg.batch_num_spk, dv_y_cfg.spk_num_seq, 1, 1))
    tau_val = numpy.reshape(tau, (dv_y_cfg.batch_num_spk, dv_y_cfg.spk_num_seq, 1, 1))
    if dv_y_cfg.train_by_window:
        # S --> S*B
        y_val = numpy.repeat(dv, dv_y_cfg.spk_num_seq)
    else:
        y_val = dv

    feed_dict = {'x':x_val, 'y':y_val, 'nlf':nlf_val, 'tau':tau_val}
    return_list = [feed_dict, gen_finish, batch_size, BTD_feat_remain]
    return return_list

class dv_y_wav_cmp_configuration(dv_y_configuration):
    
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
        self.batch_seq_total_len = 12800 # Number of frames at 16kHz; 32000 for 2s
        self.batch_seq_len   = 3200 # T
        self.batch_seq_shift = 3200
        self.learning_rate   = 0.0001
        self.batch_num_spk = 100
        self.dv_dim = 8
        self.nn_layer_config_list = [
            # Must contain: type, size; num_channels, dropout_p are optional, default 0, 1
            # {'type':'SineAttenCNN', 'size':512, 'num_channels':1, 'dropout_p':1, 'CNN_filter_size':5, 'Sine_filter_size':200,'lf0_mean':5.04976, 'lf0_var':0.361811},
            # {'type':'CNNAttenCNNWav', 'size':1024, 'num_channels':1, 'dropout_p':1, 'CNN_kernel_size':[1,3200], 'CNN_stride':[1,80], 'CNN_activation':'ReLU'},
            {'type':'SinenetV2', 'size':128, 'num_channels':8, 'channel_combi':'stack', 'dropout_p':0, 'batch_norm':False},
            {'type':'ReLUDV', 'size':256, 'dropout_p':0, 'batch_norm':False},
            {'type':'ReLUDV', 'size':256, 'dropout_p':0, 'batch_norm':True},
            {'type':'ReLUDV', 'size':self.dv_dim, 'dropout_p':0.2, 'batch_norm':False}
            # {'type':'LinDV', 'size':self.dv_dim, 'num_channels':1, 'dropout_p':0.5}
        ]

        # self.gpu_id = 'cpu'
        self.gpu_id = 1

        from modules_torch import DV_Y_F0_Tau_model
        self.dv_y_model_class = DV_Y_F0_Tau_model

        self.make_feed_dict_method_train = make_feed_dict_y_wav_f0_tau_train
        self.make_feed_dict_method_test  = make_feed_dict_y_wav_f0_tau_test
        self.auto_complete(cfg)

        self.a_val = None
        self.phi_val = None

    def additional_action_epoch(self, logger, dv_y_model):
        # Print values of a and phi to see if they are updated
        sinenet_layer = dv_y_model.nn_model.layer_list[0].layer_fn.sinenet_layer

        a_val = sinenet_layer.return_a_value()
        phi_val = sinenet_layer.return_phi_value()

        if self.a_val is not None:
            dist = numpy.linalg.norm(a_val-self.a_val)
            logger.info('Amplitude distance is %f' % dist)

        if self.phi_val is not None:
            dist = numpy.linalg.norm(phi_val-self.phi_val)
            logger.info('Phi distance is %f' % dist)
            
        self.a_val   = a_val
        self.phi_val = phi_val

        # If phi is too large or small, change it to between +- 2pi
        sinenet_layer.keep_phi_within_2pi(self.gpu_id)

def train_dv_y_wav_model(cfg, dv_y_cfg=None):
    if dv_y_cfg is None: dv_y_cfg = dv_y_wav_cmp_configuration(cfg)
    train_dv_y_model(cfg, dv_y_cfg)

def test_dv_y_wav_model(cfg, dv_y_cfg=None):
    if dv_y_cfg is None: dv_y_cfg = dv_y_wav_cmp_configuration(cfg)
    class_test_dv_y_model(cfg, dv_y_cfg)

def train_dv_y_model_plot_wav_f0_tau(cfg, dv_y_cfg):

    # Feed data use feed_dict style

    logger = make_logger("dv_y_config")
    log_class_attri(dv_y_cfg, logger, except_list=dv_y_cfg.log_except_list)

    logger = make_logger("train_dvy")
    logger.info('Creating data lists')
    speaker_id_list = dv_y_cfg.speaker_id_list_dict['train'] # For DV training and evaluation, use train speakers only
    speaker_loader  = list_random_loader(speaker_id_list)
    file_id_list    = read_file_list(cfg.file_id_list_file)
    file_list_dict  = make_dv_file_list(file_id_list, speaker_id_list, dv_y_cfg.data_split_file_number) # In the form of: file_list[(speaker_id, 'train')]
    make_feed_dict_method_train = dv_y_cfg.make_feed_dict_method_train

    # Draw random speakers
    batch_speaker_list = speaker_loader.draw_n_samples(dv_y_cfg.batch_num_spk)
    # Make feed_dict for training
    return_list = make_feed_dict_method_train(dv_y_cfg, file_list_dict, cfg.nn_feat_scratch_dirs, batch_speaker_list,  utter_tvt='train', return_dv=False, return_y=False, return_frame_index=True, return_file_name=True)

