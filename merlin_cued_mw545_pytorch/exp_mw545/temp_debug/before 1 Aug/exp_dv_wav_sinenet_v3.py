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
    return pitch_t_list

def cal_win_pitch_loc(pitch_loc_data, t_start, t_end, t_removed=0):
    t_start_total = t_start + t_removed
    t_end_total   = t_end + t_removed
    for t in pitch_loc_data:
        if t > (t_start_total):
            if t < (t_end_total):
                t_r = t - t_start_total
                return t_r, True
        elif t > t_end_total:
            # No pitch found in interval
            return 0, False
    # No pitch, return 0
    return 0, False

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
    wav = numpy.zeros((dv_y_cfg.batch_num_spk, dv_y_cfg.spk_num_seq, dv_y_cfg.seq_num_win, dv_y_cfg.seq_win_len))
    nlf = numpy.zeros((dv_y_cfg.batch_num_spk, dv_y_cfg.spk_num_seq, dv_y_cfg.seq_num_win))
    tau = numpy.zeros((dv_y_cfg.batch_num_spk, dv_y_cfg.spk_num_seq, dv_y_cfg.seq_num_win))
    one_hot = numpy.zeros((dv_y_cfg.batch_num_spk))
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
    
    for speaker_idx in range(dv_y_cfg.batch_num_spk):

        speaker_id = batch_speaker_list[speaker_idx]
        # Make classification targets, index sequence
        true_speaker_index = dv_y_cfg.speaker_id_list_dict['train'].index(speaker_id)
        one_hot[speaker_idx] = true_speaker_index

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

        # Load cmp and pitch data
        cmp_file_name = os.path.join(file_dir_dict['cmp'], file_name+'.cmp')
        lf0_index     = dv_y_cfg.cfg.acoustic_start_index['lf0']
        cmp_dim       = dv_y_cfg.cfg.nn_feature_dims['cmp']
        lf0_norm_data = load_cmp_file(cmp_file_name, cmp_dim=cmp_dim, feat_dim_index=lf0_index)
        pitch_file_name = os.path.join(file_dir_dict['pitch'], file_name+'.pm')
        pitch_loc_data = read_pitch_file(pitch_file_name)

        # Slice data into seq and win
        for seq_idx in range(dv_y_cfg.spk_num_seq):
            spk_seq_index = seq_idx
            seq_start = utter_start_frame_index + seq_idx * dv_y_cfg.batch_seq_shift
            for win_idx in range(dv_y_cfg.seq_num_win):
                win_start = seq_start + win_idx * dv_y_cfg.seq_win_shift
                win_end   = win_start + dv_y_cfg.seq_win_len - 1 # Inclusive index
                t_start = (win_start) / wav_sr
                t_end   = (win_end+1) / wav_sr

                lf0_mid = cal_win_lf0_mid(lf0_norm_data, cmp_sr, t_start, t_end) # lf0_norm_data should have same length as wav_file
                win_pitch_loc, vuv_temp = cal_win_pitch_loc(pitch_loc_data, t_start, t_end)
            
                wav[speaker_idx, spk_seq_index, win_idx, :] = wav_file[win_start:win_end+1]
                nlf[speaker_idx, spk_seq_index, win_idx] = lf0_mid
                tau[speaker_idx, spk_seq_index, win_idx] = win_pitch_loc
                vuv[speaker_idx, spk_seq_index, win_idx] = vuv_temp

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

def make_feed_dict_y_wav_sinenet_train_voiced_only(dv_y_cfg, file_list_dict, file_dir_dict, batch_speaker_list, utter_tvt, all_utt_start_frame_index=None, return_one_hot=False, return_y=False, return_frame_index=False, return_file_name=False):
    ''' This make_dict method returns 200ms windows of voiced only '''
    logger = make_logger("make_dict"),

    '''
    Draw Utterances; Load Data
    Draw starting frame; Slice; Fit into numpy holders
    '''
    feat_name_list = ['wav'] # Load wav
    feat_dim_list  = [1]
    # Make i/o shape arrays
    # This is numpy shape, not Tensor shape!
    wav = numpy.zeros((dv_y_cfg.batch_num_spk, dv_y_cfg.spk_num_seq, dv_y_cfg.seq_num_win, dv_y_cfg.seq_win_len))
    nlf = numpy.zeros((dv_y_cfg.batch_num_spk, dv_y_cfg.spk_num_seq, dv_y_cfg.seq_num_win))
    tau = numpy.zeros((dv_y_cfg.batch_num_spk, dv_y_cfg.spk_num_seq, dv_y_cfg.seq_num_win))
    one_hot = numpy.zeros((dv_y_cfg.batch_num_spk))

    wav_sr  = dv_y_cfg.cfg.wav_sr
    cmp_sr  = dv_y_cfg.cfg.frame_sr
    wav_cmp_ratio = int(wav_sr / cmp_sr)
    # Do not use silence frames at the beginning or the end
    total_sil_one_side_cmp = dv_y_cfg.frames_silence_to_keep + dv_y_cfg.sil_pad  # This is at 200Hz
    total_sil_one_side_wav = total_sil_one_side_cmp * wav_cmp_ratio              # This is at 16kHz
    min_file_len = dv_y_cfg.batch_seq_total_len + 2 * total_sil_one_side_wav # This is at 16kHz
    voiced_seq_win_threshold = int(dv_y_cfg.use_voiced_threshold * dv_y_cfg.seq_num_win)

    file_name_list = [[] for i in range(dv_y_cfg.batch_num_spk)]
    start_frame_index_list = [[] for i in range(dv_y_cfg.batch_num_spk)]
    
    for speaker_idx in range(dv_y_cfg.batch_num_spk):

        speaker_id = batch_speaker_list[speaker_idx]
        # Make classification targets, index sequence
        true_speaker_index = dv_y_cfg.speaker_id_list_dict['train'].index(speaker_id)
        one_hot[speaker_idx] = true_speaker_index

        spk_num_seq_need = dv_y_cfg.spk_num_seq
        spk_seq_index = 0

        while spk_num_seq_need > 0:
            # Draw 1 utterance
            # Draw multiple windows per utterance: dv_y_cfg.spk_num_seq
            # Check vuv of all sub-windows, find "good" windows
            # Use all, or draw randomly if more than needed
            # Stach them along B

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

            # Load cmp and pitch data
            cmp_file_name = os.path.join(file_dir_dict['cmp'], file_name+'.cmp')
            lf0_index     = dv_y_cfg.cfg.acoustic_start_index['lf0']
            cmp_dim       = dv_y_cfg.cfg.nn_feature_dims['cmp']
            lf0_norm_data = load_cmp_file(cmp_file_name, cmp_dim=cmp_dim, feat_dim_index=lf0_index)
            pitch_file_name = os.path.join(file_dir_dict['pitch'], file_name+'.pm')
            pitch_loc_data = read_pitch_file(pitch_file_name)

            # Load pitch data and find good windows first
            voiced_win_idx_list = []
            for seq_idx in range(dv_y_cfg.spk_num_seq):
                seq_start = utter_start_frame_index + seq_idx * dv_y_cfg.batch_seq_shift
                voiced_count = 0
                for win_idx in range(dv_y_cfg.seq_num_win):
                    win_start = seq_start + win_idx * dv_y_cfg.seq_win_shift
                    win_end   = win_start + dv_y_cfg.seq_win_len - 1 # Inclusive index
                    t_start = (win_start) / wav_sr
                    t_end   = (win_end+1) / wav_sr

                    win_pitch_loc, vuv_temp = cal_win_pitch_loc(pitch_loc_data, t_start, t_end)
                    if vuv_temp:
                        voiced_count += 1
                if voiced_count >= voiced_seq_win_threshold:
                    voiced_win_idx_list.append(seq_idx)

            # Use all, or draw randomly if more than needed
            num_voiced_win = len(voiced_win_idx_list)
            if num_voiced_win < spk_num_seq_need:
                # Use all
                spk_num_seq_need -= num_voiced_win
            else:
                # Draw randomly
                voiced_win_idx_list = numpy.random.choice(voiced_win_idx_list, spk_num_seq_need, replace=False)
                spk_num_seq_need = 0

            # Slice data into seq and win
            for seq_idx in voiced_win_idx_list:
                seq_start = utter_start_frame_index + seq_idx * dv_y_cfg.batch_seq_shift
                for win_idx in range(dv_y_cfg.seq_num_win):
                    win_start = seq_start + win_idx * dv_y_cfg.seq_win_shift
                    win_end   = win_start + dv_y_cfg.seq_win_len - 1 # Inclusive index
                    t_start = (win_start) / wav_sr
                    t_end   = (win_end+1) / wav_sr

                    lf0_mid = cal_win_lf0_mid(lf0_norm_data, cmp_sr, t_start, t_end) # lf0_norm_data should have same length as wav_file
                    win_pitch_loc, vuv_temp = cal_win_pitch_loc(pitch_loc_data, t_start, t_end)
                
                    wav[speaker_idx, spk_seq_index, win_idx, :] = wav_file[win_start:win_end+1]
                    nlf[speaker_idx, spk_seq_index, win_idx] = lf0_mid
                    tau[speaker_idx, spk_seq_index, win_idx] = win_pitch_loc
                spk_seq_index += 1
    
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

def make_feed_dict_y_wav_sinenet_test(dv_y_cfg, file_dir_dict, speaker_id, file_name, start_frame_index, BTD_feat_remain):
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
    nlf = numpy.zeros((dv_y_cfg.batch_num_spk, dv_y_cfg.spk_num_seq, dv_y_cfg.seq_num_win))
    tau = numpy.zeros((dv_y_cfg.batch_num_spk, dv_y_cfg.spk_num_seq, dv_y_cfg.seq_num_win))
    one_hot = numpy.zeros((dv_y_cfg.batch_num_spk))

    # Make classification targets, index sequence
    try: true_speaker_index = dv_y_cfg.speaker_id_list_dict['train'].index(speaker_id)
    except ValueError: true_speaker_index = 0 # At generation time, since one_hot is not used, a non-train speaker is given an arbituary speaker index
    one_hot[0] = true_speaker_index

    if BTD_feat_remain is not None:
        wav_feat_current, nlf_feat_current, tau_feat_current = BTD_feat_remain
        B_total = wav_feat_current.shape[0]
    else:
        # Get new file, make BTD
        file_min_len, features = get_one_utter_by_name(file_name, file_dir_dict, feat_name_list=feat_name_list, feat_dim_list=feat_dim_list)

        wav_file = features['wav'] # T * 1; 16kHz
        wav_file = numpy.squeeze(wav_file, axis=1)      # T*1 -> T
        wav_file_len = file_min_len
        if start_frame_index > 0:
            # Discard some features at beginning
            wav_file = wav_file[start_frame_index:]
            wav_file_len -= start_frame_index

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
        nlf_feat_current = numpy.zeros((B_total, dv_y_cfg.seq_num_win))
        tau_feat_current = numpy.zeros((B_total, dv_y_cfg.seq_num_win))

        # Load cmp and pitch data
        cmp_file_name = os.path.join(file_dir_dict['cmp'], file_name+'.cmp')
        lf0_index     = dv_y_cfg.cfg.acoustic_start_index['lf0']
        cmp_dim       = dv_y_cfg.cfg.nn_feature_dims['cmp']
        lf0_norm_data = load_cmp_file(cmp_file_name, cmp_dim=cmp_dim, feat_dim_index=lf0_index)
        pitch_file_name = os.path.join(file_dir_dict['pitch'], file_name+'.pm')
        pitch_loc_data = read_pitch_file(pitch_file_name)

        # Slice data into seq and win
        for seq_idx in range(B_total):
            spk_seq_index = seq_idx
            seq_start = 0 + seq_idx * dv_y_cfg.batch_seq_shift
            for win_idx in range(dv_y_cfg.seq_num_win):
                win_start = seq_start + win_idx * dv_y_cfg.seq_win_shift
                win_end   = win_start + dv_y_cfg.seq_win_len - 1 # Inclusive index
                t_start = (win_start) / wav_sr
                t_end   = (win_end+1) / wav_sr

                lf0_mid = cal_win_lf0_mid(lf0_norm_data, cmp_sr, t_start, t_end) # lf0_norm_data should have same length as wav_file
                win_pitch_loc, vuv = cal_win_pitch_loc(pitch_loc_data, t_start, t_end)
            
                wav_feat_current[spk_seq_index, win_idx, :] = wav_file[win_start:win_end+1]
                nlf_feat_current[spk_seq_index, win_idx] = lf0_mid
                tau_feat_current[spk_seq_index, win_idx] = win_pitch_loc

    if B_total > dv_y_cfg.spk_num_seq:
        B_actual = dv_y_cfg.spk_num_seq
        B_remain = B_total - B_actual
        gen_finish = False
        wav_feat_remain = wav_feat_current[B_actual:]
        nlf_feat_remain = nlf_feat_current[B_actual:]
        tau_feat_remain = tau_feat_current[B_actual:]
        BTD_feat_remain = (wav_feat_remain, nlf_feat_remain, tau_feat_remain)
    else:
        B_actual = B_total
        B_remain = 0
        gen_finish = True
        BTD_feat_remain = None

    wav[0,:B_actual] = wav_feat_current[:B_actual]
    nlf[0,:B_actual] = nlf_feat_current[:B_actual]
    tau[0,:B_actual] = tau_feat_current[:B_actual]
    batch_size = B_actual

    # B,T,D --> S(1),B,T*D
    x_val = wav
    # B,1,1 --> S(1),B,1,1
    nlf_val = nlf
    tau_val = tau
    if dv_y_cfg.train_by_window:
        # S --> S*B
        y_val = numpy.repeat(one_hot, dv_y_cfg.spk_num_seq)
    else:
        y_val = one_hot

    feed_dict = {'x':x_val, 'y':y_val, 'nlf':nlf_val, 'tau':tau_val}
    return_list = [feed_dict, gen_finish, batch_size, BTD_feat_remain]
    return return_list

def make_feed_dict_y_wav_sinenet_test_voiced_only(dv_y_cfg, file_dir_dict, speaker_id, file_name, start_frame_index, BTD_feat_remain):
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
    nlf = numpy.zeros((dv_y_cfg.batch_num_spk, dv_y_cfg.spk_num_seq, dv_y_cfg.seq_num_win))
    tau = numpy.zeros((dv_y_cfg.batch_num_spk, dv_y_cfg.spk_num_seq, dv_y_cfg.seq_num_win))
    one_hot = numpy.zeros((dv_y_cfg.batch_num_spk))

    # Make classification targets, index sequence
    try: true_speaker_index = dv_y_cfg.speaker_id_list_dict['train'].index(speaker_id)
    except ValueError: true_speaker_index = 0 # At generation time, since one_hot is not used, a non-train speaker is given an arbituary speaker index
    one_hot[0] = true_speaker_index

    if BTD_feat_remain is not None:
        wav_feat_current, nlf_feat_current, tau_feat_current = BTD_feat_remain
        B_total = wav_feat_current.shape[0]
    else:
        # Get new file, make BTD
        file_min_len, features = get_one_utter_by_name(file_name, file_dir_dict, feat_name_list=feat_name_list, feat_dim_list=feat_dim_list)

        wav_file = features['wav'] # T * 1; 16kHz
        wav_file = numpy.squeeze(wav_file, axis=1)      # T*1 -> T
        wav_file_len = file_min_len
        if start_frame_index > 0:
            # Discard some features at beginning
            wav_file = wav_file[start_frame_index:]
            wav_file_len -= start_frame_index

        wav_sr = dv_y_cfg.cfg.wav_sr
        cmp_sr = dv_y_cfg.cfg.frame_sr
        wav_cmp_ratio = int(wav_sr / cmp_sr)

        # Do not use silence frames at the beginning or the end
        total_sil_one_side_cmp = dv_y_cfg.frames_silence_to_keep + dv_y_cfg.sil_pad
        total_sil_one_side_wav = total_sil_one_side_cmp * wav_cmp_ratio
        len_no_sil_wav = wav_file_len - 2 * total_sil_one_side_wav
        voiced_seq_win_threshold = int(dv_y_cfg.use_voiced_threshold * dv_y_cfg.seq_num_win)

        # Find number of sequences, then select the ones with good voicing        
        wav_features_no_sil = wav_file[total_sil_one_side_wav:total_sil_one_side_wav+len_no_sil_wav]
        B_total = int((len_no_sil_wav - dv_y_cfg.batch_seq_len) / dv_y_cfg.batch_seq_shift) + 1

        # Load cmp and pitch data
        cmp_file_name = os.path.join(file_dir_dict['cmp'], file_name+'.cmp')
        lf0_index     = dv_y_cfg.cfg.acoustic_start_index['lf0']
        cmp_dim       = dv_y_cfg.cfg.nn_feature_dims['cmp']
        lf0_norm_data = load_cmp_file(cmp_file_name, cmp_dim=cmp_dim, feat_dim_index=lf0_index)
        pitch_file_name = os.path.join(file_dir_dict['pitch'], file_name+'.pm')
        pitch_loc_data = read_pitch_file(pitch_file_name)

        # Load pitch data and find good windows first
        voiced_win_idx_list = []
        for seq_idx in range(B_total):
            seq_start = 0 + seq_idx * dv_y_cfg.batch_seq_shift
            voiced_count = 0
            for win_idx in range(dv_y_cfg.seq_num_win):
                win_start = seq_start + win_idx * dv_y_cfg.seq_win_shift
                win_end   = win_start + dv_y_cfg.seq_win_len - 1 # Inclusive index
                t_start = (win_start) / wav_sr
                t_end   = (win_end+1) / wav_sr

                win_pitch_loc, vuv_temp = cal_win_pitch_loc(pitch_loc_data, t_start, t_end)
                if vuv_temp:
                    voiced_count += 1
            if voiced_count >= voiced_seq_win_threshold:
                voiced_win_idx_list.append(seq_idx)

        # Make numpy holders for no_sil data
        B_total = len(voiced_win_idx_list)
        wav_feat_current = numpy.zeros((B_total, dv_y_cfg.seq_num_win, dv_y_cfg.seq_win_len))
        nlf_feat_current = numpy.zeros((B_total, dv_y_cfg.seq_num_win))
        tau_feat_current = numpy.zeros((B_total, dv_y_cfg.seq_num_win))

        # Slice data into seq and win
        for seq_idx in range(B_total):
            spk_seq_index = seq_idx
            seq_start = voiced_win_idx_list[seq_idx]
            for win_idx in range(dv_y_cfg.seq_num_win):
                win_start = seq_start + win_idx * dv_y_cfg.seq_win_shift
                win_end   = win_start + dv_y_cfg.seq_win_len - 1 # Inclusive index
                t_start = (win_start) / wav_sr
                t_end   = (win_end+1) / wav_sr

                lf0_mid = cal_win_lf0_mid(lf0_norm_data, cmp_sr, t_start, t_end) # lf0_norm_data should have same length as wav_file
                win_pitch_loc, vuv = cal_win_pitch_loc(pitch_loc_data, t_start, t_end)
            
                wav_feat_current[spk_seq_index, win_idx, :] = wav_file[win_start:win_end+1]
                nlf_feat_current[spk_seq_index, win_idx] = lf0_mid
                tau_feat_current[spk_seq_index, win_idx] = win_pitch_loc

    if B_total > dv_y_cfg.spk_num_seq:
        B_actual = dv_y_cfg.spk_num_seq
        B_remain = B_total - B_actual
        gen_finish = False
        wav_feat_remain = wav_feat_current[B_actual:]
        nlf_feat_remain = nlf_feat_current[B_actual:]
        tau_feat_remain = tau_feat_current[B_actual:]
        BTD_feat_remain = (wav_feat_remain, nlf_feat_remain, tau_feat_remain)
    else:
        B_actual = B_total
        B_remain = 0
        gen_finish = True
        BTD_feat_remain = None

    wav[0,:B_actual] = wav_feat_current[:B_actual]
    nlf[0,:B_actual] = nlf_feat_current[:B_actual]
    tau[0,:B_actual] = tau_feat_current[:B_actual]
    batch_size = B_actual

    # B,T,D --> S(1),B,T*D
    x_val = wav
    # B,1,1 --> S(1),B,1,1
    nlf_val = nlf
    tau_val = tau
    if dv_y_cfg.train_by_window:
        # S --> S*B
        y_val = numpy.repeat(one_hot, dv_y_cfg.spk_num_seq)
    else:
        y_val = one_hot

    feed_dict = {'x':x_val, 'y':y_val, 'nlf':nlf_val, 'tau':tau_val}
    return_list = [feed_dict, gen_finish, batch_size, BTD_feat_remain]
    return return_list

def make_feed_dict_y_wav_sinenet_distance(dv_y_cfg, file_list_dict, file_dir_dict, batch_speaker_list, utter_tvt, all_utt_start_frame_index=None,  return_y=False, return_frame_index=False, return_file_name=False):
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
    nlf_list = []
    tau_list = []
    for plot_idx in range(dv_y_cfg.num_to_plot + 1):
        wav = numpy.zeros((dv_y_cfg.batch_num_spk, dv_y_cfg.spk_num_seq, dv_y_cfg.seq_num_win, dv_y_cfg.seq_win_len))
        nlf = numpy.zeros((dv_y_cfg.batch_num_spk, dv_y_cfg.spk_num_seq, dv_y_cfg.seq_num_win))
        tau = numpy.zeros((dv_y_cfg.batch_num_spk, dv_y_cfg.spk_num_seq, dv_y_cfg.seq_num_win))
        wav_list.append(wav)
        nlf_list.append(nlf)
        tau_list.append(tau)

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

        # Draw 1 utterance per speaker
        # Draw multiple windows per utterance:  dv_y_cfg.spk_num_seq
        # Stack them along B
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

        # Load cmp and pitch data
        cmp_file_name = os.path.join(file_dir_dict['cmp'], file_name+'.cmp')
        lf0_index     = dv_y_cfg.cfg.acoustic_start_index['lf0']
        cmp_dim       = dv_y_cfg.cfg.nn_feature_dims['cmp']
        lf0_norm_data = load_cmp_file(cmp_file_name, cmp_dim=cmp_dim, feat_dim_index=lf0_index)
        pitch_file_name = os.path.join(file_dir_dict['pitch'], file_name+'.pm')
        pitch_loc_data = read_pitch_file(pitch_file_name)

        for plot_idx in range(dv_y_cfg.num_to_plot+1):
            plot_start_frame_index = utter_start_frame_index + plot_idx * dv_y_cfg.gap_len_to_plot
            # Slice data into seq and win
            for seq_idx in range(dv_y_cfg.spk_num_seq):
                spk_seq_index = seq_idx
                seq_start = plot_start_frame_index + seq_idx * dv_y_cfg.batch_seq_shift
                for win_idx in range(dv_y_cfg.seq_num_win):
                    win_start = seq_start + win_idx * dv_y_cfg.seq_win_shift
                    win_end   = win_start + dv_y_cfg.seq_win_len - 1 # Inclusive index
                    t_start = (win_start) / wav_sr
                    t_end   = (win_end+1) / wav_sr

                    lf0_mid = cal_win_lf0_mid(lf0_norm_data, cmp_sr, t_start, t_end) # lf0_norm_data should have same length as wav_file
                    win_pitch_loc, vuv = cal_win_pitch_loc(pitch_loc_data, t_start, t_end)
                
                    wav_list[plot_idx][speaker_idx, spk_seq_index, win_idx, :] = wav_file[win_start:win_end+1]
                    nlf_list[plot_idx][speaker_idx, spk_seq_index, win_idx] = lf0_mid
                    tau_list[plot_idx][speaker_idx, spk_seq_index, win_idx] = win_pitch_loc

    # S,B,M,D
    feed_dict_list = [{}] * (dv_y_cfg.num_to_plot+1)
    for plot_idx in range(dv_y_cfg.num_to_plot+1):
        x_val = wav_list[plot_idx]
        feed_dict_list[plot_idx] = {'x':x_val}
        feed_dict_list[plot_idx]['nlf'] = nlf_list[plot_idx]
        feed_dict_list[plot_idx]['tau'] = tau_list[plot_idx]
    batch_size = dv_y_cfg.batch_num_spk * dv_y_cfg.spk_num_seq

    return_list = [feed_dict_list, batch_size]
    
    if return_y:
        return_list.append(y)
    if return_frame_index:
        return_list.append(start_frame_index_list)
    if return_file_name:
        return_list.append(file_name_list)
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
        self.batch_seq_total_len = 32000 # Number of frames at 16kHz; 32000 for 2s
        self.batch_seq_len   = 3200 # T
        self.batch_seq_shift = 3200
        self.seq_win_len   = 640
        self.seq_win_shift = 80
        self.seq_num_win   = int((self.batch_seq_len - self.seq_win_len) / self.seq_win_shift) + 1

        self.batch_num_spk = 100
        self.dv_dim = 8
        self.nn_layer_config_list = [
            # Must contain: type, size; num_channels, dropout_p are optional, default 0, 1
            {'type':'SinenetV3', 'size':81, 'sine_size':80, 'num_freq':10, 'win_len':self.seq_win_len, 'num_win':self.seq_num_win, 'dropout_p':0, 'batch_norm':False},
            {'type':'LReLUDV', 'size':256, 'dropout_p':0, 'batch_norm':True},
            {'type':'LReLUDV', 'size':256, 'dropout_p':0, 'batch_norm':True},
            {'type':'LReLUDV', 'size':self.dv_dim, 'dropout_p':0.2, 'batch_norm':False}
        ]

        # self.gpu_id = 'cpu'
        self.gpu_id = 0

        from modules_torch import DV_Y_Wav_SubWin_model
        self.dv_y_model_class = DV_Y_Wav_SubWin_model
        # from exp_mw545.exp_dv_wav_baseline import make_feed_dict_y_wav_cmp_test
        if self.use_voiced_only:
            self.make_feed_dict_method_train = make_feed_dict_y_wav_sinenet_train_voiced_only
            self.make_feed_dict_method_test  = make_feed_dict_y_wav_sinenet_test_voiced_only
        else:
            self.make_feed_dict_method_train = make_feed_dict_y_wav_sinenet_train
            self.make_feed_dict_method_test  = make_feed_dict_y_wav_sinenet_test
            self.make_feed_dict_method_distance  = make_feed_dict_y_wav_sinenet_distance
        self.make_feed_dict_method_vuv_test = make_feed_dict_y_wav_sinenet_train
        self.auto_complete(cfg)


    def additional_action_epoch(self, logger, dv_y_model):
        pass
        '''
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
        '''

def train_dv_y_wav_model(cfg, dv_y_cfg=None):
    if dv_y_cfg is None: dv_y_cfg = dv_y_wav_sinenet_configuration(cfg)
    train_dv_y_model(cfg, dv_y_cfg)

def test_dv_y_wav_model(cfg, dv_y_cfg=None):
    if dv_y_cfg is None: dv_y_cfg = dv_y_wav_sinenet_configuration(cfg)
    # class_test_dv_y_model(cfg, dv_y_cfg)
    # distance_test_dv_y_model(cfg, dv_y_cfg)
    # plot_sinenet(cfg, dv_y_cfg)
    vuv_test_sinenet(cfg, dv_y_cfg)