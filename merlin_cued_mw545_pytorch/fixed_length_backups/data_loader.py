# data_loader.py

import os, sys, pickle, time, shutil, logging, copy
import math, numpy
numpy.random.seed(545)

from frontend_mw545.modules import make_logger, File_List_Selecter, List_Random_Loader
# from frontend_mw545.modules import keep_by_speaker, remove_by_speaker, keep_by_file_number, remove_by_file_number

from frontend_mw545.data_io import Data_File_IO, Data_List_File_IO

############################
# Basic methods to be used #
############################

class Build_dv_selecter(object):
    """
    Common functions for dv experiments
    1. Speaker random loader
    2. File random loader
    3. Make One-hot output
    """
    def __init__(self, cfg=None, dv_y_cfg=None):
        super(Build_dv_selecter, self).__init__()
        self.logger = make_logger("DV Selecter")

        self.cfg = cfg
        self.dv_y_cfg = dv_y_cfg

        self.DLIO = Data_List_File_IO(cfg)
        self.file_list_selecter = File_List_Selecter()

        self.speaker_id_list = self.cfg.speaker_id_list_dict['train']
        self.speaker_random_loader = List_Random_Loader(self.speaker_id_list)

        self.file_id_list = self.DLIO.read_file_list(self.cfg.file_id_list_file['dv_enough'])
        self.file_list_dict = self.make_dv_file_list_dict()
        self.file_list_random_loader_dict = self.make_dv_file_list_random_loader()

        self.one_hot_S = numpy.zeros((dv_y_cfg.input_data_dim['S']))

    def make_dv_file_list_dict(self):
        ''' Make a dict for file lists, In the form of: file_list_dict[(speaker_id, 'train')] '''
        file_list_dict = self.file_list_selecter.sort_by_speaker_list(self.file_id_list, self.speaker_id_list)

        for speaker_id in self.speaker_id_list:
            speaker_file_dict = self.file_list_selecter.sort_by_file_number(file_list_dict[speaker_id], self.dv_y_cfg.data_split_file_number)
            for utter_tvt_name in ['train', 'valid', 'test']:
                file_list_dict[(speaker_id, utter_tvt_name)] = speaker_file_dict[utter_tvt_name]

        return file_list_dict

    def make_dv_file_list_random_loader(self):
        ''' Make a dict for List_Random_Loader, In the form of: file_list_random_loader_dict[(speaker_id, 'train')] '''
        file_list_random_loader_dict = {}
        
        for speaker_id in self.speaker_id_list:
            for utter_tvt_name in ['train', 'valid', 'test']:
                file_list_random_loader_dict[(speaker_id, utter_tvt_name)] = List_Random_Loader(self.file_list_dict[(speaker_id, utter_tvt_name)])

        return file_list_random_loader_dict

    def draw_n_speakers(self, n):
        return self.speaker_random_loader.draw_n_samples(n)

    def draw_n_files(self, speaker_id, utter_tvt_name, n):
        return self.file_list_random_loader_dict[(speaker_id,utter_tvt_name)].draw_n_samples(n)

    def make_one_hot(self, speaker_id_list, ref_speaker_id_list=None):
        dv_y_cfg = self.dv_y_cfg
        if ref_speaker_id_list is None:
            ref_speaker_id_list = self.cfg.speaker_id_list_dict['train']

        # Make classification targets, index sequence
        for i in range(dv_y_cfg.input_data_dim['S']):
            speaker_id = speaker_id_list[i]
            try:
                true_speaker_index = ref_speaker_id_list.index(speaker_id)
            except ValueError: # New speaker, not in the training set
                true_speaker_index = -1
            self.one_hot_S[i] = true_speaker_index
        if dv_y_cfg.train_by_window:
            # S --> S*B
            self.one_hot = numpy.repeat(self.one_hot_S, dv_y_cfg.input_data_dim['B'])
            batch_size = dv_y_cfg.input_data_dim['S'] * dv_y_cfg.input_data_dim['B']
        else:
            self.one_hot = self.one_hot_S
            batch_size = dv_y_cfg.input_data_dim['S']

        return self.one_hot, self.one_hot_S, batch_size

    def file_id_list_2_speaker_list(self, file_id_list):
        '''
        extract speaker_id from file_id
        '''
        speaker_list = []
        for file_id in file_id_list:
            speaker_id = file_id.split('_')[0]
            speaker_list.append(speaker_id)
        return speaker_list

#############################
# Methods for waveform data #
#############################

class Build_dv_y_wav_data_loader_Ref(object):
    """
    This is a reference data loader 
    It can load only for one file
    The pitch method is very slow:
      search in pitch_list:
      if pitch_t >= start_t and pitch_t < end_t:
        tau = pitch_t - start_t
    The f0 method extract f0 in the middle (round down) of the window
    """
    def __init__(self, cfg=None, dv_y_cfg=None):
        super().__init__()
        self.logger = make_logger("Data_Loader Ref")

        self.cfg = cfg
        self.dv_y_cfg = dv_y_cfg
        self.DIO = Data_File_IO(cfg)

        self.load_seq_win_config(dv_y_cfg)
        # silence_data_meta_dict, load if necessary
        self.silence_data_meta_dict = None

    def load_seq_win_config(self, dv_y_cfg):
        '''
        Load from dv_y_cfg, or hard-code here
        '''
        self.input_data_dim = {}
        if dv_y_cfg is None:
            self.input_data_dim['T_S'] = 6000 # Number of samples at wav_sr
            self.input_data_dim['T_B']   = 3200 # T
            self.input_data_dim['B_shift'] = 80
            self.input_data_dim['T_M']   = 640
            self.input_data_dim['M_shift'] = 80
        else:
            self.input_data_dim['T_S'] = dv_y_cfg.input_data_dim['T_S']
            self.input_data_dim['T_B'] = dv_y_cfg.input_data_dim['T_B']
            self.input_data_dim['B_shift'] = dv_y_cfg.input_data_dim['B_shift']
            self.input_data_dim['T_M']   = dv_y_cfg.input_data_dim['T_M']
            self.input_data_dim['M_shift'] = dv_y_cfg.input_data_dim['M_shift']

        self.input_data_dim['B'] = int((self.input_data_dim['T_S'] - self.input_data_dim['T_B']) / self.input_data_dim['B_shift']) + 1
        self.input_data_dim['M'] = int((self.input_data_dim['T_B'] - self.input_data_dim['T_M']) / self.input_data_dim['M_shift']) + 1

        wav_sr  = self.cfg.wav_sr
        cmp_sr  = self.cfg.frame_sr
        wav_cmp_ratio = int(wav_sr / cmp_sr)
        # Do not use silence frames at the beginning or the end
        total_sil_one_side_cmp = self.cfg.frames_silence_to_keep + self.cfg.sil_pad   # This is at 200Hz
        self.total_sil_one_side_wav = total_sil_one_side_cmp * wav_cmp_ratio          # This is at wav_sr

    def make_feed_dict(self, file_id, start_sample_no_sil):
        '''
        1. Compute start_sample_include_sil
        (Optional, not used here) 2. Compute all win_start_t_list, win_end_t_list (B*M)
        3. Load x (use wav_resil_norm, start with start_sample_no_sil)
        4. Load pitch and vuv
        5. Load f0
        '''
        feed_dict = {}

        wav_resil_norm_file_name = os.path.join(self.cfg.nn_feat_resil_norm_dirs['wav'], file_id+'.wav')
        wav_T = self.make_wav_T(wav_resil_norm_file_name, start_sample_no_sil)
        feed_dict['wav_T'] = wav_T

        start_sample_include_sil = self.compute_start_sample_include_sil(file_id, start_sample_no_sil)

        pitch_text_file_name = os.path.join(self.cfg.pitch_dir, file_id+'.pitch')
        tau_BM, vuv_BM = self.make_tau_BM(pitch_text_file_name, start_sample_include_sil)
        feed_dict['tau_BM'] = tau_BM
        feed_dict['vuv_BM'] = vuv_BM

        f0_16k_file_name = os.path.join(self.cfg.nn_feat_dirs['f016k'], file_id+'.f016k')
        f_BM = self.make_f_BM(f0_16k_file_name, start_sample_include_sil)
        feed_dict['f_BM'] = f_BM

        return feed_dict

    def make_wav_T(self, wav_resil_norm_file_name, start_sample_no_sil):
        '''
        Waveform data is S*T; window slicing is done in Pytorch
        '''
        wav_data, sample_number = self.DIO.load_data_file_frame(wav_resil_norm_file_name, 1)
        wav_data = numpy.squeeze(wav_data)

        start_sample_include_sil = start_sample_no_sil+self.total_sil_one_side_wav

        wav_T = wav_data[start_sample_include_sil:start_sample_include_sil+self.input_data_dim['T_S']]

        return wav_T

    def compute_start_sample_include_sil(self, file_id, start_sample_no_sil):
        '''
        Find start of silence, convert from frame to sample, then add
        '''
        if self.silence_data_meta_dict is None:
            data_meta_file_name='/home/dawna/tts/mw545/TorchDV/file_id_lists/data_meta/file_id_list_num_sil_frame.scp'
            from frontend_mw545.data_io import Data_Meta_List_File_IO
            DMLIO = Data_Meta_List_File_IO(self.cfg)
            self.silence_data_meta_dict = DMLIO.read_file_list_num_silence_frame(data_meta_file_name)

        num_frame_wav_cmp, first_non_sil_index, last_non_sil_index = self.silence_data_meta_dict[file_id]
        wav_sr  = self.cfg.wav_sr
        cmp_sr  = self.cfg.frame_sr
        wav_cmp_ratio = int(wav_sr / cmp_sr)
        start_sample_include_sil = start_sample_no_sil + first_non_sil_index * wav_cmp_ratio
        return start_sample_include_sil
    

    def make_tau_BM(self, pitch_text_file_name, start_sample_include_sil):
        '''
        vuv=1. if pitch found inside the window; vuv=0. otherwise
        tau=pitch_t - win_start_t; 0 <= tau < win_T
        '''
        wav_sr = float(self.cfg.wav_sr)
        tau_matrix = numpy.zeros((self.input_data_dim['B'], self.input_data_dim['M']))
        vuv_matrix = numpy.zeros((self.input_data_dim['B'], self.input_data_dim['M']))

        pitch_t_list = self.DIO.read_pitch_reaper(pitch_text_file_name)

        for b in range(self.input_data_dim['B']):
            for m in range(self.input_data_dim['M']):
                win_start_n = start_sample_include_sil + b * self.input_data_dim['B_shift'] + m * self.input_data_dim['M_shift']
                win_start_t = float(win_start_n) / wav_sr
                win_end_t   = float(win_start_n + self.input_data_dim['T_M']) / wav_sr

                for pitch_t in pitch_t_list:
                    if pitch_t >= win_start_t:
                        if pitch_t < win_end_t:
                            tau_matrix[b,m] = pitch_t - win_start_t
                            vuv_matrix[b,m] = 1.
                            break
                        else:
                            # No pitch found inside window, break for loop
                            break

        return tau_matrix, vuv_matrix

    def make_f_BM(self, f0_16k_file_name, start_sample_include_sil):
        '''
        Extract f0 in the middle (round down) of the window
        '''
        wav_sr = float(self.cfg.wav_sr)
        f0_matrix = numpy.zeros((self.input_data_dim['B'], self.input_data_dim['M']))

        f0_16k_data, sample_number = self.DIO.load_data_file_frame(f0_16k_file_name, 1)

        for b in range(self.input_data_dim['B']):
            for m in range(self.input_data_dim['M']):
                win_start_n = start_sample_include_sil + b * self.input_data_dim['B_shift'] + m * self.input_data_dim['M_shift']
                win_mid_n   = int(win_start_n + self.input_data_dim['T_M'] / 2)
                f0_matrix[b,m] = f0_16k_data[win_mid_n, 0]

        return f0_matrix

class Build_dv_y_wav_data_loader_Single_File(object):
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
        if dv_y_cfg is None:
            self.input_data_dim = {'T_S': 6000, 'T_B': 3200, 'B_shift': 80, 'T_M': 640, 'M_shift': 80}
            self.input_data_dim['B'] = int((self.input_data_dim['T_S'] - self.input_data_dim['T_B']) / self.input_data_dim['B_shift']) + 1
            self.input_data_dim['M'] = int((self.input_data_dim['T_B'] - self.input_data_dim['T_M']) / self.input_data_dim['M_shift']) + 1
            self.out_feat_list = ['wav_ST', 'f_SBM', 'tau_SBM', 'vuv_SBM']
        else:
            self.input_data_dim = dv_y_cfg.input_data_dim
            self.out_feat_list = dv_y_cfg.out_feat_list

        self.win_T = float(self.input_data_dim['T_M']) / float(self.cfg.wav_sr)

        wav_sr  = self.cfg.wav_sr
        cmp_sr  = self.cfg.frame_sr
        wav_cmp_ratio = int(wav_sr / cmp_sr)
        # Do not use silence frames at the beginning or the end
        total_sil_one_side_cmp = self.cfg.frames_silence_to_keep + self.cfg.sil_pad   # This is at frame_sr
        self.total_sil_one_side_wav = total_sil_one_side_cmp * wav_cmp_ratio          # This is at wav_sr

    def init_n_matrix(self):
        '''
        Make matrices of index, B*M
        '''
        B_M_grid = numpy.mgrid[0:self.input_data_dim['B'],0:self.input_data_dim['M']]
        start_n_matrix = self.input_data_dim['B_shift'] * B_M_grid[0] + self.input_data_dim['M_shift'] * B_M_grid[1]
        self.start_n_matrix = start_n_matrix.astype(int)
        mid_n_matrix   = start_n_matrix + self.input_data_dim['T_M'] / 2
        self.mid_n_matrix = mid_n_matrix.astype(int)

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
        return self.make_wav_T_from_data(self, start_sample_no_sil)

    def make_wav_T_from_data(self, wav_data, start_sample_no_sil):
        start_sample_include_sil = start_sample_no_sil+self.total_sil_one_side_wav
        wav_T = wav_data[start_sample_include_sil:start_sample_include_sil+self.input_data_dim['T_S']]

        return wav_T

    def make_tau_BM(self, pitch_resil_norm_file_name, start_sample_no_sil):
        '''
        B * M
        vuv=1. if pitch found inside the window; vuv=0. otherwise
        pitch found: 0 <= tau < win_T
        '''
        pitch_16k_data, sample_number = self.DIO.load_data_file_frame(pitch_resil_norm_file_name, 1)
        pitch_16k_data = numpy.squeeze(pitch_16k_data)
        return self.make_tau_BM_from_data(pitch_16k_data, start_sample_no_sil)

    def make_tau_BM_from_data(self, pitch_16k_data, start_sample_no_sil):

        start_sample_include_sil = start_sample_no_sil+self.total_sil_one_side_wav
        pitch_T = pitch_16k_data[start_sample_include_sil:start_sample_include_sil+self.input_data_dim['T_S']]

        tau_BM = pitch_T[self.start_n_matrix]

        vuv_BM = numpy.ones((self.input_data_dim['B'], self.input_data_dim['M']))
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
        return self.make_f_BM_from_data(f0_16k_data, start_sample_no_sil)

    def make_f_BM_from_data(self, f0_16k_data, start_sample_no_sil):
        start_sample_include_sil = start_sample_no_sil+self.total_sil_one_side_wav
        f0_T = f0_16k_data[start_sample_include_sil:start_sample_include_sil+self.input_data_dim['T_S']]

        f_BM = f0_T[self.mid_n_matrix]
        # mid_n_matrix = self.mid_n_matrix + start_sample_no_sil
        # f_BM = f0_16k_data[mid_n_matrix.astype(int)]
        return f_BM

class Build_dv_y_wav_data_loader_Multi_Speaker(object):
    """
    """
    def __init__(self, cfg=None, dv_y_cfg=None):
        super(Build_dv_y_wav_data_loader_Multi_Speaker, self).__init__()
        self.logger = make_logger("Data_Loader Multi")

        self.cfg = cfg
        self.dv_y_cfg = dv_y_cfg
        self.DIO = Data_File_IO(cfg)

        self.total_sil_one_side_wav = (cfg.frames_silence_to_keep + cfg.sil_pad) * int(cfg.wav_sr / cfg.frame_sr)
        self.min_file_len = dv_y_cfg.input_data_dim['T_S'] + 2 * self.total_sil_one_side_wav

        self.init_feed_dict()
        self.init_directories()

        self.dv_y_wav_data_loader_Single_File = Build_dv_y_wav_data_loader_Single_File(cfg, dv_y_cfg)

    def init_directories(self):
        sr_k = int(self.cfg.wav_sr / 1000)
        if self.dv_y_cfg.data_dir_mode == 'scratch':
            self.wav_dir   = self.cfg.nn_feat_scratch_dirs['wav']
            self.pitch_dir = self.cfg.nn_feat_scratch_dirs['pitch']
            self.f0_dir = self.cfg.nn_feat_scratch_dirs['f0%ik'%sr_k]
            
        elif self.dv_y_cfg.data_dir_mode == 'data':
            self.wav_dir   = self.cfg.nn_feat_resil_norm_dirs['wav']
            self.pitch_dir = self.cfg.nn_feat_resil_dirs['pitch']
            self.f0_dir = self.cfg.nn_feat_resil_dirs['f0%ik'%sr_k]

    def init_feed_dict(self):
        dv_y_cfg = self.dv_y_cfg
        self.feed_dict = {}
        S = dv_y_cfg.input_data_dim['S']
        B = dv_y_cfg.input_data_dim['B']
        M = dv_y_cfg.input_data_dim['M']

        if 'wav_ST' in dv_y_cfg.out_feat_list:
            self.feed_dict['wav_ST'] = numpy.zeros((S, dv_y_cfg.input_data_dim['T_S']))
        if 'wav_SBT' in dv_y_cfg.out_feat_list:
            self.feed_dict['wav_SBT'] = numpy.zeros((S, B, dv_y_cfg.input_data_dim['T_B']))
            B_T_grid = numpy.mgrid[0:B,0:self.dv_y_cfg.input_data_dim['T_B']]
            B_T_matrix = self.dv_y_cfg.input_data_dim['B_shift'] * B_T_grid[0] + B_T_grid[1]
            self.B_T_matrix = B_T_matrix.astype(int)
        if 'wav_SBMT' in dv_y_cfg.out_feat_list:
            self.feed_dict['wav_SBMT'] = numpy.zeros((S, B, M, dv_y_cfg.input_data_dim['T_M']))
            B_M_T_grid = numpy.mgrid[0:B,0:M,0:self.dv_y_cfg.input_data_dim['T_M']]
            B_M_T_matrix = self.dv_y_cfg.input_data_dim['B_shift'] * B_M_T_grid[0] + self.dv_y_cfg.input_data_dim['M_shift'] * B_M_T_grid[1] + B_M_T_grid[2]
            self.B_M_T_matrix = B_M_T_matrix.astype(int)

        if ('tau_SBM' in dv_y_cfg.out_feat_list) or ('vuv_SBM' in dv_y_cfg.out_feat_list):
            self.feed_dict['tau_SBM'] = numpy.zeros((S,B,M))
            self.feed_dict['vuv_SBM'] = numpy.zeros((S,B,M))
        if 'f_SBM' in dv_y_cfg.out_feat_list:
            self.feed_dict['f_SBM'] = numpy.zeros((S,B,M))
        
    def make_feed_dict(self, file_id_list, start_sample_no_sil_list=None):
        '''
        1. Load x
            If start_sample_no_sil_list is None, randomly draw a start index, from extra file length
        2. Load pitch and vuv
        3. Load f0 
        '''
        dv_y_cfg = self.dv_y_cfg

        if start_sample_no_sil_list is None:
            start_sample_no_sil_list = [None] * dv_y_cfg.input_data_dim['S']
            extra_file_len_ratio_list = numpy.random.rand(dv_y_cfg.input_data_dim['S'])
        else:
            extra_file_len_ratio_list = [None] * dv_y_cfg.input_data_dim['S']

        sr_k = int(self.cfg.wav_sr / 1000)

        for i, file_id in enumerate(file_id_list):
            start_sample_no_sil  = start_sample_no_sil_list[i]
            extra_file_len_ratio = extra_file_len_ratio_list[i]
            speaker_id = file_id.split('_')[0]
            wav_resil_norm_file_name = os.path.join(self.wav_dir, speaker_id, file_id+'.wav')
            wav_T, start_sample_no_sil = self.make_wav_T(wav_resil_norm_file_name, start_sample_no_sil, extra_file_len_ratio)
            if 'wav_ST' in dv_y_cfg.out_feat_list:
                self.feed_dict['wav_ST'][i] = wav_T
            if 'wav_SBT' in dv_y_cfg.out_feat_list:
                wav_BT = wav_T[self.B_T_matrix]
                self.feed_dict['wav_SBT'][i] = wav_BT
            if 'wav_SBMT' in dv_y_cfg.out_feat_list:
                wav_BMT = wav_T[self.B_M_T_matrix]
                self.feed_dict['wav_SBMT'][i] = wav_BMT
            # try:
            #     self.feed_dict['wav_ST'][i] = wav_T
            # except:
            #     print(file_id)
            #     print(wav_T)
            #     print(self.feed_dict['wav_ST'].shape)

            if ('tau_SBM' in dv_y_cfg.out_feat_list) or ('vuv_SBM' in dv_y_cfg.out_feat_list):
                pitch_resil_file_name = os.path.join(self.pitch_dir, speaker_id, file_id+'.pitch')
                tau_BM, vuv_BM = self.dv_y_wav_data_loader_Single_File.make_tau_BM(pitch_resil_file_name, start_sample_no_sil)
                self.feed_dict['tau_SBM'][i] = tau_BM
                self.feed_dict['vuv_SBM'][i] = vuv_BM

            if 'f_SBM' in dv_y_cfg.out_feat_list:
                f0_file_name = os.path.join(self.f0_dir, speaker_id, file_id+'.f0%ik'%sr_k)
                f_BM = self.dv_y_wav_data_loader_Single_File.make_f_BM(f0_file_name, start_sample_no_sil)
                self.feed_dict['f_SBM'][i] = f_BM

        # Unconventional Ones, mostly for testing
        if 'f_SBM_2' in dv_y_cfg.out_feat_list:
            # replace self.feed_dict['f_SBM'] with a new version
            # average every 2 values
            self.feed_dict['f_SBM'] = self.average_last_dim(self.feed_dict['f_SBM'], 2)
        if 'f_SBM_4' in dv_y_cfg.out_feat_list:
            # replace self.feed_dict['f_SBM'] with a new version
            # average every 2 values
            self.feed_dict['f_SBM'] = self.average_last_dim(self.feed_dict['f_SBM'], 4)

        return self.feed_dict

    def make_wav_T(self, wav_resil_norm_file_name, start_sample_no_sil, extra_file_len_ratio):
        '''
        Load waveform data
        If start index is given, simply extract T samples; 
        unfolding is done elsewhere, Pytorch or this data loader e.g. wav_BT = wav_T[self.B_T_matrix]
        If not, use random ratio
        '''
        wav_data, sample_number = self.DIO.load_data_file_frame(wav_resil_norm_file_name, 1)
        wav_data = numpy.squeeze(wav_data)

        if start_sample_no_sil is None:
            extra_file_len = sample_number - self.min_file_len
            start_sample_no_sil = int(extra_file_len_ratio * (extra_file_len+1))

        start_sample_include_sil = start_sample_no_sil+self.total_sil_one_side_wav
        wav_T = wav_data[start_sample_include_sil:start_sample_include_sil+self.dv_y_cfg.input_data_dim['T_S']]
        return wav_T, start_sample_no_sil

    def average_last_dim(self, data, num):
        '''
        average the last dimension
        '''
        n = num
        a = data
        
        n_last = a.shape[-1]
        K = int(n_last / n)
        n_res = n_last - K * n
        
        b = numpy.zeros(a.shape)
        d = a.ndim

        # Deal with K*n first
        for k in range(K):
            index_list = range(k*n, (k+1)*n)
            for i in index_list:
                if d == 3:
                    b[:,:,i] = numpy.mean(a[:,:,index_list], -1)

        # Deal with residuals
        if n_res > 0:
            index_list = range(K*n, n_last)
            for i in index_list:
                if d == 3:
                    b[:,:,i] = numpy.mean(a[:,:,index_list], -1)

        return b


############################
# Methods for vocoder data #
############################

class Build_dv_y_cmp_data_loader_Single_File(object):
    """
    This is a reference data loader 
    It can load only for one file
    """
    def __init__(self, cfg=None, dv_y_cfg=None):
        super().__init__()
        self.logger = make_logger("Data_Loader Ref")

        self.cfg = cfg
        self.dv_y_cfg = dv_y_cfg
        self.DIO = Data_File_IO(cfg)

        self.total_sil_one_side_cmp = cfg.frames_silence_to_keep + cfg.sil_pad

        self.load_seq_feat_config(dv_y_cfg)

    def load_seq_feat_config(self, dv_y_cfg):
        '''
        Load from dv_y_cfg, or hard-code here
        Compute input dimension and index list, based on out_feat_list
        '''
        self.input_data_dim = {}
        if dv_y_cfg is None:
            self.out_feat_list = ['mgc', 'lf0', 'bap']
            self.cmp_dim = 86
            self.input_data_dim['T_S'] = 200 # Number of frames at 200Hz
            self.input_data_dim['T_B']   = 40 # T
            self.input_data_dim['B_shift'] = 1
        else:
            self.out_feat_list = dv_y_cfg.out_feat_list
            self.cmp_dim = dv_y_cfg.cmp_dim
            for k in dv_y_cfg.input_data_dim:
                self.input_data_dim[k] = dv_y_cfg.input_data_dim[k]
        # Feature, data dimension, and data dimension index
        self.cfg_cmp_dim = self.cfg.nn_feature_dims['cmp']

    def make_feed_dict(self, file_id, start_frame_no_sil):
        '''
        1. Load cmp, based on start position and features
        2. Slice into sequences
        '''
        feed_dict = {}

        cmp_resil_norm_file_name = os.path.join(self.cfg.nn_feat_resil_norm_dirs['cmp'], file_id+'.cmp')
        cmp_BD = self.make_cmp_BD(cmp_resil_norm_file_name, start_frame_no_sil)
        feed_dict['cmp_BD'] = cmp_BD

        return feed_dict

    def make_cmp_BD(self, cmp_resil_norm_file_name, start_frame_no_sil):
        '''
        Output: B*D; D = self.input_data_dim['D'] = self.cmp_dim * self.input_data_dim['T_B']
        '''
        cmp_data, sample_number = self.DIO.load_data_file_frame(cmp_resil_norm_file_name, self.cfg_cmp_dim)
        return self.make_cmp_BD_from_data(cmp_data, start_frame_no_sil)

    def make_cmp_BD_from_data(self, cmp_data, start_frame_no_sil):

        cmp_BD = numpy.zeros((self.input_data_dim['B'], self.input_data_dim['D']))
        if self.cmp_dim == self.cfg_cmp_dim:
            pass
        else:
            # Extract dimensions of features we want
            cmp_data = self.feat_extract(cmp_data)

        start_frame_include_sil = start_frame_no_sil + self.total_sil_one_side_cmp
        for b in range(self.input_data_dim['B']):
            n_start = start_frame_include_sil + b * self.input_data_dim['B_shift']
            n_end   = n_start + self.input_data_dim['T_B']
            cmp_TD  = cmp_data[n_start:n_end]
            cmp_D   = cmp_TD.reshape(-1)
            cmp_BD[b] = cmp_D

        return cmp_BD

    def feat_extract(self, cmp_data):
        '''
        TODO: implement this if needed
        Extract dimensions of features based on self.out_feat_list and cfg.acoustic_start_index
        '''
        feat_dim_index_list = [] # To be implemented
        cmp_data = cmp_data[:,feat_dim_index_list]
        return cmp_data

class Build_dv_y_cmp_data_loader_Multi_Speaker(object):
    """
    Output: feed_dict['h'], SBD
    D = input_data_dim['T_B'] * cmp_dim
    """
    def __init__(self, cfg=None, dv_y_cfg=None):
        super().__init__()
        self.logger = make_logger("Data_Loader Multi")

        self.cfg = cfg
        self.dv_y_cfg = dv_y_cfg
        self.DIO = Data_File_IO(cfg)

        self.total_sil_one_side_cmp = cfg.frames_silence_to_keep + cfg.sil_pad
        self.min_file_len = dv_y_cfg.input_data_dim['T_S'] + 2 * self.total_sil_one_side_cmp

        self.init_feed_dict()
        self.init_directories()

        self.dv_y_cmp_data_loader_Single_File = Build_dv_y_cmp_data_loader_Single_File(cfg, dv_y_cfg)

    def init_directories(self):
        if self.dv_y_cfg.data_dir_mode == 'scratch':
            self.cmp_dir   = self.cfg.nn_feat_scratch_dirs['cmp']
        elif self.dv_y_cfg.data_dir_mode == 'data':
            self.cmp_dir   = self.cfg.nn_feat_resil_norm_dirs['cmp']

    def init_feed_dict(self):
        dv_y_cfg = self.dv_y_cfg
        self.feed_dict = {}
        self.feed_dict['h'] = numpy.zeros((dv_y_cfg.input_data_dim['S'], dv_y_cfg.input_data_dim['B'], dv_y_cfg.input_data_dim['D']))

    def make_feed_dict(self, file_id_list, start_sample_no_sil_list=None):
        dv_y_cfg = self.dv_y_cfg

        if start_sample_no_sil_list is None:
            start_sample_no_sil_list = [None] * dv_y_cfg.input_data_dim['S']
            extra_file_len_ratio_list = numpy.random.rand(dv_y_cfg.input_data_dim['S'])
        else:
            extra_file_len_ratio_list = [None] * dv_y_cfg.input_data_dim['S']

        for i, file_id in enumerate(file_id_list):
            start_sample_no_sil  = start_sample_no_sil_list[i]
            extra_file_len_ratio = extra_file_len_ratio_list[i]
            speaker_id = file_id.split('_')[0]
            cmp_resil_norm_file_name = os.path.join(self.cmp_dir, speaker_id, file_id+'.cmp')
            cmp_BD, start_sample_no_sil = self.make_cmp_BD(cmp_resil_norm_file_name, start_sample_no_sil, extra_file_len_ratio)
            try:
                self.feed_dict['h'][i] = cmp_BD
            except:
                print(file_id)
                print(cmp_BD)
                print(self.feed_dict['h'].shape)

        return self.feed_dict

    def make_cmp_BD(self, cmp_resil_norm_file_name, start_sample_no_sil, extra_file_len_ratio):
        '''
        Load cmp data
        If start index is given, simply extract T samples
        If not, use random ratio
        '''
        cmp_data, sample_number = self.DIO.load_data_file_frame(cmp_resil_norm_file_name, self.dv_y_cmp_data_loader_Single_File.cfg_cmp_dim)

        if start_sample_no_sil is None:
            extra_file_len = sample_number - self.min_file_len
            start_sample_no_sil = int(extra_file_len_ratio * (extra_file_len+1))

        cmp_BD = self.dv_y_cmp_data_loader_Single_File.make_cmp_BD(cmp_resil_norm_file_name, start_sample_no_sil)
        return cmp_BD, start_sample_no_sil


########################
# Data loading methods #
########################

class Build_dv_y_train_data_loader(object):
    '''
    This Data Loader does the following when called:
    1. Draw Speaker List;
    2. Draw file list
    3. Draw starting index (implicit, inside Build_dv_y_cmp_data_loader_Multi_Speaker or Build_dv_y_wav_data_loader_Multi_Speaker)
    4. Make matrices and feed_dict
    '''
    def __init__(self, cfg=None, dv_y_cfg=None):
        super(Build_dv_y_train_data_loader, self).__init__()
        self.logger = make_logger("Data_Loader Train")

        self.cfg = cfg
        self.dv_y_cfg = dv_y_cfg

        self.dv_selecter = Build_dv_selecter(cfg, dv_y_cfg)

        if dv_y_cfg.y_feat_name == 'wav':
            self.init_wav()
        elif dv_y_cfg.y_feat_name == 'cmp':
            self.init_cmp()

    def init_wav(self):
        self.dv_y_data_loader = Build_dv_y_wav_data_loader_Multi_Speaker(self.cfg, self.dv_y_cfg)

    def init_cmp(self):
        self.dv_y_data_loader = Build_dv_y_cmp_data_loader_Multi_Speaker(self.cfg, self.dv_y_cfg)

    def make_feed_dict(self, utter_tvt_name, file_id_list=None, start_sample_no_sil_list=None):
        '''
        Draw n speakers
        Draw one file name per speaker
        Load data of each file
        start_sample_no_sil_list=None: draw random starting index in the utterance
        '''

        if file_id_list is None:
            batch_speaker_id_list = self.dv_selecter.draw_n_speakers(self.dv_y_cfg.input_data_dim['S'])
            self.file_id_list = self.draw_n_files(batch_speaker_id_list, utter_tvt_name)
        else:
            self.file_id_list = file_id_list
            batch_speaker_id_list = self.dv_selecter.file_id_list_2_speaker_list(file_id_list)

        feed_dict = self.dv_y_data_loader.make_feed_dict(self.file_id_list, start_sample_no_sil_list=start_sample_no_sil_list)

        one_hot, one_hot_S, batch_size = self.dv_selecter.make_one_hot(batch_speaker_id_list)
        feed_dict['one_hot'] = one_hot
        feed_dict['one_hot_S'] = one_hot_S
        
        return feed_dict, batch_size

    def draw_n_files(self, batch_speaker_id_list, utter_tvt_name):
        '''
        Draw one file name per speaker        
        '''
        file_id_list = []
        for speaker_id in batch_speaker_id_list:
            speaker_file_id_list = self.dv_selecter.draw_n_files(speaker_id, utter_tvt_name, 1)
            file_id_list.extend(speaker_file_id_list)
        
        return file_id_list

class Build_Sinenet_Numpy(object):
    """
    Build_Sinenet_Numpy
    Numpy version of SineNet
    1. For debug
    2. For pre-processing
    """
    def __init__(self, params):
        super().__init__()
        self.params = params

        self.num_freq = self.params["layer_config"]['num_freq']
        self.win_len  = self.params["input_dim_values"]['T']
        self.k_space  = self.params["layer_config"]['k_space']

        self.t_wav = 1./16000

        self.k_2pi_tensor = self.make_k_2pi_tensor(self.num_freq, self.k_space) # K
        self.n_T_tensor   = self.make_n_T_tensor(self.win_len, self.t_wav)   # T


    def __call__(self, x, f, tau):
        return self.forward(x, f, tau)

    def forward(self, x, f, tau):
        sin_cos_matrix = self.construct_w_sin_cos_matrix(f, tau) # S*B*M*2K*T
        sin_cos_x = numpy.einsum('sbmkt,sbmt->sbmk', sin_cos_matrix, x) 
        return sin_cos_x

    def make_k_2pi_tensor(self, num_freq, k_space):
        ''' indices of frequency components '''
        k_vec = numpy.zeros(num_freq)
        for k in range(num_freq):
            k_vec[k] = k + 1
        k_vec = k_vec * 2 * numpy.pi * k_space
        # k_vec_tensor = torch.tensor(k_vec, dtype=torch.float, requires_grad=False)
        # k_vec_tensor = torch.nn.Parameter(k_vec_tensor, requires_grad=False)
        # return k_vec_tensor
        return k_vec

    def make_n_T_tensor(self, win_len, t_wav):
        ''' indices along time '''
        n_T_vec = numpy.zeros(win_len)
        for n in range(win_len):
            n_T_vec[n] = float(n) * t_wav
        # n_T_tensor = torch.tensor(n_T_vec, dtype=torch.float, requires_grad=False)
        # n_T_tensor = torch.nn.Parameter(n_T_tensor, requires_grad=False)
        # return n_T_tensor
        return n_T_vec

    def compute_deg(self, f, tau):
        ''' Return degree in radian '''
        # Time
        tau_1 = numpy.expand_dims(tau, 3) # S*B*M --> # S*B*M*1
        t = self.n_T_tensor - tau_1 # T + S*B*M*1 -> S*B*M*T

        # Degree in radian
        f_1 = numpy.expand_dims(f, 3) # S*B*M --> # S*B*M*1
        k_2pi_f = numpy.multiply(self.k_2pi_tensor, f_1) # K + S*B*M*1 -> S*B*M*K
        k_2pi_f_1 = numpy.expand_dims(k_2pi_f, 4) # S*B*M*K -> S*B*M*K*1
        t_1 = numpy.expand_dims(t, 3) # S*B*M*T -> S*B*M*1*T
        deg = numpy.multiply(k_2pi_f_1, t_1) # S*B*M*K*1, S*B*M*1*T -> S*B*M*K*T
        return deg

    def construct_w_sin_cos_matrix(self, f, tau):
        deg = self.compute_deg(f, tau) # S*B*M*K*T
        s   = numpy.sin(deg)             # S*B*M*K*T
        c   = numpy.cos(deg)             # S*B*M*K*T
        s_c = numpy.concatenate([s,c], axis=3)    # S*B*M*2K*T
        return s_c


class Build_dv_y_wav_data_loader_Ref(object):
    """
    This is a reference data loader 
    It can load only for one file
    The pitch method is very slow:
      search in pitch_list:
      if pitch_t >= start_t and pitch_t < end_t:
        tau = pitch_t - start_t
    The f0 method extract f0 in the middle (round down) of the window
    """
    def __init__(self, cfg=None, dv_y_cfg=None):
        super().__init__()
        self.logger = make_logger("Data_Loader Ref")

        self.cfg = cfg
        self.dv_y_cfg = dv_y_cfg
        self.DIO = Data_File_IO(cfg)

        self.load_seq_win_config(dv_y_cfg)
        # silence_data_meta_dict, load if necessary
        self.silence_data_meta_dict = None

    def load_seq_win_config(self, dv_y_cfg):
        '''
        Load from dv_y_cfg, or hard-code here
        '''
        self.input_data_dim = {}
        if dv_y_cfg is None:
            self.input_data_dim['T_S'] = 6000 # Number of samples at wav_sr
            self.input_data_dim['T_B']   = 3200 # T
            self.input_data_dim['B_stride'] = 80
            self.input_data_dim['T_M']   = 640
            self.input_data_dim['M_shift'] = 80
        else:
            self.input_data_dim['T_S'] = dv_y_cfg.input_data_dim['T_S']
            self.input_data_dim['T_B'] = dv_y_cfg.input_data_dim['T_B']
            self.input_data_dim['B_stride'] = dv_y_cfg.input_data_dim['B_stride']
            self.input_data_dim['T_M']   = dv_y_cfg.input_data_dim['T_M']
            self.input_data_dim['M_shift'] = dv_y_cfg.input_data_dim['M_shift']

        self.input_data_dim['B'] = int((self.input_data_dim['T_S'] - self.input_data_dim['T_B']) / self.input_data_dim['B_stride']) + 1
        self.input_data_dim['M'] = int((self.input_data_dim['T_B'] - self.input_data_dim['T_M']) / self.input_data_dim['M_shift']) + 1

        wav_sr  = self.cfg.wav_sr
        cmp_sr  = self.cfg.frame_sr
        wav_cmp_ratio = int(wav_sr / cmp_sr)
        # Do not use silence frames at the beginning or the end
        total_sil_one_side_cmp = self.cfg.frames_silence_to_keep + self.cfg.sil_pad   # This is at 200Hz
        self.total_sil_one_side_wav = total_sil_one_side_cmp * wav_cmp_ratio          # This is at wav_sr

    def make_feed_dict(self, file_id, start_sample_no_sil):
        '''
        1. Compute start_sample_include_sil
        (Optional, not used here) 2. Compute all win_start_t_list, win_end_t_list (B*M)
        3. Load x (use wav_resil_norm, start with start_sample_no_sil)
        4. Load pitch and vuv
        5. Load f0
        '''
        feed_dict = {}

        wav_resil_norm_file_name = os.path.join(self.cfg.nn_feat_resil_norm_dirs['wav'], file_id+'.wav')
        wav_T = self.make_wav_T(wav_resil_norm_file_name, start_sample_no_sil)
        feed_dict['wav_T'] = wav_T

        start_sample_include_sil = self.compute_start_sample_include_sil(file_id, start_sample_no_sil)

        pitch_text_file_name = os.path.join(self.cfg.pitch_dir, file_id+'.pitch')
        tau_BM, vuv_BM = self.make_tau_BM(pitch_text_file_name, start_sample_include_sil)
        feed_dict['tau_BM'] = tau_BM
        feed_dict['vuv_BM'] = vuv_BM

        f0_16k_file_name = os.path.join(self.cfg.nn_feat_dirs['f016k'], file_id+'.f016k')
        f_BM = self.make_f_BM(f0_16k_file_name, start_sample_include_sil)
        feed_dict['f_BM'] = f_BM

        return feed_dict

    def make_wav_T(self, wav_resil_norm_file_name, start_sample_no_sil):
        '''
        Waveform data is S*T; window slicing is done in Pytorch
        '''
        wav_data, sample_number = self.DIO.load_data_file_frame(wav_resil_norm_file_name, 1)
        wav_data = numpy.squeeze(wav_data)

        start_sample_include_sil = start_sample_no_sil+self.total_sil_one_side_wav

        wav_T = wav_data[start_sample_include_sil:start_sample_include_sil+self.input_data_dim['T_S']]

        return wav_T

    def compute_start_sample_include_sil(self, file_id, start_sample_no_sil):
        '''
        Find start of silence, convert from frame to sample, then add
        '''
        if self.silence_data_meta_dict is None:
            data_meta_file_name='/home/dawna/tts/mw545/TorchDV/file_id_lists/data_meta/file_id_list_num_sil_frame.scp'
            from frontend_mw545.data_io import Data_Meta_List_File_IO
            DMLIO = Data_Meta_List_File_IO(self.cfg)
            self.silence_data_meta_dict = DMLIO.read_file_list_num_silence_frame(data_meta_file_name)

        num_frame_wav_cmp, first_non_sil_index, last_non_sil_index = self.silence_data_meta_dict[file_id]
        wav_sr  = self.cfg.wav_sr
        cmp_sr  = self.cfg.frame_sr
        wav_cmp_ratio = int(wav_sr / cmp_sr)
        start_sample_include_sil = start_sample_no_sil + first_non_sil_index * wav_cmp_ratio
        return start_sample_include_sil
    

    def make_tau_BM(self, pitch_text_file_name, start_sample_include_sil):
        '''
        vuv=1. if pitch found inside the window; vuv=0. otherwise
        tau=pitch_t - win_start_t; 0 <= tau < win_T
        '''
        wav_sr = float(self.cfg.wav_sr)
        tau_matrix = numpy.zeros((self.input_data_dim['B'], self.input_data_dim['M']))
        vuv_matrix = numpy.zeros((self.input_data_dim['B'], self.input_data_dim['M']))

        pitch_t_list = self.DIO.read_pitch_reaper(pitch_text_file_name)

        for b in range(self.input_data_dim['B']):
            for m in range(self.input_data_dim['M']):
                win_start_n = start_sample_include_sil + b * self.input_data_dim['B_stride'] + m * self.input_data_dim['M_shift']
                win_start_t = float(win_start_n) / wav_sr
                win_end_t   = float(win_start_n + self.input_data_dim['T_M']) / wav_sr

                for pitch_t in pitch_t_list:
                    if pitch_t >= win_start_t:
                        if pitch_t < win_end_t:
                            tau_matrix[b,m] = pitch_t - win_start_t
                            vuv_matrix[b,m] = 1.
                            break
                        else:
                            # No pitch found inside window, break for loop
                            break

        return tau_matrix, vuv_matrix

    def make_f_BM(self, f0_16k_file_name, start_sample_include_sil):
        '''
        Extract f0 in the middle (round down) of the window
        '''
        wav_sr = float(self.cfg.wav_sr)
        f0_matrix = numpy.zeros((self.input_data_dim['B'], self.input_data_dim['M']))

        f0_16k_data, sample_number = self.DIO.load_data_file_frame(f0_16k_file_name, 1)

        for b in range(self.input_data_dim['B']):
            for m in range(self.input_data_dim['M']):
                win_start_n = start_sample_include_sil + b * self.input_data_dim['B_stride'] + m * self.input_data_dim['M_shift']
                win_mid_n   = int(win_start_n + self.input_data_dim['T_M'] / 2)
                f0_matrix[b,m] = f0_16k_data[win_mid_n, 0]

        return f0_matrix

class Build_dv_y_wav_data_loader_Single_File(object):
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
        if dv_y_cfg is None:
            self.input_data_dim = {'T_S': 6000, 'T_B': 3200, 'B_stride': 80, 'T_M': 640, 'M_shift': 80}
            self.input_data_dim['B'] = int((self.input_data_dim['T_S'] - self.input_data_dim['T_B']) / self.input_data_dim['B_stride']) + 1
            self.input_data_dim['M'] = int((self.input_data_dim['T_B'] - self.input_data_dim['T_M']) / self.input_data_dim['M_shift']) + 1
            self.out_feat_list = ['wav_ST', 'f_SBM', 'tau_SBM', 'vuv_SBM']
        else:
            self.input_data_dim = dv_y_cfg.input_data_dim
            self.out_feat_list = dv_y_cfg.out_feat_list

        self.win_T = float(self.input_data_dim['T_M']) / float(self.cfg.wav_sr)

        wav_sr  = self.cfg.wav_sr
        cmp_sr  = self.cfg.frame_sr
        wav_cmp_ratio = int(wav_sr / cmp_sr)
        # Do not use silence frames at the beginning or the end
        total_sil_one_side_cmp = self.cfg.frames_silence_to_keep + self.cfg.sil_pad   # This is at frame_sr
        self.total_sil_one_side_wav = total_sil_one_side_cmp * wav_cmp_ratio          # This is at wav_sr

    def init_n_matrix(self):
        '''
        Make matrices of index, B*M
        '''
        B_M_grid = numpy.mgrid[0:self.input_data_dim['B'],0:self.input_data_dim['M']]
        start_n_matrix = self.input_data_dim['B_stride'] * B_M_grid[0] + self.input_data_dim['M_shift'] * B_M_grid[1]
        self.start_n_matrix = start_n_matrix.astype(int)
        mid_n_matrix   = start_n_matrix + self.input_data_dim['T_M'] / 2
        self.mid_n_matrix = mid_n_matrix.astype(int)

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
        return self.make_wav_T_from_data(self, start_sample_no_sil)

    def make_wav_T_from_data(self, wav_data, start_sample_no_sil):
        start_sample_include_sil = start_sample_no_sil+self.total_sil_one_side_wav
        wav_T = wav_data[start_sample_include_sil:start_sample_include_sil+self.input_data_dim['T_S']]

        return wav_T

    def make_tau_BM(self, pitch_resil_norm_file_name, start_sample_no_sil):
        '''
        B * M
        vuv=1. if pitch found inside the window; vuv=0. otherwise
        pitch found: 0 <= tau < win_T
        '''
        pitch_16k_data, sample_number = self.DIO.load_data_file_frame(pitch_resil_norm_file_name, 1)
        pitch_16k_data = numpy.squeeze(pitch_16k_data)
        return self.make_tau_BM_from_data(pitch_16k_data, start_sample_no_sil)

    def make_tau_BM_from_data(self, pitch_16k_data, start_sample_no_sil):

        start_sample_include_sil = start_sample_no_sil+self.total_sil_one_side_wav
        pitch_T = pitch_16k_data[start_sample_include_sil:start_sample_include_sil+self.input_data_dim['T_S']]

        tau_BM = pitch_T[self.start_n_matrix]

        vuv_BM = numpy.ones((self.input_data_dim['B'], self.input_data_dim['M']))
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
        return self.make_f_BM_from_data(f0_16k_data, start_sample_no_sil)

    def make_f_BM_from_data(self, f0_16k_data, start_sample_no_sil):
        start_sample_include_sil = start_sample_no_sil+self.total_sil_one_side_wav
        f0_T = f0_16k_data[start_sample_include_sil:start_sample_include_sil+self.input_data_dim['T_S']]

        f_BM = f0_T[self.mid_n_matrix]
        # mid_n_matrix = self.mid_n_matrix + start_sample_no_sil
        # f_BM = f0_16k_data[mid_n_matrix.astype(int)]
        return f_BM
