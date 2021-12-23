# data_loader.py

import os, sys, pickle, time, shutil, logging, copy
import math, numpy
numpy.random.seed(547)

from frontend_mw545.modules import make_logger, File_List_Selecter, List_Random_Loader
# from frontend_mw545.modules import keep_by_speaker, remove_by_speaker, keep_by_file_number, remove_by_file_number

from frontend_mw545.data_io import Data_File_IO, Data_List_File_IO, Data_Meta_List_File_IO

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
        self.DMLIO = Data_Meta_List_File_IO(cfg)
        self.FLS = File_List_Selecter()

        self.speaker_id_list = self.cfg.speaker_id_list_dict['train']
        self.speaker_random_loader = List_Random_Loader(self.speaker_id_list)

        self.file_id_list = self.DLIO.read_file_list(self.cfg.file_id_list_file['dv_enough'])
        self.file_list_dict = self.make_dv_file_list_dict()
        self.file_list_random_loader_dict = self.make_dv_file_list_random_loader()

        self.file_num_non_sil_frame_dict = self.make_file_num_non_sil_frame_dict()

    def make_file_num_non_sil_frame_dict(self):
        cfg = self.cfg
        DMLIO = Data_Meta_List_File_IO(cfg)
        in_file_name = os.path.join(cfg.file_id_list_dir, 'data_meta/file_id_list_num_sil_frame.scp')
        file_frame_dict = DMLIO.read_file_list_num_silence_frame(in_file_name) # [num_frame_wav_cmp, first_non_sil_index, last_non_sil_index]
        file_num_non_sil_frame_dict = {}
        for k in file_frame_dict:
            # resil files contain pads at both ends
            v = file_frame_dict[k][2] - file_frame_dict[k][1] + 1 + 2 * (cfg.frames_silence_to_keep + cfg.sil_pad)
            file_num_non_sil_frame_dict[k] = v
        return file_num_non_sil_frame_dict

    def make_dv_file_list_dict(self):
        ''' 
        Make a dict for file lists, In the form of: file_list_dict[(speaker_id, 'train')] 
        1. Only train speakers
        2. split in 3 parts, dv_train, dv_valid, dv_test
        '''
        file_list_dict = self.FLS.sort_by_speaker_list(self.file_id_list, self.speaker_id_list)

        for speaker_id in self.speaker_id_list:
            speaker_file_dict = self.FLS.sort_by_file_number(file_list_dict[speaker_id], self.dv_y_cfg.data_split_file_number)
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
        file_id_list = self.file_list_random_loader_dict[(speaker_id,utter_tvt_name)].draw_n_samples(n)
        file_id_str = '|'.join(file_id_list)
        return file_id_str

    def draw_n_seconds(self, speaker_id, utter_tvt_name, num_seconds):
        '''
        Output: string, "file_id_1|file_id_2|..."
        Draw files from list, no more than n seconds
        1. Draw at least 1 file
        2. Stop when the new file exceeds; do not include the new file
        '''
        n_draw = 0
        n_frames_max = num_seconds * self.cfg.frame_sr
        file_id_list = []
        while n_draw < n_frames_max:
            file_id = self.file_list_random_loader_dict[(speaker_id,utter_tvt_name)].draw_n_samples(1)[0]
            n_draw += self.file_num_non_sil_frame_dict[file_id]
            if n_draw <= n_frames_max:
                file_id_list.append(file_id)
        if len(file_id_list) == 0:
            file_id_list = self.file_list_random_loader_dict[(speaker_id,utter_tvt_name)].draw_n_samples(1)
        file_id_str = '|'.join(file_id_list)
        return file_id_str

    def make_one_hot(self, speaker_id_list, out_len_max, ref_speaker_id_list=None):
        dv_y_cfg = self.dv_y_cfg
        if ref_speaker_id_list is None:
            ref_speaker_id_list = self.cfg.speaker_id_list_dict['train']

        S = len(speaker_id_list)
        one_hot_S = numpy.zeros((S))
        # Make classification targets, index sequence
        for i in range(S):
            speaker_id = speaker_id_list[i]
            try:
                true_speaker_index = ref_speaker_id_list.index(speaker_id)
            except ValueError: # New speaker, not in the training set
                true_speaker_index = -1
            one_hot_S[i] = true_speaker_index

        one_hot_S_B = numpy.repeat(one_hot_S, out_len_max).reshape(-1, out_len_max)
        if dv_y_cfg.train_by_window:
            # S --> S*B
            one_hot = one_hot_S_B.reshape(-1)
            batch_size = dv_y_cfg.input_data_dim['S'] * out_len_max
        else:
            one_hot = one_hot_S
            batch_size = dv_y_cfg.input_data_dim['S']

        return one_hot, one_hot_S, one_hot_S_B

    def file_id_list_2_speaker_list(self, file_id_list):
        '''
        extract speaker_id from file_id
        '''
        speaker_list = []
        for file_id in file_id_list:
            speaker_id = file_id.split('_')[0]
            speaker_list.append(speaker_id)
        return speaker_list

class Build_dv_TTS_selecter(Build_dv_selecter):
    '''
    This selecter is for drawing speaker representation files for TTS training
    The formats are slightly different, file_list_random_loader_dict[(speaker_id, 'SR')]
    No more 'train_valid_test' split
    '''
    def __init__(self, cfg=None, dv_y_cfg=None):
        self.logger = make_logger("DV TTS Selecter")

        self.cfg = cfg
        self.dv_y_cfg = dv_y_cfg

        self.DLIO = Data_List_File_IO(cfg)
        self.DMLIO = Data_Meta_List_File_IO(cfg)
        self.FLS = File_List_Selecter()

        self.speaker_id_list_dict = self.cfg.speaker_id_list_dict

        self.speaker_id_list = self.cfg.speaker_id_list_dict['all']
        self.file_list_dict = self.make_dv_file_list_dict()
        self.file_list_random_loader_dict = self.make_dv_file_list_random_loader()

        self.file_num_non_sil_frame_dict = self.make_file_num_non_sil_frame_dict()

    def make_dv_file_list_dict(self):
        ''' 
        Make a dict for file lists, In the form of: file_list_dict[(speaker_id, 'SR')] 
        1. all speaker_id has 'SR'
        2. split in 3 parts, dv_train, dv_valid, dv_test
        '''
        self.file_id_list_dir = os.path.join(self.cfg.file_id_list_dir, 'train_valid_test_SR')
        # /data/vectra2/tts/mw545/Data/exp_dirs/data_voicebank_24kHz/file_id_lists/train_valid_test_SR
        file_list_dict = {}

        for utter_tvt_name in ['train', 'valid', 'test']:
            file_id_list_file = os.path.join(self.file_id_list_dir, '%s_SR.scp' % utter_tvt_name)
            file_id_list = self.DLIO.read_file_list(file_id_list_file)
            speaker_id_list = self.cfg.speaker_id_list_dict[utter_tvt_name]
            file_list_dict.update(self.FLS.sort_by_speaker_list(file_id_list, speaker_id_list))

            for speaker_id in speaker_id_list:
                file_list_dict[(speaker_id, 'SR')] = file_list_dict[speaker_id]

        return file_list_dict

    def make_dv_file_list_random_loader(self):
        ''' Make a dict for List_Random_Loader, In the form of: file_list_random_loader_dict[(speaker_id, 'train')] '''
        file_list_random_loader_dict = {}
        
        for speaker_id in self.speaker_id_list:
            file_list_random_loader_dict[(speaker_id, 'SR')] = List_Random_Loader(self.file_list_dict[(speaker_id, 'SR')])

        return file_list_random_loader_dict

#############################
# Methods for loading data #
#############################

class Build_BD_data_loader_Multi_Speaker_Base(object):
    """docstring for Build_BD_data_loader_Multi_Speaker"""
    def __init__(self, cfg, train_cfg):
        super().__init__()
        self.logger = make_logger("Data_Loader Multi")

        self.cfg = cfg
        self.train_cfg = train_cfg
        self.DIO = Data_File_IO(cfg)

        self.init_feed_dict()
        self.init_directories()

    def init_feed_dict(self):
        train_cfg = self.train_cfg
        self.feed_dict = {}
        self.feed_dict['in_lens'] = numpy.zeros(train_cfg.input_data_dim['S'], dtype=int)
        self.feed_dict['out_lens'] = numpy.zeros(train_cfg.input_data_dim['S'], dtype=int)

    def init_directories(self):
        pass

    def make_BD_data_single_file_id(self, file_id, start_sample_number):
        # Main method to write per loader
        pass

    def make_BD_data(self, file_id_str, start_sample_number=None):
        # Possible multiple files, joined by '|'
        if '|' not in file_id_str:
            # Single file
            file_id = file_id_str
            BD_data, B, start_sample_number = self.make_BD_data_single_file_id(file_id, start_sample_number)
        else:
            # Multiple files
            file_id_list_temp = file_id.split('|')
            B = 0
            BD_data_list_f = []
            for file_id in file_id_list_temp:
                BD_data_f, B_f, start_sample_number = self.make_BD_data_single_file_id(file_id, start_sample_number)
                B += B_f
                BD_data_list_f.append(BD_data_f)
            BD_data = numpy.concatenate(BD_data_list_f)

        return BD_data, B, start_sample_number

    def make_feed_dict(self, file_id_list, start_sample_list=None):
        train_cfg = self.train_cfg

        if start_sample_list is None:
            start_sample_list = [None] * train_cfg.input_data_dim['S']

        BD_data_list = []
        for i, file_id_str in enumerate(file_id_list):
            BD_data, B, start_sample_number = self.make_BD_data(file_id_str, start_sample_list[i])
            BD_data_list.append(BD_data)
            self.feed_dict['in_lens'][i] = B

        B_in_max = numpy.max(self.feed_dict['in_lens'])
        self.feed_dict['h'] = numpy.zeros((dv_y_cfg.input_data_dim['S'], B_in_max, dv_y_cfg.input_data_dim['D']))
        for i, BD_data in enumerate(BD_data_list):
            self.feed_dict['h'][i,:self.feed_dict['in_lens'][i],:] = cmp_data

        self.feed_dict['out_lens'] = self.feed_dict['in_lens']

        return self.feed_dict
        

class Build_dv_y_wav_data_loader_Multi_Speaker(object):
    """
    """
    def __init__(self, cfg=None, dv_y_cfg=None):
        super(Build_dv_y_wav_data_loader_Multi_Speaker, self).__init__()
        self.logger = make_logger("Data_Loader Multi")

        self.cfg = cfg
        self.dv_y_cfg = dv_y_cfg
        self.DIO = Data_File_IO(cfg)

        self.init_feed_dict()
        self.init_directories()

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
        self.feed_dict['in_lens'] = numpy.zeros(dv_y_cfg.input_data_dim['S'], dtype=int)
        self.feed_dict['out_lens'] = numpy.zeros(dv_y_cfg.input_data_dim['S'], dtype=int)

        self.data_list = {k: [] for k in dv_y_cfg.out_feat_list}

    def reset_data_list(self):
        for k in self.data_list:
            self.data_list[k] = []

        '''
        S = dv_y_cfg.input_data_dim['S']
        B = dv_y_cfg.input_data_dim['B']
        M = dv_y_cfg.input_data_dim['M']

        if 'wav_ST' in dv_y_cfg.out_feat_list:
            self.feed_dict['wav_ST'] = numpy.zeros((S, dv_y_cfg.input_data_dim['T_S']))
        if 'wav_SBT' in dv_y_cfg.out_feat_list:
            self.feed_dict['wav_SBT'] = numpy.zeros((S, B, dv_y_cfg.input_data_dim['T_B']))
            B_T_grid = numpy.mgrid[0:B,0:self.dv_y_cfg.input_data_dim['T_B']]
            B_T_matrix = self.dv_y_cfg.input_data_dim['B_stride'] * B_T_grid[0] + B_T_grid[1]
            self.B_T_matrix = B_T_matrix.astype(int)
        if 'wav_SBMT' in dv_y_cfg.out_feat_list:
            self.feed_dict['wav_SBMT'] = numpy.zeros((S, B, M, dv_y_cfg.input_data_dim['T_M']))
            B_M_T_grid = numpy.mgrid[0:B,0:M,0:self.dv_y_cfg.input_data_dim['T_M']]
            B_M_T_matrix = self.dv_y_cfg.input_data_dim['B_stride'] * B_M_T_grid[0] + self.dv_y_cfg.input_data_dim['M_shift'] * B_M_T_grid[1] + B_M_T_grid[2]
            self.B_M_T_matrix = B_M_T_matrix.astype(int)

        if ('tau_SBM' in dv_y_cfg.out_feat_list) or ('vuv_SBM' in dv_y_cfg.out_feat_list):
            self.feed_dict['tau_SBM'] = numpy.zeros((S,B,M))
            self.feed_dict['vuv_SBM'] = numpy.zeros((S,B,M))
        if 'f_SBM' in dv_y_cfg.out_feat_list:
            self.feed_dict['f_SBM'] = numpy.zeros((S,B,M))

        '''

    def modify_wav_data(self, wav_data, sample_number, start_sample_number, max_len, win_len):
        # modify the length of cmp data according to start_sample_number (if given). maximum length, and window length
        # 1. The final length does not exceed max_len
        # 2. Use start_sample_number if given
        # 3. If not, start with a random integer [0, win_len-1]

        if start_sample_number is not None:
            # If given, start here, and use up to max_len
            sample_number = sample_number - start_sample_number
            if sample_number > max_len:
                sample_number = max_len
                wav_data = wav_data[start_sample_number:start_sample_number+max_len]
            else:
                wav_data = wav_data[start_sample_number:]
        else:
            if sample_number > max_len:
                # Use up to max_len, with random starting position
                extra_file_len = sample_number - max_len
                start_sample_number = int(numpy.random.rand() * extra_file_len)
                sample_number = max_len
                wav_data = wav_data[start_sample_number:start_sample_number+self.max_len]
            else:
                # start with a random integer [0, win_len-1]
                start_sample_number = int(numpy.random.rand() * win_len)
                sample_number = sample_number - start_sample_number
                wav_data = wav_data[start_sample_number:]


        return wav_data, sample_number, start_sample_number

    def compute_B(self, in_lens):
        # Compute lengths after CNN
        # This is output lengths
        # given kernel size, assume shift=sride=1

        kernel_size = self.dv_y_cfg.input_data_dim['T_B']
        kernel_stride = self.dv_y_cfg.input_data_dim['B_stride']
        # kernel_size = self.dv_y_cfg.kernel_size
        out_lens = int((in_lens-kernel_size)/kernel_stride) + 1
        return out_lens

    def make_wav_BT(self, wav_data, sample_number=None):
        wav_data = numpy.squeeze(wav_data)
        if sample_number is None:
            sample_number = wav_data.shape[0]
        B = self.compute_B(sample_number)

        dv_y_cfg = self.dv_y_cfg
        wav_BT_data = numpy.zeros((B, dv_y_cfg.input_data_dim['T_B']))

        kernel_size = self.dv_y_cfg.input_data_dim['T_B']
        kernel_stride = self.dv_y_cfg.input_data_dim['B_stride']
        for b in range(B):
            i_start = b * kernel_stride
            i_end   = i_start + kernel_size
            wav_BT_data[b] = wav_data[i_start:i_end]

        return wav_BT_data, B

    def make_BD_data_single_file_id(self, file_id, start_sample_number):
        speaker_id = file_id.split('_')[0]
        wav_resil_norm_file_name = os.path.join(self.wav_dir, speaker_id, file_id+'.wav')
        wav_data, sample_number = self.DIO.load_data_file_frame(wav_resil_norm_file_name, 1)

        new_wav_data, new_sample_number, start_sample_number = self.modify_wav_data(wav_data, sample_number, start_sample_number, dv_y_cfg.input_data_dim['T_S_max'], dv_y_cfg.input_data_dim['T_B'])
        wav_BT, B = self.make_wav_BT(new_wav_data, new_sample_number)

        return wav_BT, B, start_sample_number

        # cmp_BD_data, B, start_sample_number = self.make_cmp_BD_data_single_file_id(file_id, start_sample_number)
        
    def make_feed_dict(self, file_id_list, start_sample_list=None):
        '''
        1. Load waveform
            Default: wav_SBT
        2. Load pitch and vuv
        3. Load f0 
        '''
        dv_y_cfg = self.dv_y_cfg
        
        if start_sample_list is None:
            start_sample_list = [None] * dv_y_cfg.input_data_dim['S']

        sr_k = int(self.cfg.wav_sr / 1000)

        self.reset_data_list()
        for i, file_id in enumerate(file_id_list):
            wav_BT, B, start_sample_number = self.make_BD_data_single_file_id(file_id, start_sample_list[i])
            self.data_list['wav_SBT'].append(wav_BT)
            self.feed_dict['in_lens'][i] = B

        B_in_max = numpy.max(self.feed_dict['in_lens'])
        self.feed_dict['wav_SBT'] = numpy.zeros((dv_y_cfg.input_data_dim['S'], B_in_max, dv_y_cfg.input_data_dim['T_B']))
        for i, wav_BT in enumerate(self.data_list['wav_SBT']):
            self.feed_dict['wav_SBT'][i,:self.feed_dict['in_lens'][i],:] = wav_BT

        self.feed_dict['out_lens'] = self.feed_dict['in_lens']
        return self.feed_dict



        '''
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
        '''

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

class Build_dv_y_cmp_data_loader_Multi_Speaker(object):
    """
    Output: feed_dict['h'], STD; feed_dict['h_lens'], S
    Pad and stack T*D features from each file
    """
    def __init__(self, cfg=None, dv_y_cfg=None):
        super().__init__()
        self.logger = make_logger("Data_Loader Multi")

        self.cfg = cfg
        self.dv_y_cfg = dv_y_cfg
        self.DIO = Data_File_IO(cfg)

        self.max_len = dv_y_cfg.input_data_dim['T_S_max']
        self.cfg_cmp_dim = self.cfg.nn_feature_dims['cmp']

        self.init_feed_dict()
        self.init_directories()

    def init_directories(self):
        if self.dv_y_cfg.data_dir_mode == 'scratch':
            self.cmp_dir = self.cfg.nn_feat_scratch_dirs['cmp']
        elif self.dv_y_cfg.data_dir_mode == 'data':
            self.cmp_dir = self.cfg.nn_feat_resil_norm_dirs['cmp']

    def init_feed_dict(self):
        dv_y_cfg = self.dv_y_cfg
        self.feed_dict = {}
        self.feed_dict['in_lens'] = numpy.zeros(dv_y_cfg.input_data_dim['S'], dtype=int)
        self.feed_dict['out_lens'] = numpy.zeros(dv_y_cfg.input_data_dim['S'], dtype=int)
        # self.feed_dict['h'] = numpy.zeros((dv_y_cfg.input_data_dim['S'], dv_y_cfg.input_data_dim['B'], dv_y_cfg.input_data_dim['D']))

    def modify_cmp_data(self, cmp_data, sample_number, start_sample_number, max_len):
        # modify the length of cmp data according to start_sample_number (if given) and maximum length
        if start_sample_number is not None:
            # If given, start here, and use up to max_len
            sample_number = sample_number - start_sample_number
            if sample_number > max_len:
                sample_number = max_len
                cmp_data = cmp_data[start_sample_number:start_sample_number+max_len]
            else:
                cmp_data = cmp_data[start_sample_number:]
        else:
            if sample_number > max_len:
                # Use up to max_len, with random starting position
                extra_file_len = sample_number - max_len
                start_sample_number = int(numpy.random.rand() * (extra_file_len+1))
                sample_number = max_len
                cmp_data = cmp_data[start_sample_number:start_sample_number+self.max_len]
            else:
                start_sample_number = 0

        return cmp_data, sample_number, start_sample_number

    def compute_B(self, in_lens):
        # Compute lengths after CNN
        # This is output lengths
        # given kernel size, assume shift=sride=1

        kernel_size = self.dv_y_cfg.input_data_dim['T_B']
        # kernel_size = self.dv_y_cfg.kernel_size
        out_lens = in_lens - kernel_size + 1
        return out_lens

    def make_BD_data(self, cmp_data, sample_number=None):
        if sample_number is None:
            sample_number = cmp_data.shape[0]
        B = self.compute_B(sample_number)

        dv_y_cfg = self.dv_y_cfg
        cmp_BD_data = numpy.zeros((B, dv_y_cfg.input_data_dim['D']))

        for b in range(B):
            cmp_TD  = cmp_data[b:b+dv_y_cfg.input_data_dim['T_B']]
            cmp_D   = cmp_TD.reshape(-1)
            cmp_BD_data[b] = cmp_D

        return cmp_BD_data, B

    def make_cmp_BD_data_single_file_id(self, file_id, start_sample_number=None):
        speaker_id = file_id.split('_')[0]
        cmp_resil_norm_file_name = os.path.join(self.cmp_dir, speaker_id, file_id+'.cmp')
        return self.make_cmp_BD_data_single_file(cmp_resil_norm_file_name, start_sample_number)

    def make_cmp_BD_data_single_file(self, cmp_resil_norm_file_name, start_sample_number=None):
        cmp_data, sample_number = self.DIO.load_data_file_frame(cmp_resil_norm_file_name, self.cfg_cmp_dim)
        new_cmp_data, new_sample_number, start_sample_number = self.modify_cmp_data(cmp_data, sample_number, start_sample_number, self.max_len)
        cmp_BD_data, B = self.make_BD_data(new_cmp_data, new_sample_number)
        return cmp_BD_data, B, start_sample_number

    def make_cmp_BD_data(self, file_id, start_sample_number=None):
        # Possible multiple files, joined by '|'
        if '|' not in file_id:
            # Single file
            cmp_BD_data, B, start_sample_number = self.make_cmp_BD_data_single_file_id(file_id, start_sample_number)
        else:
            # Multiple files
            file_id_list_temp = file_id.split('|')
            B = 0
            cmp_data_list_f = []
            for f in file_id_list_temp:
                cmp_BD_data_f, B_f, start_sample_number = self.make_cmp_BD_data_single_file_id(f, start_sample_number)
                B += B_f
                cmp_data_list_f.append(cmp_BD_data_f)
            cmp_BD_data = numpy.concatenate(cmp_data_list_f)

        return cmp_BD_data, B, start_sample_number


    def make_feed_dict(self, file_id_list, start_sample_list=None):
        dv_y_cfg = self.dv_y_cfg

        if start_sample_list is None:
            start_sample_list = [None] * dv_y_cfg.input_data_dim['S']

        cmp_data_list = []
        for i, file_id in enumerate(file_id_list):
            cmp_BD_data, B, start_sample_number = self.make_cmp_BD_data(file_id, start_sample_list[i])
            cmp_data_list.append(cmp_BD_data)
            self.feed_dict['in_lens'][i] = B

        B_in_max = numpy.max(self.feed_dict['in_lens'])
        self.feed_dict['h'] = numpy.zeros((dv_y_cfg.input_data_dim['S'], B_in_max, dv_y_cfg.input_data_dim['D']))
        for i, cmp_data in enumerate(cmp_data_list):
            self.feed_dict['h'][i,:self.feed_dict['in_lens'][i],:] = cmp_data

        self.feed_dict['out_lens'] = self.feed_dict['in_lens']

        return self.feed_dict


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
    def __init__(self, cfg=None, dv_y_cfg=None, dv_selecter_type='DV'):
        super(Build_dv_y_train_data_loader, self).__init__()
        self.logger = make_logger("Data_Loader Train")

        self.cfg = cfg
        self.dv_y_cfg = dv_y_cfg

        if self.dv_y_cfg.data_loader_random_seed > 0:
            numpy.random.seed(547+self.dv_y_cfg.data_loader_random_seed)

        if dv_selecter_type == 'DV':
            self.dv_selecter = Build_dv_selecter(cfg, dv_y_cfg)
        elif dv_selecter_type == 'TTS':
            self.dv_selecter = Build_dv_TTS_selecter(cfg, dv_y_cfg)

        if dv_y_cfg.y_feat_name == 'wav':
            self.init_wav()
        elif dv_y_cfg.y_feat_name == 'cmp':
            self.init_cmp()

    def init_wav(self):
        self.dv_y_data_loader = Build_dv_y_wav_data_loader_Multi_Speaker(self.cfg, self.dv_y_cfg)

    def init_cmp(self):
        self.dv_y_data_loader = Build_dv_y_cmp_data_loader_Multi_Speaker(self.cfg, self.dv_y_cfg)

    def make_feed_dict(self, utter_tvt_name='train', file_id_list=None, start_sample_list=None):
        if file_id_list is None:
            # Draw n speakers
            batch_speaker_id_list = self.dv_selecter.draw_n_speakers(self.dv_y_cfg.input_data_dim['S'])
            if self.dv_y_cfg.train_num_seconds == 0:
                # Use 1 file
                self.file_id_list = self.draw_n_files(batch_speaker_id_list, utter_tvt_name, n=1)
            else:
                # Each "file_id" is a string, file_ids joined by '|'
                self.file_id_list = self.draw_n_seconds(batch_speaker_id_list, utter_tvt_name, n=self.dv_y_cfg.train_num_seconds)
        else:
            self.file_id_list = file_id_list
            batch_speaker_id_list = self.dv_selecter.file_id_list_2_speaker_list(file_id_list)

        feed_dict = self.dv_y_data_loader.make_feed_dict(self.file_id_list, start_sample_list=start_sample_list)

        out_lens = feed_dict['out_lens']
        out_len_max = numpy.max(out_lens)
        batch_size = numpy.sum(out_lens)
        feed_dict['output_mask_S_B'] = self.make_output_lens_mask(out_lens)

        one_hot, one_hot_S, one_hot_S_B = self.dv_selecter.make_one_hot(batch_speaker_id_list, out_len_max)
        feed_dict['one_hot'] = one_hot
        feed_dict['one_hot_S'] = one_hot_S
        feed_dict['one_hot_S_B'] = one_hot_S_B
        
        return feed_dict, batch_size

    def draw_n_files(self, batch_speaker_id_list, utter_tvt_name, n=1):
        '''
        Draw one file name per speaker
        '''
        file_id_list = []
        for speaker_id in batch_speaker_id_list:
            speaker_file_id_str = self.dv_selecter.draw_n_files(speaker_id, utter_tvt_name, n)
            file_id_list.append(speaker_file_id_str)
        
        return file_id_list

    def draw_n_seconds(self, batch_speaker_id_list, utter_tvt_name, n=1):
        '''
        Draw n seconds per speaker
        Each 'file_id' is a string of file_ids joined by '|'
        '''
        file_id_list = []
        for speaker_id in batch_speaker_id_list:
            speaker_file_id_str = self.dv_selecter.draw_n_seconds(speaker_id, utter_tvt_name, n)
            file_id_list.append(speaker_file_id_str)
        
        return file_id_list

    def make_output_lens_mask(self, out_lens):
        # Make mask and weight matrices based on output lengths
        out_len_max = numpy.max(out_lens)
        S = out_lens.shape[0]

        output_mask_SB = numpy.zeros((S, out_len_max))
        for i in range(S):
            output_mask_SB[i][:out_lens[i]] = 1.

        return output_mask_SB


###########
# Others? #
###########

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

