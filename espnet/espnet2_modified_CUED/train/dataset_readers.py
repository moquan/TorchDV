import collections.abc
from pathlib import Path
from typing import Union

import os, sys, numpy, pickle
from typeguard import check_argument_types

from espnet2.fileio.read_text import read_2column_text

sys.path.append("/data/vectra2/tts/mw545/TorchTTS/espnet_py36/espnet/espnet2_modified_CUED/merlin_cued_mw545")
from config_24kHz import configuration
from frontend_mw545.modules import File_List_Selecter, List_Random_Loader, read_file_list
from frontend_mw545.data_io import Data_File_IO

from frontend_mw545.data_loader import Build_dv_TTS_selecter

        
class data_reader_base(collections.abc.Mapping):
    """
    Abstract Reader class for a scp file
    """

    def __init__(self, fname: Union[Path, str], loader_type: str):
        assert check_argument_types()
        self.fname = Path(fname)
        self.data = read_2column_text(fname)

        self.cfg = configuration(cache_files=False)
        self.dv_tts_selector = Build_dv_TTS_selecter(self.cfg, dv_y_cfg=None)

        # Window shift threshold: 
        # if actual window shift is larger, use a random draw start_sample_number
        self.window_shift_threshold = 120
        self.window_shift = 120

    def __getitem__(self, key) -> numpy.ndarray:
        # This is the main method to write for all data readers
        p = self.data[key]
        return self.input_string_handler(p)

    def input_string_handler(self, p):
        '''
        Possible string formats:
        1. speaker_id (mostly unused)
            p001_001 p001
        2. file_id
            p001_001 p001_060
        3. one string, some rules; contains 3 parts when split by '_'
            p001_001 p001_5_seconds
        4. file_ids, split by ' '
            p001_001 p001_060 p001_064 p001_065
        return:
            BD data
        '''
        file_list = p.split(' ')
        if len(file_list) == 1:
            num_p_parts = len(p.split('_'))
            if num_p_parts == 1:
                # e.g. p001, use speaker method
                return self.speaker_id_handler(p)
            elif num_p_parts == 2:
                # e.g. p001_002, use file embedding
                return self.file_id_str_handler(p)
            elif num_p_parts == 3:
                # e.g. p001_5_seconds
                file_id_str = self.draw_data_list(p)
                return self.file_id_str_handler(file_id_str)
        else:
            file_id_str = '|'.join(file_list)
            return self.file_id_str_handler(file_id_str)

    def speaker_id_handler(self, speaker_id):
        pass

    def file_id_str_handler(self, file_id_str):
        '''
        Input: file_ids joined by '|'
        '''
        pass

    def draw_data_list(self, p):
        '''
        Input: string, has 3 parts when split by '_'
            p001_5_seconds      Draw no more than 5 seconds from p001
            p001_5_files        Draw 5 files from p001
        Output:
            string, file_ids joined by '|'
        '''
        speaker_id, n, draw_rule = p.split('_')

        if draw_rule == 'files':
            # Draw n files
            n = int(n)
            file_list_str = self.dv_tts_selector.draw_n_files(speaker_id, 'SR', n)
        elif draw_rule == 'seconds':
            n = float(n)
            file_list_str = self.dv_tts_selector.draw_n_seconds(speaker_id, 'SR', n)
        return file_list_str

    def draw_start_sample_number(self, window_shift=None, window_shift_threshold=None):
        # if actual window shift is larger than window_shift_threshold, use a random draw start_sample_number

        if window_shift is None:
            window_shift=self.window_shift
        if window_shift_threshold is None:
            window_shift_threshold=self.window_shift_threshold

        if window_shift > window_shift_threshold:
            start_sample_number = int(numpy.random.rand() * window_shift)
        else:
            start_sample_number = 0
        return start_sample_number

    '''
    Below are methods from original code, maybe just leave them
    '''
    def get_path(self, key):
        return self.data[key]

    def __contains__(self, item):
        return item

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def keys(self):
        return self.data.keys()






class spk_embed_reader(data_reader_base):
    """
    loader_type: spk_embed_%i % dimension
    """

    def __init__(self, fname: Union[Path, str], loader_type: str):
        super().__init__(fname, loader_type)

        self.feat_dim = int(loader_type.split('_')[-1])

        print("input file name is %s" % fname)

        self.dv_dir = self.get_dv_dir_from_fname(fname)
        self.load_dv_data()

    def get_dv_dir_from_fname(self, fname):
        '''
        e.g. 
        fname = 'dump/xvector/cmp_5s/tr_no_dev/dynamic_5_seconds.scp'
        return: 'dump/xvector/cmp_5s'
        '''
        f_list = fname.split('/')
        f_list_useful = f_list[:-2]
        return '/'.join(f_list_useful)

    def load_dv_data(self):
        dv_spk_dict_file = os.path.join(self.dv_dir, 'dv_spk_dict.dat')
        self.dv_spk_dict = pickle.load(open(dv_spk_dict_file, 'rb'))
        dv_file_dict_file = os.path.join(self.dv_dir, 'dv_file_dict.dat')
        self.dv_file_dict = pickle.load(open(dv_file_dict_file, 'rb'))

    def speaker_id_handler(self, speaker_id):
        return self.dv_spk_dict[speaker_id]

    def file_id_str_handler(self, file_id_str):
        file_list = file_id_str.split('|')
        total_frames = 0.
        dv_value = numpy.zeros(self.feat_dim)
        for f in file_list:
            num_frames, dv_value_f = self.dv_file_dict[f]
            total_frames += num_frames
            dv_value = dv_value + dv_value_f * num_frames
        return dv_value / total_frames


class cmp_reader(data_reader_base):
    """
    Reader class for a scp file of cmp file.
    loader_type: cmp_binary_%i_%i % dimension, window_size
    """

    def __init__(self, fname: Union[Path, str], loader_type: str):
        super().__init__(fname, loader_type)

        self.feat_dim, self.window_size = map(int, loader_type[len("cmp_binary_") :].split("_"))

        from exp_mw545.exp_dv_cmp_baseline import dv_y_cmp_configuration
        from frontend_mw545.data_loader import Build_dv_y_cmp_data_loader_Multi_Speaker
        self.cfg = configuration(cache_files=False)
        self.dv_y_cfg = dv_y_cmp_configuration(self.cfg, cache_files=False)
        self.dv_y_cfg.input_data_dim['T_B'] = self.window_size
        self.dv_y_cfg.input_data_dim['D'] = (self.window_size * self.feat_dim)
        self.dv_y_cfg.update_cmp_dim()
        self.data_loader = Build_dv_y_cmp_data_loader_Multi_Speaker(self.cfg, self.dv_y_cfg)

    def file_id_str_handler(self, file_id_str):
        BD_data, B, start_sample_number = self.data_loader.make_BD_data(file_id_str, start_sample_number=0)
        return BD_data

class wav_reader(data_reader_base):
    """
    Reader class for a scp file of wav file.
    loader_type: wav_binary_%i_%i % window_size, window_shift
    """

    def __init__(self, fname: Union[Path, str], loader_type: str):
        super().__init__(fname, loader_type)

        self.window_size, self.window_shift = map(int, loader_type[len("wav_binary_") :].split("_"))

        from exp_mw545.exp_dv_wav_sincnet import dv_y_wav_sincnet_configuration
        from frontend_mw545.data_loader import Build_dv_y_wav_data_loader_Multi_Speaker
        self.cfg = configuration(cache_files=False)
        self.dv_y_cfg = dv_y_wav_sincnet_configuration(self.cfg, cache_files=False)
        self.dv_y_cfg.input_data_dim['T_B'] = self.window_size
        self.dv_y_cfg.input_data_dim['B_stride'] = self.window_shift
        self.dv_y_cfg.update_wav_dim()
        self.data_loader = Build_dv_y_wav_data_loader_Multi_Speaker(self.cfg, self.dv_y_cfg)

        # assert self.dv_y_cfg.input_data_dim['T_B'] == self.window_size
        # assert self.dv_y_cfg.input_data_dim['B_stride'] == self.window_shift

    def file_id_str_handler(self, file_id_str):
        start_sample_number = self.draw_start_sample_number()
        BD_data, B, start_sample_number = self.data_loader.make_BD_data(file_id_str, start_sample_number=start_sample_number)
        return BD_data

class wav_f_tau_vuv_reader(data_reader_base):
    """
    Reader class for a scp file of wav file.
    loader_type: wav_f_tau_vuv_binary_%i_%i % window_size, window_shift
    """

    def __init__(self, fname: Union[Path, str], loader_type: str):
        super().__init__(fname, loader_type)

        self.window_size, self.window_shift = map(int, loader_type[len("wav_f_tau_vuv_binary_") :].split("_"))

        from exp_mw545.exp_dv_wav_sinenet_v0 import dv_y_wav_sinenet_configuration
        from frontend_mw545.data_loader import Build_dv_y_wav_data_loader_Multi_Speaker
        self.cfg = configuration(cache_files=False)
        self.dv_y_cfg = dv_y_wav_sinenet_configuration(self.cfg, cache_files=False)
        self.dv_y_cfg.input_data_dim['T_B'] = self.window_size
        self.dv_y_cfg.input_data_dim['B_stride'] = self.window_shift
        # self.dv_y_cfg.out_feat_list = ['wav_SBT', 'f_SBM', 'tau_SBM', 'vuv_SBM']
        self.dv_y_cfg.update_wav_dim()
        self.data_loader = Build_dv_y_wav_data_loader_Multi_Speaker(self.cfg, self.dv_y_cfg)

        # assert self.dv_y_cfg.input_data_dim['T_B'] == self.window_size
        # assert self.dv_y_cfg.input_data_dim['B_stride'] == self.window_shift

    def file_id_str_handler(self, file_id_str):
        start_sample_number = self.draw_start_sample_number()
        BD_data, B, start_sample_number = self.data_loader.make_BD_data(file_id_str, start_sample_number=start_sample_number)
        return BD_data
        

def compute_label_index_list(window_size, num_labs):
    if num_labs == 5:
        if window_size == 40:
            label_index_list = [0,10,20,30,39]
        if window_size == 3000:
            label_index_list = [0,6,12,18,24]
    return label_index_list

class cmp_lab_reader(data_reader_base):
    """
    Reader class for a scp file of cmp with lab
    loader_type: cmp_binary_%i_%i_%i % dimension, window_size, num_labs
    """

    def __init__(self, fname: Union[Path, str], loader_type: str):
        super().__init__(fname, loader_type)

        self.feat_dim, self.window_size, self.num_labs = map(int, loader_type[len("cmp_lab_binary_") :].split("_"))

        from exp_mw545.exp_dv_cmp_lab_attention import dv_y_cmp_configuration, dv_cmp_lab_attention_configuration
        from frontend_mw545.data_loader import Build_dv_y_cmp_data_loader_Multi_Speaker, Build_dv_atten_lab_data_loader_Multi_Speaker

        self.cfg = configuration(cache_files=False)
        self.dv_y_cfg = dv_y_cmp_configuration(self.cfg, cache_files=False)
        self.dv_y_cfg.input_data_dim['T_B'] = self.window_size
        self.dv_y_cfg.input_data_dim['D'] = (self.window_size * self.feat_dim)
        self.dv_y_cfg.update_cmp_dim()

        self.dv_attn_cfg = dv_cmp_lab_attention_configuration(self.cfg, self.dv_y_cfg, cache_files=False)
        self.dv_attn_cfg.label_index_list = compute_label_index_list(self.window_size, self.num_labs)
        self.dv_attn_cfg.update_lab_dim()

        self.y_data_loader = Build_dv_y_cmp_data_loader_Multi_Speaker(self.cfg, self.dv_y_cfg)
        self.l_data_loader = Build_dv_atten_lab_data_loader_Multi_Speaker(self.cfg, self.dv_attn_cfg)

    def file_id_str_handler(self, file_id_str):
        BD_data_y, B, start_sample_number = self.y_data_loader.make_BD_data(file_id_str, start_sample_number=0)
        BD_data_l, B, start_sample_number = self.l_data_loader.make_BD_data(file_id_str, start_sample_number=0)
        BD_data = numpy.concatenate((BD_data_y, BD_data_l), axis=1)
        return BD_data


class wav_lab_reader(data_reader_base):
    """
    Reader class for a scp file of wav with lab
    loader_type: wav_lab_binary_%i_%i_%i % window_size, window_shift, num_labs
    """

    def __init__(self, fname: Union[Path, str], loader_type: str):
        super().__init__(fname, loader_type)

        self.window_size, self.window_shift, self.num_labs = map(int, loader_type[len("wav_lab_binary_") :].split("_"))

        from exp_mw545.exp_dv_wav_sincnet_lab_attention import dv_y_wav_sincnet_configuration, dv_wav_sincnet_lab_attention_configuration
        from frontend_mw545.data_loader import Build_dv_y_wav_data_loader_Multi_Speaker, Build_dv_atten_lab_data_loader_Multi_Speaker


        self.cfg = configuration(cache_files=False)
        self.dv_y_cfg = dv_y_wav_sincnet_configuration(self.cfg, cache_files=False)
        self.dv_y_cfg.input_data_dim['T_B'] = self.window_size
        self.dv_y_cfg.input_data_dim['B_stride'] = self.window_shift
        self.dv_y_cfg.update_wav_dim()

        self.dv_attn_cfg = dv_wav_sincnet_lab_attention_configuration(self.cfg, self.dv_y_cfg, cache_files=False)
        self.dv_attn_cfg.label_index_list = compute_label_index_list(self.window_size, self.num_labs)
        self.dv_attn_cfg.update_lab_dim()

        self.y_data_loader = Build_dv_y_wav_data_loader_Multi_Speaker(self.cfg, self.dv_y_cfg)
        self.l_data_loader = Build_dv_atten_lab_data_loader_Multi_Speaker(self.cfg, self.dv_attn_cfg)

    def file_id_str_handler(self, file_id_str):
        start_sample_number = self.draw_start_sample_number()
        BD_data_y, B_y, start_sample_number = self.y_data_loader.make_BD_data(file_id_str, start_sample_number=start_sample_number)
        start_sample_number = int(start_sample_number / 120.)
        BD_data_l, B, start_sample_number = self.l_data_loader.make_BD_data(file_id_str, start_sample_number=start_sample_number)
        BD_data_l = BD_data_l[:B_y]
        BD_data = numpy.concatenate((BD_data_y, BD_data_l), axis=1)
        return BD_data



class wav_f_tau_vuv_lab_reader(data_reader_base):
    """
    Reader class for a scp file of wav with lab
    loader_type: wav_f_tau_vuv_lab_binary_%i_%i_%i % window_size, window_shift, num_labs
    """

    def __init__(self, fname: Union[Path, str], loader_type: str):
        super().__init__(fname, loader_type)

        self.window_size, self.window_shift, self.num_labs = map(int, loader_type[len("wav_f_tau_vuv_lab_binary_") :].split("_"))

        from exp_mw545.exp_dv_wav_sinenet_v2_lab_attention import dv_y_wav_sinenet_configuration, dv_wav_sinenet_lab_attention_configuration
        from frontend_mw545.data_loader import Build_dv_y_wav_data_loader_Multi_Speaker, Build_dv_atten_lab_data_loader_Multi_Speaker

        self.cfg = configuration(cache_files=False)
        self.dv_y_cfg = dv_y_wav_sinenet_configuration(self.cfg, cache_files=False)
        self.dv_y_cfg.input_data_dim['T_B'] = self.window_size
        self.dv_y_cfg.input_data_dim['B_stride'] = self.window_shift
        self.dv_y_cfg.update_wav_dim()

        self.dv_attn_cfg = dv_wav_sinenet_lab_attention_configuration(self.cfg, self.dv_y_cfg, cache_files=False)
        self.dv_attn_cfg.label_index_list = compute_label_index_list(self.window_size, self.num_labs)
        self.dv_attn_cfg.update_lab_dim()

        self.y_data_loader = Build_dv_y_wav_data_loader_Multi_Speaker(self.cfg, self.dv_y_cfg)
        self.l_data_loader = Build_dv_atten_lab_data_loader_Multi_Speaker(self.cfg, self.dv_attn_cfg)

    def file_id_str_handler(self, file_id_str):
        start_sample_number = self.draw_start_sample_number()
        BD_data_y, B_y, start_sample_number = self.y_data_loader.make_BD_data(file_id_str, start_sample_number=start_sample_number)
        start_sample_number = int(start_sample_number / 120.)
        BD_data_l, B, start_sample_number = self.l_data_loader.make_BD_data(file_id_str, start_sample_number=start_sample_number)
        BD_data_l = BD_data_l[:B_y]
        BD_data = numpy.concatenate((BD_data_y, BD_data_l), axis=1)
        return BD_data

        