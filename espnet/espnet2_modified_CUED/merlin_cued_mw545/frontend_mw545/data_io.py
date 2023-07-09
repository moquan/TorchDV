# data_io.py

import os, sys, pickle, time, shutil, logging, copy
import math, numpy, scipy

from frontend_mw545.modules import make_logger, read_file_list, File_List_Selecter

class Data_File_IO(object):
    """
    Contains methods to read and write data files
    """
    def __init__(self, cfg=None):
        super(Data_File_IO, self).__init__()
        self.cfg = cfg

    def load_data_file_frame(self, file_name, feat_dim=1, return_frame_number=True):
        '''
        Return data and frame number
        '''
        with open(file_name, 'rb') as f:
            data = numpy.fromfile(f, dtype=numpy.float32)

        assert data.size % float(feat_dim) == 0.0,'specified dimension %s not compatible with file %s'%(feat_dim, file_name)

        frame_number = int(data.size / feat_dim)
        data = data[:(feat_dim * frame_number)]
        data = data.reshape((-1, feat_dim))
            
        if return_frame_number:
            return data, frame_number
        else:
            return data

    def save_data_file(self, data, file_name):
        data = numpy.array(data, 'float32')
        with open(file_name, 'wb') as f:
            data.tofile(f)

    def read_wav_2_wav_1D_data(self, in_file_name=None, file_id=None, return_sample_rate=False):
        '''
        Input either complete file name, or specify file id
        '''
        if in_file_name is None:
            in_file_name = os.path.join(self.cfg.wav_dir, file_id+'.wav')
        sr, wav_1D_data = scipy.io.wavfile.read(in_file_name)
        wav_1D_data = numpy.array(wav_1D_data, dtype='float32')
        if return_sample_rate:
            return wav_1D_data, sr
        else:
            return wav_1D_data

    def write_wav_1D_data_2_wav(self, wav_1D_data, out_file_name, sample_rate=None, cfg=None):
        '''
        Output is audio file
        '''
        if cfg is None:
            cfg = self.cfg
        if sample_rate is None:
            sample_rate = cfg.wav_sr
        wav_1D_data = numpy.array(wav_1D_data, dtype='int16')
        scipy.io.wavfile.write(out_file_name, sample_rate, wav_1D_data)

    def read_pml_by_name_feat(self, file_id, feat_name, return_frame_number=False, cfg=None):
        '''
        Faster method to load a PML file, by ID and feat_name; no need extension or whatever
        '''
        if cfg is None:
            cfg = self.cfg
        file_name = os.path.join(cfg.acoustic_dir_dict[feat_name], file_id+cfg.acoustic_file_ext_dict[feat_name])
        feat_dim  = cfg.acoustic_in_dimension_dict[feat_name]
        feat_data, frame_number = self.load_data_file_frame(file_name, feat_dim, return_frame_number=True)

        if return_frame_number:
            return feat_data, frame_number
        else:
            return feat_data

    def read_pitch_reaper(self, pitch_file_name):
        ''' Return a list of time where vuv=1 '''
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

    def read_f0_reaper(self, f0_file_name, return_vuv_list=False):
        ''' 
        Return a list of f0 values; REAPER output: 
            200Hz, 0.005s intervals; 
            when vuv=0, f0=-1
        '''
        f0_list  = []
        vuv_list = []
        with open(f0_file_name, 'r') as f:
            file_lines = f.readlines()
        for l in file_lines:
            x = l.strip().split(' ')
            # Content lines should have 3 values
            # Time stamp, vuv, F0 value
            if len(x) == 3:
                f0_list.append(float(x[2]))
                vuv_list.append(int(x[1]))

        if return_vuv_list:
            return f0_list, vuv_list
        else:
            return f0_list

    def read_lab_no_sil_indices(self, label_align_file, silence_pattern=['*-#+*']):
        '''
        Return a list of non-silence indices
        Note: not a continuous list; may omit silent indices in between
        '''
        if self.silence_reducer is None:
            from frontend.silence_reducer_keep_sil import SilenceReducer
            self.silence_reducer = SilenceReducer(n_cmp = 1, silence_pattern = silence_pattern)
        nonsilence_indices = self.silence_reducer.load_alignment(label_align_file)
        return nonsilence_indices

class Data_List_File_IO(object):
    """
    Contains methods to read or write file_id_list
    1 read method, multiple write methods, different filters
    """
    def __init__(self, cfg=None):
        super(Data_List_File_IO, self).__init__()
        self.cfg = cfg
        self.FL_Selecter = File_List_Selecter()
        self.logger = make_logger("file_list_IO")

    def split_file_list_cfg_used(self, file_id_list):
        '''
        Return 2 file lists, 1 contains all files is used in our experiment, 1 all others
        train speaker & not AM_held_out || valid+test speaker & held_out
        1. Split by speakers, train   or    valid+test
        2. Within each list, split by held_out_file_number, into used and not_used lists
        Note: careful about files for lambda valid/test/gen
        '''
        cfg = self.cfg

        train_list, not_train_list = self.FL_Selecter.split_by_speaker(file_id_list, cfg.speaker_id_list_dict['train'])
        valid_test_list, not_used_list = self.FL_Selecter.split_by_speaker(not_train_list, cfg.speaker_id_list_dict['valid']+cfg.speaker_id_list_dict['test'])
        not_used_t_list, used_t_list = self.FL_Selecter.split_by_min_max_file_number(train_list, cfg.AM_held_out_file_number[0], cfg.AM_held_out_file_number[1])
        used_vt_file_list, not_used_vt_file_list = self.FL_Selecter.split_by_min_max_file_number(valid_test_list, cfg.held_out_file_number[0], cfg.held_out_file_number[1])

        used_list = used_t_list + used_vt_file_list
        not_used_list = not_used_list + not_used_t_list + not_used_vt_file_list

        return used_list, not_used_list

    def write_file_list_cfg_used(self, in_file_name=None, out_file_name=None, not_used_file_name=None):
        cfg = self.cfg

        if in_file_name is None:
            in_file_name = cfg.file_id_list_file['all']
        if out_file_name is None:
            out_file_name = cfg.file_id_list_file['used']
        if not_used_file_name is None:
            not_used_file_name = cfg.file_id_list_file['excluded']
            
        file_id_list = read_file_list(in_file_name)

        file_id_list_used, file_id_list_not_used = self.split_file_list_cfg_used(file_id_list)
        
        self.logger.info('Write file list to %s' % out_file_name)
        self.write_file_list(file_id_list_used, out_file_name)

        self.logger.info('Write not used file list to %s' % not_used_file_name)
        self.write_file_list(file_id_list_not_used, not_used_file_name)

    def split_file_list_used_dv_test(self, file_id_list):
        '''
        Return 1 file list, files used for dv_test
        train speaker & not AM_held_out & held_out
        '''
        cfg = self.cfg

        train_list, not_train_list = self.FL_Selecter.split_by_speaker(file_id_list, cfg.speaker_id_list_dict['train'])
        not_used_t_list, used_t_list = self.FL_Selecter.split_by_min_max_file_number(train_list, cfg.AM_held_out_file_number[0], cfg.AM_held_out_file_number[1])
        dv_list, not_dv_list = self.FL_Selecter.split_by_min_max_file_number(used_t_list, cfg.held_out_file_number[0], cfg.held_out_file_number[1])

        return dv_list

    def write_file_list_dv_test(self, in_file_name=None, out_file_name=None):
        '''
        This is a make-up function for some previously missing files
        '''
        cfg = self.cfg

        if in_file_name is None:
            in_file_name = cfg.file_id_list_file['all']
        if out_file_name is None:
            out_file_name = cfg.file_id_list_file['dv_test']
            
        file_id_list = read_file_list(in_file_name)
        file_id_list_used = self.split_file_list_used_dv_test(file_id_list)

        self.logger.info('Write file list to %s' % out_file_name)
        self.write_file_list(file_id_list_used, out_file_name)

    def write_file_list_long_enough(self, meta_file_name=None, out_file_name=None):
        '''
        Keep files long enough
        '''
        DMLF_IO = Data_Meta_List_File_IO(self.cfg)
        if meta_file_name is None:
            meta_file_name = '/home/dawna/tts/mw545/TorchDV/file_id_lists/data_meta/file_id_list_num_sil_frame.scp'
        if out_file_name is None:
            out_file_name  = self.cfg.file_id_list_file['dv_enough']

        file_frame_dict = DMLF_IO.read_file_list_num_silence_frame(meta_file_name)

        min_file_len = 200
        file_list = []

        for file_id in file_frame_dict:
            l, x, y = file_frame_dict[file_id]
            file_len = y - x + 1

            if file_len >= min_file_len:
                file_list.append(file_id)

        self.write_file_list(file_list, out_file_name)

    def write_file_list_compute_norm_info(self, in_file_name=None, out_file_name=None):
        '''
        Return the file_id_list for computing mean and std
        Train speaker && not in cfg.held_out_file_number
        '''
        cfg = self.cfg

        if in_file_name is None:
            in_file_name = cfg.file_id_list_file['used']
        if out_file_name is None:
            out_file_name = cfg.file_id_list_file['compute_norm_info']

        file_id_list = read_file_list(in_file_name)

        train_speaker_list, discard_list = self.FL_Selecter.split_by_speaker(file_id_list, cfg.speaker_id_list_dict['train'])
        discard_list, keep_list = self.FL_Selecter.split_by_min_max_file_number(train_speaker_list, cfg.held_out_file_number[0], cfg.held_out_file_number[1])

        # print(keep_list)
        # print(len(keep_list))
        self.write_file_list(keep_list, out_file_name)

    def write_file_list(self, file_list, file_name):
        '''
        Write the file list into the file name
        '''
        with open(file_name, 'w') as f:
            for file_id in file_list:
                f.write(file_id+'\n')

    def read_file_list(self, file_name):
        '''
        Return a list of file IDs
        '''
        file_lists = []
        fid = open(file_name)
        for line in fid.readlines():
            line = line.strip()
            if len(line) < 1:
                continue
            file_lists.append(line)
        fid.close()
        self.logger.info('Read file list from %s' % file_name)
        return file_lists

class Data_Meta_List_File_IO(object):
    """
    Contains methods to read or write meta data
    Methods should be in pairs; one read, one write
    Each line contains file_id, followed by values defined by the methods
    """
    def __init__(self, cfg=None):
        super(Data_Meta_List_File_IO, self).__init__()

        self.cfg = cfg
        self.logger = make_logger("write_data_meta")

        self.DF_IO = Data_File_IO(cfg)

    def write_file_list_num_silence_frame(self, in_file_name=None, out_file_name=None):
        '''
        Each line contains:
            file_id  num_frame_wav_cmp  first_non_sil_index  last_non_sil_index
            (Both inclusive index)
        '''
        cfg = self.cfg

        from frontend_mw545.data_silence_reducer import Data_Silence_Reducer
        self.DSR = Data_Silence_Reducer(cfg)

        if in_file_name is None:
            in_file_name = cfg.file_id_list_file['used']
        if out_file_name is None:
            out_file_name = os.path.join('/home/dawna/tts/mw545/TorchDV/file_id_lists/data_meta', 'file_id_list_num_sil_frame.scp')

        file_id_list = read_file_list(in_file_name)
        self.logger.info('Write file list to %s' % out_file_name)

        wav_cmp_dir = cfg.nn_feat_dirs['wav']
        wav_dim     = cfg.nn_feature_dims['wav']

        with open(out_file_name, 'w') as f_1:
            for file_id in file_id_list:
                wav_cmp_file = os.path.join(wav_cmp_dir, file_id+'.wav')
                wav_data, wav_num_frame = self.DF_IO.load_data_file_frame(wav_cmp_file, wav_dim)
                lab_dir  = cfg.lab_dir
                lab_file = os.path.join(lab_dir, file_id+'.lab')
                nonsilence_frame_index_list = self.DSR.load_alignment(lab_file)

                l = '%s %i %i %i' %(file_id, wav_num_frame, nonsilence_frame_index_list[0], nonsilence_frame_index_list[-1])
                f_1.write(l+'\n')

    def read_file_list_num_silence_frame(self, in_file_name='/home/dawna/tts/mw545/TorchDV/file_id_lists/data_meta/file_id_list_num_sil_frame.scp'):
        '''
        Return a dict of file frame numbers; key is file ID
            file_id:  [num_frame_wav_cmp, first_non_sil_index, last_non_sil_index]
        '''
        file_frame_dict = {}

        fid = open(in_file_name)
        for line in fid.readlines():
            line = line.strip()
            if len(line) < 1:
                continue
            x_list = line.split(' ')

            file_id = x_list[0]
            l = int(x_list[1])
            x = int(x_list[2])
            y = int(x_list[3])

            file_frame_dict[file_id] = [l, x, y]

        return file_frame_dict

    def read_dv_spk_values_from_file(self, dv_file_name, file_type='text'):
        if file_type == 'text':
            dv_values = {}
            with open(dv_file_name, 'r') as f:
                f_lines = f.readlines()
            for x in f_lines:
                x_id = x.split('|')[0]
                y = x.split('|')[1].strip()[1:-1].split(',')
                x_value = [float(i) for i in y]
                dv_values[x_id] = numpy.asarray(x_value,dtype=numpy.float32).reshape([1,-1])
        elif file_type == 'pickle':
            dv_values = pickle.load(open(dv_file_name, 'rb'))
        return dv_values

    def write_dv_spk_values_to_file(self, dv_values, dv_file_name, file_type='text'):
        if file_type == 'text':
            speaker_id_list = dv_values.keys()
            with open(dv_file_name, 'w') as f:
                for speaker_id in speaker_id_list:
                    f.write(speaker_id+"|[")
                    dv_size = len(dv_values[speaker_id])
                    for i in range(dv_size):
                        if i > 0:
                            f.write(',')
                        f.write(str(dv_values[speaker_id][i]))
                    f.write("]\n")
        elif file_type == 'pickle':
            pickle.dump(dv_values, open(dv_file_name, 'wb'))

    def read_dv_file_values_from_file(self, dv_file_name, file_type='text'):
        if file_type == 'text':
            dv_values = {}
            with open(dv_file_name, 'r') as f:
                f_lines = f.readlines()
            for x in f_lines:
                x_id = x.split('|')[0]
                x_len = int(x.split('|')[1])
                y = x.split('|')[2].strip()[1:-1].split(',')
                x_value = [float(i) for i in y]
                dv_values[x_id] = (x_len, numpy.asarray(x_value,dtype=numpy.float32).reshape([1,-1]))
        elif file_type == 'pickle':
            dv_values = pickle.load(open(dv_file_name, 'rb'))
        return dv_values

    def write_dv_file_values_to_file(self, dv_values, dv_file_name, file_type='text'):
        # file_id|B_total|dv_values
        # dv_file_dict[file_id] = (B_total, dv_file)
        if file_type == 'text':
            file_id_list = dv_values.keys()
            with open(dv_file_name, 'w') as f:
                for file_id in file_id_list:
                    f.write("'"+file_id+"'|%i|[" % dv_values[file_id][0])
                    dv_size = len(dv_values[file_id][1])
                    for i in range(dv_size):
                        if i > 0:
                            f.write(',')
                        f.write(str(dv_values[file_id][1][i]))
                    f.write("]\n")
        elif file_type == 'pickle':
            pickle.dump(dv_values, open(dv_file_name, 'wb'))

class Data_File_Directory_Utils(object):
    """
    functions related to file list and directory
    cfg is optional
    """
    def __init__(self, cfg=None):
        super(Data_File_Directory_Utils, self).__init__()
        self.cfg = cfg
        self.logger = make_logger("Dir_Utils")

    def prepare_file_path_list(self, file_id_list, file_dir, file_extension, new_dir_switch=True):
        if not os.path.exists(file_dir) and new_dir_switch:
            os.makedirs(file_dir)
        file_name_list = []
        for file_id in file_id_list:
            file_name = file_dir + '/' + file_id + file_extension
            file_name_list.append(file_name)
        return  file_name_list


    def copy_to_scratch(self, remove_tar_dir=False):
        '''
        Copy files to scratch; Things to specify: 
            file_id_list, original directory, target directory, file extension
        '''
        cfg = self.cfg
        file_id_list = read_file_list(cfg.file_id_list_file['used'])

        data_dir = cfg.data_dir
        scratch_dir = cfg.nn_feat_scratch_dir_root
        # data_dir = '/data/vectra2/tts/mw545/Data/exp_dirs/data_voicebank_24kHz'
        # scratch_dir = '/scratch/tmp-mw545/exp_dirs/data_voicebank_24kHz'

        dir_pair_list = [] # [ori_dir, tar_dir, file_ext]

        dir_pair_list.append(['nn_lab_resil_norm_601', 'nn_lab_resil_norm_601', '.lab'])
        dir_pair_list.append(['nn_cmp_resil_norm_86', 'nn_cmp_resil_norm_86', '.cmp'])
        dir_pair_list.append(['nn_wav_resil_norm_120', 'nn_wav_resil_norm_120', '.wav'])
        dir_pair_list.append(['nn_f024k_resil', 'nn_f024k_resil', '.f024k'])
        dir_pair_list.append(['nn_pitch_resil', 'nn_pitch_resil', '.pitch'])

        assert len(dir_pair_list) > 0

        for dir_pair in dir_pair_list:
            ori_dir  = os.path.join(data_dir , dir_pair[0])
            tar_dir  = os.path.join(scratch_dir , dir_pair[1])
            # file_ext = dir_pair[2]

            if remove_tar_dir:
                try:
                    shutil.rmtree(tar_dir)
                    self.logger.info("Removing target directory: "+tar_dir)
                except:
                    self.logger.info("Target directory does not exist yet: "+tar_dir)
                # os.makedirs(tar_dir)

            # ori_file_list = self.prepare_file_path_list(file_id_list, ori_dir, file_ext)
            # tar_file_list = self.prepare_file_path_list(file_id_list, tar_dir, file_ext)

            self.logger.info("Copying... Original directory: "+ori_dir)
            self.logger.info("Copying... Target directory: "+tar_dir)
            shutil.copytree(ori_dir, tar_dir)
            # for x, y in zip(ori_file_list, tar_file_list):
            #     shutil.copyfile(x, y)

    def clean_data(self):
        ''' 
        Clean data; remove excluded files
        '''
        cfg = self.cfg
        file_id_list = read_file_list(cfg.file_id_list_file['excluded'])

        # for feat_name in ['cmp','wav','lab']:
        #     nn_resil_norm_file_list_scratch = self.prepare_file_path_list(file_id_list, cfg.nn_feat_scratch_dirs[feat_name], '.'+feat_name)
        #     for file_name in nn_resil_norm_file_list_scratch:
        #         os.remove(file_name)

        for feat_name in ['cmp','wav']:
            nn_file_list            = self.prepare_file_path_list(file_id_list, cfg.nn_feat_dirs[feat_name], '.'+feat_name)
            nn_resil_file_list      = self.prepare_file_path_list(file_id_list, cfg.nn_feat_resil_dirs[feat_name], '.'+feat_name)
            nn_resil_norm_file_list = self.prepare_file_path_list(file_id_list, cfg.nn_feat_resil_norm_dirs[feat_name], '.'+feat_name)
            for file_list_temp in [nn_file_list, nn_resil_file_list, nn_resil_norm_file_list]:
                for file_name in file_list_temp:
                    try:
                        os.remove(file_name)
                    except:
                        pass

    def clean_directory(self, target_dir):
        # remove all files under this directory
        # but keep this directory
        file_list = os.listdir(target_dir)
        for file_name in file_list:
            full_file_name = os.path.join(target_dir, file_name)
            os.remove(full_file_name)

    def rename_in_directory(self):
        '''
        Rename all files in a directory
        e.g. change their extensions
        Temporary function: subject to changes
        '''

        '''
        Change all extensions of pitch files
        .f0 --> .f016k
        '''
        cfg = self.cfg
        file_id_list = read_file_list(cfg.file_id_list_file['used'])

        dir_name = '/home/dawna/tts/mw545/TorchDV/debug_nausicaa/data/nn_f016k'
        ori_ext  = '.f0'
        tar_ext  = '.f016k'

        self.logger.info('Dir %s, %s --> %s' %(dir_name, ori_ext, tar_ext))

        for file_id in file_id_list:
            ori_file_name = os.path.join(dir_name, file_id + ori_ext)
            tar_file_name = os.path.join(dir_name, file_id + tar_ext)
            os.rename(ori_file_name, tar_file_name)

        dir_name = '/home/dawna/tts/mw545/TorchDV/debug_nausicaa/data/nn_f0200'
        ori_ext  = '.f0'
        tar_ext  = '.f0200'
        self.logger.info('Dir %s, %s --> %s' %(dir_name, ori_ext, tar_ext))

        for file_id in file_id_list:
            ori_file_name = os.path.join(dir_name, file_id + ori_ext)
            tar_file_name = os.path.join(dir_name, file_id + tar_ext)
            os.rename(ori_file_name, tar_file_name)        

#########################
# Main function to call #
#########################

def run_data_io(cfg):
    pass