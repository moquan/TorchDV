# data_silence_reducer.py

import os, sys, pickle, time, shutil, logging
import math, numpy

from frontend_mw545.modules import make_logger, read_file_list, prepare_file_path_list
from frontend_mw545.data_io import Data_File_IO

import re

class Data_Silence_Reducer(object):
    """docstring for Data_Silence_Reducer"""
    def __init__(self, cfg=None):
        super(Data_Silence_Reducer, self).__init__()
        self.logger = make_logger("DSR")

        self.cfg = cfg
        self.DIO = Data_File_IO(cfg)

        self.silence_pattern = ['*-#+*']
        self.silence_pattern_size = len(self.silence_pattern)

        self.total_sil_one_side_cmp = cfg.frames_silence_to_keep + cfg.sil_pad

    def check_silence_pattern(self, label):
        '''
        Return 1 if is silence, 0 if not
        Code copied from Merlin (https://github.com/CSTR-Edinburgh/merlin)
        '''
        label_size = len(label)
        binary_flag = 0
        for si in range(self.silence_pattern_size):
            current_pattern = self.silence_pattern[si]
            current_size = len(current_pattern)
            if current_pattern[0] == '*' and current_pattern[current_size - 1] == '*':
                temp_pattern = current_pattern[1:current_size - 1]
                for il in range(1, label_size - current_size + 2):
                    if temp_pattern == label[il:il + current_size - 2]:
                        binary_flag = 1
            elif current_pattern[current_size-1] != '*':
                temp_pattern = current_pattern[1:current_size]
                if temp_pattern == label[label_size - current_size + 1:label_size]:
                    binary_flag = 1
            elif current_pattern[0] != '*':
                temp_pattern = current_pattern[0:current_size - 1]
                if temp_pattern == label[0:current_size - 1]:
                    binary_flag = 1
            if binary_flag == 1:
                break
        
        return binary_flag # one means yes, zero means no

    def load_alignment(self, alignment_file_name):
        '''
        Return a list of nonsilence_indices
        Note: may be discontinuous; silence inside utterance
        Code copied from Merlin (https://github.com/CSTR-Edinburgh/merlin)
        '''
        base_frame_index = 0
        nonsilence_frame_index_list = []
        fid = open(alignment_file_name)
        for line in fid.readlines():
            line = line.strip()
            if len(line) < 1:
                continue
            temp_list = re.split('\s+', line)
            start_time = int(temp_list[0])
            end_time = int(temp_list[1])
            full_label = temp_list[2]
            frame_number = int((end_time - start_time)/50000)

            label_binary_flag = self.check_silence_pattern(full_label)

            if label_binary_flag == 0:
                for frame_index in range(frame_number):
                    nonsilence_frame_index_list.append(base_frame_index + frame_index)
            base_frame_index = base_frame_index + frame_number
            # print   start_time, end_time, frame_number, base_frame_index
        fid.close()
        
        return nonsilence_frame_index_list

    def reduce_silence_file(self, alignment_file_name, in_file_name, out_file_name, feat_dim=None, feat_name=None):
        '''
        Input:
            Lab File
            Input NN File
            Resil NN File
            Either feat_dim, or feat_name to lookup in cfg
        '''
        if feat_dim is None:
            feat_dim = self.cfg.nn_feature_dims[feat_name]

        nonsilence_indices = self.load_alignment(alignment_file_name)
        in_data, in_frame_number = self.DIO.load_data_file_frame(in_file_name, feat_dim)
        
        if len(nonsilence_indices) == in_frame_number:
            print('WARNING: no silence found!')
            # previsouly: continue -- in fact we should keep non-silent data!

        no_sil_start = nonsilence_indices[0]
        no_sil_end   = nonsilence_indices[-1]

        # Inclusive Index
        sil_pad_first_idx = no_sil_start - self.total_sil_one_side_cmp
        sil_pad_last_idx  = no_sil_end + self.total_sil_one_side_cmp
        out_data = in_data[sil_pad_first_idx:sil_pad_last_idx+1]
        self.DIO.save_data_file(out_data, out_file_name)

    def reduce_silence_file_old(self, alignment_file_name, in_file_name, out_file_name, feat_dim=None, feat_name=None):
        '''
        Old method: complicated indexing due to possible padding
        New update: all silence are >=5, therefore no need padding
        '''
        if feat_dim is None:
            feat_dim = self.cfg.nn_feature_dims[feat_name]

        nonsilence_indices = self.load_alignment(alignment_file_name)
        in_data, in_frame_number = self.DIO.load_data_file_frame(in_file_name, feat_dim)
        
        if len(nonsilence_indices) == in_frame_number:
            print('WARNING: no silence found!')
            # previsouly: continue -- in fact we should keep non-silent data!

        no_sil_start = nonsilence_indices[0]
        no_sil_end   = nonsilence_indices[-1]

        # Trim / Pad the end first
        sil_pad_last_idx = no_sil_end + self.total_sil_one_side_cmp
        if sil_pad_last_idx > (frame_number-1):
            # Need to pad, repeat last frame
            num_to_pad = sil_pad_last_idx - (frame_number-1)
            last_frame = ori_cmp_data[-1]
            last_frame_2D  = numpy.expand_dims(last_frame, axis=0) # (n_cmp) --> (1, n_cmp)
            last_frame_pad = numpy.repeat(last_frame_2D, num_to_pad, axis=0) # (1, n_cmp) --> (n, n_cmp)
            pad_cmp_data = numpy.concatenate((ori_cmp_data, last_frame_pad), axis=0)
        else:
            # Trim but keep enough silence pad
            pad_cmp_data = ori_cmp_data[:sil_pad_last_idx+1]

        # Trim / Pad the start
        sil_pad_first_idx = no_sil_start - self.total_sil_one_side_cmp
        if sil_pad_first_idx < 0:
            # Need to pad, repeat first frame
            num_to_pad = 0 - sil_pad_first_idx
            first_frame = ori_cmp_data[0]
            first_frame_2D  = numpy.expand_dims(first_frame, axis=0) # (n_cmp) --> (1, n_cmp)
            first_frame_pad = numpy.repeat(first_frame_2D, num_to_pad, axis=0)
            new_cmp_data = numpy.concatenate((first_frame_pad, pad_cmp_data), axis=0)
        else:
            # Trim but keep enough silence pad
            new_cmp_data = pad_cmp_data[sil_pad_first_idx:]

        actual_frame_number = new_cmp_data.shape[0]
        expect_frame_number = (no_sil_end - no_sil_start + 1) + 2 * self.total_sil_one_side_cmp
        assert actual_frame_number == expect_frame_number

class Data_Silence_List_Reducer(object):
    """docstring for Data_Silence_List_Reducer"""
    def __init__(self, cfg=None):
        super(Data_Silence_List_Reducer, self).__init__()
        self.logger = make_logger("DSR_List")

        self.cfg = cfg
        self.DSR = Data_Silence_Reducer(cfg)

        file_id_list_file = cfg.file_id_list_file['used']
        # file_id_list_file = cfg.file_id_dv_test_list_file
        
        self.logger.info('Reading file list from %s' % file_id_list_file)
        self.file_id_list = read_file_list(file_id_list_file)

    def reduce_silence_file_list(self, feat_name):
        '''
        Reduce silence; get directory and feat_dim by feat_name        
        '''

        cfg = self.cfg

        in_file_dir = cfg.nn_feat_dirs[feat_name]
        out_file_dir = cfg.nn_feat_resil_dirs[feat_name]
        file_ext = '.' + feat_name

        for file_id in self.file_id_list:
            alignment_file_name = os.path.join(cfg.lab_dir, file_id + '.lab')
            in_file_name  = os.path.join(in_file_dir,  file_id + file_ext)
            out_file_name = os.path.join(out_file_dir, file_id + file_ext)

            self.logger.info('Saving to file %s' % out_file_name)
            self.DSR.reduce_silence_file(alignment_file_name, in_file_name, out_file_name, feat_name=feat_name)

#########################
# Main function to call #
#########################

def run_Data_Silence_List_Reducer(cfg):
    DSR_List = Data_Silence_List_Reducer(cfg)
    feat_name = 'f016k'
    DSR_List.reduce_silence_file_list(feat_name)

