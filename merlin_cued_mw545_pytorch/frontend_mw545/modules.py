# modules.py

import os, sys, pickle, time, shutil, logging, copy
import math, numpy, scipy, scipy.io.wavfile #, sigproc, sigproc.pystraight
numpy.random.seed(545)

'''
This file contains handy modules of using Merlin
All file lists and directories should be provided elsewhere
'''

class File_List_Selecter(object):
    """
    Split the input file list into (dict of) lists
    """
    def __init__(self):
        super(File_List_Selecter, self).__init__()
        pass

    def sort_by_speaker_list(self, file_list, speaker_id_list):
        '''
        Return a dict, file_list_dict[speaker_id]
        '''
        file_list_dict = {}
        for speaker_id in speaker_id_list:
            file_list_dict[speaker_id] = []

        for file_name in file_list:
            speaker_id = file_name.split('_')[0]
            if speaker_id in file_list_dict:
                try:
                    file_list_dict[speaker_id].append(file_name)
                except KeyError:
                    print('speaker_id not in speaker_id_list; file name is %s' % file_name)

        return file_list_dict

    def split_by_speaker(self, file_list, speaker_id_list):
        '''
        Input:  2 lists
        Output: 2 lists, one "keep" list, one "discard" list
        '''
        keep_list = []
        discard_list = []
        speaker_id_list = check_and_change_to_list(speaker_id_list)
        for y in file_list:
            speaker_id = y.split('/')[-1].split('.')[0].split('_')[0]
            if speaker_id in speaker_id_list:
                keep_list.append(y)
            else:
                discard_list.append(y)
        return keep_list, discard_list

    def sort_by_file_number(self, file_list, data_split_file_number):
        '''
        Input:  1 list, 1 dict
        Output: 1 dict
            data_split_file_number should be a dict of list of 2 values:
            [file_number_min, file_number_max]
            output: each key corresponds to a key in data_split_file_number dict
        '''
        file_list_dict = {}
        remain_file_list = file_list
        for k in data_split_file_number:
            keep_list, discard_list = self.split_by_min_max_file_number(remain_file_list, data_split_file_number[k][0], data_split_file_number[k][1])
            file_list_dict[k] = keep_list
            remain_file_list = discard_list

        if len(remain_file_list) > 0:
            print('Files remain un-sorted!')
            print(remain_file_list)

        return file_list_dict

    def split_by_min_max_file_number(self, file_list, file_number_min, file_number_max):
        '''
        Input:  1 list, 2 values
        Output: 2 lists, one "keep" list, one "discard" list
        Check if file number is within min max (inclusive)
        '''
        keep_list = []
        discard_list = []

        min_minus_1 = int(file_number_min) - 1
        max_plus_1  = int(file_number_max) + 1
        for y in file_list:
            file_number = int(y.split('/')[-1].split('.')[0].split('_')[1])
            if (file_number > min_minus_1) and (file_number < max_plus_1):
                keep_list.append(y)
            else:
                discard_list.append(y)
        return keep_list, discard_list

class List_Random_Loader(object):
    def __init__(self, list_to_draw):
        self.list_total  = list_to_draw
        self.list_remain = copy.deepcopy(self.list_total)

    def draw_n_samples(self, n):
        list_return = []
        n_remain = len(self.list_remain)
        n_need   = n
        while n_need > 0:
            if n_remain > n_need:
                # Enough, draw a subset
                list_draw = numpy.random.choice(self.list_remain, n_need, replace=False)
                for f in list_draw:
                    list_return.append(f)
                    self.list_remain.remove(f)
                n_need = 0
            else:
                # Use them all
                list_return.extend(self.list_remain)
                # Reset the list
                self.list_remain = copy.deepcopy(self.list_total)
                n_need -= n_remain
                n_remain = len(self.list_remain)
        return list_return

def make_logger(logger_name):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    # create console handler and set level to debug
    if not logger.handlers:
        ch = logging.StreamHandler(sys.stdout)
        # ch.setLevel(logging.DEBUG)
        # create formatter
        formatter = logging.Formatter('%(asctime)s %(levelname)8s%(name)15s: %(message)s')
        # add formatter to ch
        ch.setFormatter(formatter)
        # add ch to logger
        logger.addHandler(ch)
    return logger

def make_held_out_file_number(last_index, start_index=1):
    '''
    List of 3-digit strings of file numbers
    pad 0 in front if needed e.g. 3 --> 003
    '''
    held_out_file_number = []
    for i in range(start_index, last_index+1):
        # held_out_file_number.append('0'*(3-len(str(i)))+str(i))
        held_out_file_number.append("%3i" % i)
    return held_out_file_number

def read_file_list(file_name):
    logger = make_logger("read_file_list")
    file_lists = []
    fid = open(file_name)
    for line in fid.readlines():
        line = line.strip()
        if len(line) < 1:
            continue
        file_lists.append(line)
    fid.close()
    logger.info('Read file list from %s' % file_name)
    return file_lists

def log_class_attri(cfg, logger, except_list=['feat_index']):
    attri_list = vars(cfg)
    for i in attri_list.keys():
        if i not in except_list:
            logger.info(i+ ' is '+str(attri_list[i]))

def check_and_change_to_list(sub_list):
    if not isinstance(sub_list, list):
        sub_list = [sub_list]
    return sub_list

def prepare_script_file_path(file_dir, new_dir_switch=True, script_name=''):
    if not os.path.exists(file_dir) and new_dir_switch:
        os.makedirs(file_dir)
    if len(script_name) > 0:
        target_script_name = os.path.join(file_dir, os.path.basename(script_name))
        if os.path.exists(target_script_name):
            for i in range(100000):
                temp_script_name = target_script_name+'_'+str(i)
                if not os.path.exists(temp_script_name):
                    os.rename(target_script_name, temp_script_name)
                    break
        shutil.copyfile(script_name, target_script_name)

def prepare_file_path_list(file_id_list, file_dir, file_extension, new_dir_switch=True):
    if not os.path.exists(file_dir) and new_dir_switch:
        os.makedirs(file_dir)
    file_name_list = []
    for file_id in file_id_list:
        file_name = file_dir + '/' + file_id + file_extension
        file_name_list.append(file_name)
    return  file_name_list
