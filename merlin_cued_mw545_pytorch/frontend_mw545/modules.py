# modules.py

import os, sys, pickle, time, shutil, logging, copy
import math, numpy

numpy.random.seed(545)

'''
This file contains handy modules
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
        Input: 2 lists
        Output: 1 dict
            file_list_dict[speaker_id]
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

    def sort_by_file_number(self, file_list, data_split_file_number_dict):
        '''
        Input:  1 list, 1 dict
        Output: 1 dict
            data_split_file_number_dict should be a dict of list of 2 values:
            [file_number_min, file_number_max], inclusive
            output: each key corresponds to a key in data_split_file_number_dict
        '''
        file_list_dict = {}
        remain_file_list = file_list
        for k in data_split_file_number_dict:
            keep_list, discard_list = self.split_by_min_max_file_number(remain_file_list, data_split_file_number_dict[k][0], data_split_file_number_dict[k][1])
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

def copy_dict(x_dict, except_List=[]):
    '''
    Copy every key-value pair to the new dict, except keys in the list
    '''
    y_dict = {}
    for k in x_dict:
        if k not in except_List:
            y_dict[k] = x_dict[k]
    return y_dict

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

#######################
# Test tools to use #
#######################

class Data_Replicate_Test(object):
    """ 
    Make a few files in a new test directory
    Compare with files already generated
    """
    def __init__(self, cfg=None):
        super(Data_Replicate_Test, self).__init__()
        self.cfg = cfg
        self.logger = make_logger("Replicate_Test")

        from frontend_mw545.data_io import Data_File_IO
        self.DF_IO = Data_File_IO(cfg)

    def check_data_same(self, data_1, data_2, l_1=None, l_2=None, tol=0):
        '''
        check file length first
        check maximum data difference, compare with tolerance
        No logger info when data are same
        '''
        if l_1 is None:
            l_1 = data_1.shape[0]
        if l_2 is None:
            l_2 = data_2.shape[0]

        if l_1 != l_2:
            self.logger.info('Different Files Lengths! %i %i' % (l_1, l_2))
            return False
        if (data_1 == data_2).all():
            # self.logger.info('0 Difference')
            return True
        else:
            data_diff = data_1[data_1 != data_2] - data_2[data_1 != data_2]
            # print(data_diff)
            max_data_diff = numpy.max(data_diff)
            if max_data_diff > tol:
                self.logger.info('Different Data! Max Difference is %s' % str(max_data_diff))
                return False
            else:
                self.logger.info('Difference within Tolerence')
                return True

    def check_file_same(self, file_1, file_2):
        '''
        Return True if the 2 files are exactly the same:
        1. same length
        2. same values
        '''
        data_1, l_1 = self.DF_IO.load_data_file_frame(file_1, 1)
        data_2, l_2 = self.DF_IO.load_data_file_frame(file_2, 1)

        return self.check_data_same(data_1, data_2, l_1, l_2)

    def check_file_dict_same(self, file_dict_1, file_dict_2):
        bool_all_same = True
        for k in file_dict_1:
            if self.check_file_same(file_dict_1[k], file_dict_2[k]):
                continue
            else:
                self.logger.info('Data of key %s is different' % k)
                bool_all_same = False
        return bool_all_same

    def check_data_dict_same(self, data_dict_1, data_dict_2, tol=0):
        bool_all_same = True
        for k in data_dict_1:
            if self.check_data_same(data_dict_1[k], data_dict_2[k], tol=tol):
                continue
            else:
                self.logger.info('Data of key %s is different' % k)
                bool_all_same = False
        return bool_all_same

class Graph_Plotting(object):
    """
    Functions for plotting
    """
    def __init__(self):
        super(Graph_Plotting, self).__init__()
        self.logger = make_logger("Graph_Plot")

        import matplotlib.pyplot as plt
        self.plt = plt


    def change_default_x_list(self, x_list, y_list):
        '''
        Make x-axis if there is none
        '''
        l = len(y_list)

        if x_list is None:
            x_list = [None] * l

        for i in range(l):
            if x_list[i] is None:
                x_list[i] = range(len(y_list[i]))

        return x_list
        
    def single_plot(self, fig_file_name, x_list, y_list, legend_list, title=None, x_label=None, y_label=None):
        '''
        Line plots
        Plot multiple lines on the same graph
        '''
        x_list = self.change_default_x_list(x_list, y_list)
        fig, ax = self.plt.subplots()
        fig.set_tight_layout(True)

        for x, y, l in zip(x_list, y_list, legend_list):
            if l is None:
                ax.plot(x, y)
            else:
                ax.plot(x, y, label=l)

        self.set_title_labels(ax, title, x_label, y_label)

        self.logger.info('Saving to %s' % fig_file_name)
        fig.savefig(fig_file_name, format="png")
        self.plt.close(fig)

    def one_line_one_scatter(self, fig_file_name, x_list, y_list, legend_list, title=None, x_label=None, y_label=None):
        '''
        One line plot, one scatter plot
        Useful for data samples on a curve
        '''
        x_list = self.change_default_x_list(x_list, y_list)
        fig, ax = self.plt.subplots()
        ax.plot(x_list[0], y_list[0], label=legend_list[0])
        ax.scatter(x_list[1], y_list[1], label=legend_list[1], c='r', marker='.')

        self.set_title_labels(ax, title, x_label, y_label)
        
        self.logger.info('Saving to %s' % fig_file_name)
        fig.savefig(fig_file_name, format="png")
        self.plt.close(fig)

    def single_bar_plot(self, fig_file_name, x, y, w, title=None, x_label=None, y_label=None):
        '''
        Single Bar Plot
        '''
        fig, ax = self.plt.subplots()
        ax.bar(x, y, w)
        self.set_title_labels(ax, title, x_label, y_label)

        self.logger.info('Saving to %s' % fig_file_name)
        fig.savefig(fig_file_name, format="png")
        self.plt.close(fig)

    def mean_std_plot(self, fig_file_name, x_list, y_list, s_list, legend_list, colour_list=['b','r','g','y','m','c'], title=None, x_label=None, y_label=None):
        x_list = self.change_default_x_list(x_list, y_list)
        fig, ax = self.plt.subplots()

        for i in range(len(y_list)):
            x = numpy.array(x_list[i])
            y = numpy.array(y_list[i])
            s = numpy.array(s_list[i])
            l = legend_list[i]
            c = colour_list[i]

            ax.plot(x, y, c+'-', label=l)
            ax.fill(numpy.concatenate([x, x[::-1]]),
                 numpy.concatenate([y - 1.9600 * s,
                                (y + 1.9600 * s)[::-1]]),
                 alpha=.5, fc=c, ec='None', label='95% confidence')

        self.set_title_labels(ax, title, x_label, y_label)

        self.logger.info('Saving to %s' % fig_file_name)
        fig.savefig(fig_file_name, format="png")
        self.plt.close(fig)

    def set_title_labels(self, ax, title, x_label, y_label):
        '''
        Set title and x/y labels for the graph
        To be used for all plotting functions
        '''
        if title is not None:
            ax.set_title(title)  # Add a title to the axes
        if x_label is not None:
            ax.set_xlabel(x_label)  # Add a title to the axes
        if y_label is not None:
            ax.set_ylabel(y_label)  # Add a title to the axes
        ax.legend()

class Build_Log_File_Reader(object):
    """
    Functions for reading log files e.g. errors, train_accuracy
    Return 3 lists, of train, valid, and test
    """
    def __init__(self):
        super().__init__()
        pass

    def extract_errors_from_log_file(self, log_file_name):
        file_lines = []
        with open(log_file_name) as f:
            for line in f.readlines():
                line = line.strip()
                if len(line) < 1:
                    continue
                file_lines.append(line)

        train_error = []
        valid_error = []
        test_error  = []

        for single_line in file_lines:
            words = single_line.strip().split(' ')
            if 'epoch' in words and 'loss:' in words:
                # words_new = words
                if '&' in words:
                    train_index = words.index('loss:')+2
                    valid_index = train_index+2
                    test_index  = valid_index+2
                else:
                    if 'train' in words:
                        train_index = words.index('train')+1
                    elif 'training' in words:
                        train_index = words.index('training')+1
                    if 'validation' in words:
                        valid_index = words.index('validation')+1
                    elif 'valid' in words:
                        valid_index = words.index('valid')+1
                    test_index  = words.index('test')+1
                train_error.append(float(words[train_index][:-1]))
                valid_error.append(float(words[valid_index][:-1]))
                test_error.append(float(words[test_index][:-1]))
                
        return (train_error, valid_error, test_error)

    def extract_accuracy_from_log_file(self, log_file_name):
        # Note: not compatible if window-level accuracy is also present
        # use self.extract_accuracy_win_from_log_file(log_file_name) in this case
        file_lines = []
        with open(log_file_name) as f:
            for line in f.readlines():
                line = line.strip()
                if len(line) < 1:
                    continue
                file_lines.append(line)

        train_accuracy = []
        valid_accuracy = []
        test_accuracy  = []

        for single_line in file_lines:
            words = single_line.strip().split(' ')
            if 'epoch' in words and 'accu:' in words:
                if words[words.index('accu:')-1] == 'win':
                    print('Warning: window-level accuracy in log file, check again')
                # words_new = words
                if '&' in words:
                    train_index = words.index('accu:')+2
                    valid_index = train_index+2
                    test_index  = valid_index+2
                else:
                    if 'train' in words:
                        train_index = words.index('train')+1
                    elif 'training' in words:
                        train_index = words.index('training')+1
                    if 'validation' in words:
                        valid_index = words.index('validation')+1
                    elif 'valid' in words:
                        valid_index = words.index('valid')+1
                    test_index  = words.index('test')+1
                
                train_accuracy.append(float(words[train_index][:-1]))
                valid_accuracy.append(float(words[valid_index][:-1]))
                test_accuracy.append(float(words[test_index][:-1]))
                    
        return (train_accuracy, valid_accuracy, test_accuracy)

    def extract_accuracy_win_from_log_file(self, log_file_name):
        # Extract both utterance level and window level accuracy
        #
        # 2021-10-12 20:59:59,537     INFO    train_model: epoch 84 train & valid & test loss: & 0.0442 & 0.2732 & 0.4700
        # 2021-10-12 20:59:59,537     INFO    train_model: epoch 84 train & valid & test accu: & 1.0000 & 0.9986 & 0.9934
        # 2021-10-12 20:59:59,537     INFO    train_model: epoch 84 train & valid & test win accu: & 0.9900 & 0.9259 & 0.8841
        file_lines = []
        with open(log_file_name) as f:
            for line in f.readlines():
                line = line.strip()
                if len(line) < 1:
                    continue
                file_lines.append(line)

        train_accuracy = []
        valid_accuracy = []
        test_accuracy  = []

        train_accuracy_win = []
        valid_accuracy_win = []
        test_accuracy_win  = []

        for single_line in file_lines:
            words = single_line.strip().split(' ')
            if 'epoch' in words and 'accu:' in words:
                # words_new = words
                if '&' in words:
                    train_index = words.index('accu:')+2
                    valid_index = train_index+2
                    test_index  = valid_index+2
                else:
                    if 'train' in words:
                        train_index = words.index('train')+1
                    elif 'training' in words:
                        train_index = words.index('training')+1
                    if 'validation' in words:
                        valid_index = words.index('validation')+1
                    elif 'valid' in words:
                        valid_index = words.index('valid')+1
                    test_index  = words.index('test')+1
                if words[words.index('accu:')-1] == 'win':
                    # This is window level accuracy
                    train_accuracy_win.append(float(words[train_index][:-1]))
                    valid_accuracy_win.append(float(words[valid_index][:-1]))
                    test_accuracy_win.append(float(words[test_index][:-1]))
                else:
                    train_accuracy.append(float(words[train_index][:-1]))
                    valid_accuracy.append(float(words[valid_index][:-1]))
                    test_accuracy.append(float(words[test_index][:-1]))
                    
        return (train_accuracy, valid_accuracy, test_accuracy, train_accuracy_win, valid_accuracy_win, test_accuracy_win)
  