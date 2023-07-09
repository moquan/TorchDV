# exp_dv_cmp_pytorch.py

# This file uses dv_cmp experiments to slowly progress with pytorch

import os, sys, pickle, time, shutil, logging, copy
import math, numpy, scipy
numpy.random.seed(545)
import matplotlib
import matplotlib.pyplot as plt
from modules import make_logger, read_file_list, read_sil_index_file, prepare_file_path, prepare_file_path_list, copy_to_scratch
from modules import keep_by_speaker, sort_by_speaker_list, keep_by_file_number, check_and_change_to_list
from modules_2 import compute_feat_dim, log_class_attri, resil_nn_file_list, norm_nn_file_list, get_utters_from_binary_dict, get_one_utter_by_name, count_male_female_class_errors
from modules_2 import compute_cosine_distance, compute_Euclidean_distance
from modules_torch import torch_initialisation

from io_funcs.binary_io import BinaryIOCollection
io_fun = BinaryIOCollection()

from frontend_mw545.data_converter import Data_File_Converter


class list_random_loader(object):
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

class dv_y_configuration(object):
    
    def __init__(self, cfg):
        
        # Things to be filled
        self.python_script_name = None
        self.dv_y_model_class = None
        self.make_feed_dict_method_train = None
        self.make_feed_dict_method_test  = None
        self.make_feed_dict_method_gen   = None
        self.y_feat_name   = None
        self.out_feat_list = None
        self.nn_layer_config_list = None
        self.finetune_model = False
        self.prev_nnets_file_name = ''

        self.train_by_window = True # Optimise lambda_w; False: optimise speaker level lambda
        self.classify_in_training = True # Compute classification accuracy after validation errors during training
        self.batch_output_form = 'mean' # Method to convert from SBD to SD
        self.use_voiced_only = False # Use voiced regions only
        
        # Things no need to change
        self.learning_rate    = 0.0001
        self.num_train_epoch  = 1000
        self.warmup_epoch     = 10
        self.early_stop_epoch = 2    # After this number of non-improvement, roll-back to best previous model and decay learning rate
        self.max_num_decay    = 10
        self.epoch_num_batch  = {'train': 400, 'valid':400}

        self.batch_num_spk = 40 # S
        # self.spk_num_utter = 1  # Deprecated and useless; to use multiple utterances from same speaker, use same speaker along self.batch_num_spk
        self.dv_dim = 8
        
        self.data_split_file_number = {}
        # self.data_split_file_number['train'] = make_held_out_file_number(1000, 120)
        # self.data_split_file_number['valid'] = make_held_out_file_number(120, 81)
        # self.data_split_file_number['test']  = make_held_out_file_number(80, 41)
        self.data_split_file_number['train'] = [120, 1000]
        self.data_split_file_number['valid'] = [81, 120]
        self.data_split_file_number['test']  = [41, 80]

        # From cfg: Features
        # self.dv_dim = cfg.dv_dim
        self.wav_sr = cfg.wav_sr
        self.cmp_use_delta = False
        self.frames_silence_to_keep = cfg.frames_silence_to_keep
        self.sil_pad = cfg.sil_pad

        self.speaker_id_list_dict = cfg.speaker_id_list_dict
        self.num_speaker_dict     = cfg.num_speaker_dict

        self.log_except_list = ['data_split_file_number', 'speaker_id_list_dict', 'feat_index', 'sil_index_dict']
        self.log_except_list.extend(['n_mid_0', 'win_start_matrix'])

    def auto_complete(self, cfg):
        ''' Remember to call this after __init__ !!! '''
        self.compute_spk_num_seq() # B
        # Features
        self.nn_feature_dims = cfg.nn_feature_dims[self.y_feat_name]
        self.feat_dim, self.feat_index = compute_feat_dim(self, cfg, self.out_feat_list) # D

        self.num_nn_layers = len(self.nn_layer_config_list)

        # Directories
        self.work_dir = cfg.work_dir
        self.exp_dir  = make_dv_y_exp_dir_name(self, cfg)
        if 'debug' in self.work_dir: self.change_to_debug_mode()
        nnets_file_name = "Model" # self.make_nnets_file_name(cfg)
        self.nnets_file_name = os.path.join(self.exp_dir, nnets_file_name)
        dv_file_name = "DV.dat"
        self.dv_file_name = os.path.join(self.exp_dir, dv_file_name)
        prepare_file_path(file_dir=self.exp_dir, script_name=cfg.python_script_name)
        prepare_file_path(file_dir=self.exp_dir, script_name=self.python_script_name)

        try: self.gpu_id 
        except: self.gpu_id = 0
        self.gpu_per_process_gpu_memory_fraction = 0.8

        self.cfg = cfg
        # self.sil_index_dict = read_sil_index_file(sil_index_file='/home/dawna/tts/mw545/TorchDV/sil_index_list.scp')

    def compute_spk_num_seq(self):
        self.spk_num_seq = int((self.batch_seq_total_len - self.batch_seq_len) / self.batch_seq_shift) + 1  # B

    def reload_model_param(self):
        ''' Change model parameters '''
        ''' Possible changes: sizes of S,B,M '''
        pass

    def change_to_debug_mode(self, process=None):
        if 'debug' in self.work_dir:
            self.batch_num_spk = 1
            for k in self.epoch_num_batch:
                self.epoch_num_batch[k] = 10 
            if '_smallbatch' not in self.exp_dir:
                self.exp_dir = self.exp_dir + '_smallbatch'
            self.num_train_epoch = 5

    def change_to_class_test_mode(self):
        self.epoch_num_batch = {'test':40}
        self.batch_num_spk = 1
        spk_num_utter_list = [1,2,5,10]
        self.spk_num_utter_list = check_and_change_to_list(spk_num_utter_list)
        lambda_u_dict_file_name = 'lambda_u_class_test.dat'
        self.lambda_u_dict_file_name = os.path.join(self.exp_dir, lambda_u_dict_file_name)

        # if self.y_feat_name == 'cmp':
        #     self.batch_seq_shift = 1
        # elif self.y_feat_name == 'wav':
        #     self.batch_seq_shift = 80

        self.compute_spk_num_seq() # B
        self.reload_model_param()
        if 'debug' in self.work_dir: self.change_to_debug_mode(process="class_test")

    def change_to_distance_test_mode(self):
        if self.y_feat_name == 'cmp':
            self.max_len_to_plot = 5
            self.gap_len_to_plot = 1
            self.batch_seq_total_len = 150 # Number of frames at 16kHz
            self.batch_seq_len   = 40 # T
            self.batch_seq_shift = 5
        elif self.y_feat_name == 'wav':
            self.max_len_to_plot = 5*80
            self.gap_len_to_plot = 5
            self.batch_seq_total_len = 12000 # Number of frames at 16kHz
            self.batch_seq_len   = 3200 # T
            self.batch_seq_shift = 5*80

        self.epoch_num_batch = {'test':10*4}
        self.batch_num_spk = int(self.batch_num_spk / 4)
        self.num_to_plot = int(self.max_len_to_plot / self.gap_len_to_plot)

        self.compute_spk_num_seq() # B
        self.reload_model_param()
        if 'debug' in self.work_dir: self.change_to_debug_mode()

    def change_to_gen_h_mode(self):
        self.batch_speaker_list = ['p15', 'p28', 'p122', 'p68'] # Males 2, Females 2
        self.utter_name = '003'
        self.batch_num_spk = len(self.batch_speaker_list)
        self.h_list_file_name = os.path.join(self.exp_dir, "h_spk_list.dat")
        self.file_list_dict = {(spk_id, 'gen'): [spk_id+'_'+self.utter_name] for spk_id in self.batch_speaker_list}

    def additional_action_epoch(self, logger, dv_y_model):
        # Run every epoch, after train and eval; Add tests if necessary
        pass

    def return_n_mid_0_matrix(self):
        '''
         n_mid_0 matrix: cmp index of sub-windows
         temporary storage of matrix
        '''
        try:
            return self.n_mid_0
        except AttributeError:
            self.n_mid_0 = make_n_mid_0_matrix(self)
            return self.n_mid_0

    def return_win_start_0_matrix(self):
        '''
         win_start matrix: starting index of sub-windows
         temporary storage of matrix
        '''
        try:
            return self.win_start_matrix
        except AttributeError:
            self.win_start_matrix = make_win_start_0_matrix(self)
            return self.win_start_matrix

def make_dv_y_exp_dir_name(model_cfg, cfg):
    exp_dir = cfg.work_dir + '/dv_y_%s_lr_%f_' %(model_cfg.y_feat_name, model_cfg.learning_rate)
    for nn_layer_config in model_cfg.nn_layer_config_list:
        layer_str = '%s%i' % (nn_layer_config['type'][:3], nn_layer_config['size'])
        # exp_dir = exp_dir + str(nn_layer_config['type'])[:3] + str(nn_layer_config['size'])
        # {'type':'SinenetV3', 'size':80, 'num_freq':10, 'win_len':self.seq_win_len, 'num_win':self.seq_num_win, 'dropout_p':0, 'batch_norm':False},
        if 'num_freq' in nn_layer_config:
            layer_str = layer_str + 'f' + str(nn_layer_config['num_freq'])
        if 'relu_size' in nn_layer_config:
            layer_str = layer_str + 'r' + str(nn_layer_config['relu_size'])
        if 'batch_norm' in nn_layer_config and nn_layer_config['batch_norm']:
            layer_str = layer_str + 'BN'
        if 'dropout_p' in nn_layer_config and nn_layer_config['dropout_p'] > 0:
            layer_str = layer_str + 'DR'
        exp_dir = exp_dir + layer_str + "_"
    exp_dir = exp_dir + "DV%iS%iB%iT%iD%i" %(model_cfg.dv_dim, model_cfg.batch_num_spk, model_cfg.spk_num_seq, model_cfg.batch_seq_len, model_cfg.feat_dim)
    if model_cfg.use_voiced_only:
        exp_dir = exp_dir + "_vO"+str(model_cfg.use_voiced_threshold)
    return exp_dir

def make_dv_file_list(file_id_list, speaker_id_list, data_split_file_number):
    file_list = {}
    for speaker_id in speaker_id_list:
        file_list[speaker_id], _ = keep_by_speaker(file_id_list, [speaker_id])
        file_list[(speaker_id, 'all')] = file_list[speaker_id]
        for utter_tvt_name in ['train', 'valid', 'test']:
            file_list[(speaker_id, utter_tvt_name)], temp_discard_list = keep_by_file_number(file_list[speaker_id], data_split_file_number[utter_tvt_name])
    return file_list

#############
# Processes #
#############

def train_dv_y_model(cfg, dv_y_cfg):
    train_dv_y_model_v2(cfg, dv_y_cfg)

def train_dv_y_model_v1(cfg, dv_y_cfg):
    numpy.random.seed(545)
    # Feed data use feed_dict style

    logger = make_logger("dv_y_config")
    dv_y_cfg = copy.deepcopy(dv_y_cfg)
    log_class_attri(dv_y_cfg, logger, except_list=dv_y_cfg.log_except_list)

    logger = make_logger("train_dvy")
    logger.info('Creating data lists')
    # dv_y_data_loader = Build_dv_y_data_loader(cfg, dv_y_cfg)

    speaker_id_list = dv_y_cfg.speaker_id_list_dict['train'] # For DV training and evaluation, use train speakers only
    speaker_loader  = list_random_loader(speaker_id_list)
    file_id_list    = read_file_list(cfg.file_id_list_file)
    file_list_dict  = make_dv_file_list(file_id_list, speaker_id_list, dv_y_cfg.data_split_file_number) # In the form of: file_list[(speaker_id, 'train')]    
    make_feed_dict_method_train = dv_y_cfg.make_feed_dict_method_train

    dv_y_model = torch_initialisation(dv_y_cfg)
    dv_y_model.build_optimiser()
    if dv_y_cfg.finetune_model:
        logger.info('Loading %s for finetune' % dv_y_cfg.prev_nnets_file_name)
        dv_y_model.load_nn_model_optim(dv_y_cfg.prev_nnets_file_name)
    dv_y_model.print_model_parameters(logger)

    epoch      = 0
    early_stop = 0
    num_decay  = 0    
    best_valid_loss  = sys.float_info.max
    num_train_epoch  = dv_y_cfg.num_train_epoch
    early_stop_epoch = dv_y_cfg.early_stop_epoch
    max_num_decay    = dv_y_cfg.max_num_decay
    previous_valid_loss = sys.float_info.max

    while (epoch < num_train_epoch):
        epoch = epoch + 1

        logger.info('start training Epoch '+str(epoch))
        epoch_start_time = time.time()
        epoch_train_load_time = 0.
        epoch_train_model_time = 0.

        for batch_idx in range(dv_y_cfg.epoch_num_batch['train']):
            batch_start_time = time.time()
            # logger.info('start loading batch '+str(batch_idx))
            # Draw random speakers
            batch_speaker_list = speaker_loader.draw_n_samples(dv_y_cfg.batch_num_spk)
            # Make feed_dict for training
            feed_dict, batch_size = make_feed_dict_method_train(dv_y_cfg, file_list_dict, cfg.nn_feat_scratch_dirs, batch_speaker_list, utter_tvt='train')
            batch_load_time = time.time()
            epoch_train_load_time += (batch_load_time - batch_start_time)
            # logger.info('start training batch '+str(batch_idx))
            dv_y_model.nn_model.train()
            dv_y_model.update_parameters(feed_dict=feed_dict)
            batch_train_time = time.time()
            epoch_train_model_time += (batch_train_time - batch_load_time)
        epoch_train_time = time.time()

        logger.info('epoch train load time is %s, train model time is %s' % (str(epoch_train_load_time), str(epoch_train_model_time)))

        logger.info('start evaluating Epoch '+str(epoch))
        output_string = {'loss':'epoch %i' % epoch, 'accuracy':'epoch %i' % epoch, 'time':'epoch %i' % epoch}
        epoch_valid_load_time = 0.
        epoch_valid_model_time = 0.
        for utter_tvt_name in ['train', 'valid', 'test']:
            total_batch_size = 0.
            total_loss       = 0.
            total_accuracy   = 0.
            for batch_idx in range(dv_y_cfg.epoch_num_batch['valid']):
                batch_start_time = time.time()
                # Draw random speakers
                batch_speaker_list = speaker_loader.draw_n_samples(dv_y_cfg.batch_num_spk)
                # Make feed_dict for evaluation
                feed_dict, batch_size = make_feed_dict_method_train(dv_y_cfg, file_list_dict, cfg.nn_feat_scratch_dirs, batch_speaker_list, utter_tvt=utter_tvt_name)
                batch_load_time = time.time()
                epoch_valid_load_time += (batch_load_time - batch_start_time)
                dv_y_model.eval()
                with dv_y_model.no_grad():
                    batch_mean_loss = dv_y_model.gen_loss_value(feed_dict=feed_dict)
                    batch_train_time = time.time()
                    epoch_valid_model_time += (batch_train_time - batch_load_time)
                    total_batch_size += batch_size
                    total_loss       += batch_mean_loss
                    if dv_y_cfg.classify_in_training:
                        _c, _t, accuracy = dv_y_model.cal_accuracy(feed_dict=feed_dict)
                        total_accuracy   += accuracy
            average_loss = total_loss/float(dv_y_cfg.epoch_num_batch['valid'])
            output_string['loss'] = output_string['loss'] + ';  %s loss %.4f' % (utter_tvt_name, average_loss)

            if dv_y_cfg.classify_in_training:
                average_accu = total_accuracy/float(dv_y_cfg.epoch_num_batch['valid'])
                output_string['accuracy'] = output_string['accuracy'] + '; %s accuracy %.4f' % (utter_tvt_name, average_accu)

            if utter_tvt_name == 'valid':
                nnets_file_name = dv_y_cfg.nnets_file_name
                # Compare validation error
                valid_error = average_loss
                if valid_error < best_valid_loss:
                    early_stop = 0
                    logger.info('valid error reduced, saving model, %s' % nnets_file_name)
                    dv_y_model.save_nn_model_optim(nnets_file_name)
                    best_valid_loss = valid_error
                elif valid_error > previous_valid_loss:
                    early_stop = early_stop + 1
                    logger.info('valid error increased, early stop %i' % early_stop)
                if (early_stop > early_stop_epoch) and (epoch > dv_y_cfg.warmup_epoch):
                    early_stop = 0
                    num_decay = num_decay + 1
                    if num_decay > max_num_decay:
                        logger.info('stopping early, best model, %s, best valid error %.4f' % (nnets_file_name, best_valid_loss))
                        return best_valid_loss
                    else:
                        new_learning_rate = dv_y_model.learning_rate*0.5
                        logger.info('reduce learning rate to '+str(new_learning_rate)) # Use str(lr) for full length
                        dv_y_model.update_learning_rate(new_learning_rate)
                    logger.info('loading previous best model, %s ' % nnets_file_name)
                    dv_y_model.load_nn_model_optim(nnets_file_name)
                    # logger.info('reduce learning rate to '+str(new_learning_rate))
                    # dv_y_model.update_learning_rate(new_learning_rate)
                previous_valid_loss = valid_error

        logger.info('epoch valid load time is %s, train model time is %s' % (str(epoch_valid_load_time), str(epoch_valid_model_time)))

        epoch_valid_time = time.time()
        output_string['time'] = output_string['time'] + '; train time is %.2f, valid time is %.2f' %((epoch_train_time - epoch_start_time), (epoch_valid_time - epoch_train_time))
        logger.info(output_string['loss'])
        if dv_y_cfg.classify_in_training:
            logger.info(output_string['accuracy'])
        logger.info(output_string['time'])
        

        dv_y_cfg.additional_action_epoch(logger, dv_y_model)

    logger.info('Reach num_train_epoch, best model, %s, best valid error %.4f' % (nnets_file_name, best_valid_loss))
    return best_valid_loss

def train_dv_y_model_v2(cfg, dv_y_cfg):
    ''' New version: data_loader is a class, not a function '''
    numpy.random.seed(545)
    # Feed data use feed_dict style

    logger = make_logger("dv_y_config")
    dv_y_cfg = copy.deepcopy(dv_y_cfg)
    log_class_attri(dv_y_cfg, logger, except_list=dv_y_cfg.log_except_list)

    logger = make_logger("train_dvy")
    logger.info('Creating data loader')
    dv_y_data_loader = Build_dv_y_train_data_loader(cfg, dv_y_cfg)

    dv_y_model = torch_initialisation(dv_y_cfg)
    dv_y_model.build_optimiser()
    if dv_y_cfg.finetune_model:
        logger.info('Loading %s for finetune' % dv_y_cfg.prev_nnets_file_name)
        dv_y_model.load_nn_model_optim(dv_y_cfg.prev_nnets_file_name)
    dv_y_model.print_model_parameters(logger)

    epoch      = 0
    early_stop = 0
    num_decay  = 0    
    best_valid_loss  = sys.float_info.max
    num_train_epoch  = dv_y_cfg.num_train_epoch
    early_stop_epoch = dv_y_cfg.early_stop_epoch
    max_num_decay    = dv_y_cfg.max_num_decay
    previous_valid_loss = sys.float_info.max

    while (epoch < num_train_epoch):
        epoch = epoch + 1

        logger.info('start training Epoch '+str(epoch))
        epoch_start_time = time.time()
        epoch_train_load_time = 0.
        epoch_train_model_time = 0.

        for batch_idx in range(dv_y_cfg.epoch_num_batch['train']):
            batch_start_time = time.time()
            # logger.info('start loading batch '+str(batch_idx))
            # Make feed_dict for training
            feed_dict, batch_size = dv_y_data_loader.make_feed_dict_method_train(utter_tvt='train')
            batch_load_time = time.time()
            epoch_train_load_time += (batch_load_time - batch_start_time)
            # logger.info('start training batch '+str(batch_idx))
            dv_y_model.nn_model.train()
            dv_y_model.update_parameters(feed_dict=feed_dict)
            batch_train_time = time.time()
            epoch_train_model_time += (batch_train_time - batch_load_time)
        epoch_train_time = time.time()

        logger.info('epoch train load time is %s, train model time is %s' % (str(epoch_train_load_time), str(epoch_train_model_time)))

        logger.info('start evaluating Epoch '+str(epoch))
        output_string = {'loss':'epoch %i' % epoch, 'accuracy':'epoch %i' % epoch, 'time':'epoch %i' % epoch}
        epoch_valid_load_time = 0.
        epoch_valid_model_time = 0.
        for utter_tvt_name in ['train', 'valid', 'test']:
            total_batch_size = 0.
            total_loss       = 0.
            total_accuracy   = 0.
            for batch_idx in range(dv_y_cfg.epoch_num_batch['valid']):
                batch_start_time = time.time()
                # Make feed_dict for evaluation
                feed_dict, batch_size = dv_y_data_loader.make_feed_dict_method_train(utter_tvt=utter_tvt_name)
                batch_load_time = time.time()
                epoch_valid_load_time += (batch_load_time - batch_start_time)
                dv_y_model.eval()
                with dv_y_model.no_grad():
                    batch_mean_loss = dv_y_model.gen_loss_value(feed_dict=feed_dict)
                    batch_train_time = time.time()
                    epoch_valid_model_time += (batch_train_time - batch_load_time)
                    total_batch_size += batch_size
                    total_loss       += batch_mean_loss
                    if dv_y_cfg.classify_in_training:
                        _c, _t, accuracy = dv_y_model.cal_accuracy(feed_dict=feed_dict)
                        total_accuracy   += accuracy
            average_loss = total_loss/float(dv_y_cfg.epoch_num_batch['valid'])
            output_string['loss'] = output_string['loss'] + ';  %s loss %.4f' % (utter_tvt_name, average_loss)

            if dv_y_cfg.classify_in_training:
                average_accu = total_accuracy/float(dv_y_cfg.epoch_num_batch['valid'])
                output_string['accuracy'] = output_string['accuracy'] + '; %s accuracy %.4f' % (utter_tvt_name, average_accu)

            if utter_tvt_name == 'valid':
                nnets_file_name = dv_y_cfg.nnets_file_name
                # Compare validation error
                valid_error = average_loss
                if valid_error < best_valid_loss:
                    early_stop = 0
                    logger.info('valid error reduced, saving model, %s' % nnets_file_name)
                    dv_y_model.save_nn_model_optim(nnets_file_name)
                    best_valid_loss = valid_error
                elif valid_error > previous_valid_loss:
                    early_stop = early_stop + 1
                    logger.info('valid error increased, early stop %i' % early_stop)
                if (early_stop > early_stop_epoch) and (epoch > dv_y_cfg.warmup_epoch):
                    early_stop = 0
                    num_decay = num_decay + 1
                    if num_decay > max_num_decay:
                        logger.info('stopping early, best model, %s, best valid error %.4f' % (nnets_file_name, best_valid_loss))
                        return best_valid_loss
                    else:
                        new_learning_rate = dv_y_model.learning_rate*0.5
                        logger.info('reduce learning rate to '+str(new_learning_rate)) # Use str(lr) for full length
                        dv_y_model.update_learning_rate(new_learning_rate)
                    logger.info('loading previous best model, %s ' % nnets_file_name)
                    dv_y_model.load_nn_model_optim(nnets_file_name)
                    # logger.info('reduce learning rate to '+str(new_learning_rate))
                    # dv_y_model.update_learning_rate(new_learning_rate)
                previous_valid_loss = valid_error

        logger.info('epoch valid load time is %s, train model time is %s' % (str(epoch_valid_load_time), str(epoch_valid_model_time)))

        epoch_valid_time = time.time()
        output_string['time'] = output_string['time'] + '; train time is %.2f, valid time is %.2f' %((epoch_train_time - epoch_start_time), (epoch_valid_time - epoch_train_time))
        logger.info(output_string['loss'])
        if dv_y_cfg.classify_in_training:
            logger.info(output_string['accuracy'])
        logger.info(output_string['time'])
        

        dv_y_cfg.additional_action_epoch(logger, dv_y_model)

    logger.info('Reach num_train_epoch, best model, %s, best valid error %.4f' % (nnets_file_name, best_valid_loss))
    return best_valid_loss

def class_test_dv_y_model(cfg, dv_y_cfg):
    class_test_dv_y_model_v2(cfg, dv_y_cfg)

def class_test_dv_y_model_v1(cfg, dv_y_cfg):
    numpy.random.seed(546)
    # Classification test
    # Also generates lambda_u per utterance; store in lambda_u_dict[file_name]
    # Use test utterances only
    # Make or load lambda_u_dict
    # lambda_u_dict[file_name] = [lambda_u, B_u]
    # Draw random groups of files, weighted average lambda, then classify

    logger = make_logger("dv_y_config")
    dv_y_cfg = copy.deepcopy(dv_y_cfg)
    dv_y_cfg.change_to_class_test_mode()
    log_class_attri(dv_y_cfg, logger, except_list=dv_y_cfg.log_except_list)

    logger = make_logger("class_dvy")
    logger.info('Creating data lists')
    speaker_id_list = dv_y_cfg.speaker_id_list_dict['train'] # For classification, use train speakers only
    file_id_list    = read_file_list(cfg.file_id_list_file)
    file_list_dict  = make_dv_file_list(file_id_list, speaker_id_list, dv_y_cfg.data_split_file_number) # In the form of: file_list[(speaker_id, 'train')]
    make_feed_dict_method_test = dv_y_cfg.make_feed_dict_method_test

    dv_y_model = torch_initialisation(dv_y_cfg)
    dv_y_model.load_nn_model(dv_y_cfg.nnets_file_name)
    dv_y_model.eval()

    try: 
        lambda_u_dict = pickle.load(open(dv_y_cfg.lambda_u_dict_file_name, 'rb'))
        logger.info('Loaded lambda_u_dict from %s' % dv_y_cfg.lambda_u_dict_file_name)
    # Generate
    except:
        logger.info('Cannot load from %s, generate instead' % dv_y_cfg.lambda_u_dict_file_name)
        lambda_u_dict = {}   # lambda_u[file_name] = [lambda_speaker, total_batch_size]
        for speaker_id in speaker_id_list:
            logger.info('Generating %s' % speaker_id)
            for file_name in file_list_dict[(speaker_id, 'test')]:
                lambda_temp_list = []
                batch_size_list  = []
                gen_finish = False
                start_frame_index = 0
                BTD_feat_remain = None
                while not (gen_finish):
                    feed_dict, gen_finish, batch_size, BTD_feat_remain = make_feed_dict_method_test(dv_y_cfg, cfg.nn_feat_scratch_dirs, speaker_id, file_name, start_frame_index, BTD_feat_remain)
                    with dv_y_model.no_grad():
                        lambda_temp = dv_y_model.gen_lambda_SBD_value(feed_dict=feed_dict)
                    lambda_temp_list.append(lambda_temp)
                    batch_size_list.append(batch_size)
                B_u = numpy.sum(batch_size_list)
                lambda_u = numpy.zeros(dv_y_cfg.dv_dim)
                for lambda_temp, batch_size in zip(lambda_temp_list, batch_size_list):
                    for b in range(batch_size):
                        lambda_u += lambda_temp[0,b]
                lambda_u /= float(B_u)
                lambda_u_dict[file_name] = [lambda_u, B_u]
        logger.info('Saving lambda_u_dict to %s' % dv_y_cfg.lambda_u_dict_file_name)
        pickle.dump(lambda_u_dict, open(dv_y_cfg.lambda_u_dict_file_name, 'wb'))

    # Classify
    for spk_num_utter in dv_y_cfg.spk_num_utter_list:
        logger.info('Testing with %i utterances per speaker' % spk_num_utter)
        accuracy_list = []
        for speaker_id in speaker_id_list:
            logger.info('testing speaker %s' % speaker_id)
            speaker_lambda_list = []
            speaker_file_loader = list_random_loader(file_list_dict[(speaker_id, 'test')])
            for batch_idx in range(dv_y_cfg.epoch_num_batch['test']):
                logger.info('batch %i' % batch_idx)
                batch_file_list = speaker_file_loader.draw_n_samples(spk_num_utter)

                # Weighted average of lambda_u
                batch_lambda = numpy.zeros(dv_y_cfg.dv_dim)
                B_total = 0.
                for file_name in batch_file_list:
                    lambda_u, B_u = lambda_u_dict[file_name]
                    batch_lambda += lambda_u * B_u
                    B_total += B_u
                batch_lambda /= B_total
                speaker_lambda_list.append(batch_lambda)

            true_speaker_index = dv_y_cfg.speaker_id_list_dict['train'].index(speaker_id)
            B_remain = dv_y_cfg.epoch_num_batch['test']
            b_index = 0 # Track counter, instead of removing elements
            correct_counter = 0.
            while B_remain > 0:
                lambda_val = numpy.zeros((dv_y_cfg.batch_num_spk, dv_y_cfg.spk_num_seq, dv_y_cfg.dv_dim))
                if B_remain > dv_y_cfg.spk_num_seq:
                    # Fill all dv_y_cfg.spk_num_seq, keep remain for later
                    B_actual = dv_y_cfg.spk_num_seq
                    B_remain -= dv_y_cfg.spk_num_seq
                else:
                    # No more remain
                    B_actual = B_remain
                    B_remain = 0

                for b in range(B_actual):
                    lambda_val[0, b] = speaker_lambda_list[b_index + b]

                # Set up for next round (if dv_y_cfg.spk_num_seq)
                b_index += B_actual

                feed_dict = {'lambda': lambda_val}
                with dv_y_model.no_grad():
                    idx_list_S_B = dv_y_model.lambda_to_indices(feed_dict=feed_dict)
                for b in range(B_actual):
                    if idx_list_S_B[0, b] == true_speaker_index: 
                        correct_counter += 1.
            speaker_accuracy = correct_counter/float(dv_y_cfg.epoch_num_batch['test'])
            logger.info('speaker %s accuracy is %f' % (speaker_id, speaker_accuracy))
            accuracy_list.append(speaker_accuracy)
        mean_accuracy = numpy.mean(accuracy_list)
        logger.info('Accuracy with %i utterances per speaker is %.4f' % (spk_num_utter, mean_accuracy))

def class_test_dv_y_model_v2(cfg, dv_y_cfg):
    numpy.random.seed(546)
    # Classification test
    # Also generates lambda_u per utterance; store in lambda_u_dict[file_name]
    # Use test utterances only
    # Make or load lambda_u_dict
    # lambda_u_dict[file_name] = [lambda_u, B_u]
    # Draw random groups of files, weighted average lambda, then classify

    logger = make_logger("dv_y_config")
    dv_y_cfg = copy.deepcopy(dv_y_cfg)
    dv_y_cfg.change_to_class_test_mode()
    log_class_attri(dv_y_cfg, logger, except_list=dv_y_cfg.log_except_list)

    logger = make_logger("class_dvy")
    logger.info('Creating data lists')
    dv_y_data_loader = Build_dv_y_test_data_loader(cfg, dv_y_cfg)

    dv_y_model = torch_initialisation(dv_y_cfg)
    dv_y_model.load_nn_model(dv_y_cfg.nnets_file_name)
    dv_y_model.eval()

    try: 
        lambda_u_dict = pickle.load(open(dv_y_cfg.lambda_u_dict_file_name, 'rb'))
        logger.info('Loaded lambda_u_dict from %s' % dv_y_cfg.lambda_u_dict_file_name)
    # Generate
    except:
        logger.info('Cannot load from %s, generate instead' % dv_y_cfg.lambda_u_dict_file_name)
        lambda_u_dict = {}   # lambda_u[file_name] = [lambda_speaker, total_batch_size]
        for speaker_id in dv_y_data_loader.speaker_id_list:
            logger.info('Generating %s' % speaker_id)
            for file_id in dv_y_data_loader.file_list_dict[(speaker_id, 'test')]:
                # logger.info('Generating %s' % file_id)
                lambda_temp_list = []
                batch_size_list  = []
                dv_y_data_loader.file_gen_finish = False
                dv_y_data_loader.load_new_file_bool = True
                while not (dv_y_data_loader.file_gen_finish):
                    feed_dict, batch_size = dv_y_data_loader.make_feed_dict_method_test(speaker_id, file_id)
                    with dv_y_model.no_grad():
                        lambda_temp = dv_y_model.gen_lambda_SBD_value(feed_dict=feed_dict)
                    lambda_temp_list.append(lambda_temp)
                    batch_size_list.append(batch_size)
                B_u = numpy.sum(batch_size_list)
                lambda_u = numpy.zeros(dv_y_cfg.dv_dim)
                for lambda_temp, batch_size in zip(lambda_temp_list, batch_size_list):
                    for b in range(batch_size):
                        lambda_u += lambda_temp[0,b]
                lambda_u /= float(B_u)
                lambda_u_dict[file_id] = [lambda_u, B_u]
        logger.info('Saving lambda_u_dict to %s' % dv_y_cfg.lambda_u_dict_file_name)
        pickle.dump(lambda_u_dict, open(dv_y_cfg.lambda_u_dict_file_name, 'wb'))

    # for k in lambda_u_dict:
    #     print(k)
    #     print(lambda_u_dict[k])      # [array([nan, nan, nan, nan, nan, nan, nan, nan]), 0.0]

    # Classify
    for spk_num_utter in dv_y_cfg.spk_num_utter_list:
        logger.info('Testing with %i utterances per speaker' % spk_num_utter)
        accuracy_list = []
        for speaker_id in dv_y_data_loader.speaker_id_list:
            logger.info('testing speaker %s' % speaker_id)
            speaker_lambda_list = []
            for batch_idx in range(dv_y_cfg.epoch_num_batch['test']):
                logger.info('batch %i' % batch_idx)
                batch_file_list = dv_y_data_loader.file_loader_dict[(speaker_id, 'test')].draw_n_samples(spk_num_utter)

                # Weighted average of lambda_u
                batch_lambda = numpy.zeros(dv_y_cfg.dv_dim)
                B_total = 0.
                for file_id in batch_file_list:
                    lambda_u, B_u = lambda_u_dict[file_id]
                    batch_lambda += lambda_u * B_u
                    B_total += B_u
                batch_lambda /= B_total
                speaker_lambda_list.append(batch_lambda)

            true_speaker_index = dv_y_cfg.speaker_id_list_dict['train'].index(speaker_id)
            B_remain = dv_y_cfg.epoch_num_batch['test']
            b_index = 0 # Track counter, instead of removing elements
            correct_counter = 0.
            while B_remain > 0:
                lambda_val = numpy.zeros((dv_y_cfg.batch_num_spk, dv_y_cfg.spk_num_seq, dv_y_cfg.dv_dim))
                if B_remain > dv_y_cfg.spk_num_seq:
                    # Fill all dv_y_cfg.spk_num_seq, keep remain for later
                    B_actual = dv_y_cfg.spk_num_seq
                    B_remain -= dv_y_cfg.spk_num_seq
                else:
                    # No more remain
                    B_actual = B_remain
                    B_remain = 0

                for b in range(B_actual):
                    lambda_val[0, b] = speaker_lambda_list[b_index + b]

                # Set up for next round (if dv_y_cfg.spk_num_seq)
                b_index += B_actual

                feed_dict = {'lambda': lambda_val}
                with dv_y_model.no_grad():
                    idx_list_S_B = dv_y_model.lambda_to_indices(feed_dict=feed_dict)
                for b in range(B_actual):
                    if idx_list_S_B[0, b] == true_speaker_index: 
                        correct_counter += 1.
            speaker_accuracy = correct_counter/float(dv_y_cfg.epoch_num_batch['test'])
            logger.info('speaker %s accuracy is %f' % (speaker_id, speaker_accuracy))
            accuracy_list.append(speaker_accuracy)
        mean_accuracy = numpy.mean(accuracy_list)
        logger.info('Accuracy with %i utterances per speaker is %.4f' % (spk_num_utter, mean_accuracy))

def distance_test_dv_y_model(cfg, dv_y_cfg, test_type='Euc'):
    numpy.random.seed(547)
    '''
        Make a list of feed_dict, compute distance between them
        test_type='Euc': Generate lambdas, compute euclidean distance, [lambda_i - lambda_0]^2
        test_type='EucNorm': Generate lambdas, compute euclidean distance, [(lambda_i - lambda_0)/lambda_0]^2; 
            cannot implement, contain 0 issue
        test_type='BiCE': Generate probabilities, make binary (correct/wrong), cross-entropy; -sum(p_0 log(p_i))
            not implemented yet
    '''
    logger = make_logger("dv_y_config")
    dv_y_cfg = copy.deepcopy(dv_y_cfg)
    dv_y_cfg.change_to_distance_test_mode()
    log_class_attri(dv_y_cfg, logger, except_list=dv_y_cfg.log_except_list)

    logger = make_logger("dist_dvy")
    logger.info('Creating data lists')
    speaker_id_list = dv_y_cfg.speaker_id_list_dict['train'] # For DV training and evaluation, use train speakers only
    speaker_loader  = list_random_loader(speaker_id_list)
    file_id_list    = read_file_list(cfg.file_id_list_file)
    file_list_dict  = make_dv_file_list(file_id_list, speaker_id_list, dv_y_cfg.data_split_file_number) # In the form of: file_list[(speaker_id, 'train')]
    make_feed_dict_method_distance = dv_y_cfg.make_feed_dict_method_distance

    dv_y_model = torch_initialisation(dv_y_cfg)
    dv_y_model.load_nn_model(dv_y_cfg.nnets_file_name)
    dv_y_model.eval()
    
    distance_sum = [0.] * (dv_y_cfg.num_to_plot+1) # First is 0.; keep here for easier plotting
    num_batch = dv_y_cfg.epoch_num_batch['test']

    for batch_idx in range(num_batch):
        logger.info('start generating for batch '+str(batch_idx+1))
        # Draw random speakers
        batch_speaker_list = speaker_loader.draw_n_samples(dv_y_cfg.batch_num_spk)
        # Make feed_dict for training
        feed_dict_list, batch_size = make_feed_dict_method_distance(dv_y_cfg, file_list_dict, cfg.nn_feat_scratch_dirs, batch_speaker_list, utter_tvt='train')
        with dv_y_model.no_grad():
            if test_type == 'Euc':
                for plot_idx in range(dv_y_cfg.num_to_plot+1):
                    lambda_temp = dv_y_model.gen_lambda_SBD_value(feed_dict=feed_dict_list[plot_idx])
                    if plot_idx == 0:
                        lambda_0 = lambda_temp
                    else:
                        dist_temp = compute_Euclidean_distance(lambda_temp, lambda_0)
                        distance_sum[plot_idx] += dist_temp
            if test_type == 'BiCE':
                for plot_idx in range(dv_y_cfg.num_to_plot+1):
                    p_temp = dv_y_model.gen_p_SBD_value(feed_dict=feed_dict_list[plot_idx])
                    # TODO: make binary p_temp_binary
                    if plot_idx == 0:
                        p_0_binary = p_temp_binary
                    else:
                        # TODO: implement cross entropy
                        # dist_temp = compute_cross_entropy(p_temp_binary, p_0_binary)
                        distance_sum[plot_idx] += dist_temp
        del feed_dict_list # Save memory

    logger.info('Printing %s distances' % test_type)
    num_lambda = batch_size*num_batch
    print([float(distance_sum[i]/(num_lambda)) for i in range(dv_y_cfg.num_to_plot+1)])

def plot_sinenet(cfg, dv_y_cfg):
    numpy.random.seed(548)
    '''
    Plot all filters in sinenet
    If use real data, plot real data too
    '''
    logger = make_logger("dv_y_config")
    dv_y_cfg = copy.deepcopy(dv_y_cfg)
    dv_y_cfg.batch_num_spk = 1
    dv_y_cfg.spk_num_seq   = 20 # Use this for different frequency
    dv_y_cfg.seq_num_win   = 40 # Use this for different pitch locations    
    log_class_attri(dv_y_cfg, logger, except_list=dv_y_cfg.log_except_list)

    dv_y_model = torch_initialisation(dv_y_cfg)
    dv_y_model.load_nn_model(dv_y_cfg.nnets_file_name)
    dv_y_model.eval()

    BTD_feat_remain = None
    start_frame_index = int(51000)
    speaker_id = 'p153'
    file_name  = 'p153_003'
    make_feed_dict_method_test = dv_y_cfg.make_feed_dict_method_test
    feed_dict, gen_finish, batch_size, BTD_feat_remain = make_feed_dict_method_test(dv_y_cfg, cfg.nn_feat_scratch_dirs, speaker_id, file_name, start_frame_index, BTD_feat_remain)

    W_a_b = dv_y_model.nn_model.layer_list[0].layer_fn.sinenet_layer.fc_fn.weight.cpu().detach().numpy()
    W_s_c = dv_y_model.gen_w_sin_cos(feed_dict)
    x = feed_dict['x']

    print(W_a_b.shape) # (80, 32)
    print(W_s_c.shape) # (1, 20, 40, 32, 640)
    print(x.shape)     # (1, 12000)

    num_freq = 16
    seq_win_len = 640
    sine_size = 80

    W_s_c = W_s_c[0,0,0]
    x = x[0,start_frame_index:start_frame_index+seq_win_len]

    print(W_a_b.shape) # (80, 32)
    print(W_s_c.shape) # (32, 640)
    print(x.shape)     # (640,)

    # Heatmap of W_a_b
    fig, ax = plt.subplots()
    im = ax.imshow(W_a_b)
    fig_name = '/home/dawna/tts/mw545/Export_Temp/PNG_out/' + "heatmap.png"
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('mag', rotation=-90, va="bottom")
    logger.info('Saving heatmap to %s' % fig_name)
    fig.savefig(fig_name)
    plt.close(fig)

    W_ab_combine = numpy.zeros((sine_size, num_freq))
    for i in range(sine_size):
        for j in range(num_freq):
            a = W_a_b[i,j]
            b = W_a_b[i,j+num_freq]
            W_ab_combine[i,j] = numpy.sqrt(numpy.square(a)+numpy.square(b))

    # Heatmap of W_ab_combine
    fig, ax = plt.subplots()
    im = ax.imshow(W_ab_combine)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('', rotation=-90, va="bottom")
    ax.set_title('Amplitude; 80 filters, 16 frequencies')
    fig_name = '/home/dawna/tts/mw545/Export_Temp/PNG_out/' + "heatmap_combine.png"
    logger.info('Saving heatmap to %s' % fig_name)
    fig.savefig(fig_name)
    plt.close(fig)

    phase_ab_combine = numpy.zeros((sine_size, num_freq))
    for i in range(sine_size):
        for j in range(num_freq):
            a = W_a_b[i,j]
            b = W_a_b[i,j+num_freq]
            phase_ab_combine[i,j] = numpy.arctan(a/b)

    # Heatmap of phase_ab_combine
    fig, ax = plt.subplots()
    im = ax.imshow(phase_ab_combine)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('', rotation=-90, va="bottom")
    ax.set_title('Phase; 80 filters, 16 frequencies')
    fig_name = '/home/dawna/tts/mw545/Export_Temp/PNG_out/' + "heatmap_phase.png"
    logger.info('Saving heatmap to %s' % fig_name)
    fig.savefig(fig_name)
    plt.close(fig)

    # Combine the filters by frequency?

    W_absc = numpy.zeros((sine_size, seq_win_len))
    for i in range(sine_size):
        for j in range(seq_win_len):
            a = W_a_b[i,0]
            b = W_a_b[i,num_freq]
            s = W_s_c[0,j]
            c = W_s_c[num_freq,j]

            W_absc[i,j] = a * s + b * c

    # Plot the lowest frequency only, for all filters
    D_size = 5
    D_tot  = int(sine_size/D_size)
    for d_1 in range(D_tot):
        fig, ax_list = plt.subplots(D_size+1)        

        for d_2 in range(D_size):
            d = d_1 * D_size + d_2
            ax_list[d_2].plot(W_absc[d])
        # Plot x as well
        ax_list[D_size].plot(x)

        fig_name = '/home/dawna/tts/mw545/Export_Temp/PNG_out/' + "sinenet_filter_%i.png" % d_1
        logger.info('Saving h to %s' % fig_name)
        fig.savefig(fig_name)
        plt.close(fig)







    # W = dv_y_model.gen_w_mul_w_sin_cos(feed_dict)

    # S = dv_y_cfg.batch_num_spk
    # B = dv_y_cfg.spk_num_seq
    # M = dv_y_cfg.seq_num_win
    # D = 80
    
    # D_size = 5
    # D_tot  = int(D/D_size)

    # for d_1 in range(D_tot):
    #     fig, ax_list = plt.subplots(D_size+1)        

    #     for d_2 in range(D_size):
    #         d = d_1 * D_size + d_2
    #         ax_list[d_2].plot(W[0,0,0,d])
    #     if is_use_true_data:
    #         # Plot x as well
    #         x = feed_dict['x']
    #         ax_list[D_size].plot(x[0,0,0])

    #     fig_name = '/home/dawna/tts/mw545/Export_Temp/PNG_out/' + "sinenet_filter_%i.png" % d_1
    #     logger.info('Saving h to %s' % fig_name)
    #     fig.savefig(fig_name)
    #     plt.close(fig)

def plot_sinenet_old(cfg, dv_y_cfg):
    numpy.random.seed(548)
    '''
    Plot all filters in sinenet
    If use real data, plot real data too
    '''
    logger = make_logger("dv_y_config")
    dv_y_cfg = copy.deepcopy(dv_y_cfg)
    dv_y_cfg.batch_num_spk = 1
    dv_y_cfg.spk_num_seq   = 20 # Use this for different frequency
    dv_y_cfg.seq_num_win   = 40 # Use this for different pitch locations    
    log_class_attri(dv_y_cfg, logger, except_list=dv_y_cfg.log_except_list)

    dv_y_model = torch_initialisation(dv_y_cfg)
    dv_y_model.load_nn_model(dv_y_cfg.nnets_file_name)
    dv_y_model.eval()

    is_use_true_data = True
    if is_use_true_data:
        BTD_feat_remain = None
        start_frame_index = int(51000)
        speaker_id = 'p15'
        file_name  = 'p15_003'
        make_feed_dict_method_test = dv_y_cfg.make_feed_dict_method_test
        feed_dict, gen_finish, batch_size, BTD_feat_remain = make_feed_dict_method_test(dv_y_cfg, cfg.nn_feat_scratch_dirs, speaker_id, file_name, start_frame_index, BTD_feat_remain)
    else:
        f_0     = 120.
        f_inc   = 10.
        tau_0   = 0.
        tau_inc = 10./16000
        f = numpy.zeros((dv_y_cfg.batch_num_spk, dv_y_cfg.spk_num_seq, dv_y_cfg.seq_num_win))
        for i in range(dv_y_cfg.spk_num_seq):
            f[0,i] = f_0 + i * f_inc
        lf = numpy.log(f)
        log_f_mean = 5.04418
        log_f_std  = 0.358402
        nlf = (lf - log_f_mean) / log_f_std
        # lf = torch.add(torch.mul(nlf, self.log_f_std), self.log_f_mean) # S*B*M
        tau = numpy.zeros((dv_y_cfg.batch_num_spk, dv_y_cfg.spk_num_seq, dv_y_cfg.seq_num_win))
        for i in range(dv_y_cfg.seq_num_win):
            tau[0,:,i] = tau_0 + i * tau_inc
        feed_dict = {'nlf': nlf, 'tau': tau}

    W = dv_y_model.gen_w_mul_w_sin_cos(feed_dict)

    S = dv_y_cfg.batch_num_spk
    B = dv_y_cfg.spk_num_seq
    M = dv_y_cfg.seq_num_win
    D = 80
    
    D_size = 5
    D_tot  = int(D/D_size)

    for d_1 in range(D_tot):
        fig, ax_list = plt.subplots(D_size+1)        

        for d_2 in range(D_size):
            d = d_1 * D_size + d_2
            ax_list[d_2].plot(W[0,0,0,d])
        if is_use_true_data:
            # Plot x as well
            x = feed_dict['x']
            ax_list[D_size].plot(x[0,0,0])

        fig_name = '/home/dawna/tts/mw545/Export_Temp/PNG_out/' + "sinenet_filter_%i.png" % d_1
        logger.info('Saving h to %s' % fig_name)
        fig.savefig(fig_name)
        plt.close(fig)

def vuv_test_dv_y_model(cfg, dv_y_cfg):
    numpy.random.seed(549)
    '''
    Run the evaluation part of the training procedure
    Store the results based on v/uv
    Print: amount of data, and CE, vs amount of v/uv
    '''
    logger = make_logger("dv_y_config")
    dv_y_cfg = copy.deepcopy(dv_y_cfg)
    log_class_attri(dv_y_cfg, logger, except_list=dv_y_cfg.log_except_list)

    # Need to extract vuv information
    if 'vuv' not in dv_y_cfg.out_feat_list:
        dv_y_cfg.out_feat_list.append('vuv')

    logger = make_logger("vuv_test_dvy")
    logger.info('Creating data lists')
    speaker_id_list = dv_y_cfg.speaker_id_list_dict['train'] # For DV training and evaluation, use train speakers only
    speaker_loader  = list_random_loader(speaker_id_list)
    file_id_list    = read_file_list(cfg.file_id_list_file)
    file_list_dict  = make_dv_file_list(file_id_list, speaker_id_list, dv_y_cfg.data_split_file_number) # In the form of: file_list[(speaker_id, 'train')]
    make_feed_dict_method_vuv_test = dv_y_cfg.make_feed_dict_method_vuv_test

    dv_y_model = torch_initialisation(dv_y_cfg)
    dv_y_model.load_nn_model(dv_y_cfg.nnets_file_name)
    dv_y_model.eval()

    for utter_tvt_name in ['train', 'valid', 'test']:
        ce_holders = [[] for i in range(dv_y_cfg.seq_num_win+1)]
        # vuv = numpy.zeros((dv_y_cfg.batch_num_spk, dv_y_cfg.spk_num_seq, dv_y_cfg.seq_num_win))
        num_batch = dv_y_cfg.epoch_num_batch['valid'] * 10
        # num_batch = 1
        for batch_idx in range(num_batch):
            # Draw random speakers
            batch_speaker_list = speaker_loader.draw_n_samples(dv_y_cfg.batch_num_spk)
            # Make feed_dict for evaluation
            feed_dict, batch_size = make_feed_dict_method_vuv_test(dv_y_cfg, file_list_dict, cfg.nn_feat_scratch_dirs, batch_speaker_list, utter_tvt=utter_tvt_name)
            vuv_SBM = feed_dict['vuv']
            with dv_y_model.no_grad():
                ce_SB = dv_y_model.gen_SB_loss_value(feed_dict=feed_dict) # 1D vector
                vuv_SB = numpy.sum(vuv_SBM, axis=2).reshape(-1)
                s_b = dv_y_cfg.batch_num_spk * dv_y_cfg.spk_num_seq
                for i in range(s_b):
                    vuv = int(vuv_SB[i])
                    ce  = int(ce_SB[i])
                    ce_holders[vuv].append(ce)

        len_list = [len(ce_list) for ce_list in ce_holders]
        mean_list = [numpy.mean(ce_list) for ce_list in ce_holders]
        print(len_list)
        print(mean_list)
        ce_sum = 0.
        num_sum = 0
        for (l,m) in zip(len_list, mean_list):
            ce_sum += l*m
            num_sum += l
        ce_mean = ce_sum / float(num_sum)
        logger.info('Mean Cross Entropy Results of %s Dataset is %.4f' % (utter_tvt_name, ce_mean))

def ce_vs_var_nlf_test(cfg, dv_y_cfg):
    numpy.random.seed(550)
    '''
    Run the evaluation part of the training procedure
    Print: amount of data, and CE, vs amount of var(nlf); only use voiced region data
    '''
    logger = make_logger("dv_y_config")
    dv_y_cfg = copy.deepcopy(dv_y_cfg)
    log_class_attri(dv_y_cfg, logger, except_list=dv_y_cfg.log_except_list)
    if 'vuv' not in dv_y_cfg.out_feat_list:
        dv_y_cfg.out_feat_list.append('vuv')
    if 'nlf' not in dv_y_cfg.out_feat_list:
        dv_y_cfg.out_feat_list.append('nlf')

    logger = make_logger("vuv_test_dvy")
    logger.info('Creating data lists')
    speaker_id_list = dv_y_cfg.speaker_id_list_dict['train'] # For DV training and evaluation, use train speakers only
    speaker_loader  = list_random_loader(speaker_id_list)
    file_id_list    = read_file_list(cfg.file_id_list_file)
    file_list_dict  = make_dv_file_list(file_id_list, speaker_id_list, dv_y_cfg.data_split_file_number) # In the form of: file_list[(speaker_id, 'train')]
    make_feed_dict_method_train = dv_y_cfg.make_feed_dict_method_train

    dv_y_model = torch_initialisation(dv_y_cfg)
    dv_y_model.load_nn_model(dv_y_cfg.nnets_file_name)
    dv_y_model.eval()

    dv_y_cfg.out_feat_list.append('nlf_var')
    nlf_var_list = {}
    ce_list = {}

    # DNN-based model part
    from exp_mw545.exp_dv_wav_subwin import dv_y_wav_subwin_configuration
    cfg_dnn = copy.deepcopy(cfg)
    cfg_dnn.work_dir = "/home/dawna/tts/mw545/TorchDV/dv_wav_subwin"
    dnn_cfg = dv_y_wav_subwin_configuration(cfg_dnn)
    dnn_model = torch_initialisation(dnn_cfg)
    dnn_model.load_nn_model(dnn_cfg.nnets_file_name)
    dnn_model.eval()

    for utter_tvt_name in ['train', 'valid', 'test']:
        nlf_var_list[utter_tvt_name] = []
        ce_list[utter_tvt_name] = []
        # vuv = numpy.zeros((dv_y_cfg.batch_num_spk, dv_y_cfg.spk_num_seq, dv_y_cfg.seq_num_win))
        num_batch = dv_y_cfg.epoch_num_batch['valid']
        # num_batch = 1
        for batch_idx in range(num_batch):
            # Draw random speakers
            batch_speaker_list = speaker_loader.draw_n_samples(dv_y_cfg.batch_num_spk)
            # Make feed_dict for evaluation
            feed_dict, batch_size = make_feed_dict_method_train(dv_y_cfg, file_list_dict, cfg.nn_feat_scratch_dirs, batch_speaker_list, utter_tvt=utter_tvt_name)
            nlf_var = feed_dict['nlf_var']
            vuv_SBM = feed_dict['vuv']
            with dv_y_model.no_grad():
                ce_SB = dv_y_model.gen_SB_loss_value(feed_dict=feed_dict) # 1D vector
            with dnn_model.no_grad():
                ce_SB_dnn = dnn_model.gen_SB_loss_value(feed_dict=feed_dict) # 1D vector
                vuv_SB = numpy.sum(vuv_SBM, axis=2).reshape(-1)
                nlf_var_SB = numpy.sum(nlf_var, axis=2).reshape(-1)
                s_b = dv_y_cfg.batch_num_spk * dv_y_cfg.spk_num_seq
                for i in range(s_b):
                    if vuv_SB[i] == dv_y_cfg.seq_num_win:
                        nlf_var_list[utter_tvt_name].append(nlf_var_SB[i])
                        ce_list[utter_tvt_name].append(ce_SB_dnn[i]-ce_SB[i])
        from scipy.stats.stats import pearsonr
        corr_coef = pearsonr(nlf_var_list[utter_tvt_name], ce_list[utter_tvt_name])
        print(corr_coef)
        # logger.info('Corr coef is %4.f for %s' % (corr_coef, utter_tvt_name))
        # fig, ax = plt.subplots()
        # ax.plot(nlf_var_list, ce_list, '.')
        # ax.set_title('ce vs var(nlf) %s' % utter_tvt_name)
        # fig_name = '/home/dawna/tts/mw545/Export_Temp/PNG_out/' + "ce_var_nlf_%s.png" % utter_tvt_name
        # logger.info('Saving to %s' % fig_name)
        # fig.savefig(fig_name)
        # plt.close(fig)
    # Dump results
    pickle.dump(nlf_var_list, open(os.path.join(dv_y_cfg.exp_dir, 'nlf_var_list.data'), 'wb'))
    pickle.dump(ce_list, open(os.path.join(dv_y_cfg.exp_dir, 'ce_list.data'), 'wb'))


########################
# Data loading methods #
########################


class Build_dv_y_data_loader(object):
    def __init__(self, cfg, dv_y_cfg):
        super().__init__()
        self.cfg = copy.deepcopy(cfg)
        self.dv_y_cfg = copy.deepcopy(dv_y_cfg)
        self.nn_data_dir = dv_y_cfg.nn_data_dir
        self.logger = make_logger("Data_Loader")

        self.file_id_list    = read_file_list(cfg.file_id_used_list_file)
        self.speaker_id_list = dv_y_cfg.speaker_id_list_dict['train'] # For DV training and evaluation, use train speakers only
        self.speaker_loader  = list_random_loader(self.speaker_id_list)
        # self.file_list_dict  = self.make_dv_file_list(self.file_id_list, self.speaker_id_list, dv_y_cfg.data_split_file_number) # In the form of: file_list[(speaker_id, 'train')]
        self.file_list_dict = self.make_dv_file_list(self.file_id_list, self.speaker_id_list, dv_y_cfg.data_split_file_number) # In the form of: file_loader_dict[(speaker_id, 'train')]
        self.file_loader_dict = {k: list_random_loader(self.file_list_dict[k]) for k in self.file_list_dict}

        self.feed_dict = {}
        # Numpy holders
        self.one_hot = numpy.zeros((dv_y_cfg.batch_num_spk))
        # For file starting index; either frame or sample
        self.file_start_index_list = numpy.zeros((dv_y_cfg.batch_num_spk))

        # Additional Initialisation depending on features to load
        if 'wav' in dv_y_cfg.out_feat_list:
            self.init_wav()

        # File finish flag and remain dict for lambda generation for classification
        self.file_gen_finish = False
        self.file_gen_remain = {}

    def make_dv_file_list(self, file_id_list, speaker_id_list, data_split_file_number):
        ''' Make a dict of list_random_loader, In the form of: file_loader_dict[(speaker_id, 'train')] '''
        file_list_dict = {}
        all_remain_file_list = file_id_list
        for speaker_id in speaker_id_list:
            speaker_remain_file_list, all_remain_file_list = keep_by_speaker(all_remain_file_list, [speaker_id])
            for utter_tvt_name in ['train', 'valid', 'test']:
                file_list_dict[(speaker_id, utter_tvt_name)], speaker_remain_file_list = keep_by_file_number(speaker_remain_file_list, data_split_file_number[utter_tvt_name])

        return file_list_dict

    def make_one_hot_y_val(self, batch_speaker_list):
        dv_y_cfg = self.dv_y_cfg
        # Make classification targets, index sequence
        for speaker_idx in range(dv_y_cfg.batch_num_spk):
            speaker_id = batch_speaker_list[speaker_idx]
            true_speaker_index = dv_y_cfg.speaker_id_list_dict['train'].index(speaker_id)
            self.one_hot[speaker_idx] = true_speaker_index
        if dv_y_cfg.train_by_window:
            # S --> S*B
            y_val = numpy.repeat(self.one_hot, dv_y_cfg.spk_num_seq)
            batch_size = dv_y_cfg.batch_num_spk * dv_y_cfg.spk_num_seq
        else:
            y_val = self.one_hot
            batch_size = dv_y_cfg.batch_num_spk

        return y_val, batch_size

    def draw_file_id_list(self, batch_speaker_list, utter_tvt):
        file_id_list = []
        for speaker_id in batch_speaker_list:
            file_id = self.file_loader_dict[(speaker_id, utter_tvt)].draw_n_samples(1)
            file_id_list.extend(file_id)
        return file_id_list

    def init_wav(self):
        dv_y_cfg = self.dv_y_cfg
        # Numpy data holders
        self.wav = numpy.zeros((dv_y_cfg.batch_num_spk, dv_y_cfg.batch_seq_total_len))
        if ('tau' in dv_y_cfg.out_feat_list) or ('vuv' in dv_y_cfg.out_feat_list):
            self.tau_SBM = numpy.zeros((dv_y_cfg.batch_num_spk, dv_y_cfg.spk_num_seq, dv_y_cfg.seq_num_win))
            self.vuv_SBM = numpy.zeros((dv_y_cfg.batch_num_spk, dv_y_cfg.spk_num_seq, dv_y_cfg.seq_num_win))
            self.tau_idx = self.make_tau_idx()
        if 'f' in dv_y_cfg.out_feat_list:
            self.f_SBM = numpy.zeros((dv_y_cfg.batch_num_spk, dv_y_cfg.spk_num_seq, dv_y_cfg.seq_num_win))
            self.f_idx = self.make_f_idx()
        
        # Parameters
        wav_sr  = dv_y_cfg.cfg.wav_sr
        cmp_sr  = dv_y_cfg.cfg.frame_sr
        wav_cmp_ratio = int(wav_sr / cmp_sr)
        # Do not use silence frames at the beginning or the end
        total_sil_one_side_cmp = dv_y_cfg.frames_silence_to_keep + dv_y_cfg.sil_pad   # This is at 200Hz
        self.total_sil_one_side_wav = total_sil_one_side_cmp * wav_cmp_ratio          # This is at 16kHz
        self.min_file_len = dv_y_cfg.batch_seq_total_len + 2 * self.total_sil_one_side_wav # This is at 16kHz
        # Window length in time, for pitch detection and vuv
        self.window_T = float(dv_y_cfg.seq_win_len) / float(wav_sr)

        # Directories and extensions
        self.wav_dir =  os.path.join(self.nn_data_dir, 'nn_wav_resil_norm_80')
        self.pitch_dir = os.path.join(self.nn_data_dir, 'nn_pitch_resil')
        self.f_dir = os.path.join(self.nn_data_dir, 'nn_f016k_resil')

        self.wav_ext = '.wav'
        self.pitch_ext = '.pitch'
        self.f_ext = '.f016k'

        # Train method
        self.make_feed_dict_method_train = self.make_feed_dict_y_wav_subwin_train
        self.make_feed_dict_method_test  = self.make_feed_dict_y_wav_subwin_test

    def make_tau_idx(self):
        ''' 
        This is the index matrix for tau extraction; 
        simply add the utterance starting index to it
        '''
        dv_y_cfg = self.dv_y_cfg
        tau_idx = numpy.zeros((dv_y_cfg.spk_num_seq, dv_y_cfg.seq_num_win))

        win_idx_seq = numpy.arange(dv_y_cfg.seq_num_win) * dv_y_cfg.seq_win_shift
        for seq_idx in range(dv_y_cfg.spk_num_seq):
            seq_start = seq_idx * dv_y_cfg.batch_seq_shift
            tau_idx[seq_idx] = win_idx_seq + seq_start

        tau_idx = tau_idx.astype(int)
        return tau_idx

    def make_f_idx(self):
        ''' 
        This is the index matrix for f extraction; 
        simply add half window-width to tau index matrix
        '''
        try:
            tau_idx = self.tau_idx
        except KeyError:
            tau_idx = self.make_tau_idx()
        f_idx = tau_idx + int(self.dv_y_cfg.seq_win_len/2)
        return f_idx

    def make_feed_dict_y_wav_subwin_train(self):
        pass

    def make_feed_dict_y_wav_subwin_test(self):
        pass

class Build_dv_y_train_data_loader(Build_dv_y_data_loader):
    def __init__(self, cfg, dv_y_cfg):
        super().__init__(cfg, dv_y_cfg)

    def load_wav_ST(self, file_id_list, all_utt_start_frame_index):
        ''' 
        wav data stored in self.wav, not returned! 
        file_start_index_list stored in self.file_start_index_list, not returned! 
        '''
        dv_y_cfg = self.dv_y_cfg
        
        if all_utt_start_frame_index is None:
            extra_file_len_ratio = numpy.random.rand(dv_y_cfg.batch_num_spk)
            all_file_start_index = 0
        else:
            extra_file_len_ratio = numpy.zeros(dv_y_cfg.batch_num_spk)
            all_file_start_index = all_utt_start_frame_index

        for speaker_idx in range(dv_y_cfg.batch_num_spk):
            file_id = file_id_list[speaker_idx]
            file_name = os.path.join(self.wav_dir, file_id+self.wav_ext)

            wav_data, wav_file_len = io_fun.load_binary_file_frame(file_name, 1)
            wav_data  = numpy.squeeze(wav_data, axis=1)      # T*1 -> T

            extra_file_len = wav_file_len - self.min_file_len
            file_start_index = self.total_sil_one_side_wav + int(extra_file_len_ratio[speaker_idx]*extra_file_len) + all_file_start_index # Either extra_file_len_ratio or all_file_start_index is 0 here
            self.file_start_index_list[speaker_idx] = file_start_index

            self.wav[speaker_idx, :] = wav_data[file_start_index:file_start_index+dv_y_cfg.batch_seq_total_len]

    def load_tau_vuv_SBM(self, file_id_list, file_start_index_list=None):
        '''
        tau and vuv data stored in self.tau and self.vuv, not returned!
        '''
        dv_y_cfg = self.dv_y_cfg
        
        if file_start_index_list is None:
            file_start_index_list = self.file_start_index_list

        for speaker_idx in range(dv_y_cfg.batch_num_spk):
            file_id = file_id_list[speaker_idx]
            file_name = os.path.join(self.pitch_dir, file_id+self.pitch_ext)

            tau_data, tau_file_len = io_fun.load_binary_file_frame(file_name, 1)
            tau_data  = numpy.squeeze(tau_data, axis=1)      # T*1 -> T

            file_start_index = file_start_index_list[speaker_idx]
            tau_idx = self.tau_idx + int(file_start_index)
            for seq_idx in range(dv_y_cfg.spk_num_seq):
                self.tau_SBM[speaker_idx, seq_idx] = tau_data[tau_idx[seq_idx]]
        self.vuv_SBM[:,:,:] = 1.
        # Smaller than 0: end of sentence, no pitch found after
        self.vuv_SBM[self.tau_SBM < 0.] = 0.
        # Larger than window length, no pitch found inside
        self.vuv_SBM[self.tau_SBM > self.window_T] = 0.
        # Change tau to 0. in unvoiced regions
        self.tau_SBM[self.vuv_SBM == 0.] = 0.

    def load_f_SBM(self, file_id_list, file_start_index_list=None):
        '''
        f data stored in self.f_SBM, not returned!
        '''
        dv_y_cfg = self.dv_y_cfg
        
        
        if file_start_index_list is None:
            file_start_index_list = self.file_start_index_list

        for speaker_idx in range(dv_y_cfg.batch_num_spk):
            file_id = file_id_list[speaker_idx]
            file_name = os.path.join(self.f_dir, file_id+self.f_ext)

            f_data, f_file_len = io_fun.load_binary_file_frame(file_name, 1)
            f_data  = numpy.squeeze(f_data, axis=1)      # T*1 -> T

            file_start_index = file_start_index_list[speaker_idx]
            f_idx = self.f_idx + int(file_start_index)
            for seq_idx in range(dv_y_cfg.spk_num_seq):
                self.f_SBM[speaker_idx, seq_idx] = f_data[f_idx[seq_idx]]

    def make_feed_dict_y_wav_subwin_train(self, batch_speaker_list=None, utter_tvt='', all_utt_start_frame_index=None, return_frame_index=False, return_file_name=False):
        dv_y_cfg = self.dv_y_cfg
        if batch_speaker_list is None:
            batch_speaker_list = self.speaker_loader.draw_n_samples(dv_y_cfg.batch_num_spk)
        '''
        Draw Utterances; Load Data
        Draw starting frame; Slice; Fit into numpy holders
        '''
        # one_hot part
        y_val, batch_size = self.make_one_hot_y_val(batch_speaker_list)
        self.feed_dict['y'] = y_val

        # Waveform part
        # Also, this part computes the starting frame index of all files
        file_id_list = self.draw_file_id_list(batch_speaker_list, utter_tvt)
        self.load_wav_ST(file_id_list, all_utt_start_frame_index)
        self.feed_dict['x'] = self.wav

        # tau and vuv part
        if ('tau' in dv_y_cfg.out_feat_list) or ('vuv' in dv_y_cfg.out_feat_list):
            self.load_tau_vuv_SBM(file_id_list, self.file_start_index_list)
            self.feed_dict['tau'] = self.tau_SBM
            self.feed_dict['vuv'] = self.vuv_SBM

        # f0 part
        if 'f' in dv_y_cfg.out_feat_list:
            self.load_f_SBM(file_id_list, self.file_start_index_list)
            self.feed_dict['f'] = self.f_SBM

        return_list = [self.feed_dict, batch_size]
        
        if return_frame_index:
            return_list.append(self.file_start_index_list)
        if return_file_name:
            return_list.append(file_id_list)
        return return_list

class Build_dv_y_test_data_loader(Build_dv_y_data_loader):
    def __init__(self, cfg, dv_y_cfg):
        super().__init__(cfg, dv_y_cfg)
        self.gen_data_dict = {}
        self.file_gen_finish = False
        self.load_new_file_bool = False

    def load_data_dict(self, file_id):
        dv_y_cfg = self.dv_y_cfg

        # wav part
        file_name = os.path.join(self.wav_dir, file_id+self.wav_ext)
        wav_data, wav_file_len = io_fun.load_binary_file_frame(file_name, 1)
        wav_data  = numpy.squeeze(wav_data, axis=1)      # T*1 -> T

        total_sil_one_side_wav = self.total_sil_one_side_wav
        wav_len_no_sil = wav_file_len - 2 * total_sil_one_side_wav
        wav_data_no_sil = wav_data[total_sil_one_side_wav:total_sil_one_side_wav+wav_len_no_sil]
        self.gen_data_dict['wav'] = wav_data_no_sil

        if ('tau' in dv_y_cfg.out_feat_list) or ('vuv' in dv_y_cfg.out_feat_list):
            file_name = os.path.join(self.pitch_dir, file_id+self.pitch_ext)
            tau_data, tau_file_len = io_fun.load_binary_file_frame(file_name, 1)
            tau_data  = numpy.squeeze(tau_data, axis=1)      # T*1 -> T
            tau_data_no_sil = tau_data[total_sil_one_side_wav:total_sil_one_side_wav+wav_len_no_sil]
            self.gen_data_dict['tau'] = tau_data_no_sil

        if 'f' in dv_y_cfg.out_feat_list:
            file_name = os.path.join(self.f_dir, file_id+self.f_ext)
            f_data, f_file_len = io_fun.load_binary_file_frame(file_name, 1)
            f_data  = numpy.squeeze(f_data, axis=1)      # T*1 -> T
            f_data_no_sil = f_data[total_sil_one_side_wav:total_sil_one_side_wav+wav_len_no_sil]
            self.gen_data_dict['f'] = f_data_no_sil

        return wav_len_no_sil

    def load_tau_vuv_SBM(self, T_actual):
        '''
        tau and vuv data stored in self.tau and self.vuv, not returned!
        '''
        dv_y_cfg = self.dv_y_cfg

        tau_data = self.gen_data_dict['tau']

        # Check if padding is needed
        T_total = dv_y_cfg.batch_seq_total_len
        if T_actual < T_total:
            # Pad -1., i.e. no pitch found after
            tau_data_T = numpy.ones(T_total) * -1.
            tau_data_T[:T_actual] = tau_data
        else:
            tau_data_T = tau_data[:T_actual]

        tau_idx = self.tau_idx
        for seq_idx in range(dv_y_cfg.spk_num_seq):
            self.tau_SBM[0, seq_idx] = tau_data_T[tau_idx[seq_idx]]

        self.vuv_SBM[:,:] = 1.
        # Smaller than 0: end of sentence, no pitch found after
        self.vuv_SBM[self.tau_SBM < 0.] = 0.
        # Larger than window length, no pitch found inside
        self.vuv_SBM[self.tau_SBM > self.window_T] = 0.
        # Change tau to 0. in unvoiced regions
        self.tau_SBM[self.vuv_SBM == 0.] = 0.

    def load_f_SBM(self, T_actual):
        '''
        f data stored in self.f_SBM, not returned!
        '''
        dv_y_cfg = self.dv_y_cfg
        
        f_data = self.gen_data_dict['f']

        # Check if padding is needed
        T_total = dv_y_cfg.batch_seq_total_len
        if T_actual < T_total:
            # Pad 0., since that seq is not used anyway
            f_data_T = numpy.zeros(T_total)
            f_data_T[:T_actual] = f_data
        else:
            f_data_T = f_data[:T_actual]

        f_idx = self.f_idx
        for seq_idx in range(dv_y_cfg.spk_num_seq):
            self.f_SBM[0, seq_idx] = f_data_T[f_idx[seq_idx]]

    def make_feed_dict_y_wav_subwin_test(self, speaker_id, file_id):
        dv_y_cfg = self.dv_y_cfg
        logger = make_logger("make_dict")
        assert dv_y_cfg.batch_num_spk == 1
        '''
        Load Data; load starting frame; Slice; Fit into numpy holders
        '''
        if self.load_new_file_bool:
            self.load_new_file_bool = False
            # Load a new file
            # one_hot part
            batch_speaker_list = [speaker_id]
            try: y_val, batch_size = self.make_one_hot_y_val(batch_speaker_list)
            except: y_val = self.one_hot # Exception when speaker is not a train speaker
            self.feed_dict['y'] = y_val
            # Load all data
            wav_len_no_sil = self.load_data_dict(file_id)
            self.gen_T_total = wav_len_no_sil
        else:
            file_start_index = dv_y_cfg.spk_num_seq*dv_y_cfg.batch_seq_shift
            # Use previous file
            # one_hot part: do nothing; re-use the current self.feed_dict['y']
            # Other data file: shift and remove the used parts
            self.gen_T_total = self.gen_T_total - file_start_index
            for k in self.gen_data_dict:
                self.gen_data_dict[k] = self.gen_data_dict[k][file_start_index:]

        B_total = int((self.gen_T_total - dv_y_cfg.batch_seq_len) / dv_y_cfg.batch_seq_shift) + 1
        # Decide if there are features remain
        if B_total > dv_y_cfg.spk_num_seq:
            B_actual = dv_y_cfg.spk_num_seq
            T_actual = dv_y_cfg.batch_seq_total_len
            self.file_gen_finish = False
        else:
            B_actual = B_total
            T_actual = self.gen_T_total
            self.file_gen_finish = True
        

        self.wav[0,:T_actual] = self.gen_data_dict['wav'][:T_actual]
        self.feed_dict['x'] = self.wav

        # tau and vuv part
        if ('tau' in dv_y_cfg.out_feat_list) or ('vuv' in dv_y_cfg.out_feat_list):
            self.load_tau_vuv_SBM(T_actual)
            self.feed_dict['tau'] = self.tau_SBM
            self.feed_dict['vuv'] = self.vuv_SBM

        # f0 part
        if 'f' in dv_y_cfg.out_feat_list:
            self.load_f_SBM(T_actual)
            self.feed_dict['f'] = self.f_SBM

        batch_size = B_actual
        return_list = [self.feed_dict, batch_size]
        return return_list

class Test_dv_y_data_loader(object):
    def __init__(self, cfg, dv_y_cfg):
        super().__init__()
        self.cfg = copy.deepcopy(cfg)
        self.dv_y_cfg = copy.deepcopy(dv_y_cfg)

        self.dv_y_data_loader = Build_dv_y_train_data_loader(self.cfg, self.dv_y_cfg)
        self.data_file_io     = Data_File_Converter(self.cfg)

        self.logger = make_logger("Data_Loader Test")

    def test(self):
        # self.file_dict_test()
        self.pitch_loading_test()

        # tau_idx = self.dv_y_data_loader.make_tau_idx()
        # print(tau_idx.shape)
        # for i in range(tau_idx.shape[0]):
        #     print(tau_idx[i])

    def file_dict_test(self):
        ''' check file dict '''
        file_loader_dict = self.dv_y_data_loader.file_loader_dict
        for k in file_loader_dict:
            n_remain = len(file_loader_dict[k].list_remain)
            print(k)
            print(n_remain)

        file_list_dict = {}
        all_remain_file_list = self.dv_y_data_loader.file_id_list
        print(len(all_remain_file_list))

        speaker_id = self.dv_y_data_loader.speaker_id_list[0]

        # for speaker_id in self.dv_y_data_loader.speaker_id_list:
        speaker_remain_file_list, all_remain_file_list = keep_by_speaker(all_remain_file_list, [speaker_id])
        print(len(speaker_remain_file_list))
        print(len(all_remain_file_list))
        for utter_tvt_name in ['train', 'valid', 'test']:
            file_list_dict[(speaker_id, utter_tvt_name)], speaker_remain_file_list = keep_by_file_number(speaker_remain_file_list, self.dv_y_cfg.data_split_file_number[utter_tvt_name])
            print(len(file_list_dict[(speaker_id, utter_tvt_name)]))
            print(len(speaker_remain_file_list))

    def pitch_loading_test(self):
        '''
         Test pitch loading method
         Expectation: direct reading from pitch text file, and compute time stamps
        '''
        dv_y_cfg = self.dv_y_cfg
        dv_y_data_loader = self.dv_y_data_loader
        data_file_io = self.data_file_io

        pitch_text_dir = '/home/dawna/tts/mw545/TorchDV/debug_nausicaa/data/reaper_16kHz/pitch'
        lab_dir = self.cfg.lab_dir

        feed_dict, batch_size, file_start_index_list, file_id_list = dv_y_data_loader.make_feed_dict_y_wav_subwin_train(batch_speaker_list=None, utter_tvt='train', all_utt_start_frame_index=None, return_frame_index=True, return_file_name=True)

        tau_SBM_1 = feed_dict['tau']
        vuv_SBM_1 = feed_dict['vuv']

        tau_SBM_2 = numpy.zeros((dv_y_cfg.batch_num_spk, dv_y_cfg.spk_num_seq, dv_y_cfg.seq_num_win))
        vuv_SBM_2 = numpy.zeros((dv_y_cfg.batch_num_spk, dv_y_cfg.spk_num_seq, dv_y_cfg.seq_num_win))

        for speaker_idx in range(dv_y_cfg.batch_num_spk):
            file_id = file_id_list[speaker_idx]
            file_start_index = file_start_index_list[speaker_idx] - dv_y_data_loader.total_sil_one_side_wav # 16kHz

            pitch_file = os.path.join(pitch_text_dir, file_id+'.pitch')
            lab_file   = os.path.join(lab_dir, file_id+'.lab')

            pitch_data_list = data_file_io.read_pitch_text(pitch_file)
            nonsilence_indices = data_file_io.read_lab_no_sil_indices(lab_file)
            sil_start_wav = nonsilence_indices[0] * int(dv_y_cfg.cfg.wav_sr / dv_y_cfg.cfg.frame_sr)
            file_start_t = float(file_start_index + sil_start_wav) / float(dv_y_cfg.cfg.wav_sr)

            win_start_t = file_start_t
            for seq_idx in range(dv_y_cfg.spk_num_seq):
                for win_idx in range(dv_y_cfg.seq_num_win):
                    win_end_t = win_start_t + dv_y_data_loader.window_T
                    for pitch_t in pitch_data_list:
                        if pitch_t >= win_start_t:
                            if pitch_t <= win_end_t:
                                tau_SBM_2[speaker_idx, seq_idx, win_idx] = pitch_t - win_start_t
                                vuv_SBM_2[speaker_idx, seq_idx, win_idx] = 1
                            else:
                                break
                    win_start_t = win_start_t + dv_y_cfg.seq_win_shift / float(dv_y_cfg.cfg.wav_sr)
                win_start_t = win_start_t + dv_y_cfg.batch_seq_shift / float(dv_y_cfg.cfg.wav_sr)

        tau_SBM_diff = tau_SBM_1 - tau_SBM_2
        vuv_SBM_diff = vuv_SBM_1 - vuv_SBM_2

        # for speaker_idx in range(dv_y_cfg.batch_num_spk):
        speaker_idx = 0
        for seq_idx in range(dv_y_cfg.spk_num_seq):
            print(tau_SBM_diff[speaker_idx, seq_idx])
            print(vuv_SBM_diff[speaker_idx, seq_idx])


        








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
    return numpy.array(pitch_t_list)

def make_n_mid_0_matrix(dv_y_cfg):
    ''' 
    Return mid index value of each sub-window, before adding time/index shift
    Type is float, for interpolation
    '''
    wav_sr  = dv_y_cfg.cfg.wav_sr
    cmp_sr  = dv_y_cfg.cfg.frame_sr
    wav_cmp_ratio = int(wav_sr / cmp_sr)
    n_mid_0 = numpy.zeros([dv_y_cfg.spk_num_seq, dv_y_cfg.seq_num_win])
    for seq_idx in range(dv_y_cfg.spk_num_seq):
        for win_idx in range(dv_y_cfg.seq_num_win):
            seq_start = seq_idx * dv_y_cfg.batch_seq_shift
            win_start = seq_start + win_idx * dv_y_cfg.seq_win_shift
            t_start = (win_start) / float(wav_sr)
            t_end   = (win_start+dv_y_cfg.seq_win_len) / float(wav_sr)

            t_mid = (t_start + t_end) / 2.
            n_mid = t_mid * float(cmp_sr)

            n_mid_0[seq_idx, win_idx] = n_mid
    return n_mid_0

def make_win_start_0_matrix(dv_y_cfg):
    '''  
    Return start index of each sub-window, before adding index shift
    '''
    wav_sr  = dv_y_cfg.cfg.wav_sr
    cmp_sr  = dv_y_cfg.cfg.frame_sr
    wav_cmp_ratio = int(wav_sr / cmp_sr)
    win_start_matrix = numpy.zeros([dv_y_cfg.spk_num_seq, dv_y_cfg.seq_num_win, 1])
    for seq_idx in range(dv_y_cfg.spk_num_seq):
        seq_start = seq_idx * dv_y_cfg.batch_seq_shift
        for win_idx in range(dv_y_cfg.seq_num_win):
            win_start_matrix[seq_idx, win_idx, 0] = seq_start + win_idx * dv_y_cfg.seq_win_shift
    return win_start_matrix

def cal_seq_win_lf0_mid(lf0_norm_data, utter_start_frame_index, n_mid_0, wav_cmp_ratio):
    ''' Derive position using utter_start_frame_index and n_mid_0 matrix, then extract from lf0_norm_data '''
    l = lf0_norm_data.shape[0]
    n_mid = n_mid_0 + (utter_start_frame_index / float(wav_cmp_ratio))
    n_l = (n_mid-0.5).astype(int)
    n_r = n_l + 1
    r = n_mid - (n_l + 0.5)
    n_l[n_r>= l] = -1
    n_r[n_r>= l] = -1
    lf0_l = lf0_norm_data[n_l]
    lf0_r = lf0_norm_data[n_r]
    lf0_mid = r * lf0_r + (1-r) * lf0_l
    return lf0_mid

def cal_seq_win_tau_vuv(pitch_loc_data, utter_start_frame_index, dv_y_cfg, win_start_0, wav_sr):
    ''' Calculate pitch location per window; if not found then vuv=0 '''
    spk_num_seq = dv_y_cfg.spk_num_seq
    seq_num_win = dv_y_cfg.seq_num_win
    seq_win_len = dv_y_cfg.seq_win_len

    l = pitch_loc_data.shape[0]
    tau = numpy.zeros((spk_num_seq, seq_num_win))
    vuv = numpy.zeros((spk_num_seq, seq_num_win))
    
    pitch_max = seq_win_len / float(wav_sr)

    win_start = win_start_0 + utter_start_frame_index
    t_start = win_start / float(wav_sr)

    t_start = numpy.repeat(t_start, l, axis=2)

    pitch_start = pitch_loc_data - t_start
    pitch_start[pitch_start <= 0.] = pitch_max
    pitch_start_min = numpy.amin(pitch_start, axis=2)

    vuv[pitch_start_min < pitch_max] = 1
    pitch_start_min[pitch_start_min >= pitch_max] = 0.
    # pitch_start_min = pitch_start_min * vuv
    tau = pitch_start_min

    return tau, vuv

def cal_seq_win_lf0_var(lf0_norm_data, utter_start_frame_index, n_mid_0, wav_cmp_ratio):
    ''' Derive position using utter_start_frame_index and n_mid_0 matrix, then extract from lf0_norm_data '''
    l = lf0_norm_data.shape[0]
    n_mid = n_mid_0 + (utter_start_frame_index / float(wav_cmp_ratio))
    n_l_4 = (n_mid-3.5).astype(int)
    n_list = []
    lf0_list = []
    for i in range(8):
        n_i = n_l_4 + i
        n_i[n_i>= l] = -1
        n_list.append(n_i)
        lf0_i =  lf0_norm_data[n_i]
        lf0_list.append(lf0_i)
    lf0_stack = numpy.stack(lf0_list, axis=0)
    lf0_var = numpy.var(lf0_stack, axis=0)
    return lf0_var

def make_feed_dict_y_wav_subwin_train(dv_y_cfg, file_list_dict, file_dir_dict, batch_speaker_list, utter_tvt, all_utt_start_frame_index=None, return_frame_index=False, return_file_name=False):
    logger = make_logger("make_dict")

    '''
    Draw Utterances; Load Data
    Draw starting frame; Slice; Fit into numpy holders
    '''
    wav_sr  = dv_y_cfg.cfg.wav_sr
    cmp_sr  = dv_y_cfg.cfg.frame_sr
    wav_cmp_ratio = int(wav_sr / cmp_sr)
    # Do not use silence frames at the beginning or the end
    total_sil_one_side_cmp = dv_y_cfg.frames_silence_to_keep + dv_y_cfg.sil_pad  # This is at 200Hz
    total_sil_one_side_wav = total_sil_one_side_cmp * wav_cmp_ratio              # This is at 16kHz
    min_file_len = dv_y_cfg.batch_seq_total_len + 2 * total_sil_one_side_wav # This is at 16kHz

    # Each speaker has 1 utterance, no more need for list-of-lists
    file_name_list = []
    start_frame_index_list = []
    
    # one_hot part
    one_hot = numpy.zeros((dv_y_cfg.batch_num_spk))
    for speaker_idx in range(dv_y_cfg.batch_num_spk):
        speaker_id = batch_speaker_list[speaker_idx]
        # Make classification targets, index sequence
        true_speaker_index = dv_y_cfg.speaker_id_list_dict['train'].index(speaker_id)
        one_hot[speaker_idx] = true_speaker_index
    if dv_y_cfg.train_by_window:
        # S --> S*B
        y_val = numpy.repeat(one_hot, dv_y_cfg.spk_num_seq)
        batch_size = dv_y_cfg.batch_num_spk * dv_y_cfg.spk_num_seq
    else:
        y_val = one_hot
        batch_size = dv_y_cfg.batch_num_spk
    feed_dict = {'y':y_val}

    # Waveform part
    # Also, this part computes the length and draws random starting frame index
    wav = numpy.zeros((dv_y_cfg.batch_num_spk, dv_y_cfg.batch_seq_total_len))
    feat_name_list = ['wav'] # Load wav
    feat_dim_list  = [1]
    for speaker_idx in range(dv_y_cfg.batch_num_spk):
        speaker_id = batch_speaker_list[speaker_idx]
        # Draw 1 utterance per speaker
        # Draw multiple windows per utterance:  dv_y_cfg.spk_num_seq
        # Stack them along B
        speaker_file_name_list, speaker_utter_len_list, speaker_utter_list = get_utters_from_binary_dict(1, file_list_dict[(speaker_id, utter_tvt)], file_dir_dict, feat_name_list=feat_name_list, feat_dim_list=feat_dim_list, min_file_len=min_file_len, random_seed=None)

        file_name = speaker_file_name_list[0]
        file_name_list.append(file_name)

        wav_file  = speaker_utter_list['wav'][0] # T * 1; 16kHz
        wav_file  = numpy.squeeze(wav_file, axis=1)      # T*1 -> T
        wav_file_len = speaker_utter_len_list[0]
        # Find start frame index, random if None
        if all_utt_start_frame_index is None:
            extra_file_len = wav_file_len - min_file_len
            utter_start_frame_index = numpy.random.randint(low=total_sil_one_side_wav, high=total_sil_one_side_wav+extra_file_len+1)
        else:
            utter_start_frame_index = total_sil_one_side_wav + all_utt_start_frame_index
        start_frame_index_list.append(utter_start_frame_index)

        wav[speaker_idx, :] = wav_file[utter_start_frame_index:utter_start_frame_index+dv_y_cfg.batch_seq_total_len]
    x_val = wav
    feed_dict['x'] = x_val

    # lf0 part
    if 'nlf' in dv_y_cfg.out_feat_list:
        nlf = numpy.zeros((dv_y_cfg.batch_num_spk, dv_y_cfg.spk_num_seq, dv_y_cfg.seq_num_win))
        for speaker_idx in range(dv_y_cfg.batch_num_spk):
            speaker_id = batch_speaker_list[speaker_idx]
            file_name = file_name_list[speaker_idx]
            utter_start_frame_index = start_frame_index_list[speaker_idx]
            # Load cmp and pitch data
            cmp_file_name = os.path.join(file_dir_dict['cmp'], file_name+'.cmp')
            lf0_index     = dv_y_cfg.cfg.acoustic_start_index['lf0']
            cmp_dim       = dv_y_cfg.cfg.nn_feature_dims['cmp']
            lf0_norm_data = load_cmp_file(cmp_file_name, cmp_dim=cmp_dim, feat_dim_index=lf0_index)

            # Get lf0_mid data in forms of numpy array operations, faster than for loops
            n_mid_0 = dv_y_cfg.return_n_mid_0_matrix()
            nlf_spk = cal_seq_win_lf0_mid(lf0_norm_data, utter_start_frame_index, n_mid_0, wav_cmp_ratio)
            nlf[speaker_idx] = nlf_spk
            
        feed_dict['nlf'] = nlf

    # tau and vuv part
    if ('tau' in dv_y_cfg.out_feat_list) or ('vuv' in dv_y_cfg.out_feat_list):
        tau = numpy.zeros((dv_y_cfg.batch_num_spk, dv_y_cfg.spk_num_seq, dv_y_cfg.seq_num_win))
        vuv = numpy.zeros((dv_y_cfg.batch_num_spk, dv_y_cfg.spk_num_seq, dv_y_cfg.seq_num_win))
        for speaker_idx in range(dv_y_cfg.batch_num_spk):
            speaker_id = batch_speaker_list[speaker_idx]
            file_name = file_name_list[speaker_idx]
            utter_start_frame_index = start_frame_index_list[speaker_idx]
            # Load cmp and pitch data
            pitch_file_name = os.path.join(file_dir_dict['pitch'], file_name+'.pm')
            pitch_loc_data = read_pitch_file(pitch_file_name)

            win_start_0 = dv_y_cfg.return_win_start_0_matrix()
            tau_spk, vuv_spk = cal_seq_win_tau_vuv(pitch_loc_data, utter_start_frame_index, dv_y_cfg, win_start_0, wav_sr)
            
            tau[speaker_idx] = tau_spk
            vuv[speaker_idx] = vuv_spk

        feed_dict['tau'] = tau
        feed_dict['vuv'] = vuv
        if True:
            # Make vuv_SB as mask for CE_SB
            # Current method: Some b in B are voiced, use vuv as error mask
            # b is 1 only if all m in M are voiced
            # Need to reshape from S*B to SB since Cross-Entropy is this shape
            assert dv_y_cfg.train_by_window
            # Make binary S * B matrix
            vuv_S_B = (vuv>0).all(axis=2)
            # Reshape to SB for pytorch cross-entropy function
            vuv_SB = numpy.reshape(vuv_S_B, (dv_y_cfg.batch_num_spk * dv_y_cfg.spk_num_seq))
            feed_dict['vuv_SB'] = vuv_SB

    # var(lf0) part: temporary for a test
    if 'nlf_var' in dv_y_cfg.out_feat_list:
        nlf_var = numpy.zeros((dv_y_cfg.batch_num_spk, dv_y_cfg.spk_num_seq, dv_y_cfg.seq_num_win))
        for speaker_idx in range(dv_y_cfg.batch_num_spk):
            speaker_id = batch_speaker_list[speaker_idx]
            file_name = file_name_list[speaker_idx]
            utter_start_frame_index = start_frame_index_list[speaker_idx]
            # Load cmp and pitch data
            cmp_file_name = os.path.join(file_dir_dict['cmp'], file_name+'.cmp')
            lf0_index     = dv_y_cfg.cfg.acoustic_start_index['lf0']
            cmp_dim       = dv_y_cfg.cfg.nn_feature_dims['cmp']
            lf0_norm_data = load_cmp_file(cmp_file_name, cmp_dim=cmp_dim, feat_dim_index=lf0_index)

            # Get lf0_mid data in forms of numpy array operations, faster than for loops
            n_mid_0 = dv_y_cfg.return_n_mid_0_matrix()
            nlf_var_spk = cal_seq_win_lf0_var(lf0_norm_data, utter_start_frame_index, n_mid_0, wav_cmp_ratio)
            nlf_var[speaker_idx] = nlf_spk
            
        feed_dict['nlf_var'] = nlf_var

    return_list = [feed_dict, batch_size]
    
    if return_frame_index:
        return_list.append(start_frame_index_list)
    if return_file_name:
        return_list.append(file_name_list)
    return return_list

def make_feed_dict_y_wav_subwin_test(dv_y_cfg, file_dir_dict, speaker_id, file_name, start_frame_index, BTD_feat_remain):
    logger = make_logger("make_dict")

    '''Load Data; load starting frame; Slice; Fit into numpy holders
    '''
    # BTD_feat_remain is a dict now, keys are feat names
    assert dv_y_cfg.batch_num_spk == 1

    # one_hot part
    one_hot = numpy.zeros((dv_y_cfg.batch_num_spk))
    try: true_speaker_index = dv_y_cfg.speaker_id_list_dict['train'].index(speaker_id)
    except ValueError: true_speaker_index = 0 # At generation time, since one_hot is not used, a non-train speaker is given an arbituary speaker index
    one_hot[0] = true_speaker_index
    # Pass the actual features to feat_dict
    if dv_y_cfg.train_by_window:
        # S --> S*B
        y_val = numpy.repeat(one_hot, dv_y_cfg.spk_num_seq)
    else:
        y_val = one_hot
    feed_dict = {'y': y_val}

    # Make it a dict, for all features
    BTD_feat_current = {}
    if BTD_feat_remain is not None:
        for feat_name in dv_y_cfg.out_feat_list:
            BTD_feat_current[feat_name] = BTD_feat_remain[feat_name]
        T_total = BTD_feat_current['wav'].shape[0]
        B_total = int((T_total - dv_y_cfg.batch_seq_len) / dv_y_cfg.batch_seq_shift) + 1
    else:
        # Waveform Part
        feat_name_list = ['wav']
        feat_dim_list  = [1]
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
        BTD_feat_current['wav'] = wav_features_no_sil
        T_total = len_no_sil_wav
        B_total = int((T_total - dv_y_cfg.batch_seq_len) / dv_y_cfg.batch_seq_shift) + 1

        # Temporarily change dv_y_cfg.spk_num_seq to B_total to load nlf and tau/vuv
        # Change back after this process
        spk_num_seq_orig = dv_y_cfg.spk_num_seq
        dv_y_cfg.spk_num_seq = B_total
        utter_start_frame_index = total_sil_one_side_wav # Start from beginning of non-silence

        # lf0 part
        if 'nlf' in dv_y_cfg.out_feat_list:
            # Load cmp and pitch data
            cmp_file_name = os.path.join(file_dir_dict['cmp'], file_name+'.cmp')
            lf0_index     = dv_y_cfg.cfg.acoustic_start_index['lf0']
            cmp_dim       = dv_y_cfg.cfg.nn_feature_dims['cmp']
            lf0_norm_data = load_cmp_file(cmp_file_name, cmp_dim=cmp_dim, feat_dim_index=lf0_index)

            # Make a new n_mid_0 because file length has changed
            n_mid_0 = make_n_mid_0_matrix(dv_y_cfg) 
            nlf_spk = cal_seq_win_lf0_mid(lf0_norm_data, utter_start_frame_index, n_mid_0, wav_cmp_ratio)
            BTD_feat_current['nlf'] = nlf_spk

        # tau and vuv part
        if ('tau' in dv_y_cfg.out_feat_list) or ('vuv' in dv_y_cfg.out_feat_list):
            # Load cmp and pitch data
            pitch_file_name = os.path.join(file_dir_dict['pitch'], file_name+'.pm')
            pitch_loc_data = read_pitch_file(pitch_file_name)

            # Make a new win_start_0 because file length has changed
            win_start_0 = make_win_start_0_matrix(dv_y_cfg)
            tau_spk, vuv_spk = cal_seq_win_tau_vuv(pitch_loc_data, utter_start_frame_index, dv_y_cfg, win_start_0, wav_sr)
            
            BTD_feat_current['tau'] = tau_spk
            BTD_feat_current['vuv'] = vuv_spk

        # Change back from B_total
        dv_y_cfg.spk_num_seq = spk_num_seq_orig

    # Decide if there are features remain
    if B_total > dv_y_cfg.spk_num_seq:
        B_actual = dv_y_cfg.spk_num_seq
        T_actual = dv_y_cfg.batch_seq_total_len
        B_remain = B_total - B_actual
        gen_finish = False

        # Store the remains first
        BTD_feat_remain = {}
        for feat_name in dv_y_cfg.out_feat_list:
            if feat_name == 'wav':
                # Waveform is different, in shape of T not B*D; unfold in pytorch
                wav_feat_remain = BTD_feat_current['wav'][B_actual*dv_y_cfg.batch_seq_shift:]
                BTD_feat_remain['wav'] = wav_feat_remain
            else:
                BTD_feat_remain[feat_name] = BTD_feat_current[feat_name][B_actual:]
    else:
        B_actual = B_total
        T_actual = T_total
        B_remain = 0
        gen_finish = True
        BTD_feat_remain = None
    batch_size = B_actual

    for feat_name in dv_y_cfg.out_feat_list:
        if feat_name == 'wav':
            # Waveform is different, in shape of T not B*D; unfold in pytorch
            wav = numpy.zeros((dv_y_cfg.batch_num_spk, dv_y_cfg.batch_seq_total_len))
            wav[0,:T_actual] = BTD_feat_current['wav'][:T_actual]
            feed_dict['x'] = wav
        elif feat_name in ['nlf', 'tau', 'vuv']:
            # nlf, tau and vuv are in shape of B*M
            feat_SBM = numpy.zeros((dv_y_cfg.batch_num_spk, dv_y_cfg.spk_num_seq, dv_y_cfg.seq_num_win))
            feat_SBM[0,:B_actual] = BTD_feat_current[feat_name][:B_actual]
            feed_dict[feat_name] = feat_SBM

            if feat_name == 'vuv':
                # Make vuv_SB as mask for CE_SB
                # Current method: Some b in B are voiced, use vuv as error mask
                # b is 1 only if all m in M are voiced
                # Need to reshape from S*B to SB since Cross-Entropy is this shape
                assert dv_y_cfg.train_by_window
                # Make binary S * B matrix
                vuv_S_B = (feat_SBM>0).all(axis=2)
                # Reshape to SB for pytorch cross-entropy function
                vuv_SB = numpy.reshape(vuv_S_B, (dv_y_cfg.batch_num_spk * dv_y_cfg.spk_num_seq))
                feed_dict['vuv_SB'] = vuv_SB

    return_list = [feed_dict, gen_finish, batch_size, BTD_feat_remain]
    return return_list