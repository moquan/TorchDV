# exp_dv_cmp_pytorch.py

# This file uses dv_cmp experiments to slowly progress with pytorch

import os, sys, pickle, time, shutil, logging, copy
import math, numpy, scipy
numpy.random.seed(545)
import matplotlib
import matplotlib.pyplot as plt
from modules import make_logger, read_file_list, read_sil_index_file, prepare_file_path, prepare_file_path_list, make_held_out_file_number, copy_to_scratch
from modules import keep_by_speaker, remove_by_speaker, keep_by_file_number, remove_by_file_number, keep_by_min_max_file_number, check_and_change_to_list
from modules_2 import compute_feat_dim, log_class_attri, resil_nn_file_list, norm_nn_file_list, get_utters_from_binary_dict, get_one_utter_by_name, count_male_female_class_errors
from modules_2 import compute_cosine_distance, compute_Euclidean_distance
from modules_torch import torch_initialisation

from io_funcs.binary_io import BinaryIOCollection
io_fun = BinaryIOCollection()

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

        self.batch_num_spk = 64 # S
        self.spk_num_utter = 1  # Deprecated and useless; to use multiple utterances from same speaker, use same speaker along self.batch_num_spk
        
        self.data_split_file_number = {}
        self.data_split_file_number['train'] = make_held_out_file_number(1000, 120)
        self.data_split_file_number['valid'] = make_held_out_file_number(120, 81)
        self.data_split_file_number['test']  = make_held_out_file_number(80, 41)

        # From cfg: Features
        # self.dv_dim = cfg.dv_dim
        self.wav_sr = cfg.wav_sr
        self.cmp_use_delta = False
        self.frames_silence_to_keep = cfg.frames_silence_to_keep
        self.sil_pad = cfg.sil_pad

        self.speaker_id_list_dict = cfg.speaker_id_list_dict
        self.num_speaker_dict     = cfg.num_speaker_dict

        self.log_except_list = ['data_split_file_number', 'speaker_id_list_dict', 'feat_index', 'sil_index_dict']

    def auto_complete(self, cfg):
        ''' Remember to call this after __init__ !!! '''
        self.spk_num_seq = int((self.batch_seq_total_len - self.batch_seq_len) / self.batch_seq_shift) + 1  # B

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

    def reload_model_param(self):
        ''' Change model parameters '''
        ''' Possible changes: sizes of S,B '''
        pass

    def change_to_debug_mode(self, process=None):
        if 'debug' in self.work_dir:
            for k in self.epoch_num_batch:
                self.epoch_num_batch[k] = 10 
            if '_smallbatch' not in self.exp_dir:
                self.exp_dir = self.exp_dir + '_smallbatch'
            self.num_train_epoch = 10

    def change_to_class_test_mode(self):
        self.epoch_num_batch = {'test':40}
        self.batch_num_spk = 1
        spk_num_utter_list = [1,2,5,10]
        self.spk_num_utter_list = check_and_change_to_list(spk_num_utter_list)
        lambda_u_dict_file_name = 'lambda_u_class_test.dat'
        self.lambda_u_dict_file_name = os.path.join(self.exp_dir, lambda_u_dict_file_name)

        if self.y_feat_name == 'cmp':
            self.batch_seq_shift = 1
        elif self.y_feat_name == 'wav':
            self.batch_seq_shift = 80

        self.spk_num_seq = int((self.batch_seq_total_len - self.batch_seq_len) / self.batch_seq_shift) + 1

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
        self.spk_num_seq = int((self.batch_seq_total_len - self.batch_seq_len) / self.batch_seq_shift) + 1
        self.num_to_plot = int(self.max_len_to_plot / self.gap_len_to_plot)

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
        file_list[speaker_id] = keep_by_speaker(file_id_list, [speaker_id])
        file_list[(speaker_id, 'all')] = file_list[speaker_id]
        for utter_tvt_name in ['train', 'valid', 'test']:
            file_list[(speaker_id, utter_tvt_name)] = keep_by_file_number(file_list[speaker_id], data_split_file_number[utter_tvt_name])
    return file_list

#############
# Processes #
#############

def train_dv_y_model(cfg, dv_y_cfg):

    # Feed data use feed_dict style

    logger = make_logger("dv_y_config")
    dv_y_cfg = copy.deepcopy(dv_y_cfg)
    log_class_attri(dv_y_cfg, logger, except_list=dv_y_cfg.log_except_list)

    logger = make_logger("train_dvy")
    logger.info('Creating data lists')
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

        epoch_valid_time = time.time()
        output_string['time'] = output_string['time'] + '; train time is %.2f, valid time is %.2f' %((epoch_train_time - epoch_start_time), (epoch_valid_time - epoch_train_time))
        logger.info(output_string['loss'])
        if dv_y_cfg.classify_in_training:
            logger.info(output_string['accuracy'])
        logger.info(output_string['time'])
        logger.info('epoch valid load time is %s, train model time is %s' % (str(epoch_valid_load_time), str(epoch_valid_model_time)))

        dv_y_cfg.additional_action_epoch(logger, dv_y_model)

    logger.info('Reach num_train_epoch, best model, %s, best valid error %.4f' % (nnets_file_name, best_valid_loss))
    return best_valid_loss

def class_test_dv_y_model(cfg, dv_y_cfg):

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

def distance_test_dv_y_model(cfg, dv_y_cfg, test_type='Euc'):
    '''
        Make a list of feed_dict, compute distance between them
        test_type='Euc': Generate lambdas, compute euclidean distance, [lambda_i - lambda_0]^2
        test_type='EucNorm': Generate lambdas, compute euclidean distance, [(lambda_i - lambda_0)/lambda_0]^2; 
            cannot implement, contain 0 issue
        test_type='BiCE': Generate probabilities, make binary (correct/wrong), cross-entropy; -sum(p_0 log(p_i))
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

def vuv_test_sinenet(cfg, dv_y_cfg):
    '''
    Run the evaluation part of the training procedure
    Store the results based on v/uv
    make_feed_dict_y_wav_sinenet_train(..., return_vuv=True)
    Print: amount of data, and CE, vs amount of v/uv
    '''
    logger = make_logger("dv_y_config")
    dv_y_cfg = copy.deepcopy(dv_y_cfg)
    log_class_attri(dv_y_cfg, logger, except_list=dv_y_cfg.log_except_list)

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
            feed_dict, batch_size, vuv_SBM = make_feed_dict_method_vuv_test(dv_y_cfg, file_list_dict, cfg.nn_feat_scratch_dirs, batch_speaker_list, utter_tvt=utter_tvt_name, return_vuv=True)
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
        logger.info('Mean Cross Entropy Results of %s Dataset is %s' % (utter_tvt_name, str(ce_mean)))


