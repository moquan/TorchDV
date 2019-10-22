# exp_dv_cmp_pytorch.py

# This file uses dv_cmp experiments to slowly progress with pytorch

import os, sys, pickle, time, shutil, logging, copy
import math, numpy, scipy
numpy.random.seed(545)
from modules import make_logger, read_file_list, prepare_file_path, prepare_file_path_list, make_held_out_file_number, copy_to_scratch
from modules import keep_by_speaker, remove_by_speaker, keep_by_file_number, remove_by_file_number, keep_by_min_max_file_number, check_and_change_to_list
from modules_2 import compute_feat_dim, log_class_attri, resil_nn_file_list, norm_nn_file_list, get_utters_from_binary_dict, get_one_utter_by_name, count_male_female_class_errors
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
        
        # Things no need to change
        self.learning_rate    = 0.0001
        self.num_train_epoch  = 100
        self.warmup_epoch     = 10
        self.early_stop_epoch = 2    # After this number of non-improvement, roll-back to best previous model and decay learning rate
        self.max_num_decay    = 10
        self.epoch_num_batch  = {'train': 400, 'valid':400}

        self.batch_num_spk = 100 # S
        self.spk_num_utter = 1 # When >1, windows from different utterances are stacked along B
        self.batch_seq_total_len = 400 # Number of frames at 200Hz; 400 for 2s
        self.batch_seq_len   = 40 # T
        self.batch_seq_shift = 5

        self.data_split_file_number = {}
        self.data_split_file_number['train'] = make_held_out_file_number(1000, 120)
        self.data_split_file_number['valid'] = make_held_out_file_number(120, 81)
        self.data_split_file_number['test']  = make_held_out_file_number(80, 41)

        # From cfg: Features
        self.dv_dim = cfg.dv_dim
        self.wav_sr = cfg.wav_sr
        self.cmp_use_delta = False
        self.frames_silence_to_keep = cfg.frames_silence_to_keep
        self.sil_pad = cfg.sil_pad

        self.speaker_id_list_dict = cfg.speaker_id_list_dict
        self.num_speaker_dict     = cfg.num_speaker_dict

        self.log_except_list = ['data_split_file_number', 'speaker_id_list_dict', 'feat_index']


    def auto_complete(self, cfg):
        ''' Remember to call this after __init__ !!! '''
        self.utter_num_seq   = int((self.batch_seq_total_len - self.batch_seq_len) / self.batch_seq_shift) + 1  # Outputs of each sequence is then averaged
        self.spk_num_seq     = self.spk_num_utter * self.utter_num_seq # B

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

        self.gpu_id = 1
        self.gpu_per_process_gpu_memory_fraction = 0.8

    def change_to_debug_mode(self, process=None):
        self.epoch_num_batch  = {'train': 10, 'valid':10, 'test':10}
        if '_smallbatch' not in self.exp_dir:
            self.exp_dir = self.exp_dir + '_smallbatch'
        self.num_train_epoch = 50
        

        # Additional settings per process
        # if process == "class_test":
        #     self.num_speaker_dict['train'] = 10
        #     self.speaker_id_list_dict['train'] = self.speaker_id_list_dict['train'][:self.num_speaker_dict['train']]

    def change_to_class_test_mode(self):
        self.epoch_num_batch = {'test':40}
        self.batch_num_spk = 1
        self.spk_num_utter = 1
        spk_num_utter_list = [1,2,5,10]
        self.spk_num_utter_list = check_and_change_to_list(spk_num_utter_list)
        lambda_u_dict_file_name = 'lambda_u_class_test.dat'
        self.lambda_u_dict_file_name = os.path.join(self.exp_dir, lambda_u_dict_file_name)

        self.batch_seq_shift = 1
        self.utter_num_seq = int((self.batch_seq_total_len - self.batch_seq_len) / self.batch_seq_shift) + 1  # Outputs of each sequence is then averaged
        # self.spk_num_seq = self.spk_num_utter * self.utter_num_seq
        if 'debug' in self.work_dir: self.change_to_debug_mode(process="class_test")

    def change_to_gen_mode(self):
        self.batch_num_spk = 10
        self.spk_num_utter = 5
        self.batch_seq_shift = 1
        self.utter_num_seq = int((self.batch_seq_total_len - self.batch_seq_len) / self.batch_seq_shift) + 1  # Outputs of each sequence is then averaged
        self.spk_num_seq = self.spk_num_utter * self.utter_num_seq
        if 'debug' in self.work_dir: self.change_to_debug_mode()

def make_dv_y_exp_dir_name(model_cfg, cfg):
    exp_dir = cfg.work_dir + '/dv_y_%s_lr_%f_' %(model_cfg.y_feat_name, model_cfg.learning_rate)
    for nn_layer_config in model_cfg.nn_layer_config_list:
        layer_str = '%s%i' % (nn_layer_config['type'][:3], nn_layer_config['size'])
        # exp_dir = exp_dir + str(nn_layer_config['type'])[:3] + str(nn_layer_config['size'])
        if 'batch_norm' in nn_layer_config and nn_layer_config['batch_norm']:
            layer_str = layer_str + 'BN'
        if 'dropout_p' in nn_layer_config and nn_layer_config['dropout_p'] > 0:
            layer_str = layer_str + 'DR'
        exp_dir = exp_dir + layer_str + "_"
    exp_dir = exp_dir + "DV%iS%iB%iT%iD%i" %(model_cfg.dv_dim, model_cfg.batch_num_spk, model_cfg.spk_num_seq, model_cfg.batch_seq_len, model_cfg.feat_dim)
    # exp_dir + "DV"+str(model_cfg.dv_dim)+"_S"+str(model_cfg.batch_num_spk)+"_B"+str(model_cfg.spk_num_seq)+"_T"+str(model_cfg.batch_seq_len)
    # if cfg.exp_type_switch == 'wav_sine_attention':
    #     exp_dir = exp_dir + "_SineSize_"+str(model_cfg.nn_layer_config_list[0]['Sine_filter_size'])
    # elif cfg.exp_type_switch == 'dv_y_wav_cnn_attention':
    #     exp_dir = exp_dir + "_CNN_K%i_S%i" % (model_cfg.nn_layer_config_list[0]['CNN_kernel_size'][1], model_cfg.nn_layer_config_list[0]['CNN_stride'][1])
    return exp_dir

def make_dv_file_list(file_id_list, speaker_id_list, data_split_file_number):
    file_list = {}
    for speaker_id in speaker_id_list:
        file_list[speaker_id] = keep_by_speaker(file_id_list, [speaker_id])
        file_list[(speaker_id, 'all')]   = file_list[speaker_id]
        for utter_tvt_name in ['train', 'valid', 'test']:
            file_list[(speaker_id, utter_tvt_name)] = keep_by_file_number(file_list[speaker_id], data_split_file_number[utter_tvt_name])
    return file_list

#############
# Processes #
#############

def train_dv_y_model(cfg, dv_y_cfg):

    # Feed data use feed_dict style

    logger = make_logger("dv_y_config")
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
    # model.print_model_parameters(logger)

    epoch      = 0
    early_stop = 0
    num_decay  = 0    
    best_valid_loss  = sys.float_info.max
    num_train_epoch  = dv_y_cfg.num_train_epoch
    early_stop_epoch = dv_y_cfg.early_stop_epoch
    max_num_decay    = dv_y_cfg.max_num_decay

    while (epoch < num_train_epoch):
        epoch = epoch + 1

        logger.info('start training Epoch '+str(epoch))
        epoch_start_time = time.time()

        for batch_idx in range(dv_y_cfg.epoch_num_batch['train']):
            # Draw random speakers
            batch_speaker_list = speaker_loader.draw_n_samples(dv_y_cfg.batch_num_spk)
            # Make feed_dict for training
            feed_dict, batch_size = make_feed_dict_method_train(dv_y_cfg, file_list_dict, cfg.nn_feat_scratch_dirs, batch_speaker_list,  utter_tvt='train')
            dv_y_model.nn_model.train()
            dv_y_model.update_parameters(feed_dict=feed_dict)
        epoch_train_time = time.time()

        logger.info('start evaluating Epoch '+str(epoch))
        output_string = {'loss':'epoch %i' % epoch, 'accuracy':'epoch %i' % epoch, 'time':'epoch %i' % epoch}
        for utter_tvt_name in ['train', 'valid', 'test']:
            total_batch_size = 0.
            total_loss       = 0.
            total_accuracy   = 0.
            for batch_idx in range(dv_y_cfg.epoch_num_batch['valid']):
                # Draw random speakers
                batch_speaker_list = speaker_loader.draw_n_samples(dv_y_cfg.batch_num_spk)
                # Make feed_dict for evaluation
                feed_dict, batch_size = make_feed_dict_method_train(dv_y_cfg, file_list_dict, cfg.nn_feat_scratch_dirs, batch_speaker_list, utter_tvt=utter_tvt_name)
                dv_y_model.eval()
                batch_mean_loss = dv_y_model.gen_loss_value(feed_dict=feed_dict)
                total_batch_size += batch_size
                total_loss       += batch_mean_loss
                if dv_y_cfg.classify_in_training:
                    _c, _t, accuracy = dv_y_model.cal_accuracy(feed_dict=feed_dict)
                    total_accuracy   += accuracy
            average_loss = total_loss/float(dv_y_cfg.epoch_num_batch['valid'])
            output_string['loss'] = output_string['loss'] + '; '+utter_tvt_name+' loss '+str(average_loss)

            if dv_y_cfg.classify_in_training:
                average_accu = total_accuracy/float(dv_y_cfg.epoch_num_batch['valid'])
                output_string['accuracy'] = output_string['accuracy'] + '; %s accuracy %.2f' % (utter_tvt_name, average_accu*100.)

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
                        logger.info('stopping early, best model, %s, best valid error %.2f' % (nnets_file_name, best_valid_loss))
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

    return best_valid_loss

def class_test_dv_y_model(cfg, dv_y_cfg):

    logger = make_logger("dv_y_config")
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

    try: 
        lambda_u_dict = pickle.load(open(dv_y_cfg.lambda_u_dict_file_name, 'rb'))
        logger.info('Loaded lambda_u_dict from %s' % dv_y_cfg.lambda_u_dict_file_name)
    # Generate for all utterances, all speakers
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
                    dv_y_model.eval()
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
                B_total = 0
                for file_name in batch_file_list:
                    lambda_u, B_u = lambda_u_dict[file_name]
                    batch_lambda += lambda_u * B_u
                    B_total += B_u
                batch_lambda /= B_total
                speaker_lambda_list.append(batch_lambda)

            true_speaker_index = dv_y_cfg.speaker_id_list_dict['train'].index(speaker_id)
            lambda_list_remain = speaker_lambda_list
            B_remain = len(speaker_lambda_list)
            b_index = 0 # Track counter, instead of removing elements
            correct_counter = 0.
            while B_remain > 0:
                lambda_val = numpy.zeros((dv_y_cfg.batch_num_spk, dv_y_cfg.spk_num_seq, dv_y_cfg.dv_dim))
                if B_remain > dv_y_cfg.spk_num_seq:
                    # Fill all dv_y_cfg.spk_num_seq, keep remain for later
                    B_actual = dv_y_cfg.spk_num_seq
                    B_remain -= dv_y_cfg.spk_num_seq
                    b_index += dv_y_cfg.spk_num_seq
                else:
                    # No more remain
                    B_actual = B_remain
                    B_remain = 0

                for b in range(B_actual):
                    lambda_val[0, b] = lambda_list_remain[b_index + b]

                feed_dict = {'x': lambda_val}
                idx_list_S_B = dv_y_model.lambda_to_indices(feed_dict=feed_dict)
                print(idx_list_S_B)
                for b in range(B_actual):
                    if idx_list_S_B[0, b] == true_speaker_index: 
                        correct_counter += 1.
            speaker_accuracy = correct_counter/float(dv_y_cfg.epoch_num_batch['test'])
            logger.info('speaker %s accuracy is %f' % (speaker_id, speaker_accuracy))
            accuracy_list.append(speaker_accuracy)
        mean_accuracy = numpy.mean(accuracy_list)
        logger.info('Accuracy with %i utterances per speaker is %f' % (spk_num_utter, mean_accuracy))

################################
# dv_y_cmp; Not used any more  #
# Moved to exp_dv_cmp_baseline #
################################

def make_feed_dict_y_cmp_train(dv_y_cfg, file_list_dict, file_dir_dict, batch_speaker_list, utter_tvt, return_dv=False, return_y=False, return_frame_index=False, return_file_name=False):
    feat_name = dv_y_cfg.y_feat_name # Hard-coded here for now
    # Make i/o shape arrays
    # This is numpy shape, not Tensor shape!
    y  = numpy.zeros((dv_y_cfg.batch_num_spk, dv_y_cfg.spk_num_seq, dv_y_cfg.batch_seq_len, dv_y_cfg.feat_dim))
    dv = numpy.zeros((dv_y_cfg.batch_num_spk))

    # Do not use silence frames at the beginning or the end
    total_sil_one_side = dv_y_cfg.frames_silence_to_keep+dv_y_cfg.sil_pad
    min_file_len = dv_y_cfg.batch_seq_total_len + 2 * total_sil_one_side

    file_name_list = []
    start_frame_index_list = []
    for speaker_idx in range(dv_y_cfg.batch_num_spk):
        speaker_id = batch_speaker_list[speaker_idx]

        # Make classification targets, index sequence
        true_speaker_index = dv_y_cfg.speaker_id_list_dict['train'].index(speaker_id)
        dv[speaker_idx] = true_speaker_index

        # Draw multiple utterances per speaker: dv_y_cfg.spk_num_utter
        # Draw multiple windows per utterance:  dv_y_cfg.utter_num_seq
        # Stack them along B
        speaker_file_name_list, speaker_utter_len_list, speaker_utter_list = get_utters_from_binary_dict(dv_y_cfg.spk_num_utter, file_list_dict[(speaker_id, utter_tvt)], file_dir_dict, feat_name_list=[feat_name], feat_dim_list=[dv_y_cfg.feat_dim], min_file_len=min_file_len, random_seed=None)
        file_name_list.append(speaker_file_name_list)

        speaker_start_frame_index_list = []
        for utter_idx in range(dv_y_cfg.spk_num_utter):
            y_stack = speaker_utter_list[feat_name][utter_idx][:,dv_y_cfg.feat_index]
            frame_number   = speaker_utter_len_list[utter_idx]
            extra_file_len = frame_number - (min_file_len)
            start_frame_index = numpy.random.choice(range(total_sil_one_side, total_sil_one_side+extra_file_len+1))
            speaker_start_frame_index_list.append(start_frame_index)
            for seq_idx in range(dv_y_cfg.utter_num_seq):
                y[speaker_idx, utter_idx*dv_y_cfg.utter_num_seq+seq_idx, :, :] = y_stack[start_frame_index:start_frame_index+dv_y_cfg.batch_seq_len, :]
                start_frame_index = start_frame_index + dv_y_cfg.batch_seq_shift
        start_frame_index_list.append(speaker_start_frame_index_list)


    # S,B,T,D --> S,B,T*D
    x_val = numpy.reshape(y, (dv_y_cfg.batch_num_spk, dv_y_cfg.spk_num_seq, dv_y_cfg.batch_seq_len*dv_y_cfg.feat_dim))
    if dv_y_cfg.train_by_window:
        # S --> S*B
        y_val = numpy.repeat(dv, dv_y_cfg.spk_num_seq)
        batch_size = dv_y_cfg.batch_num_spk * dv_y_cfg.spk_num_seq
    else:
        y_val = dv
        batch_size = dv_y_cfg.batch_num_spk

    feed_dict = {'x':x_val, 'y':y_val}
    return_list = [feed_dict, batch_size]
    
    if return_dv:
        return_list.append(dv)
    if return_y:
        return_list.append(y)
    if return_frame_index:
        return_list.append(start_frame_index_list)
    if return_file_name:
        return_list.append(file_name_list)
    return return_list

def make_feed_dict_y_cmp_test(dv_y_cfg, file_dir_dict, speaker_id, file_name, start_frame_index, BTD_feat_remain):
    feat_name = dv_y_cfg.y_feat_name # Hard-coded here for now
    assert dv_y_cfg.batch_num_spk == 1
    # Make i/o shape arrays
    # This is numpy shape, not Tensor shape!
    # No speaker index here! Will add it to Tensor later
    y  = numpy.zeros((dv_y_cfg.spk_num_seq, dv_y_cfg.batch_seq_len, dv_y_cfg.feat_dim))
    dv = numpy.zeros((dv_y_cfg.batch_num_spk))

    # Do not use silence frames at the beginning or the end
    total_sil_one_side = dv_y_cfg.frames_silence_to_keep+dv_y_cfg.sil_pad

    # Make classification targets, index sequence
    try: true_speaker_index = dv_y_cfg.speaker_id_list_dict['train'].index(speaker_id)
    except ValueError: true_speaker_index = 0 # At generation time, since dv is not used, a non-train speaker is given an arbituary speaker index
    dv[0] = true_speaker_index

    if BTD_feat_remain is None:
        # Get new file, make BD
        _min_len, features = get_one_utter_by_name(file_name, file_dir_dict, feat_name_list=[feat_name], feat_dim_list=[dv_y_cfg.feat_dim])
        y_features = features[feat_name]
        l = y_features.shape[0]
        l_no_sil = l - total_sil_one_side * 2
        features = y_features[total_sil_one_side:total_sil_one_side+l_no_sil]
        B_total  = int((l_no_sil - dv_y_cfg.batch_seq_len) / dv_y_cfg.batch_seq_shift) + 1
        BTD_features = numpy.zeros((B_total, dv_y_cfg.batch_seq_len, dv_y_cfg.feat_dim))
        for b in range(B_total):
            start_i = dv_y_cfg.batch_seq_shift * b
            BTD_features[b] = features[start_i:start_i+dv_y_cfg.batch_seq_len]
    else:
        BTD_features = BTD_feat_remain
        B_total = BTD_features.shape[0]

    if B_total > dv_y_cfg.batch_seq_len:
        B_actual = dv_y_cfg.batch_seq_len
        B_remain = B_total - B_actual
        gen_finish = False
    else:
        B_actual = B_total
        B_remain = 0
        gen_finish = True

    for b in range(B_actual):
        y[b] = BTD_features[b]

    if B_remain > 0:
        BTD_feat_remain = numpy.zeros((B_remain, dv_y_cfg.batch_seq_len, dv_y_cfg.feat_dim))
        for b in range(B_remain):
            BTD_feat_remain[b] = BTD_features[b + B_actual]
    else:
        BTD_feat_remain = None

    batch_size = B_actual

    # B,T,D --> S(1),B,T*D
    x_val = numpy.reshape(y, (dv_y_cfg.batch_num_spk, dv_y_cfg.spk_num_seq, dv_y_cfg.batch_seq_len*dv_y_cfg.feat_dim))
    if dv_y_cfg.train_by_window:
        # S --> S*B
        y_val = numpy.repeat(dv, dv_y_cfg.spk_num_seq)
    else:
        y_val = dv

    feed_dict = {'x':x_val, 'y':y_val}
    return_list = [feed_dict, gen_finish, batch_size, BTD_feat_remain]
    return return_list

class dv_y_cmp_configuration(dv_y_configuration):
    """docstring for ClassName"""
    def __init__(self, cfg):
        super(dv_y_cmp_configuration, self).__init__(cfg)
        self.train_by_window = True # Optimise lambda_w; False: optimise speaker level lambda
        self.classify_in_training = True # Compute classification accuracy after validation errors during training
        self.batch_output_form = 'mean' # Method to convert from SBD to SD
        self.retrain_model = False
        self.previous_model_name = ''
        # self.python_script_name = '/home/dawna/tts/mw545/tools/merlin/merlin_cued_mw545_pytorch/exp_mw545/exp_dv_cmp_pytorch.py'
        self.python_script_name = os.path.realpath(__file__)
        self.y_feat_name   = 'cmp'
        self.out_feat_list = ['mgc', 'lf0', 'bap']
        self.nn_layer_config_list = [
            # Must contain: type, size; num_channels, dropout_p are optional, default 0, 1
            # {'type':'SineAttenCNN', 'size':512, 'num_channels':1, 'dropout_p':1, 'CNN_filter_size':5, 'Sine_filter_size':200,'lf0_mean':5.04976, 'lf0_var':0.361811},
            # {'type':'CNNAttenCNNWav', 'size':1024, 'num_channels':1, 'dropout_p':1, 'CNN_kernel_size':[1,3200], 'CNN_stride':[1,80], 'CNN_activation':'ReLU'},
            {'type':'ReLUDVMax', 'size':256, 'num_channels':2, 'channel_combi':'maxout', 'dropout_p':0, 'batch_norm':False},
            {'type':'ReLUDVMax', 'size':256, 'num_channels':2, 'channel_combi':'maxout', 'dropout_p':0, 'batch_norm':False},
            {'type':'ReLUDVMax', 'size':256, 'num_channels':2, 'channel_combi':'maxout', 'dropout_p':0.5, 'batch_norm':False},
            # {'type':'ReLUDVMax', 'size':256, 'num_channels':2, 'channel_combi':'maxout', 'dropout_p':0.5, 'batch_norm':False},
            {'type':'LinDV', 'size':self.dv_dim, 'num_channels':1, 'dropout_p':0.5}
        ]

        from modules_torch import DV_Y_CMP_model
        self.dv_y_model_class = DV_Y_CMP_model
        self.make_feed_dict_method_train = make_feed_dict_y_cmp_train
        self.make_feed_dict_method_test  = make_feed_dict_y_cmp_test
        self.auto_complete(cfg)

def train_dv_y_cmp_model(cfg, dv_y_cfg=None):
    if dv_y_cfg is None: dv_y_cfg = dv_y_cmp_configuration(cfg)
    train_dv_y_model(cfg, dv_y_cfg)

def test_dv_y_cmp_model(cfg, dv_y_cfg=None):
    if dv_y_cfg is None: dv_y_cfg = dv_y_cmp_configuration(cfg)
    # for s in [545,54,5]:
        # numpy.random.seed(s)
    class_test_dv_y_model(cfg, dv_y_cfg)

