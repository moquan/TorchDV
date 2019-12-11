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
        
        # Things no need to change
        self.learning_rate    = 0.0001
        self.num_train_epoch  = 1000
        self.warmup_epoch     = 10
        self.early_stop_epoch = 2    # After this number of non-improvement, roll-back to best previous model and decay learning rate
        self.max_num_decay    = 10
        self.epoch_num_batch  = {'train': 400, 'valid':400}

        self.batch_num_spk = 64 # S
        self.spk_num_utter = 1 # When >1, windows from different utterances are stacked along B
        

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
        self.utter_num_seq = int((self.batch_seq_total_len - self.batch_seq_len) / self.batch_seq_shift) + 1  # Outputs of each sequence is then averaged
        self.spk_num_seq   = self.spk_num_utter * self.utter_num_seq # B

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
        self.sil_index_dict = read_sil_index_file(sil_index_file='/home/dawna/tts/mw545/TorchDV/sil_index_list.scp')

    def change_to_debug_mode(self, process=None):
        if 'debug' in self.work_dir:
            self.epoch_num_batch  = {'train': 10, 'valid':10, 'test':10}
            if '_smallbatch' not in self.exp_dir:
                self.exp_dir = self.exp_dir + '_smallbatch'
            self.num_train_epoch = 10

    def change_to_class_test_mode(self):
        self.epoch_num_batch = {'test':40}
        self.batch_num_spk = 1
        self.spk_num_utter = 1
        spk_num_utter_list = [1,2,5,10]
        self.spk_num_utter_list = check_and_change_to_list(spk_num_utter_list)
        lambda_u_dict_file_name = 'lambda_u_class_test.dat'
        self.lambda_u_dict_file_name = os.path.join(self.exp_dir, lambda_u_dict_file_name)

        if self.y_feat_name == 'cmp':
            self.batch_seq_shift = 1
        elif self.y_feat_name == 'wav':
            self.batch_seq_shift = 80

        self.utter_num_seq = int((self.batch_seq_total_len - self.batch_seq_len) / self.batch_seq_shift) + 1
        # self.spk_num_seq = self.spk_num_utter * self.utter_num_seq
        if 'debug' in self.work_dir: self.change_to_debug_mode(process="class_test")

    def change_to_distance_test_mode(self):
        self.batch_num_spk = 10
        self.spk_num_utter = 5
        self.batch_seq_shift = 1
        self.utter_num_seq = int((self.batch_seq_total_len - self.batch_seq_len) / self.batch_seq_shift) + 1
        self.spk_num_seq = self.spk_num_utter * self.utter_num_seq
        if 'debug' in self.work_dir: self.change_to_debug_mode()

    def change_to_gen_h_mode(self):

        self.batch_speaker_list = ['p15', 'p28', 'p122', 'p68'] # Males 2, Females 2
        self.utter_name = '003'
        self.batch_num_spk = len(self.batch_speaker_list)
        self.spk_num_utter = 1
        self.h_list_file_name = os.path.join(self.exp_dir, "h_spk_list.dat")

    def additional_action_epoch(self, logger, dv_y_model):
        # Run every epoch, after train and eval
        # Add tests if necessary
        pass

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
                with dv_y_model.no_grad():
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

        dv_y_cfg.additional_action_epoch(logger, dv_y_model)

    logger.info('Reach num_train_epoch, best model, %s, best valid error %.4f' % (nnets_file_name, best_valid_loss))
    return best_valid_loss

def class_test_dv_y_model(cfg, dv_y_cfg):

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
        logger.info('Accuracy with %i utterances per speaker is %f' % (spk_num_utter, mean_accuracy))

def distance_test_dv_y_model(cfg, dv_y_cfg):
    distance_test_dv_y_wav_model(cfg, dv_y_cfg)

def distance_test_dv_y_cmp_model(cfg, dv_y_cfg):

    # Use test utterances only
    # Extract lambda per window, and shift the window by a few frames
    # Compute cosine distances
    # If contains nan, run lambda 0 test

    logger = make_logger("dv_y_config")
    dv_y_cfg = copy.deepcopy(dv_y_cfg)
    log_class_attri(dv_y_cfg, logger, except_list=dv_y_cfg.log_except_list)

    logger = make_logger("dist_dvy")
    logger.info('Creating data lists')
    speaker_id_list = dv_y_cfg.speaker_id_list_dict['train'] # For DV training and evaluation, use train speakers only
    speaker_loader  = list_random_loader(speaker_id_list)
    file_id_list    = read_file_list(cfg.file_id_list_file)
    file_list_dict  = make_dv_file_list(file_id_list, speaker_id_list, dv_y_cfg.data_split_file_number) # In the form of: file_list[(speaker_id, 'train')]
    make_feed_dict_method_train = dv_y_cfg.make_feed_dict_method_train

    dv_y_model = torch_initialisation(dv_y_cfg)
    dv_y_model.load_nn_model(dv_y_cfg.nnets_file_name)
    dv_y_model.eval()
    dv_y_model.detect_nan_model_parameters(logger)

    max_len_to_plot = 4
    dv_y_cfg.orig_batch_seq_len = dv_y_cfg.batch_seq_len
    dv_y_cfg.batch_seq_total_len += max_len_to_plot
    dv_y_cfg.batch_seq_len += max_len_to_plot
    # Distance sum holders
    dist_sum = {i+1:0. for i in range(max_len_to_plot)}

    num_batch = dv_y_cfg.epoch_num_batch['valid']

    for batch_idx in range(num_batch):
        batch_idx += 1
        logger.info('start generating Batch '+str(batch_idx))
        # Draw random speakers
        batch_speaker_list = speaker_loader.draw_n_samples(dv_y_cfg.batch_num_spk)
        # Make feed_dict for training
        feed_dict, batch_size = make_feed_dict_method_train(dv_y_cfg, file_list_dict, cfg.nn_feat_scratch_dirs, batch_speaker_list,  utter_tvt='test')

        for i in range(max_len_to_plot+1):
            feed_dict_i = {}
            for k in feed_dict:
                if k == 'x':
                    feed_dict_i['x'] = feed_dict['x'][:,:,i*dv_y_cfg.feat_dim:(i+dv_y_cfg.orig_batch_seq_len)*dv_y_cfg.feat_dim]
                else:
                    feed_dict_i[k] = feed_dict[k]
            with dv_y_model.no_grad():
                lambda_SBD_i = dv_y_model.gen_lambda_SBD_value(feed_dict=feed_dict_i)

            if i == 0:
                lambda_SBD_0 = lambda_SBD_i
            else:
                dist_i, nan_count = compute_Euclidean_distance(lambda_SBD_i, lambda_SBD_0)
                if nan_count == 0:
                    dist_sum[i] += dist_i
                else:
                    logger.info('NaN detected, run lambda zero test!')
                    lambda_0_test_dv_y_wav_model(cfg, dv_y_cfg)
                    return False

        logger.info('Printing distances')
        num_lambda = batch_size*batch_idx
        print([float(dist_sum[i+1]/(num_lambda)) for i in range(max_len_to_plot)])

def distance_test_dv_y_wav_model(cfg, dv_y_cfg):

    # Use test utterances only
    # Extract lambda per window, and shift the window by a few samples
    # Compute cosine distances
    # If contains nan, run lambda 0 test

    logger = make_logger("dv_y_config")
    dv_y_cfg = copy.deepcopy(dv_y_cfg)
    log_class_attri(dv_y_cfg, logger, except_list=dv_y_cfg.log_except_list)

    logger = make_logger("dist_dvy")
    logger.info('Creating data lists')
    speaker_id_list = dv_y_cfg.speaker_id_list_dict['train'] # For DV training and evaluation, use train speakers only
    speaker_loader  = list_random_loader(speaker_id_list)
    file_id_list    = read_file_list(cfg.file_id_list_file)
    file_list_dict  = make_dv_file_list(file_id_list, speaker_id_list, dv_y_cfg.data_split_file_number) # In the form of: file_list[(speaker_id, 'train')]
    make_feed_dict_method_train = dv_y_cfg.make_feed_dict_method_train

    dv_y_model = torch_initialisation(dv_y_cfg)
    dv_y_model.load_nn_model(dv_y_cfg.nnets_file_name)
    dv_y_model.eval()
    dv_y_model.detect_nan_model_parameters(logger)

    max_len_to_plot = 200
    dv_y_cfg.orig_batch_seq_len = dv_y_cfg.batch_seq_len
    dv_y_cfg.batch_seq_total_len += max_len_to_plot
    dv_y_cfg.batch_seq_len += max_len_to_plot
    # Distance sum holders
    dist_sum = {i+1:0. for i in range(max_len_to_plot)}

    num_batch = int(dv_y_cfg.epoch_num_batch['valid'] / 10)
    # num_batch = 2

    for batch_idx in range(num_batch):
        batch_idx += 1
        logger.info('start generating Batch '+str(batch_idx))
        # Draw random speakers
        batch_speaker_list = speaker_loader.draw_n_samples(dv_y_cfg.batch_num_spk)
        # Make feed_dict for training
        feed_dict, batch_size = make_feed_dict_method_train(dv_y_cfg, file_list_dict, cfg.nn_feat_scratch_dirs, batch_speaker_list,  utter_tvt='test')

        for i in range(max_len_to_plot+1):
            feed_dict_i = {}
            for k in feed_dict:
                if k == 'x':
                    feed_dict_i['x'] = feed_dict['x'][:,:,i:i+dv_y_cfg.orig_batch_seq_len]
                else:
                    feed_dict_i[k] = feed_dict[k]
            with dv_y_model.no_grad():
                lambda_SBD_i = dv_y_model.gen_lambda_SBD_value(feed_dict=feed_dict_i)

            if i == 0:
                lambda_SBD_0 = lambda_SBD_i
            else:
                dist_i, nan_count = compute_Euclidean_distance(lambda_SBD_i, lambda_SBD_0)
                if nan_count == 0:
                    dist_sum[i] += dist_i
                else:
                    logger.info('NaN detected, run lambda zero test!')
                    lambda_0_test_dv_y_wav_model(cfg, dv_y_cfg)
                    return False

        logger.info('Printing distances')
        num_lambda = batch_size*batch_idx
        print([float(dist_sum[i+1]/(num_lambda)) for i in range(max_len_to_plot)])
    
def lambda_0_test_dv_y_wav_model(cfg, dv_y_cfg):

    logger = make_logger("dv_y_config")
    dv_y_cfg = copy.deepcopy(dv_y_cfg)
    log_class_attri(dv_y_cfg, logger, except_list=dv_y_cfg.log_except_list)

    logger = make_logger("lamda_0_dvy")
    logger.info('Creating data lists')
    speaker_id_list = dv_y_cfg.speaker_id_list_dict['train'] # For DV training and evaluation, use train speakers only
    speaker_loader  = list_random_loader(speaker_id_list)
    file_id_list    = read_file_list(cfg.file_id_list_file)
    file_list_dict  = make_dv_file_list(file_id_list, speaker_id_list, dv_y_cfg.data_split_file_number) # In the form of: file_list[(speaker_id, 'train')]
    make_feed_dict_method_train = dv_y_cfg.make_feed_dict_method_train

    dv_y_model = torch_initialisation(dv_y_cfg)
    dv_y_model.load_nn_model(dv_y_cfg.nnets_file_name)
    dv_y_model.eval()
    dv_y_model.detect_nan_model_parameters(logger)

    logger.info('Printing bias of expansion layer')
    b = dv_y_model.nn_model.expansion_layer.bias
    print(b)

    num_batch = dv_y_cfg.epoch_num_batch['valid']
    # Collect x that produce 0 lambda
    x_list = []
    speaker_counter = {}
    for batch_idx in range(num_batch):
        batch_idx += 1
        logger.info('start generating Batch '+str(batch_idx))
        # Draw random speakers
        batch_speaker_list = speaker_loader.draw_n_samples(dv_y_cfg.batch_num_spk)
        # Make feed_dict for training
        feed_dict, batch_size = make_feed_dict_method_train(dv_y_cfg, file_list_dict, cfg.nn_feat_scratch_dirs, batch_speaker_list,  utter_tvt='test')
        with dv_y_model.no_grad():
            lambda_SBD = dv_y_model.gen_lambda_SBD_value(feed_dict=feed_dict)

        S,B,D = lambda_SBD.shape

        for s in range(S):
            for b in range(B):
                lambda_D = lambda_SBD[s,b]
                n = numpy.count_nonzero(lambda_D)
                if n == 0:
                    x = feed_dict['x'][s,b]
                    x_list.append(x)
                    speaker_id = batch_speaker_list[s]
                    try: speaker_counter[speaker_id] += 1
                    except: speaker_counter[speaker_id] = 1

    logger.info('Number of windows give 0 lambda are %i out of %i ' % (len(x_list), batch_size*num_batch))
    print(speaker_counter)

    # Plot these waveforms
    num_to_print = 5
    if len(x_list) > num_to_print:
        logger.info('PLot waveforms that give 0 lambda')
        fig, ax_list = plt.subplots(num_to_print)
        fig.suptitle('%i waveforms that give 0 lambda' % (num_to_print))
        for i in range(num_to_print):
            x = x_list[i]
            # Plot x, waveform
            ax_list[i].plot(x)
        fig_name = '/home/dawna/tts/mw545/Export_Temp' + "/wav_0_lambda.png"
        logger.info('Saving Waveform to %s' % fig_name)
        fig.savefig(fig_name)

    # Feed in waveforms that produce 0 lambda
    feed_dict_0 = feed_dict
    i = 0
    assert len(x_list) > (S*B)
    for s in range(S):
        for b in range(B):
            feed_dict_0['x'][s,b,:] = x_list[i]
            i += 1

    with dv_y_model.no_grad():
        h_list = dv_y_model.gen_all_h_values(feed_dict=feed_dict_0)

    # Insert x in h_list for plotting as well
    h_list.insert(0, feed_dict['x'])
    h_list_file_name = os.path.join(dv_y_cfg.exp_dir, "h_0_list.dat")
    pickle.dump(h_list, open(h_list_file_name, "wb" ))
    return h_list

def generate_all_h_dv_y_model(cfg, dv_y_cfg):

    # Generate h of all layers
    # File names: see dv_y_cfg.change_to_gen_h_mode

    logger = make_logger("dv_y_config")
    dv_y_cfg = copy.deepcopy(dv_y_cfg)
    dv_y_cfg.change_to_gen_h_mode()
    log_class_attri(dv_y_cfg, logger, except_list=dv_y_cfg.log_except_list)

    logger = make_logger("gen_h_dvy")
    make_feed_dict_method_train = dv_y_cfg.make_feed_dict_method_train
    
    batch_speaker_list = dv_y_cfg.batch_speaker_list
    file_list_dict = {}
    for speaker_name in batch_speaker_list:
        file_list_dict[(speaker_name, 'eval')] = ['%s_%s' % (speaker_name, dv_y_cfg.utter_name)]

    dv_y_model = torch_initialisation(dv_y_cfg)
    dv_y_model.load_nn_model(dv_y_cfg.nnets_file_name)
    dv_y_model.eval()

    # Make feed_dict for training
    feed_dict, batch_size = make_feed_dict_method_train(dv_y_cfg, file_list_dict, cfg.nn_feat_scratch_dirs, batch_speaker_list, all_utt_start_frame_index=10, utter_tvt='eval')
    with dv_y_model.no_grad():
        h_list = dv_y_model.gen_all_h_values(feed_dict=feed_dict)

    # Insert x in h_list for plotting as well
    h_list.insert(0, feed_dict['x'])
    h_list_file_name = dv_y_cfg.h_list_file_name
    pickle.dump(h_list, open(h_list_file_name, "wb" ))
    for h in h_list:
        print(h.shape)
    return h_list

def plot_all_h_dv_y_model(cfg, dv_y_cfg):
    logger = make_logger("dv_y_config")
    dv_y_cfg = copy.deepcopy(dv_y_cfg)
    dv_y_cfg.change_to_gen_h_mode()
    log_class_attri(dv_y_cfg, logger, except_list=dv_y_cfg.log_except_list)

    logger = make_logger("plot_h_dvy")
    h_list_file_name = dv_y_cfg.h_list_file_name
    try:
        h_list = pickle.load(open(h_list_file_name, "rb" ))
        logger.info('Loaded %s' % h_list_file_name)
    except:
        h_list = generate_all_h_dv_y_model(cfg, dv_y_cfg)

    S = dv_y_cfg.batch_num_spk
    B = dv_y_cfg.spk_num_seq

    for s in range(S):
        for b in range(B):
            fig, ax_list = plt.subplots(len(h_list))
            for i,h in enumerate(h_list):
                # logger.info('Layer %i ' % (i))
                # Print first row
                if len(h.shape) > 3:
                    for h_i in h:
                        ax_list[i].plot(h_i[s,b])
                else:
                    ax_list[i].plot(h[s,b])

            b_str = '0'*(3-len(str(b)))+str(b)
            fig_name = '/home/dawna/tts/mw545/Export_Temp/PNG_out/' + "h_spk_%i_seq_%s.png" % (s,b_str)
            logger.info('Saving h to %s' % fig_name)
            fig.savefig(fig_name)
            plt.close(fig)

def eval_logit_dv_y_model(cfg, dv_y_cfg):
    logger = make_logger("dv_y_config")
    dv_y_cfg = copy.deepcopy(dv_y_cfg)
    log_class_attri(dv_y_cfg, logger, except_list=dv_y_cfg.log_except_list)

    logger = make_logger("eval_logit")
    logger.info('Creating data lists')
    speaker_id_list = dv_y_cfg.speaker_id_list_dict['train'] # For DV training and evaluation, use train speakers only

    make_feed_dict_method_train = dv_y_cfg.make_feed_dict_method_train

    dv_y_cfg.batch_num_spk = 4
    batch_speaker_list = ['p15', 'p28', 'p122', 'p68'] # Males 2, Females 2
    file_list_dict = {}
    for speaker_name in batch_speaker_list:
        file_list_dict[(speaker_name, 'eval')] = ['%s_003' % speaker_name]

    dv_y_model = torch_initialisation(dv_y_cfg)
    dv_y_model.load_nn_model(dv_y_cfg.nnets_file_name)
    dv_y_model.eval()

    fig = plt.figure(figsize=(200,100))
    num_spk = dv_y_cfg.batch_num_spk
    num_win = 5
    # fig.set_size_inches(185, 105)
    fig, ax_list = plt.subplots(num_spk, num_win)
    fig.suptitle('%i speakers, %i windows' % (num_spk, num_win))
    # Make feed_dict for training
    feed_dict, batch_size = make_feed_dict_method_train(dv_y_cfg, file_list_dict, cfg.nn_feat_scratch_dirs, batch_speaker_list, all_utt_start_frame_index=4000, utter_tvt='eval')
    with dv_y_model.no_grad():
        logit_SBD = dv_y_model.gen_logit_SBD_value(feed_dict=feed_dict)
    for i in range(num_spk):
        for j in range(num_win):
            logit_D = logit_SBD[i,j]
            ax_list[i,j].plot(logit_D)

    fig_name = '/home/dawna/tts/mw545/Export_Temp' + "/gen_logit.png"
    logger.info('Saving logits to %s' % fig_name)
    fig.savefig(fig_name)


    # SinenetV3 specific
    plot_f0_tau = False
    plot_h      = False
    for nn_layer_config in dv_y_cfg.nn_layer_config_list:
        if nn_layer_config['type'] == 'SinenetV3':
            plot_f0_tau = True
            plot_h = True
            break
    if plot_f0_tau:
    # if False:
        with dv_y_model.no_grad():
            nlf, tau, tau_list = dv_y_model.gen_nlf_tau_values(feed_dict=feed_dict)
        # Plot f0
        num_spk = dv_y_cfg.batch_num_spk
        num_win = 5
        fig, ax = plt.subplots()
        fig.suptitle('F0, %i speakers, %i windows' % (num_spk, num_win))
        ax.plot(numpy.squeeze(nlf).T)
        fig_name = '/home/dawna/tts/mw545/Export_Temp' + "/nlf.png"
        logger.info('Saving NLF to %s' % fig_name)
        fig.savefig(fig_name)
        # Plot tau
        num_spk = dv_y_cfg.batch_num_spk
        num_win = 5
        fig, ax = plt.subplots()
        fig.suptitle('Tau, %i speakers, %i windows' % (num_spk, num_win))
        ax.plot(numpy.squeeze(tau).T)
        fig_name = '/home/dawna/tts/mw545/Export_Temp' + "/tau.png"
        logger.info('Saving Tau to %s' % fig_name)
        fig.savefig(fig_name)
        # Plot tau trajectories
        tau_SBT = numpy.stack(tau_list, axis=-1)
        num_spk = dv_y_cfg.batch_num_spk
        num_win = 5
        fig, ax_list = plt.subplots(num_spk, num_win)
        fig.suptitle('Tau trajectory, %i speakers, %i windows' % (num_spk, num_win))
        for i in range(num_spk):
            for j in range(num_win):
                tau_T = tau_SBT[i,j]
                ax_list[i,j].plot(numpy.squeeze(tau_T))
        fig_name = '/home/dawna/tts/mw545/Export_Temp' + "/tau_list.png"
        logger.info('Saving Tau trajectory to %s' % fig_name)
        fig.savefig(fig_name)

    if plot_h:
        with dv_y_model.no_grad():
            h = dv_y_model.gen_sinenet_h_value(feed_dict=feed_dict)
        # Plot different speaker
        fig, ax_list = plt.subplots(num_spk)
        fig.suptitle('h, %i speakers' % (num_spk))
        for i in range(num_spk):
            h_BD = h[i]
            h_D  = h_BD[0]
            ax_list[i].plot(h_D)
        fig_name = '/home/dawna/tts/mw545/Export_Temp' + "/h_speaker.png"
        logger.info('Saving h_speaker to %s' % fig_name)
        fig.savefig(fig_name)
        # Plot different window
        fig, ax_list = plt.subplots(num_win)
        fig.suptitle('h, %i windows' % (num_win))
        h_BD = h[0]
        for i in range(num_win):
            h_D  = h_BD[i]
            ax_list[i].plot(h_D)
        fig_name = '/home/dawna/tts/mw545/Export_Temp' + "/h_window.png"
        logger.info('Saving h_window to %s' % fig_name)
        fig.savefig(fig_name)

def relu_0_stats(cfg, dv_y_cfg):

    # Generate a lot of all_h
    # For each layer, each dimension, compute:
    # zero/non-zero ratio

    # Use all train speakers, train utterances first
    logger = make_logger("dv_y_config")
    dv_y_cfg = copy.deepcopy(dv_y_cfg)
    log_class_attri(dv_y_cfg, logger, except_list=dv_y_cfg.log_except_list)

    logger = make_logger("relu_0_stats")
    logger.info('Creating data lists')

    num_batch  = dv_y_cfg.epoch_num_batch['valid']
    if dv_y_cfg.train_by_window:
        batch_size = dv_y_cfg.batch_num_spk * dv_y_cfg.spk_num_seq
    else:
        batch_size = dv_y_cfg.batch_num_spk

    all_h_list_file_name = os.path.join(dv_y_cfg.exp_dir, 'all_h_list.dat')
    try:
        all_h_list = pickle.load(open(all_h_list_file_name, "rb" ))
        logger.info('Loaded %s' % all_h_list_file_name)
    except:
        speaker_id_list = dv_y_cfg.speaker_id_list_dict['train']
        speaker_loader  = list_random_loader(speaker_id_list)
        file_id_list    = read_file_list(cfg.file_id_list_file)
        file_list_dict  = make_dv_file_list(file_id_list, speaker_id_list, dv_y_cfg.data_split_file_number) # In the form of: file_list[(speaker_id, 'train')]
        make_feed_dict_method_train = dv_y_cfg.make_feed_dict_method_train

        dv_y_model = torch_initialisation(dv_y_cfg)
        dv_y_model.load_nn_model(dv_y_cfg.nnets_file_name)
        dv_y_model.eval()

        all_h_list = []
        for batch_idx in range(num_batch):
            batch_idx += 1
            if batch_idx % 10 == 0:
                logger.info('start generating Batch '+str(batch_idx))
            # Draw random speakers
            batch_speaker_list = speaker_loader.draw_n_samples(dv_y_cfg.batch_num_spk)
            # Make feed_dict for training
            feed_dict, batch_size = make_feed_dict_method_train(dv_y_cfg, file_list_dict, cfg.nn_feat_scratch_dirs, batch_speaker_list,  utter_tvt='train')
            with dv_y_model.no_grad():
                h_list = dv_y_model.gen_all_h_values(feed_dict=feed_dict)
            all_h_list.append(h_list)
        logger.info('Saving all_h_list to %s' % all_h_list_file_name)
        pickle.dump(all_h_list, open(all_h_list_file_name, "wb" ))

    # Create holders for stats
    h_list = all_h_list[0]
    h_stats = {}
    for k in ['non_zero_count']:
        h_stats[k] = []
        for h in h_list:
            h_stats[k].append(numpy.zeros(h.shape[-1]))

    for h_list in all_h_list:
        for i,h in enumerate(h_list):
            l = len(h.shape)
            # Detect non-zero values, change to 1
            h_temp = (h != 0).astype(int)
            # Sum over all dimensions except the last one
            for j in range(l-1):
                h_temp = numpy.sum(h_temp, axis=0)
            h_stats['non_zero_count'][i] += h_temp

    h_stats['non_zero_count'] = [h / (num_batch * batch_size) for h in h_stats['non_zero_count']]
    logger.info('Printing non-zero ratios')
    for h in h_stats['non_zero_count']:
        print(h)