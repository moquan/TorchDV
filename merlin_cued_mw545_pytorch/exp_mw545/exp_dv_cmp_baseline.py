# exp_dv_cmp_baseline.py
import os, sys, cPickle, time, shutil, logging, copy
import math, numpy, scipy
from modules import make_logger, read_file_list, prepare_file_path, prepare_file_path_list, make_held_out_file_number, copy_to_scratch
from modules import keep_by_speaker, remove_by_speaker, keep_by_file_number, remove_by_file_number, keep_by_min_max_file_number, check_and_change_to_list
from modules_2 import compute_feat_dim, log_class_attri, resil_nn_file_list, norm_nn_file_list, get_utters_from_binary_dict, count_male_female_class_errors

from io_funcs.binary_io import BinaryIOCollection
io_fun = BinaryIOCollection()

from modules_torch import config_torch, dv_y_cmp_model

class dv_y_configuration(object):
    
    def __init__(self, cfg):
        
        # Things to be filled
        self.python_script_name = None
        self.dv_y_model_class = None
        self.make_feed_dict_method = None
        self.y_feat_name   = None
        self.out_feat_list = None
        self.nn_layer_config_list = None
        
        # Things no need to change
        self.tf_scope_name = 'dv_y_model'
        self.learning_rate    = 0.001
        self.num_train_epoch  = 100
        self.warmup_epoch     = 10
        self.early_stop_epoch = 5    # After this number of non-improvement, roll-back to best previous model and decay learning rate
        self.max_num_decay    = 10
        self.num_train_batch  = 400
        self.num_valid_batch  = self.num_train_batch

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

        self.train_speaker_list   = cfg.train_speaker_list
        self.num_train_speakers   = cfg.num_train_speakers



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

        self.gpu_id = 0
        self.gpu_per_process_gpu_memory_fraction = 0.8

    def change_to_debug_mode(self):
        self.num_train_batch  = 10
        if '_smallbatch' not in self.exp_dir:
            self.exp_dir = self.exp_dir + '_smallbatch'
        self.num_train_epoch = 5
        self.num_valid_batch = self.num_train_batch
        self.train_speaker_list   = self.train_speaker_list[:10]
        self.num_train_speakers   = 10

    def change_to_test_mode(self):
        self.num_valid_batch = 4000
        self.batch_num_spk = 10
        self.spk_num_utter = 1
        spk_num_utter_list = [1,2,5,10]
        self.spk_num_utter_list = check_and_change_to_list(spk_num_utter_list)
        self.batch_seq_shift = 1
        self.utter_num_seq = int((self.batch_seq_total_len - self.batch_seq_len) / self.batch_seq_shift) + 1  # Outputs of each sequence is then averaged
        # self.spk_num_seq = self.spk_num_utter * self.utter_num_seq
        if 'debug' in self.work_dir: self.change_to_debug_mode()

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
        exp_dir = exp_dir + str(nn_layer_config['type'])[:3] + str(nn_layer_config['size']) + "_"
        if 'batch_norm' in nn_layer_config and nn_layer_config['batch_norm']:
            exp_dir = exp_dir + 'BN_'
    exp_dir = exp_dir + "DV"+str(model_cfg.dv_dim)+"_S"+str(model_cfg.batch_num_spk)+"_B"+str(model_cfg.spk_num_seq)+"_T"+str(model_cfg.batch_seq_len)
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

def tf_session_initialisation(logger, dv_y_cfg, dv_y_model_class):
    logger.info('config tensorflow')
    tf_config = config_tf(dv_y_cfg)
    logger.info('building model')
    dv_y_model = dv_y_model_class(dv_y_cfg)
    logger.info('start running model')
    dv_y_model.sess = tf.Session(config=tf_config)
    dv_y_model.sess.run(dv_y_model.init)
    if dv_y_cfg.retrain_model:
        logger.info('restore previous model, '+dv_y_cfg.previous_model_name)
        dv_y_model.saver.restore(sess, dv_y_cfg.previous_model_name)
    return dv_y_model

def train_dv_y_model(cfg, dv_y_cfg=None, dv_y_model_class=None, make_feed_dict_method=None):

    logger = make_logger("train_dv_y_model")
    if dv_y_cfg is None:              dv_y_cfg = dv_y_configuration(cfg)
    if dv_y_model_class is None:      dv_y_model_class = dv_y_cfg.dv_y_model_class
    if make_feed_dict_method is None: make_feed_dict_method = dv_y_cfg.make_feed_dict_method
    log_class_attri(dv_y_cfg, logger, except_list=[])
    
    logger.info('Creating data lists')
    speaker_id_list = dv_y_cfg.train_speaker_list # For DV training and evaluation, use train speakers only
    file_id_list    = read_file_list(cfg.file_id_list_file)
    file_list_dict  = make_dv_file_list(file_id_list, speaker_id_list, dv_y_cfg.data_split_file_number) # In the form of: file_list[(speaker_id, 'train')]

    # Tensorflow configuration and Model Initialisation
    dv_y_model = tf_session_initialisation(logger, dv_y_cfg, dv_y_model_class)

    # Counters and Loss
    epoch      = 0
    early_stop = 0
    num_decay  = 0    
    best_validation_loss = sys.float_info.max
    # previous_valid_loss  = sys.float_info.max
    num_train_batch  = dv_y_cfg.num_train_batch
    num_valid_batch  = dv_y_cfg.num_valid_batch
    num_train_epoch  = dv_y_cfg.num_train_epoch
    early_stop_epoch = dv_y_cfg.early_stop_epoch
    max_num_decay    = dv_y_cfg.max_num_decay
    numpy.random.seed(20375)
    
    while (epoch < num_train_epoch):
        epoch = epoch + 1

        logger.info('start training Epoch '+str(epoch))
        epoch_start_time = time.time()
        # Training
        for batch_idx in range(num_train_batch):
            # Draw random speakers
            batch_speaker_list = numpy.random.choice(speaker_id_list, dv_y_cfg.batch_num_spk)
            # Make feed_dict for training
            feed_dict, batch_size = make_feed_dict_y_cmp(dv_y_cfg, file_list_dict, cfg.nn_feat_scratch_dirs, dv_y_model, batch_speaker_list,  utter_tvt='train', model_is_train=True)
            dv_y_model.train()
            dv_y_model.train_model_param(feed_dict=feed_dict)
        epoch_train_time = time.time()

        logger.info('start evaluating Epoch '+str(epoch))
        output_string = 'epoch '+str(epoch)
        for utter_tvt_name in ['train', 'valid', 'test']:
            total_batch_size = 0.
            total_loss       = 0.
            for batch_idx in range(num_valid_batch):
                # Draw random speakers
                batch_speaker_list = numpy.random.choice(speaker_id_list, dv_y_cfg.batch_num_spk)
                # Make feed_dict for evaluation
                feed_dict, batch_size = make_feed_dict_y_cmp(dv_y_cfg, file_list_dict, cfg.nn_feat_scratch_dirs, dv_y_model, batch_speaker_list, utter_tvt=utter_tvt_name, model_is_train=False)
                dv_y_model.eval()
                batch_mean_loss = dv_y_model.return_train_loss(feed_dict=feed_dict)
                total_batch_size = total_batch_size + batch_size
                total_loss       = total_loss + batch_mean_loss
            average_loss = total_loss/num_valid_batch
            output_string = output_string + ', '+utter_tvt_name+' loss is '+str(average_loss)
            
            if utter_tvt_name == 'valid':
                nnets_file_name = dv_y_cfg.nnets_file_name
                # Compare validation error
                valid_error = average_loss
                if valid_error < best_validation_loss:
                    early_stop = 0
                    logger.info('valid error reduced, saving model, '+nnets_file_name)
                    dv_y_model.save_current_model(nnets_file_name)
                    best_validation_loss = valid_error
                elif valid_error > previous_valid_loss:
                    early_stop = early_stop + 1
                    new_learning_rate = dv_y_model.learning_rate*0.5
                    logger.info('reduce learning rate to '+str(new_learning_rate))
                    dv_y_model.update_learning_rate(new_learning_rate)
                if (early_stop > early_stop_epoch) and (epoch > dv_y_cfg.warmup_epoch):
                    early_stop = 0
                    num_decay = num_decay + 1
                    if num_decay > max_num_decay:
                        logger.info('stopping early, best model, '+nnets_file_name+', best validation error '+str(best_validation_loss))
                        dv_y_model.close_tf_session_and_reset()
                        return best_validation_loss
                    logger.info('loading previous best model, '+nnets_file_name)
                    dv_y_model.load_prev_model(nnets_file_name)
                    # logger.info('reduce learning rate to '+str(new_learning_rate))
                    # dv_y_model.update_learning_rate(new_learning_rate)
                previous_valid_loss = valid_error

        epoch_valid_time = time.time()
        output_string = output_string + ', train time is %.2f, valid time is  %.2f' %((epoch_train_time - epoch_start_time), (epoch_valid_time - epoch_train_time))
        logger.info(output_string)

    dv_y_model.close_tf_session_and_reset()
    return best_validation_loss

def classification_test_dv_y_model(cfg, dv_y_cfg=None, dv_y_model_class=None, make_feed_dict_method=None):

    logger = make_logger("test_dv_y_model")
    # For DV training and evaluation, use train speakers only
    if dv_y_cfg is None:              dv_y_cfg = dv_y_configuration(cfg)
    if dv_y_model_class is None:      dv_y_model_class = dv_y_cfg.dv_y_model_class
    if make_feed_dict_method is None: make_feed_dict_method = dv_y_cfg.make_feed_dict_method
    dv_y_cfg.change_to_test_mode()
    log_class_attri(dv_y_cfg, logger, except_list=[])
    
    logger.info('Creating data lists')
    # For DV training and evaluation, use train speakers only
    speaker_id_list = dv_y_cfg.train_speaker_list
    file_id_list   = read_file_list(cfg.file_id_list_file)
    # In the form of: file_list[(speaker_id, 'train')]
    file_list_dict = make_dv_file_list(file_id_list, speaker_id_list, dv_y_cfg.data_split_file_number)

    numpy.random.seed(20375)

    logger.info('Classification Test')
    for spk_num_utter in dv_y_cfg.spk_num_utter_list:
        logger.info('Testing %i utterances per speaker' % spk_num_utter)
        dv_y_cfg.spk_num_utter = spk_num_utter
        dv_y_cfg.spk_num_seq = dv_y_cfg.spk_num_utter * dv_y_cfg.utter_num_seq # B is larger, stacked multiple utterances

        # Tensorflow configuration and Model Initialisation
        dv_y_model = tf_session_initialisation(logger, dv_y_cfg, dv_y_model_class)
        dv_y_model.load_prev_model(dv_y_cfg.nnets_file_name)

        num_valid_batch = dv_y_cfg.num_valid_batch

        total_correct_class = 0.
        total_wrong_class   = {}
        for batch_idx in range(num_valid_batch):
            if (batch_idx+1) % int(num_valid_batch/10) == 0:
                logger.info('generating batch %i / %i' % (batch_idx+1, num_valid_batch))
            # Draw random speakers
            batch_speaker_list = numpy.random.choice(speaker_id_list, dv_y_cfg.batch_num_spk)
            feed_dict, batch_size, dv = make_feed_dict_y_cmp(dv_y_cfg, file_list_dict, cfg.nn_feat_scratch_dirs, dv_y_model, batch_speaker_list, utter_tvt='test', model_is_train=False, return_dv=True)
            logit_SD = dv_y_model.gen_logit_SD(feed_dict)
            for idx1 in range(dv_y_cfg.batch_num_spk):
                print dv[idx1]
                print logit_SD[idx1]
                correct_speaker_index = numpy.argmax(dv[idx1])
                predict_speaker_index = numpy.argmax(logit_SD[idx1])
                # Correct case
                if correct_speaker_index == predict_speaker_index:
                    total_correct_class += 1.
                else:
                    correct_speaker_name = dv_y_cfg.train_speaker_list[correct_speaker_index]
                    predict_speaker_name = dv_y_cfg.train_speaker_list[predict_speaker_index]
                    try: total_wrong_class[(correct_speaker_name, predict_speaker_name)] += 1.
                    except: total_wrong_class[(correct_speaker_name, predict_speaker_name)] = 1.
                    # logger.info('Wrong: classified as '+predict_speaker_name+', should be '+correct_speaker_name)

        accuracy = total_correct_class / float(num_valid_batch * dv_y_cfg.batch_num_spk)

        for wrong_class_type in total_wrong_class.keys():
            logger.info('Wrong classification: '+str(wrong_class_type)+str(total_wrong_class[wrong_class_type]))

        wrong_list = count_male_female_class_errors(total_wrong_class, cfg.male_speaker_list)

        logger.info('Classification accuracy is '+str(accuracy)+' with '+str(dv_y_cfg.spk_num_utter)+' utterances, dv size '+str(cfg.dv_dim)+', EER is '+str((1.-accuracy)*100)+'%')
        error_string = 'Error types are '
        for t in ['mm', 'ff', 'mf', 'fm']:
            error_string = error_string + t + ' ' + str(wrong_list[t] / float(num_valid_batch * dv_y_cfg.batch_num_spk))+', '
        logger.info(error_string)

        dv_y_model.close_tf_session_and_reset()

def gen_dv_y_model(cfg, dv_y_cfg=None, dv_y_model_class=None, make_feed_dict_method=None):

    logger = make_logger("gen_dv_y_model")
    if dv_y_cfg is None:              dv_y_cfg = dv_y_configuration(cfg)
    if dv_y_model_class is None:      dv_y_model_class = dv_y_cfg.dv_y_model_class
    if make_feed_dict_method is None: make_feed_dict_method = dv_y_cfg.make_feed_dict_method
    dv_y_cfg.change_to_gen_mode()
    log_class_attri(dv_y_cfg, logger, except_list=[])
    
    logger.info('Creating data lists')
    # In the form of: file_list[(speaker_id, 'train')]
    file_id_list = read_file_list(cfg.file_id_list_file)
    speaker_id_list = cfg.all_speaker_list
    file_list_dict = make_dv_file_list(file_id_list, speaker_id_list, dv_y_cfg.data_split_file_number)

    # Tensorflow configuration and Model Initialisation
    dv_y_model = tf_session_initialisation(logger, dv_y_cfg, dv_y_model_class)
    dv_y_model.load_prev_model(dv_y_cfg.nnets_file_name)

    numpy.random.seed(20375)

    generation_finished = False

    remain_speaker_id_list = speaker_id_list
    lambda_S = {}
    while not generation_finished:
        if len(remain_speaker_id_list) < dv_y_cfg.batch_num_spk:
            padding_speaker_number = dv_y_cfg.batch_num_spk - len(remain_speaker_id_list) 
            batch_speaker_list     = remain_speaker_id_list + remain_speaker_id_list[:padding_speaker_number]
            actual_speaker_number  = len(remain_speaker_id_list)
            remain_speaker_id_list = []
        else:
            batch_speaker_list     = remain_speaker_id_list[0:dv_y_cfg.batch_num_spk]
            remain_speaker_id_list = remain_speaker_id_list[dv_y_cfg.batch_num_spk:]
            actual_speaker_number  = dv_y_cfg.batch_num_spk

        logger.info('generating for '+ str(batch_speaker_list[:actual_speaker_number]))

        feed_dict, batch_size, dv = make_feed_dict_y_cmp(dv_y_cfg, file_list_dict, cfg.nn_feat_scratch_dirs, dv_y_model, batch_speaker_list, utter_tvt='test', model_is_train=False, return_dv=True)
        lambda_SBD = dv_y_model.gen_lambda_SBD(feed_dict)
        
        for i in range(actual_speaker_number):
            speaker_id = batch_speaker_list[i]
            lambda_BD = lambda_SBD[i]
            lambda_S[speaker_id] = numpy.mean(lambda_BD, axis=0)

        if len(remain_speaker_id_list) == 0:
            generation_finished = True
    logger.info('saving dv of %i speakers ' % len(lambda_S.keys()))
    logger.info('saving dv file to '+ str(dv_y_cfg.dv_file_name))
    cPickle.dump(lambda_S, open(dv_y_cfg.dv_file_name, 'wb'))

    dv_y_model.close_tf_session_and_reset()


def make_feed_dict_y_cmp(dv_y_cfg, file_list_dict, file_dir_dict, dv_y_model, batch_speaker_list, utter_tvt, model_is_train, return_dv=False, return_y=False, return_frame_index=False, return_file_name=False):
    feat_name = dv_y_cfg.y_feat_name # Hard-coded here for now
    # Make i/o shape arrays
    y  = numpy.zeros((dv_y_cfg.batch_num_spk, dv_y_cfg.spk_num_seq, dv_y_cfg.batch_seq_len, dv_y_cfg.feat_dim))
    dv = numpy.zeros((dv_y_cfg.batch_num_spk))

    # Do not use silence frames at the beginning or the end
    total_sil_one_side = dv_y_cfg.frames_silence_to_keep+dv_y_cfg.sil_pad
    min_file_len = dv_y_cfg.batch_seq_total_len + 2 * total_sil_one_side

    file_name_list = []
    start_frame_index_list = []
    for speaker_idx in range(dv_y_cfg.batch_num_spk):
        speaker_id = batch_speaker_list[speaker_idx]

        # Make dv 1-hot output
        try: true_speaker_index = dv_y_cfg.train_speaker_list.index(speaker_id)
        except: true_speaker_index = 0 # At generation time, since dv is not used, a non-train speaker is given an arbituary speaker index
        dv[speaker_idx, true_speaker_index] = 1.

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

    feed_dict = [y, dv]

    if dv_y_cfg.train_by_window:
        batch_size = dv_y_cfg.batch_num_spk * dv_y_cfg.spk_num_seq
    else:
        batch_size = dv_y_cfg.batch_num_spk

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


class dv_y_cmp_configuration(dv_y_configuration):
    """docstring for ClassName"""
    def __init__(self, cfg):
        super(dv_y_cmp_configuration, self).__init__(cfg)
        self.train_by_window  = True # Optimise lambda_w; False: optimise speaker level lambda
        self.batch_output_form = 'mean' # Method to convert from SBD to SD
        self.retrain_model = False
        self.previous_model_name = ''
        self.python_script_name = '/home/dawna/tts/mw545/DVExp/tools/merlin_cued_mw545/exp_mw545/exp_dv_cmp_baseline.py'
        self.y_feat_name   = 'cmp'
        self.out_feat_list = ['mgc', 'lf0', 'bap']
        self.nn_layer_config_list = [
            # Must contain: type, size; num_channels, dropout_p are optional, default 0, 1
            # {'type':'SineAttenCNN', 'size':512, 'num_channels':1, 'dropout_p':1, 'CNN_filter_size':5, 'Sine_filter_size':200,'lf0_mean':5.04976, 'lf0_var':0.361811},
            # {'type':'CNNAttenCNNWav', 'size':1024, 'num_channels':1, 'dropout_p':1, 'CNN_kernel_size':[1,3200], 'CNN_stride':[1,80], 'CNN_activation':'ReLU'},
            {'type':'ReLUDVMax', 'size':512, 'num_channels':2, 'channel_combi':'maxout', 'dropout_p':0, 'batch_norm':False},
            {'type':'ReLUDVMax', 'size':512, 'num_channels':2, 'channel_combi':'maxout', 'dropout_p':0, 'batch_norm':False},
            {'type':'ReLUDVMax', 'size':512, 'num_channels':2, 'channel_combi':'maxout', 'dropout_p':0, 'batch_norm':False},
            # {'type':'ReLUDVMaxDrop', 'size':512, 'num_channels':2, 'channel_combi':'maxout', 'dropout_p':0.5, 'batch_norm':False},
            {'type':'LinDV', 'size':self.dv_dim, 'num_channels':1, 'dropout_p':0}
        ]

        self.dv_y_model_class = dv_y_cmp_model
        self.make_feed_dict_method = make_feed_dict_y_cmp

        self.auto_complete(cfg)


def train_dv_y_cmp_model(cfg, dv_y_cfg=None):
    if dv_y_cfg is None: dv_y_cfg = dv_y_cmp_configuration(cfg)
    train_dv_y_model(cfg, dv_y_cfg)#, dv_y_model_class=dv_y_cfg.dv_y_model_class, make_feed_dict_method=dv_y_cfg.make_feed_dict_method)

def class_test_dv_y_cmp_model(cfg, dv_y_cfg=None):
    if dv_y_cfg is None: dv_y_cfg = dv_y_cmp_configuration(cfg)
    classification_test_dv_y_model(cfg, dv_y_cfg)#, dv_y_model_class=dv_y_cmp_model, make_feed_dict_method=make_feed_dict_y_cmp)

def test_dv_y_cmp_model(cfg, dv_y_cfg=None):
    if dv_y_cfg is None: dv_y_cfg = dv_y_cmp_configuration(cfg)
    class_test_dv_y_cmp_model(cfg, dv_y_cfg)

def gen_dv_y_cmp_model(cfg, dv_y_cfg=None):
    if dv_y_cfg is None: dv_y_cfg = dv_y_cmp_configuration(cfg)
    gen_dv_y_model(cfg, dv_y_cfg)#, dv_y_model_class=dv_y_cmp_model, make_feed_dict_method=make_feed_dict_y_cmp)