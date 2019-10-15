# exp_dv_cmp_pytorch.py

# This file uses dv_cmp experiments to slowly progress with pytorch

import os, sys, pickle, time, shutil, logging, copy
import math, numpy, scipy
numpy.random.seed(545)
from modules import make_logger, read_file_list, prepare_file_path, prepare_file_path_list, make_held_out_file_number, copy_to_scratch
from modules import keep_by_speaker, remove_by_speaker, keep_by_file_number, remove_by_file_number, keep_by_min_max_file_number, check_and_change_to_list
from modules_2 import compute_feat_dim, log_class_attri, resil_nn_file_list, norm_nn_file_list, get_utters_from_binary_dict, count_male_female_class_errors

from io_funcs.binary_io import BinaryIOCollection
io_fun = BinaryIOCollection()



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

    def change_to_debug_mode(self):
        self.epoch_num_batch  = {'train': 10, 'valid':10}
        if '_smallbatch' not in self.exp_dir:
            self.exp_dir = self.exp_dir + '_smallbatch'
        self.num_train_epoch = 5
        # self.num_speaker_dict['train'] = 10
        # self.speaker_id_list_dict['train'] = self.speaker_id_list_dict['train'][:self.num_speaker_dict['train']]

    def change_to_test_mode(self):
        self.epoch_num_batch  = {'train': 0, 'valid':4000}
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


class dv_y_cmp_configuration(dv_y_configuration):
    """docstring for ClassName"""
    def __init__(self, cfg):
        super(dv_y_cmp_configuration, self).__init__(cfg)
        self.train_by_window  = True # Optimise lambda_w; False: optimise speaker level lambda
        self.batch_output_form = 'mean' # Method to convert from SBD to SD
        self.retrain_model = False
        self.previous_model_name = ''
        self.python_script_name = '/home/dawna/tts/mw545/tools/merlin/merlin_cued_mw545_pytorch/debug_nausicaa/exp_dv_cmp_pytorch.py'
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

        # self.dv_y_model_class = dv_y_cmp_model
        # self.make_feed_dict_method = make_feed_dict_y_cmp

        self.auto_complete(cfg)


import torch
torch.manual_seed(545)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ReLUDVMax(nn.Module):
    def __init__(self, input_dim, output_dim, num_channels):
        super().__init__()
        self.input_dim    = input_dim
        self.output_dim   = output_dim
        self.num_channels = num_channels

        self.fc_list = nn.ModuleList([nn.Linear(input_dim, output_dim) for i in range(self.num_channels)])
        self.relu_fn = nn.ReLU()

    def forward(self, x):
        h_list = []
        for i in range(self.num_channels):
            # Linear
            h_i = self.fc_list[i](x)
            # ReLU
            h_i = self.relu_fn(h_i)
            h_list.append(h_i)

        h_stack = torch.stack(h_list, dim=0)
        # MaxOut
        h_max, _indices = torch.max(h_stack, dim=0, keepdim=False)
        return h_max

class DV_Y_CMP_NN_model(nn.Module):
    def __init__(self, dv_y_cfg):
        super().__init__()
        self.input_dim = dv_y_cfg.batch_seq_len * dv_y_cfg.feat_dim
        prev_output_dim = self.input_dim
        self.num_nn_layers = dv_y_cfg.num_nn_layers
        self.train_by_window = dv_y_cfg.train_by_window

        # Hidden layers
        # The last is bottleneck, output_dim is lambda_dim
        self.layer_list = nn.ModuleList()
        for i in range(self.num_nn_layers):
            layer_config = dv_y_cfg.nn_layer_config_list[i]
            layer_type = layer_config['type']
            input_dim  = prev_output_dim
            output_dim = layer_config['size']
            prev_output_dim = output_dim
            if layer_type == 'ReLUDVMax':
                num_channels = layer_config['num_channels']
                layer_temp = ReLUDVMax(input_dim, output_dim, num_channels)
            elif layer_type == 'LinDV':
                layer_temp = nn.Linear(input_dim, output_dim)
            self.layer_list.append(layer_temp)

        # Expansion layer, from lambda to logit
        input_dim  = prev_output_dim
        self.output_dim = dv_y_cfg.num_speaker_dict['train']
        self.expansion_layer = nn.Linear(input_dim, self.output_dim)

    def gen_lambda_SBD(self, x):
        for i in range(self.num_nn_layers):
            layer_temp = self.layer_list[i]
            x = layer_temp(x)
        return x

    def gen_logit_SBD(self, x):
        lambda_SBD = self.gen_lambda_SBD(x)
        logit_SBD  = self.expansion_layer(lambda_SBD)
        return logit_SBD

    def forward(self, x):
        logit_SBD = self.gen_logit_SBD(x)
        if self.train_by_window:
            # Flatten to 2D for cross-entropy
            logit_SB_D = logit_SBD.view(-1, self.output_dim)
            return logit_SB_D
        else:
            # Average over B
            logit_S_D = torch.mean(logit_SBD, dim=1, keepdim=False)

class DV_Y_CMP_model(object):
    def __init__(self, dv_y_cfg):
        self.nn_model = DV_Y_CMP_NN_model(dv_y_cfg)

    def build_optimiser(self):
        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        self.optimizer = torch.optim.Adam(self.nn_model.parameters(), lr=1e-4)
        # Zero gradients
        self.optimizer.zero_grad()

    def gen_loss(self, feed_dict):
        x, y = self.numpy_to_tensor(feed_dict)
        y_pred = self.nn_model(x)
        # Compute and print loss
        self.loss = self.criterion(y_pred, y)
        return self.loss

    def cal_accuracy(self, feed_dict):
        x, y = self.numpy_to_tensor(feed_dict)
        outputs = self.nn_model(x)
        _, predicted = torch.max(outputs.data, 1)
        total = y.size(0)
        correct = (predicted == y).sum().item()
        accuracy = correct/total
        return correct, total, accuracy

    def numpy_to_tensor(self, feed_dict):
        x_val = feed_dict['x']
        y_val = feed_dict['y']
        x = torch.tensor(x_val, dtype=torch.float)
        y = torch.tensor(y_val, dtype=torch.long)
        x = x.to(self.device_id)
        y = y.to(self.device_id)
        return (x, y)




    

    def __call__(self, x):
        ''' Simulate PyTorch forward() method '''
        ''' Note that x could be feed_dict '''
        return self.nn_model(x)

    def eval(self):
        ''' Simulate PyTorch eval() method '''
        self.nn_model.eval()

    def train(self):
        ''' Simulate PyTorch train() method '''
        self.nn_model.train()

    def to_device(self, device_id):
        self.device_id = device_id
        self.nn_model.to(device_id)

    def print_model_parameters(self, logger):
        logger.info('Print Parameter Sizes')
        for name, param in self.nn_model.named_parameters():
            print(str(name)+'  '+str(param.size())+'  '+str(param.type()))

    def update_parameters(self, feed_dict):
        self.loss = self.gen_loss(feed_dict)
        # perform a backward pass, and update the weights.
        self.loss.backward()
        self.optimizer.step()

    def gen_loss_value(self, feed_dict):
        self.loss = self.gen_loss(feed_dict)
        return self.loss.item()

    def save_nn_model(self, nnets_file_name):
        save_dict = {'model_state_dict': self.nn_model.state_dict()}
        torch.save(save_dict, nnets_file_name)

    def load_nn_model(self, nnets_file_name):
        checkpoint = torch.load(nnets_file_name)
        self.nn_model.load_state_dict(checkpoint['model_state_dict'])

    def save_nn_model_optim(self, nnets_file_name):
        save_dict = {'model_state_dict': self.nn_model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict()}
        torch.save(save_dict, nnets_file_name)

    def load_nn_model_optim(self, nnets_file_name):
        checkpoint = torch.load(nnets_file_name)
        self.nn_model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])



def torch_initialisation(dv_y_cfg):
    logger = make_logger("torch initialisation")
    if torch.cuda.is_available():
    # if False:
        logger.info('Using GPU cuda:%i' % dv_y_cfg.gpu_id)
        device_id = torch.device("cuda:%i" % dv_y_cfg.gpu_id)
    else:
        logger.info('Using CPU; No GPU')
        device_id = torch.device("cpu")

    # Construct our model by instantiating the class defined above
    model = DV_Y_CMP_model(dv_y_cfg)
    model.to_device(device_id)
    return model

def train_dv_y_cmp_model(cfg, dv_y_cfg=None):

    # First attempt: 3 layers of ReLUMax
    # Also, feed data use feed_dict style

    if dv_y_cfg is None: dv_y_cfg = dv_y_cmp_configuration(cfg)

    logger = make_logger("dv_y_config")
    log_class_attri(dv_y_cfg, logger, except_list=dv_y_cfg.log_except_list)

    logger = make_logger("train_dv_y_model")
    logger.info('Creating data lists')
    speaker_id_list = dv_y_cfg.speaker_id_list_dict['train'] # For DV training and evaluation, use train speakers only
    file_id_list    = read_file_list(cfg.file_id_list_file)
    file_list_dict  = make_dv_file_list(file_id_list, speaker_id_list, dv_y_cfg.data_split_file_number) # In the form of: file_list[(speaker_id, 'train')]

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
            batch_speaker_list = numpy.random.choice(speaker_id_list, dv_y_cfg.batch_num_spk)
            # Make feed_dict for training
            feed_dict, batch_size = make_feed_dict_y_cmp(dv_y_cfg, file_list_dict, cfg.nn_feat_scratch_dirs, dv_y_model, batch_speaker_list,  utter_tvt='train')
            dv_y_model.nn_model.train()
            dv_y_model.update_parameters(feed_dict=feed_dict)
        epoch_train_time = time.time()

        logger.info('start evaluating Epoch '+str(epoch))
        output_string_1 = 'epoch '+str(epoch)
        output_string_2 = 'epoch '+str(epoch)
        for utter_tvt_name in ['train', 'valid', 'test']:
            total_batch_size = 0.
            total_loss       = 0.
            total_accuracy   = 0.
            for batch_idx in range(dv_y_cfg.epoch_num_batch['valid']):
                # Draw random speakers
                batch_speaker_list = numpy.random.choice(speaker_id_list, dv_y_cfg.batch_num_spk)
                # Make feed_dict for evaluation
                feed_dict, batch_size = make_feed_dict_y_cmp(dv_y_cfg, file_list_dict, cfg.nn_feat_scratch_dirs, dv_y_model, batch_speaker_list, utter_tvt=utter_tvt_name)
                dv_y_model.eval()
                batch_mean_loss = dv_y_model.gen_loss_value(feed_dict=feed_dict)
                _c, _t, accuracy = dv_y_model.cal_accuracy(feed_dict=feed_dict)
                total_batch_size += batch_size
                total_loss       += batch_mean_loss
                total_accuracy   += accuracy
            average_loss = total_loss/float(dv_y_cfg.epoch_num_batch['valid'])
            average_accu = total_accuracy/float(dv_y_cfg.epoch_num_batch['valid'])
            output_string_1 = output_string_1 + '; '+utter_tvt_name+' loss '+str(average_loss)
            output_string_2 = output_string_2 + '; '+utter_tvt_name+' accuracy '+str(average_accu)

            if utter_tvt_name == 'valid':
                nnets_file_name = dv_y_cfg.nnets_file_name
                # Compare validation error
                valid_error = average_loss
                if valid_error < best_valid_loss:
                    early_stop = 0
                    logger.info('valid error reduced, saving model, '+nnets_file_name)
                    dv_y_model.save_nn_model_optim(nnets_file_name)
                    best_valid_loss = valid_error
                elif valid_error > previous_valid_loss:
                    early_stop = early_stop + 1
                    new_learning_rate = dv_y_model.learning_rate*0.5
                    logger.info('reduce learning rate to '+str(new_learning_rate))
                    dv_y_model.update_learning_rate(new_learning_rate)
                if (early_stop > early_stop_epoch) and (epoch > dv_y_cfg.warmup_epoch):
                    early_stop = 0
                    num_decay = num_decay + 1
                    if num_decay > max_num_decay:
                        logger.info('stopping early, best model, '+nnets_file_name+', best valid error '+str(best_valid_loss))
                        return best_valid_loss
                    logger.info('loading previous best model, '+nnets_file_name)
                    dv_y_model.load_nn_model_optim(nnets_file_name)
                    # logger.info('reduce learning rate to '+str(new_learning_rate))
                    # dv_y_model.update_learning_rate(new_learning_rate)
                previous_valid_loss = valid_error

        epoch_valid_time = time.time()
        output_string_3 = 'epoch '+str(epoch) + '; train time is %.2f, valid time is  %.2f' %((epoch_train_time - epoch_start_time), (epoch_valid_time - epoch_train_time))
        logger.info(output_string_1)
        logger.info(output_string_2)
        logger.info(output_string_3)

    return best_valid_loss




def make_feed_dict_y_cmp(dv_y_cfg, file_list_dict, file_dir_dict, dv_y_model, batch_speaker_list, utter_tvt, return_dv=False, return_y=False, return_frame_index=False, return_file_name=False):
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



def data_format_test(dv_y_cfg, dv_y_model):
    logger = make_logger("data_format_test")
    S = dv_y_cfg.batch_num_spk
    B = dv_y_cfg.spk_num_seq
    T = dv_y_cfg.batch_seq_len
    D = dv_y_cfg.feat_dim
    D_in  = T * D
    D_out = dv_y_cfg.num_speaker_dict['train']
    
    # Create random Tensors to hold inputs and outputs
    x_val = numpy.random.rand(S,B,D_in)
    y_val = numpy.ones(S*B)
    x = torch.tensor(x_val, dtype=torch.float)
    y = torch.tensor(y_val, dtype=torch.long)

    feed_dict = {'x':x_val, 'y':y_val}
    
    for t in range(1,501):
        dv_y_model.nn_model.train()
        dv_y_model.update_parameters(feed_dict)
        if t % 100 == 0:
            dv_y_model.nn_model.eval()
            loss = dv_y_model.gen_loss(feed_dict)
            logger.info('%i, %f' % (t, loss.item()))
        

