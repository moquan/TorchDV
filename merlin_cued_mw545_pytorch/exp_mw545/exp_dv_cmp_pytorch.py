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
        # Flatten to 2D for cross-entropy
        logit_SB_D = logit_SBD.view(-1, self.output_dim)
        return logit_SB_D

class DV_Y_CMP_model(object):
    def __init__(self, dv_y_cfg):
        self.nn_model = DV_Y_CMP_NN_model(dv_y_cfg)

    def __call__(self, x):
        ''' Simulate PyTorch forward() method '''
        ''' Note that x could be feed_dict '''
        return self.nn_model(x)

    def to_device(self, device_id):
        self.device_id = device_id
        self.nn_model.to(device_id)

    def print_model_parameters(self, logger):
        logger.info('Print Parameter Sizes')
        for name, param in self.nn_model.named_parameters():
            print(str(name)+'  '+str(param.size()))

    def build_optimiser(self):
        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        self.optimizer = torch.optim.Adam(self.nn_model.parameters(), lr=1e-4)
        # Zero gradients
        self.optimizer.zero_grad()

    def update_parameters(self, x, y):
        y_pred = self.nn_model(x)
        # Compute and print loss
        self.loss = self.criterion(y_pred, y)
        # perform a backward pass, and update the weights.
        self.loss.backward()
        self.optimizer.step()


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

    model = torch_initialisation(dv_y_cfg)
    model.build_optimiser()
    model.print_model_parameters(logger)
    
    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    S = dv_y_cfg.batch_num_spk
    B = dv_y_cfg.spk_num_seq
    T = dv_y_cfg.batch_seq_len
    D = dv_y_cfg.feat_dim
    D_in  = T * D
    D_out = dv_y_cfg.num_speaker_dict['train']
    

    # Create random Tensors to hold inputs and outputs
    x = torch.randn(S,B,D_in)
    y = torch.ones(S*B, dtype=torch.long)
    x = x.to(model.device_id)
    y = y.to(model.device_id)

    # Construct our loss function and an Optimizer. The call to model.parameters()
    # in the SGD constructor will contain the learnable parameters of the two
    # nn.Linear modules which are members of the model.
    
    for t in range(1,501):
        # Forward pass: Compute predicted y by passing x to the model
        model.update_parameters(x, y)
        if t % 100 == 0:
            logger.info('%i, %f' % (t, model.loss.item()))
        

