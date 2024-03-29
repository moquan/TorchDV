# exp_dv_config.py

import os, sys, pickle, time, shutil, logging, copy
import math, numpy, scipy

from frontend_mw545.modules import make_logger, prepare_script_file_path, check_and_change_to_list

class dv_configuration_base(object):
    """docstring for dv_configuration_base"""
    def __init__(self, cfg):
        super(dv_configuration_base, self).__init__()
        self.cfg = cfg
        self.log_except_list = ['cfg', 'feat_index']
        self.run_mode = cfg.run_mode
        
    def init_dir(self):
        # Things to be filled
        self.python_script_name = os.path.realpath(__file__)
        self.make_feed_dict_method_train = None
        self.make_feed_dict_method_test  = None
        self.make_feed_dict_method_gen   = None
        self.nn_layer_config_list = None
        self.exp_dir = None
        self.prev_nnets_file_name = None

    def init_data(self):
        self.data_dir_mode = 'scratch'
        self.input_data_dim = {}
        self.input_data_dim['S'] = 4 # S
        self.data_loader_random_seed = 0

    def init_model(self):
        self.nn_layer_config_list = []

        self.train_by_window = True      # Optimise lambda_w; False: optimise speaker level lambda
        self.train_num_seconds = 0       # Number of seconds per speaker during training; if 0, use 1 file per speaker
        self.classify_in_training = True # Compute classification accuracy after validation errors during training
        # self.batch_output_form = 'mean'  # Method to convert from SBD to SD
        self.use_voiced_only = False     # Use voiced regions only
        self.dv_dim = 512

        try: 
            self.gpu_id 
        except AttributeError: 
            self.gpu_id = 0
        # self.gpu_per_process_gpu_memory_fraction = 0.8

    def init_train(self):
        # Things no need to change
        self.learning_rate    = 0.0001
        self.feed_per_update  = 1    # Number of feed_dict per batch; step() only once per batch
        self.num_train_epoch  = 1000
        self.warmup_epoch     = 10
        self.early_stop_epoch = 2    # After this number of non-improvement, roll-back to best previous model and decay learning rate
        self.max_num_decay    = 10
        self.epoch_num_batch  = {'train': 13000, 'valid':2000}
        
        self.data_split_file_number = {}
        self.data_split_file_number['train'] = [120, 3000]
        self.data_split_file_number['valid'] = [81, 120]
        self.data_split_file_number['test']  = [41, 80]

    def change_to_debug_mode(self, process=None):
        self.logger.info('Change to Debug Mode')
        for k in self.epoch_num_batch:
            self.epoch_num_batch[k] = 1
        self.exp_dir = "debug"
        self.num_train_epoch = 100
        self.warmup_epoch    = 1
        self.early_stop_epoch = 1

    def change_to_retrain_mode(self):
        # Mode for fine-tuning
        # 2 differences from train_normal
        # 1. Load previous model
        # 2. zero_grad after each epoch
        self.logger.info('Change to retrain Mode')
        self.run_mode = 'retrain'

    def auto_complete(self, cfg=None, cache_files=True):
        if cfg is None: cfg = self.cfg
        ''' Remember to call this after __init__ !!! '''
        # Features
        self.num_nn_layers = len(self.nn_layer_config_list)
        # Directories
        self.work_dir = cfg.work_dir
        if self.exp_dir is None:
            self.exp_dir = cfg.script_name
        self.model_dir = self.make_model_dir_name(cfg)

        if self.run_mode == 'debug': self.change_to_debug_mode()
        if self.retrain_model: self.change_to_retrain_mode()
        
        self.complete_model_dir = os.path.join(self.work_dir, "exps", self.exp_dir, self.model_dir)
        self.nnets_file_name = os.path.join(self.complete_model_dir, "Model")
        self.dv_file_name = os.path.join(self.complete_model_dir, "DV.dat")

        if cache_files:
            prepare_script_file_path(file_dir=self.complete_model_dir, script_name=self.python_script_name)
            prepare_script_file_path(file_dir=self.complete_model_dir, script_name=cfg.json_config_file)

    def additional_action_epoch(self, logger, dv_y_model):
        # Run every epoch, after train and eval; Add tests if necessary
        pass

    def make_model_dir_name(self, cfg):
        pass


class dv_y_configuration(dv_configuration_base):
    
    def __init__(self, cfg):
        super().__init__(cfg)
        self.logger = make_logger('dv_y_conf')

        self.init_dir()
        self.init_data()
        self.init_model()
        self.init_train()
        
    def init_wav_data(self):
        self.y_feat_name   = 'wav'
        self.out_feat_list = ['wav_SBT', 'f_SBM', 'tau_SBM', 'vuv_SBM'] # This is the order, if features are included
        
        # self.input_data_dim['T_S_max'] = int(10 * self.cfg.wav_sr) # This is the maximum input length
        self.input_data_dim['T_S_max'] = numpy.inf # This is the maximum input length
        self.input_data_dim['T_B'] = int(0.2 * self.cfg.wav_sr)
        self.input_data_dim['B_stride'] = int(0.005 * self.cfg.wav_sr)
        self.input_data_dim['D'] = self.input_data_dim['T_B']
        # self.input_data_dim['T_M'] = int(0.04 * self.cfg.wav_sr)
        # self.input_data_dim['M_stride'] = int(0.005 * self.cfg.wav_sr)

    def init_cmp_data(self):
        self.y_feat_name   = 'cmp'
        self.out_feat_list = ['mgc', 'lf0', 'bap']
        self.cmp_dim = self.cfg.nn_feature_dims['cmp']
        
        self.input_data_dim['T_S_max'] = numpy.inf # This is the maximum input length
        self.input_data_dim['T_B'] = int(0.2 * self.cfg.frame_sr) # T, 0.2s, 40 frames
        self.input_data_dim['B_stride'] = 1
        self.input_data_dim['D'] = self.cmp_dim * self.input_data_dim['T_B']

    def init_mfcc_data(self):
        self.y_feat_name   = 'mfcc'
        self.out_feat_list = ['mfcc']
        self.cmp_dim = 23
        
        self.input_data_dim['T_S_max'] = numpy.inf # This is the maximum input length of waveform
        self.input_data_dim['T_B'] = int(0.025 * self.cfg.wav_sr) # each mfcc frame
        self.input_data_dim['B_stride'] = int(0.005 * self.cfg.wav_sr) # each mfcc frame
        self.input_data_dim['D'] = self.cmp_dim

    def update_cmp_dim(self):
        '''
        Compute new acoustic feature dimension
        Based on the features in out_feat_list
        '''
        self.cmp_dim = 0
        for feat_name in self.out_feat_list:
            feat_dim = self.cfg.acoustic_in_dimension_dict[feat_name]
            self.cmp_dim += feat_dim
        self.input_data_dim['D'] = self.cmp_dim * self.input_data_dim['T_B']

    def update_wav_dim(self):
        '''
        Compute new acoustic feature dimension
        Based on the features in out_feat_list
        '''
        self.compute_M()
        self.cmp_dim = 0
        for feat_name in self.out_feat_list:
            if feat_name == 'wav_SBT':
                self.cmp_dim += self.input_data_dim['T_B']
            elif feat_name in ['f_SBM', 'tau_SBM', 'vuv_SBM']:
                self.cmp_dim += self.input_data_dim['M']
        self.input_data_dim['D'] = self.cmp_dim

    def update_mfcc_dim(self):
        '''
        '''
        self.input_data_dim['D'] = self.cmp_dim
        pass

    def compute_M(self):
        if 'T_M' in self.input_data_dim and 'M_stride' in self.input_data_dim:
            self.input_data_dim['M'] = int((self.input_data_dim['T_B'] - self.input_data_dim['T_M']) / self.input_data_dim['M_stride']) + 1

    def change_to_class_test_mode(self):
        self.epoch_num_batch = {'test':40}
        self.input_data_dim['S'] = 1
        spk_num_utter_list = [1,2,5,10]
        self.spk_num_utter_list = check_and_change_to_list(spk_num_utter_list)
        lambda_u_dict_file_name = 'lambda_u_class_test.dat'
        self.lambda_u_dict_file_name = os.path.join(self.exp_dir, lambda_u_dict_file_name)

        if 'debug' in self.work_dir: self.change_to_debug_mode(process="class_test")

    def change_to_distance_test_mode(self):
        if self.y_feat_name == 'cmp':
            self.max_len_to_plot = 10
            self.gap_len_to_plot = 1
        elif self.y_feat_name == 'wav':
            self.max_len_to_plot = 10*80
            self.gap_len_to_plot = 5

        self.epoch_num_batch = {'test':10*4}
        self.input_data_dim['S'] = int(self.input_data_dim['S'] / 4)
        self.num_to_plot = int(self.max_len_to_plot / self.gap_len_to_plot)

        if 'debug' in self.work_dir: self.change_to_debug_mode()

    def change_to_gen_h_mode(self):
        self.batch_speaker_list = ['p15', 'p28', 'p122', 'p68'] # Males 2, Females 2
        self.utter_name = '003'
        self.input_data_dim['S'] = len(self.batch_speaker_list)
        self.h_list_file_name = os.path.join(self.exp_dir, "h_spk_list.dat")
        self.file_list_dict = {(spk_id, 'gen'): [spk_id+'_'+self.utter_name] for spk_id in self.batch_speaker_list}

    def make_model_dir_name(self, cfg=None):
        if cfg is None: cfg = self.cfg
        dv_model_dir = 'dvy_%s_lr%.0E_fpu%i_' %(self.y_feat_name, self.learning_rate, self.feed_per_update)

        for nn_layer_config in self.nn_layer_config_list:
            if nn_layer_config['type'] == 'Tensor_Reshape':
                layer_str = ''
            else:
                layer_str = '%s' % (nn_layer_config['type'][:3])
                if 'size' in nn_layer_config:
                    layer_str = layer_str + '%i' % (nn_layer_config['size'])
                if 'inc_a' in nn_layer_config and nn_layer_config['inc_a']:
                    layer_str = layer_str + 'a'
                if 'batch_norm' in nn_layer_config and nn_layer_config['batch_norm']:
                    layer_str = layer_str + 'B'
                if 'dropout_p' in nn_layer_config and nn_layer_config['dropout_p'] > 0:
                    layer_str = layer_str + 'D'
                if 'num_freq' in nn_layer_config:
                    layer_str = layer_str + 'f' + str(nn_layer_config['num_freq'])
                if 'k_space' in nn_layer_config and nn_layer_config['k_space'] != 1:
                    layer_str = layer_str + 'ks' + str(nn_layer_config['k_space'])
                if 'layer_norm' in nn_layer_config and nn_layer_config['layer_norm']:
                    layer_str = layer_str + 'L'
                if 'k_train' in nn_layer_config and nn_layer_config['k_train']:
                    layer_str = layer_str + 'T'
                layer_str = layer_str + "_"
            dv_model_dir = dv_model_dir + layer_str

        if self.y_feat_name == 'wav':
            if 'wav_SBT' in self.out_feat_list:
                dv_model_dir = dv_model_dir + "DV%iS%iT%i" %(self.dv_dim, self.input_data_dim['S'], self.input_data_dim['T_B'])
                if 'T_M' in self.input_data_dim:
                    dv_model_dir = dv_model_dir + 'TM%i' % self.input_data_dim['T_M']
        elif self.y_feat_name == 'cmp':
            dv_model_dir = dv_model_dir + "DV%iS%iT%iD%i" %(self.dv_dim, self.input_data_dim['S'], self.input_data_dim['T_B'], self.input_data_dim['D'])
        if self.use_voiced_only:
            dv_model_dir = dv_model_dir + "_vO"+str(self.use_voiced_threshold)
        if not self.train_by_window:
            dv_model_dir = dv_model_dir + "_nTW"
            if self.train_num_seconds > 0:
                dv_model_dir = dv_model_dir + str(self.train_num_seconds) + 's'
        if self.data_loader_random_seed > 0:
            dv_model_dir = dv_model_dir  + '_seed' + str(self.data_loader_random_seed)
        
        self.model_dir = dv_model_dir
        return dv_model_dir


class dv_attention_configuration(dv_configuration_base):
    """
    Given the S*B*D output of dv_y_model, apply a S*B*1 attention to generate S*D output
    """
    def __init__(self, cfg, dv_y_cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.dv_y_cfg = dv_y_cfg
        self.logger = make_logger('dv_attn_conf')
        self.log_except_list = ['cfg', 'dv_y_cfg', 'feat_index']

        self.init_dir()
        self.init_data()
        self.init_model()
        self.init_train()

    def init_dir(self):
        super().init_dir()
        self.y_model_name = None
        self.attention_model_name = None
        self.load_y_model = False
        self.load_attention_model = False
        self.prev_y_model_name = ''
        self.prev_attention_model_name = ''

    def init_data(self):
        super().init_data()
        self.input_data_dim['S'] = self.dv_y_cfg.input_data_dim['S']
        # Get T_B and B_stride from dv_y_cfg; convert to frame_sr

    def init_lab_data(self):
        self.feat_name = 'lab'
        self.input_data_dim['S'] = self.dv_y_cfg.input_data_dim['S']
        self.input_data_dim['T_S_max'] = numpy.inf # This is the maximum input length
        self.input_data_dim['T_B'] = int(0.2 * self.cfg.frame_sr)
        self.label_index_list = [0,10,20,30,39]         #indices of labels within the frame, to stack and use
        self.label_index_list = numpy.array(self.label_index_list)
        self.lab_dim = self.cfg.nn_feature_dims['lab']
        self.input_data_dim['B_stride'] = 1
        self.update_lab_dim()

    def update_lab_dim(self):
        '''
        label_index_list may change, update accordingly
        '''
        self.input_data_dim['M']  = len(self.label_index_list)
        self.label_index_list = numpy.array(self.label_index_list)
        self.input_data_dim['D'] = self.lab_dim * self.input_data_dim['M']
        if self.dv_y_cfg.y_feat_name == 'wav':
            self.input_data_dim['T_B'] = self.dv_y_cfg.input_data_dim['T_B'] * self.cfg.frame_sr / self.cfg.wav_sr
            self.input_data_dim['B_stride'] = self.dv_y_cfg.input_data_dim['B_stride'] * self.cfg.frame_sr / self.cfg.wav_sr
        elif self.dv_y_cfg.y_feat_name == 'cmp':
            self.input_data_dim['T_B'] = self.dv_y_cfg.input_data_dim['T_B']
            self.input_data_dim['B_stride'] = self.dv_y_cfg.input_data_dim['B_stride']

    def init_model(self):
        super().init_model()
        self.train_by_window = False
        self.train_num_seconds = self.dv_y_cfg.train_num_seconds
        self.dv_dim = self.dv_y_cfg.dv_dim

        try: 
            self.gpu_id 
        except AttributeError: 
            self.gpu_id = 0

    def init_train(self):
        super().init_train()
        self.feed_per_update = self.dv_y_cfg.feed_per_update
        self.epoch_num_batch  = self.dv_y_cfg.epoch_num_batch

    def make_model_dir_name(self, cfg=None, dv_y_cfg=None):
        # exp_dir_name contains 3 parts:
        # work_dir, atten_model_dir, dv_y_model_dir
        if cfg is None: cfg = self.cfg
        if dv_y_cfg is None: dv_y_cfg = self.dv_y_cfg

        self.dv_y_model_dir = dv_y_cfg.dv_model_dir
        if self.load_y_model:
            self.dv_y_model_dir += '_LY'
        
        atten_model_dir = 'dva_%s_%s_lr%.0E_fpu%i_' %(self.feat_name, self.y_model_name, self.learning_rate, self.feed_per_update)
        for nn_layer_config in self.nn_layer_config_list:
            if nn_layer_config['type'] == 'Tensor_Reshape':
                layer_str = ''
            else:
                layer_str = '%s%i' % (nn_layer_config['type'][:3], nn_layer_config['size'])
                
                if 'batch_norm' in nn_layer_config and nn_layer_config['batch_norm']:
                    layer_str = layer_str + 'B'
                if 'layer_norm' in nn_layer_config and nn_layer_config['layer_norm']:
                    layer_str = layer_str + 'L'
                if 'dropout_p' in nn_layer_config and nn_layer_config['dropout_p'] > 0:
                    layer_str = layer_str + 'D'
                # if 'inc_a' in nn_layer_config and nn_layer_config['inc_a']:
                #     layer_str = layer_str + 'a'
                # if 'num_freq' in nn_layer_config:
                #     layer_str = layer_str + 'f' + str(nn_layer_config['num_freq'])
                # if 'k_space' in nn_layer_config and nn_layer_config['k_space'] != 1:
                #     layer_str = layer_str + 'ks' + str(nn_layer_config['k_space'])
                
                # if 'k_train' in nn_layer_config and nn_layer_config['k_train']:
                #     layer_str = layer_str + 'T'
                
                layer_str = layer_str + "_"
            atten_model_dir = atten_model_dir + layer_str

        if self.feat_name == 'lab':
            atten_model_dir = atten_model_dir + "S%iM%iD%i" %(self.input_data_dim['S'], self.input_data_dim['M'], self.input_data_dim['D'])

        if self.data_loader_random_seed > 0:
            atten_model_dir = atten_model_dir  + '_seed' + str(self.data_loader_random_seed)

        self.atten_model_dir = atten_model_dir
        model_dir = os.path.join(self.dv_y_model_dir, self.atten_model_dir)
        return model_dir


    def auto_complete(self, cfg=None, cache_files=True):
        if cfg is None: cfg = self.cfg
        ''' Remember to call this after __init__ !!! '''
        # Features
        self.dv_y_cfg.gpu_id = self.gpu_id
        self.dv_y_cfg.auto_complete(cfg, cache_files=False)
        super().auto_complete(cfg, cache_files)

