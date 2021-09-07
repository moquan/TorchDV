# exp_dv_config.py

import os, sys, pickle, time, shutil, logging, copy
import math, numpy, scipy

from frontend_mw545.modules import make_logger, prepare_script_file_path
# from nn_torch.torch_tests import return_gpu_memory


class dv_y_configuration(object):
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.logger = make_logger('dv_y_conf')
        self.log_except_list = ['cfg', 'feat_index']
        self.run_mode = 'normal'

        self.init_dir()
        self.init_data()
        self.init_model()
        self.init_train()
        
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
        if 'wav' in self.cfg.work_dir:
            self.init_wav_data()
        elif 'cmp' in self.cfg.work_dir:
            self.init_cmp_data()
        else:
            self.init_wav_data()

    def init_wav_data(self):
        self.y_feat_name   = 'wav'
        self.out_feat_list = ['wav_ST', 'f_SBM', 'tau_SBM', 'vuv_SBM']
        
        self.input_data_dim['T_S'] = int(self.cfg.wav_sr) # use 1s per utterance
        self.input_data_dim['T_B'] = int(0.2 * self.cfg.wav_sr)
        self.input_data_dim['B_shift'] = int(0.005 * self.cfg.wav_sr)
        self.input_data_dim['T_M'] = int(0.04 * self.cfg.wav_sr)
        self.input_data_dim['M_shift'] = int(0.005 * self.cfg.wav_sr)

    def init_cmp_data(self):
        self.y_feat_name   = 'cmp'
        self.out_feat_list = ['mgc', 'lf0', 'bap']
        self.cmp_dim = self.cfg.nn_feature_dims['cmp']
        
        self.input_data_dim['T_S'] = numpy.inf # This is the maximum input length
        self.input_data_dim['T_B'] = int(0.2 * self.cfg.frame_sr) # T
        self.input_data_dim['B_shift'] = 1
        self.input_data_dim['D'] = self.cfg.nn_feature_dims['cmp']

    def init_model(self):
        self.nn_layer_config_list = []

        self.train_by_window = True      # Optimise lambda_w; False: optimise speaker level lambda
        self.classify_in_training = True # Compute classification accuracy after validation errors during training
        self.batch_output_form = 'mean'  # Method to convert from SBD to SD
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

    def compute_B_M(self):
        if 'T_M' in self.input_data_dim and 'M_shift' in self.input_data_dim:
            self.input_data_dim['M'] = int((self.input_data_dim['T_B'] - self.input_data_dim['T_M']) / self.input_data_dim['M_shift']) + 1

    def auto_complete(self, cfg=None, cache_files=True):
        if cfg is None: cfg = self.cfg
        ''' Remember to call this after __init__ !!! '''
        # Features
        self.compute_B_M() # B
        self.num_nn_layers = len(self.nn_layer_config_list)

        # Directories
        self.work_dir = cfg.work_dir
        if self.exp_dir is None:
            self.exp_dir  = self.make_dv_y_exp_dir_name(cfg)
        if 'debug' in self.work_dir: self.change_to_debug_mode()
        if self.prev_nnets_file_name is not None: self.change_to_retrain_mode()
        self.nnets_file_name = os.path.join(self.exp_dir, "Model")
        self.dv_file_name = os.path.join(self.exp_dir, "DV.dat")
        if cache_files:
            prepare_script_file_path(file_dir=self.exp_dir, script_name=cfg.python_script_name)
            prepare_script_file_path(file_dir=self.exp_dir, script_name=self.python_script_name)

    def change_to_debug_mode(self, process=None):
        self.logger.info('Change to Debug Mode')
        self.run_mode = 'debug'
        # self.input_data_dim['S'] = 1
        for k in self.epoch_num_batch:
            self.epoch_num_batch[k] = 1
        if '_smallbatch' not in self.exp_dir:
            self.exp_dir = self.exp_dir + '_smallbatch'
        self.num_train_epoch = 100
        self.warmup_epoch    = 1
        self.early_stop_epoch = 1

    def change_to_retrain_mode(self):
        self.logger.info('Change to retrain Mode')
        self.run_mode = 'retrain'

    def additional_action_epoch(self, logger, dv_y_model):
        # Run every epoch, after train and eval; Add tests if necessary
        pass




    def update_cmp_dim(self):
        '''
        Compute new acoustic feature dimension
        Based on the features in out_feat_list
        '''
        self.cmp_dim = 0
        for feat_name in self.out_feat_list:
            feat_dim = self.cfg.acoustic_in_dimension_dict[feat_name]
            self.cmp_dim += feat_dim
        self.input_data_dim['D'] = self.input_data_dim['T_B'] * self.cmp_dim

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

    def make_dv_y_exp_dir_name(self, cfg):
        exp_dir = cfg.work_dir + '/dvy_%s_lr%.0E_fpu%i_' %(self.y_feat_name, self.learning_rate, self.feed_per_update)
        for nn_layer_config in self.nn_layer_config_list:
            if nn_layer_config['type'] == 'Tensor_Reshape':
                layer_str = ''
            else:
                layer_str = '%s%i' % (nn_layer_config['type'][:3], nn_layer_config['size'])
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
            exp_dir = exp_dir + layer_str
        if self.y_feat_name == 'wav':
            exp_dir = exp_dir + "DV%iS%iT%i" %(self.dv_dim, self.input_data_dim['S'], self.input_data_dim['T_M'])
        elif self.y_feat_name == 'cmp':
            exp_dir = exp_dir + "DV%iS%iT%iD%i" %(self.dv_dim, self.input_data_dim['S'], self.input_data_dim['T_B'], self.input_data_dim['D'])
        if self.use_voiced_only:
            exp_dir = exp_dir + "_vO"+str(self.use_voiced_threshold)
        return exp_dir