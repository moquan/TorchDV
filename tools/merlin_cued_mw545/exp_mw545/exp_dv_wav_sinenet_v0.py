# exp_dv_wav_sinenet_v0.py

import os,numpy
from exp_mw545.exp_dv_config import dv_y_configuration

class dv_y_wav_sinenet_configuration(dv_y_configuration):
    def __init__(self, cfg, cache_files=True):
        super().__init__(cfg)

        self.retrain_model = False
        self.learning_rate  = 0.0001
        # self.prev_nnets_file_name = '/home/dawna/tts/mw545/TorchDV/dv_wav_sincnet/dvy_wav_lr1E-04_fpu40_Sin60_LRe512L_LRe512L_Lin512L_DV512S10T3000/Model'
        self.python_script_name = os.path.realpath(__file__)
        self.data_dir_mode = 'scratch' # Use scratch for speed up
        self.train_by_window = True
        self.train_num_seconds = 0
        self.data_loader_random_seed = 0

        # Waveform-level input configuration
        self.init_wav_data()
        self.y_feat_name   = 'wav'
        self.out_feat_list = ['wav_SBT', 'f_SBM', 'tau_SBM', 'vuv_SBM']
        self.input_data_dim['T_B'] = int(0.125 * self.cfg.wav_sr)
        # self.input_data_dim['T_B'] = int(0.2 * self.cfg.wav_sr)
        self.input_data_dim['B_stride'] = self.input_data_dim['T_B']
        self.input_data_dim['T_M'] = 240
        self.input_data_dim['M_stride'] = 240
        self.update_wav_dim()


        self.input_data_dim['S'] = 10
        self.feed_per_update = 40
        S_per_update = self.input_data_dim['S'] * self.feed_per_update
        self.epoch_num_batch  = {'train': int(52000/S_per_update), 'valid': int(8000/self.input_data_dim['S'])}

        self.dv_dim = 512

        self.nn_layer_config_list = [
            {'type': 'Tensor_Reshape', 'io_name': 'h_SBD_wav_feats_split', 'out_feat_list': self.out_feat_list, 'input_data_dim': self.input_data_dim},
            {'type': 'Tensor_Reshape', 'io_name': 'wav_SBT_2_wav_SBMT', 'win_len_shift':[self.input_data_dim['T_M'], self.input_data_dim['M_stride']]},
            {'type': 'Sinenet_V0', 'size':80, 'num_freq':64, 'k_space':0.5, 'dropout_p':0, 'use_f': 'D', 'use_tau': 'D', 'inc_a':True, 'k_train':True, 'batch_norm':False}, 
            {'type': 'Tensor_Reshape', 'io_name': 'h_SBMD_2_h_SBD'},     # h_SBD
            {'type': 'LReLU',  'size':256*2, 'dropout_p':0, 'layer_norm':True},
            {'type': 'LReLU',  'size':256*2, 'dropout_p':0, 'layer_norm':True},
            {'type': 'Linear', 'size':self.dv_dim, 'dropout_p':0, 'layer_norm':True}
        ]

        # self.gpu_id = 'cpu'
        self.gpu_id = 0

        self.auto_complete(cfg, cache_files)

    def additional_action_epoch(self, logger, dv_y_model):
        '''
        Print K value
        '''
        logger.info('Printing gamma_k values')
        k_2pi_tensor = dv_y_model.nn_model.layer_list[2].layer_fn.sinenet_fn.k_2pi_tensor
        k_value = k_2pi_tensor.cpu().detach().numpy()*(1/(2*numpy.pi))
        print(list(k_value))

        # logger.info('Printing tau_k values')
        # tau_tensor = dv_y_model.nn_model.layer_list[1].layer_fn.tau_tensor
        # tau_value = tau_tensor.cpu().detach().numpy()
        # print(list(tau_value))

def train_model(cfg, dv_y_cfg=None):
    if dv_y_cfg is None: dv_y_cfg = dv_y_wav_sinenet_configuration(cfg)

    from exp_mw545.exp_dv_y import Build_DV_Y_Model_Trainer
    dv_y_model_trainer = Build_DV_Y_Model_Trainer(cfg, dv_y_cfg)
    dv_y_model_trainer.train()
    # dv_y_model_trainer.train_no_validation()


def test_model(cfg, dv_y_cfg=None):
    if dv_y_cfg is None: dv_y_cfg = dv_y_wav_sinenet_configuration(cfg)
    dv_y_cfg.input_data_dim['B_stride'] = int( dv_y_cfg.cfg.wav_sr/200)   # Change stride at inference time
    dv_y_cfg.update_wav_dim()

    from exp_mw545.exp_dv_y import Build_DV_Y_Testing
    dv_y_model_test = Build_DV_Y_Testing(cfg, dv_y_cfg)
    # fig_file_name = '/home/dawna/tts/mw545/Export_Temp/PNG_out/vuv_loss_sinenet_v0.png'
    # dv_y_model_test.vuv_loss_test(fig_file_name)
    # fig_file_name = '/home/dawna/tts/mw545/Export_Temp/PNG_out/pos_sinenet_v0.png'
    # dv_y_model_test.positional_test(fig_file_name)

    # Additional output dir; also output to the exp dir
    output_dir = '/home/dawna/tts/mw545/Export_Temp/PNG_out'
    # dv_y_model_test.gen_dv(output_dir)
    # dv_y_model_test.cross_entropy_accuracy_test()
    # dv_y_model_test.number_secs_accu_test()
    dv_y_model_test.positional_test(output_dir)
    
