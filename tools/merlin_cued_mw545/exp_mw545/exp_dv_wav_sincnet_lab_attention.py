# exp_dv_wav_sincnet_lab_attention.py

# d-vector style model
# https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/41939.pdf

# For each window, network input is a vector of stacked vocoder feature vectors

import os
from exp_mw545.exp_dv_config import dv_y_configuration, dv_attention_configuration

class dv_wav_sincnet_lab_attention_configuration(dv_attention_configuration):
    """docstring for dv_cmp_attention_configuration"""
    def __init__(self, cfg, dv_y_cfg, cache_files=True):
        super().__init__(cfg, dv_y_cfg)
        self.retrain_model = False
        self.learning_rate  = 0.00001
        # self.prev_nnets_file_name = '/home/dawna/tts/mw545/TorchDV/dv_cmp_baseline/dvy_cmp_lr1E-04_fpu40_LRe512L_LRe512L_Lin512L_DV512S10T40D3440/Model'
        self.python_script_name = os.path.realpath(__file__)
        self.data_dir_mode = 'scratch' # Use scratch for speed up
        self.data_loader_random_seed = 0

        self.load_y_model = True
        self.y_model_name = 'sincnet_frame'
        # self.prev_y_model_name = '/home/dawna/tts/mw545/TorchDV/dv_wav_sincnet/dvy_wav_lr1E-04_fpu40_Sin60_LRe512L_LRe512L_Lin512L_DV512S10T3000/Model'
        self.prev_y_model_name = '/data/vectra2/tts/mw545/TorchDV/dv_wav_sincnet/dvy_wav_lr1E-04_fpu10_Sin80_LRe512L_LRe512L_Lin512L_DV512S10T3000/Model'
        # self.y_model_name = 'sincnet'
        # self.prev_y_model_name = '/home/dawna/tts/mw545/TorchDV/dv_wav_sincnet/dvy_wav_lr1E-04_fpu40_Sin60_LRe512L_LRe512L_Lin512L_DV512S10T3000_nTW5s/Model'

        self.init_lab_data()
        self.feat_name = 'lab'
        self.label_index_list = [0,6,12,18,24]
        self.update_lab_dim()

        self.nn_layer_config_list = [
            {'type':'LReLU', 'size':256, 'dropout_p':0, 'layer_norm':True},
            {'type':'LReLU', 'size':16, 'dropout_p':0, 'layer_norm':True},
            {'type':'LReLU', 'size':1, 'dropout_p':0, 'layer_norm':True},
        ]

        # self.gpu_id = 'cpu'
        self.gpu_id = 0
        self.auto_complete(cfg, cache_files)


class dv_y_wav_sincnet_configuration(dv_y_configuration):
    def __init__(self, cfg, cache_files=False):
        super().__init__(cfg)

        self.retrain_model = False
        self.learning_rate  = 0.0001
        # self.prev_nnets_file_name = '/home/dawna/tts/mw545/TorchDV/dv_wav_sincnet/dvy_wav_lr1E-04_fpu40_Sin60_LRe512L_LRe512L_Lin512L_DV512S10T3000/Model'
        self.python_script_name = os.path.realpath(__file__)
        self.data_dir_mode = 'scratch' # Use scratch for speed up
        self.train_by_window = False
        self.train_num_seconds = 5
        self.data_loader_random_seed = 0

        # Waveform-level input configuration
        self.init_wav_data()
        self.y_feat_name   = 'wav'
        self.out_feat_list = ['wav_SBT']
        self.input_data_dim['T_B'] = int(0.125 * self.cfg.wav_sr)
        self.input_data_dim['B_stride'] = int( self.cfg.wav_sr/8)
        self.update_wav_dim()

        self.input_data_dim['S'] = 10
        self.feed_per_update = 40
        S_per_update = self.input_data_dim['S'] * self.feed_per_update
        self.epoch_num_batch  = {'train': int(52000/S_per_update), 'valid': int(8000/self.input_data_dim['S'])}

        self.dv_dim = 512
        self.nn_layer_config_list = [
            {'type': 'Tensor_Reshape', 'io_name': 'h_SBD_2_wav_SBT'},
            {'type': 'Tensor_Reshape', 'io_name': 'wav_SBT_2_wav_SB_T'},
            {'type': 'SincNet', 'size':80, 'dropout_p':0, 'batch_norm':False},
            {'type': 'Tensor_Reshape', 'io_name': 'h_SB_D_2_h_SBD'},     # h_SBD
            {'type': 'LReLU', 'size':256*2, 'dropout_p':0, 'layer_norm':True},
            {'type': 'LReLU', 'size':256*2, 'dropout_p':0, 'layer_norm':True},
            {'type': 'Linear', 'size':self.dv_dim, 'dropout_p':0, 'layer_norm':True}
        ]

        # self.gpu_id = 'cpu'
        self.gpu_id = 0
        self.auto_complete(cfg, cache_files)

def train_model(cfg, dv_attn_cfg=None, dv_y_cfg=None):
    if dv_attn_cfg is None:
        if dv_y_cfg is None: 
            dv_y_cfg = dv_y_wav_sincnet_configuration(cfg, cache_files=False)
        dv_attn_cfg = dv_wav_sincnet_lab_attention_configuration(cfg, dv_y_cfg)

    from exp_mw545.exp_dv_attention import Build_DV_Attention_Model_Trainer
    dv_model_trainer = Build_DV_Attention_Model_Trainer(cfg, dv_attn_cfg)
    dv_model_trainer.train()

def test_model(cfg, dv_attn_cfg=None, dv_y_cfg=None):
    if dv_attn_cfg is None:
        if dv_y_cfg is None: 
            dv_y_cfg = dv_y_wav_sincnet_configuration(cfg, cache_files=False)
            dv_y_cfg.input_data_dim['B_stride'] = int( dv_y_cfg.cfg.wav_sr/200)   # Change stride at inference time
            dv_y_cfg.update_wav_dim()
        dv_attn_cfg = dv_wav_sincnet_lab_attention_configuration(cfg, dv_y_cfg)

    from exp_mw545.exp_dv_attention import Build_DV_Attention_Testing
    dv_model_test = Build_DV_Attention_Testing(cfg, dv_attn_cfg)


    # fig_file_name = '/home/dawna/tts/mw545/Export_Temp/PNG_out/vuv_loss_cmp.png'
    # dv_model_test.vuv_loss_test(fig_file_name)
    # fig_file_name = '/home/dawna/tts/mw545/Export_Temp/PNG_out/positional_cmp.png'
    # dv_model_test.positional_test(fig_file_name)

    # Additional output dir; also output to the exp dir
    # output_dir = '/home/dawna/tts/mw545/Export_Temp/PNG_out'
    output_dir = '/data/vectra2/tts/mw545/Export_Temp/PNG_out'
    # dv_model_test.gen_dv(output_dir)
    # dv_model_test.cross_entropy_accuracy_test()
    dv_model_test.number_secs_accu_test(output_dir,[40,45,50,55])


