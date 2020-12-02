# exp_dv_wav_sincnet.py

# d-vector style model
# https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/41939.pdf

# For each window, network input is a vector of stacked waveforms
# Then within each window, split into smaller windows (M)
# Stack this with Reaper predicted lf0 and pitch values
import os
from exp_mw545.exp_dv_config import dv_y_configuration

class dv_y_wav_sincnet_configuration(dv_y_configuration):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.use_voiced_only = False   # Use voiced regions only
        self.use_voiced_threshold = 1. # Percentage of voiced required
        self.finetune_model = False
        self.learning_rate  = 0.00001
        # self.prev_nnets_file_name = ''
        self.python_script_name = os.path.realpath(__file__)
        self.data_dir_mode = 'scratch' # Use scratch for speed up

        # Waveform-level input configuration
        self.y_feat_name   = 'wav'
        self.init_wav_data()
        # self.out_feat_list = ['wav_SBT', 'f_SBM', 'tau_SBM', 'vuv_SBM']
        self.out_feat_list = ['wav_SBT']
        # self.input_data_dim['T_M'] = 640
        # self.input_data_dim['M_shift'] = 80
        

        self.dv_dim = 2048
        self.nn_layer_config_list = [
            {'type': 'Tensor_Reshape', 'io_name': 'wav_SBT_2_wav_SB_T'},
            {'type': 'SincNet', 'size':80, 'dropout_p':0, 'batch_norm':False},
            {'type': 'Tensor_Reshape', 'io_name': 'h_SB_D_2_h_SBD'},     # h_SBD
            {'type': 'LReLU', 'size':256*8, 'dropout_p':0, 'batch_norm':True},
            {'type': 'LReLU', 'size':256*8, 'dropout_p':0, 'batch_norm':True},
            {'type': 'Linear', 'size':self.dv_dim, 'dropout_p':0, 'batch_norm':True}
        ]

        # self.gpu_id = 'cpu'
        self.gpu_id = 0
        self.auto_complete(cfg)

def train_model(cfg, dv_y_cfg=None):
    if dv_y_cfg is None: dv_y_cfg = dv_y_wav_sincnet_configuration(cfg)
    
    from exp_mw545.exp_dv_y import Build_DV_Y_Model_Trainer
    dv_y_model_trainer = Build_DV_Y_Model_Trainer(cfg, dv_y_cfg)
    dv_y_model_trainer.train()
    # dv_y_model_trainer.train_overfit_train()

def test_model(cfg, dv_y_cfg=None):
    if dv_y_cfg is None: dv_y_cfg = dv_y_wav_sincnet_configuration(cfg)

    from exp_mw545.exp_dv_y import Build_DV_Y_Testing
    dv_y_model_test = Build_DV_Y_Testing(cfg, dv_y_cfg)
    fig_file_name = '/home/dawna/tts/mw545/Export_Temp/PNG_out/vuv_loss_sincnet_2.png'
    dv_y_model_test.vuv_loss_test(fig_file_name)