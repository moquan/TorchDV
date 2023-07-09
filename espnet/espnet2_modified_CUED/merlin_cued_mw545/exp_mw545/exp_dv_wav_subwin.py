# exp_dv_wav_subwin.py

# d-vector style model
# https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/41939.pdf

# For each window, network input is a vector of stacked waveforms
# Then within each window, split into smaller windows (M)
# Stack this with Reaper predicted lf0 and pitch values
import os
from exp_mw545.exp_dv_config import dv_y_configuration

class dv_y_wav_subwin_configuration(dv_y_configuration):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.use_voiced_only = False   # Use voiced regions only
        self.use_voiced_threshold = 1. # Percentage of voiced required
        self.retrain_model = False
        self.learning_rate  = 0.0001
        # self.prev_nnets_file_name = ''
        self.python_script_name = os.path.realpath(__file__)
        # self.data_dir_mode = 'data' # Use scratch for speed up

        # Waveform-level input configuration
        self.y_feat_name   = 'wav'
        self.init_wav_data()
        # self.out_feat_list = ['wav_ST', 'f_SBM', 'tau_SBM', 'vuv_SBM']
        self.out_feat_list = ['wav_SBT', 'f_SBM', 'tau_SBM', 'vuv_SBM']

        self.input_data_dim['S'] = 1
        self.feed_per_update = 40
        S_per_update = self.input_data_dim['S'] * self.feed_per_update
        self.epoch_num_batch  = {'train': int(52000/S_per_update), 'valid': int(8000/self.input_data_dim['S'])}
        self.input_data_dim['T_M'] = 160
        self.input_data_dim['M_shift'] = 40
        

        self.dv_dim = 2048
        self.nn_layer_config_list = [
            # {'type': 'Tensor_Reshape', 'io_name': 'wav_ST_2_wav_SBMT', 'win_len_shift_list':[[self.input_data_dim['T_B'], self.input_data_dim['B_shift']], [self.input_data_dim['T_M'], self.input_data_dim['M_shift']]]},
            {'type': 'Tensor_Reshape', 'io_name': 'wav_SBT_2_wav_SBMT', 'win_len_shift':[self.input_data_dim['T_M'], self.input_data_dim['M_shift']]},
            {'type': 'DW3', 'size':80, 'dropout_p':0, 'layer_norm':True},
            {'type': 'Tensor_Reshape', 'io_name': 'h_SBMD_2_h_SBD'},     # h_SBD
            {'type': 'LReLU', 'size':256*8, 'dropout_p':0, 'layer_norm':True},
            {'type': 'LReLU', 'size':256*8, 'dropout_p':0, 'layer_norm':True},
            {'type': 'Linear', 'size':self.dv_dim, 'dropout_p':0, 'layer_norm':True}
        ]

        # self.gpu_id = 'cpu'
        self.gpu_id = 0
        self.auto_complete(cfg)

def train_model(cfg, dv_y_cfg=None):
    if dv_y_cfg is None: dv_y_cfg = dv_y_wav_subwin_configuration(cfg)
    
    from exp_mw545.exp_dv_y import Build_DV_Y_Model_Trainer
    dv_y_model_trainer = Build_DV_Y_Model_Trainer(cfg, dv_y_cfg)
    dv_y_model_trainer.train()

def test_model(cfg, dv_y_cfg=None):
    if dv_y_cfg is None: dv_y_cfg = dv_y_wav_subwin_configuration(cfg)

    from exp_mw545.exp_dv_y import Build_DV_Y_Testing
    dv_y_model_test = Build_DV_Y_Testing(cfg, dv_y_cfg)
    fig_file_name = '/home/dawna/tts/mw545/Export_Temp/PNG_out/vuv_loss_subwin.png'
    dv_y_model_test.vuv_loss_test(fig_file_name)
    fig_file_name = '/home/dawna/tts/mw545/Export_Temp/PNG_out/pos_subwin.png'
    dv_y_model_test.positional_test(fig_file_name)