# exp_dv_wav_sinenet_v0.py

# d-vector style model
# https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/41939.pdf

# For each window, network input is a vector of stacked waveforms
# Then within each window, split into smaller windows (M)
# Reaper predicted lf0 and pitch values
import os,numpy
from exp_mw545.exp_dv_config import dv_y_configuration

# from frontend_mw545.data_loader import Build_dv_y_train_data_loader
class dv_y_wav_sinenet_configuration(dv_y_configuration):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.use_voiced_only = False   # Use voiced regions only
        self.use_voiced_threshold = 1. # Percentage of voiced required
        self.learning_rate  = 0.00001
        self.retrain_model = False
        # self.prev_nnets_file_name = '/home/dawna/tts/mw545/TorchDV/dv_wav_sinenet_v1/old/variable_k/dv_y_wav_lr_0.000010_Sin80f64_LRe2048B_LRe2048B_Lin2048B_DV2048S4B161M19T320/Model'
        self.python_script_name = os.path.realpath(__file__)
        self.data_dir_mode = 'data' # Use scratch for speed up

        # Waveform-level input configuration
        self.y_feat_name   = 'wav'
        self.init_wav_data()
        self.out_feat_list = ['wav_SBT', 'f_SBM', 'tau_SBM', 'vuv_SBM']

        self.input_data_dim['T_M'] = 160
        self.input_data_dim['M_shift'] = 40

        self.dv_dim = 2048

        self.nn_layer_config_list = [
            # {'type': 'Tensor_Reshape', 'io_name': 'wav_ST_2_wav_SBMT', 'win_len_shift_list':[[self.input_data_dim['T_B'], self.input_data_dim['B_shift']], [self.input_data_dim['T_M'], self.input_data_dim['M_shift']]]},
            {'type': 'Tensor_Reshape', 'io_name': 'wav_SBT_2_wav_SBMT', 'win_len_shift':[self.input_data_dim['T_M'], self.input_data_dim['M_shift']]},
            {'type': 'Sinenet_V0', 'size':80, 'num_freq':64, 'k_space':0.5, 'dropout_p':0, 'k_train':True, 'batch_norm':False}, # S_B_D output; D <- M*(D+1)
            {'type': 'Tensor_Reshape', 'io_name': 'h_SBMD_2_h_SBD'},     # h_SBD
            {'type': 'LReLU',  'size':256*8, 'dropout_p':0, 'batch_norm':True},
            {'type': 'LReLU',  'size':256*8, 'dropout_p':0, 'batch_norm':True},
            {'type': 'Linear', 'size':self.dv_dim, 'dropout_p':0, 'batch_norm':True}
        ]

        # self.gpu_id = 'cpu'
        self.gpu_id = 0

        self.auto_complete(cfg)

    def additional_action_epoch(self, logger, dv_y_model):
        '''
        Print K value
        '''
        sinenet_config = self.nn_layer_config_list[1]
        if sinenet_config['k_train']:
            logger.info('Printing gamma_k values')
            k_vec = dv_y_model.nn_model.layer_list[1].layer_fn.sinenet_fn.k_2pi_tensor
            k_vec_value = k_vec.cpu().detach().numpy()*(1/(2*numpy.pi))
            print(list(k_vec_value))

def train_model(cfg, dv_y_cfg=None):
    if dv_y_cfg is None: dv_y_cfg = dv_y_wav_sinenet_configuration(cfg)

    from exp_mw545.exp_dv_y import Build_DV_Y_Model_Trainer
    dv_y_model_trainer = Build_DV_Y_Model_Trainer(cfg, dv_y_cfg)
    dv_y_model_trainer.train()
    # dv_y_model_trainer.train_no_validation()

def test_model(cfg, dv_y_cfg=None):
    if dv_y_cfg is None: dv_y_cfg = dv_y_wav_sinenet_configuration(cfg)

    from exp_mw545.exp_dv_y import Build_DV_Y_Testing
    dv_y_model_test = Build_DV_Y_Testing(cfg, dv_y_cfg)
    fig_file_name = '/home/dawna/tts/mw545/Export_Temp/PNG_out/vuv_loss_sinenet_v0.png'
    dv_y_model_test.vuv_loss_test(fig_file_name)

    fig_file_name = '/home/dawna/tts/mw545/Export_Temp/PNG_out/positional_sinenet_v0.png'
    dv_y_model_test.positional_test(fig_file_name)