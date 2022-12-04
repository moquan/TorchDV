# modules_torch.py

import os, sys, pickle, time, shutil, logging, copy
import math, numpy, scipy
numpy.random.seed(545)
import torch
torch.manual_seed(545)

from modules import make_logger
from exp_mw545.sincnet_models import SincNet

'''
This file contains handy modules of using PyTorch
'''

########################
# PyTorch-based Layers #
########################

class Tensor_Reshape(torch.nn.Module):
    def __init__(self, current_layer_params):
        super().__init__()
        self.params = current_layer_params

    def update_layer_params(self):
        input_dim_seq = self.params['input_dim_seq']
        input_dim_values = self.params['input_dim_values']
        expect_input_dim_seq = self.params['expect_input_dim_seq']

        # First, check if change is needed at all; pass on if not
        if input_dim_seq == expect_input_dim_seq:
            self.params['expect_input_dim_values'] = input_dim_values
            return self.params
        else:
            # Make anything into ['S', 'B', 'T', 'D']
            if input_dim_seq == ['S', 'B', 'T', 'D']:
                # Do nothing, pass on
                temp_input_dim_values = input_dim_values
            elif input_dim_seq == ['S', 'B', 'D']:
                # Add T and make it 1
                temp_input_dim_values = {'S':input_dim_values['S'], 'B':input_dim_values['B'], 'T':1, 'D':input_dim_values['D']}
            elif input_dim_seq == ['S', 'T']:
                # Add T and make it 1
                temp_input_dim_values = {'S':input_dim_values['S'], 'B':1, 'T':input_dim_values['T'], 'D':1}

        # Then, make from ['S', 'B', 'T', 'D']
        if expect_input_dim_seq == ['S', 'B', 'D']:
            # So basically, stack and remove T; last dimension D -> T * D
            self.params['expect_input_dim_values'] = {'S':temp_input_dim_values['S'], 'B':temp_input_dim_values['B'], 'T':0, 'D':temp_input_dim_values['T']*temp_input_dim_values['D'] }
        elif expect_input_dim_seq ==  ['S','B','1','T']:
            # If D>1, that is stacked waveform, so flatten it
            # So basically, stack and remove D; T -> T * D
            self.params['expect_input_dim_values'] = {'S':temp_input_dim_values['S'], 'B':temp_input_dim_values['B'], 'T':temp_input_dim_values['T']*temp_input_dim_values['D'],'D':0 }
        elif expect_input_dim_seq ==  ['S','B','T']:
            # If D>1, that is stacked waveform, so flatten it
            # So basically, stack and remove D; T -> T * D
            self.params['expect_input_dim_values'] = {'S':temp_input_dim_values['S'], 'B':temp_input_dim_values['B'], 'T':temp_input_dim_values['T']*temp_input_dim_values['D'],'D':0 }
        return self.params

    def forward(self, x_dict):
        if 'x' in x_dict:
            temp_input = x_dict['x']
        elif 'h' in x_dict:
            temp_input = x_dict['h']
        else:
            raise Exception('No input tensor found!')

        input_dim_seq = self.params['input_dim_seq']
        input_dim_values = self.params['input_dim_values']
        expect_input_dim_seq = self.params['expect_input_dim_seq']
        expect_input_dim_values = self.params['expect_input_dim_values']

        # First, check if change is needed at all; pass on if not
        if input_dim_seq == expect_input_dim_seq:
            return temp_input
        else:
            # Make anything into ['S', 'B', 'T', 'D']
            if input_dim_seq == ['S', 'B', 'T', 'D']:
                # Do nothing, pass on
                temp_input = temp_input
            elif input_dim_seq == ['S', 'B', 'D']:
                # Add T and make it 1
                temp_input_dim_values = [input_dim_values['S'], input_dim_values['B'], 1, input_dim_values['D']]
                temp_input = temp_input.view(temp_input_dim_values)

        # Then, make from ['S', 'B', 'T', 'D']
        if expect_input_dim_seq == ['S', 'B', 'D']:
            expect_input_shape_values = [expect_input_dim_values['S'], expect_input_dim_values['B'], expect_input_dim_values['D']]
            expect_input = temp_input.view(expect_input_shape_values)
        elif expect_input_dim_seq ==  ['S','B','1','T']:
            expect_input_shape_values = [expect_input_dim_values['S'], expect_input_dim_values['B'], 1, expect_input_dim_values['T']]
            expect_input = temp_input.view(expect_input_shape_values)
        elif expect_input_dim_seq ==  ['S','B','T']:
            expect_input_shape_values = [expect_input_dim_values['S'], expect_input_dim_values['B'], expect_input_dim_values['T']]
            expect_input = temp_input.view(expect_input_shape_values)
        return expect_input

class Build_S_B_TD_Input_Layer(object):
    ''' This layer has only parameters, no torch.nn.module '''
    ''' Mainly for the prev_layer argument '''
    def __init__(self, dv_y_cfg):
        self.input_dim = dv_y_cfg.batch_seq_len * dv_y_cfg.feat_dim
        self.params = {}
        self.params["output_dim_seq"]      = ['S', 'B', 'D']
        self.params["output_dim_values"]   = {'S':dv_y_cfg.batch_num_spk, 'B':dv_y_cfg.spk_num_seq, 'D':self.input_dim}
        v = self.params["output_dim_values"]
        self.params["output_shape_values"] = [v['S'], v['B'], v['D']]
        
class Build_S_B_M_T_Input_Layer(object):
    ''' This layer has only parameters, no torch.nn.module '''
    ''' Mainly for the prev_layer argument '''
    def __init__(self, dv_y_cfg):
        self.input_dim = dv_y_cfg.seq_win_len
        self.params = {}
        self.params["output_dim_seq"]      = ['S', 'B', 'M', 'T']
        self.params["output_dim_values"]   = {'S':dv_y_cfg.batch_num_spk, 'B':dv_y_cfg.spk_num_seq, 'M':dv_y_cfg.seq_num_win, 'T':dv_y_cfg.seq_win_len}
        v = self.params["output_dim_values"]
        self.params["output_shape_values"] = [v['S'], v['B'], v['M'], v['T']]

class Build_S_T_Input_Layer(object):
    ''' This layer has only parameters, no torch.nn.module '''
    ''' Mainly for the prev_layer argument '''
    def __init__(self, dv_y_cfg):
        self.input_dim = dv_y_cfg.batch_seq_total_len
        self.params = {}
        self.params["output_dim_seq"]      = ['S', 'T']
        self.params["output_dim_values"]   = {'S':dv_y_cfg.batch_num_spk, 'T':dv_y_cfg.batch_seq_total_len}
        v = self.params["output_dim_values"]
        self.params["output_shape_values"] = [v['S'], v['T']]

class Build_NN_Layer(torch.nn.Module):
    def __init__(self, layer_config, prev_layer):
        super().__init__()
        self.params = {}
        self.params["layer_config"] = layer_config
        self.params["type"] = layer_config['type']
        self.params["size"] = layer_config['size']

        self.params["input_dim_seq"]    = prev_layer.params["output_dim_seq"]
        self.params["input_dim_values"] = prev_layer.params["output_dim_values"]

        # To be set per layer type; mostly for definition of h
        self.params["expect_input_dim_seq"]    = []
        self.params["expect_input_dim_values"] = {}
        self.params["output_dim_seq"]          = []
        self.params["output_dim_values"]       = {}

        construct_layer = getattr(self, self.params["layer_config"]["type"])
        construct_layer()

        ''' Dropout '''
        try: 
            self.params["dropout_p"] = self.params["layer_config"]['dropout_p']
        except KeyError: 
            self.params["dropout_p"] = 0.
            return dropout_input
        if self.params["dropout_p"] > 0:
            self.dropout_fn = torch.nn.Dropout(p=self.params["dropout_p"])
        else:
            self.dropout_fn = lambda a: a # Do nothing, just return same tensor

    def forward(self, x_dict):
        x_dict['h_reshape'] = self.reshape_fn(x_dict)
        y_dict = self.layer_fn(x_dict)
        y_dict['h'] = self.dropout_fn(y_dict['h'])
        return y_dict

    def ReLUDV(self, activation_fn='ReLU'):
        self.params["expect_input_dim_seq"] = ['S','B','D']
        self.reshape_fn = Tensor_Reshape(self.params)
        self.params = self.reshape_fn.update_layer_params()

        self.params["output_dim_seq"]       = ['S', 'B', 'D']
        v = self.params["expect_input_dim_values"]
        self.params["output_dim_values"]    = {'S': v['S'], 'B': v['B'], 'D': self.params["size"]}

        input_dim  = self.params['expect_input_dim_values']['D']
        output_dim = self.params['output_dim_values']['D']
        batch_norm = self.params["layer_config"]["batch_norm"]
        if activation_fn == 'ReLU':
            self.layer_fn = ReLUDVLayer(input_dim, output_dim, batch_norm)
        elif activation_fn == 'LReLU':
            self.layer_fn = ReLUDVLayer(input_dim, output_dim, batch_norm, activation_fn=torch.nn.LeakyReLU())

    def LReLUDV(self):
        self.ReLUDV(activation_fn='LReLU')

    def ReLUDVMax(self, activation_fn='ReLU'):
        self.params["expect_input_dim_seq"] = ['S','B','D']
        self.reshape_fn = Tensor_Reshape(self.params)
        self.params = self.reshape_fn.update_layer_params()

        self.params["output_dim_seq"]       = ['S', 'B', 'D']
        v = self.params["expect_input_dim_values"]
        self.params["output_dim_values"]    = {'S': v['S'], 'B': v['B'], 'D': self.params["size"]}

        input_dim  = self.params['expect_input_dim_values']['D']
        output_dim = self.params['output_dim_values']['D']
        num_channels = self.params["layer_config"]["num_channels"]
        if activation_fn == 'ReLU':
            self.layer_fn = ReLUDVMaxLayer(input_dim, output_dim, num_channels)
        elif activation_fn == 'LReLU':
            self.layer_fn = ReLUDVMaxLayer(input_dim, output_dim, num_channels, activation_fn=torch.nn.LeakyReLU())

    def LReLUDVMax(self):
        self.ReLUDVMax(activation_fn='LReLU')

    def LinDV(self):
        self.params["expect_input_dim_seq"] = ['S','B','D']
        self.reshape_fn = Tensor_Reshape(self.params)
        self.params = self.reshape_fn.update_layer_params()

        self.params["output_dim_seq"]       = ['S', 'B', 'D']
        v = self.params["expect_input_dim_values"]
        self.params["output_dim_values"]    = {'S': v['S'], 'B': v['B'], 'D': self.params["size"]}

        input_dim  = self.params['expect_input_dim_values']['D']
        output_dim = self.params['output_dim_values']['D']
        self.layer_fn = LinearDVLayer(input_dim, output_dim)

    def ReLUSubWin(self, activation_fn='ReLU'):
        self.params["expect_input_dim_seq"] = ['S','B','M','T']
        self.reshape_fn = Tensor_Reshape(self.params)
        self.params = self.reshape_fn.update_layer_params()

        self.params["output_dim_seq"] = ['S', 'B', 'D']
        v = self.params["expect_input_dim_values"]
        layer_config = self.params["layer_config"]
        total_output_dim = layer_config['size'] * layer_config['num_win']
        self.params["output_dim_values"] = {'S': v['S'], 'B': v['B'], 'D': total_output_dim} 

        output_dim = layer_config['size']
        win_len    = layer_config['win_len']
        num_win    = layer_config['num_win']
        batch_norm = layer_config["batch_norm"]
        if activation_fn == 'ReLU':
            self.layer_fn = ReLUSubWinLayer(output_dim, win_len, num_win, batch_norm)
        elif activation_fn == 'LReLU':
            self.layer_fn = ReLUSubWinLayer(output_dim, win_len, num_win, batch_norm, activation_fn=torch.nn.LeakyReLU())

    def LReLUSubWin(self):
        self.ReLUSubWin(activation_fn='LReLU')

    def ReLUSubWin_ST(self, activation_fn='ReLU'):
        '''
        Input ST
        ST --> SBT --> SBMT, 2 unfold ops
        '''
        self.params["expect_input_dim_seq"] = ['S','T']
        self.reshape_fn = Tensor_Reshape(self.params)
        self.params = self.reshape_fn.update_layer_params()

        self.params["output_dim_seq"] = ['S', 'B', 'D']
        v = self.params["expect_input_dim_values"]
        layer_config = self.params["layer_config"]
        T = layer_config['total_length']
        assert v['T'] == T
        batch_seq_len, batch_seq_shift = layer_config['win_len_shift_list'][0]
        B = int((T - batch_seq_len) / batch_seq_shift) + 1
        seq_win_len, seq_win_shift = layer_config['win_len_shift_list'][1]
        M = int((batch_seq_len - seq_win_len) / seq_win_shift) + 1
        D = layer_config['size']
        total_output_dim = M * D
        self.params["output_dim_values"] = {'S': v['S'], 'B': B, 'D': total_output_dim}
        
        win_len_shift_list = layer_config['win_len_shift_list']
        batch_norm = layer_config["batch_norm"]

        if activation_fn == 'ReLU':
            self.layer_fn = ReLUSubWinLayer_ST(D, win_len_shift_list, T, M, batch_norm)
        elif activation_fn == 'LReLU':
            self.layer_fn = ReLUSubWinLayer_ST(D, win_len_shift_list, T, M, batch_norm, activation_fn=torch.nn.LeakyReLU())

    def SCNet_ST(self):
        '''
        SincNet
        Input ST
        ST --> SBT, 1 unfold op
        '''
        self.params["expect_input_dim_seq"] = ['S','T']
        self.reshape_fn = Tensor_Reshape(self.params)
        self.params = self.reshape_fn.update_layer_params()

        self.params["output_dim_seq"] = ['S', 'B', 'D']
        v = self.params["expect_input_dim_values"]
        layer_config = self.params["layer_config"]
        T = layer_config['total_length']
        assert v['T'] == T
        batch_seq_len, batch_seq_shift = layer_config['win_len_shift_list'][0]
        B = int((T - batch_seq_len) / batch_seq_shift) + 1
        # TODO: Hard-code configuration here for now
        seq_win_len, seq_win_shift = layer_config['win_len_shift_list'][1]
        M1 = int((int((batch_seq_len - seq_win_len) / seq_win_shift) + 1)/3)
        M2 = int((int((M1 - 5) / 1) + 1)/3)
        M3 = int((int((M2 - 5) / 1) + 1)/3)
        total_output_dim = M3 * 60
        self.params["output_dim_values"] = {'S': v['S'], 'B': B, 'D': total_output_dim}

        win_len_shift_list = layer_config['win_len_shift_list']
        batch_norm = layer_config["batch_norm"]

        self.layer_fn = SincNetLayer_ST(win_len_shift_list)
        assert self.layer_fn.out_dim == total_output_dim

    def ReLUSubWin_f_tau_vuv_ST(self, activation_fn='ReLU'):
        '''
        Input ST
        ST --> SBT --> SBMT, 2 unfold ops
        Also, append nlf, tau, vuv information
        '''
        self.params["expect_input_dim_seq"] = ['S','T']
        self.reshape_fn = Tensor_Reshape(self.params)
        self.params = self.reshape_fn.update_layer_params()

        self.params["output_dim_seq"] = ['S', 'B', 'D']
        v = self.params["expect_input_dim_values"]
        layer_config = self.params["layer_config"]
        T = layer_config['total_length']
        assert v['T'] == T
        batch_seq_len, batch_seq_shift = layer_config['win_len_shift_list'][0]
        B = int((T - batch_seq_len) / batch_seq_shift) + 1
        seq_win_len, seq_win_shift = layer_config['win_len_shift_list'][1]
        M = int((batch_seq_len - seq_win_len) / seq_win_shift) + 1
        D = layer_config['size']
        total_output_dim = M * D
        self.params["output_dim_values"] = {'S': v['S'], 'B': B, 'D': total_output_dim} 
        
        win_len_shift_list = layer_config['win_len_shift_list']
        batch_norm = layer_config["batch_norm"]

        if activation_fn == 'ReLU':
            self.layer_fn = ReLUSubWinLayer_f_tau_vuv_ST(D, win_len_shift_list, T, M, batch_norm)
        elif activation_fn == 'LReLU':
            self.layer_fn = ReLUSubWinLayer_f_tau_vuv_ST(D, win_len_shift_list, T, M, batch_norm, activation_fn=torch.nn.LeakyReLU())

    def Sinenet(self):
        self.params["expect_input_dim_seq"] = ['S','B','1','T']
        self.reshape_fn = Tensor_Reshape(self.params)
        self.params = self.reshape_fn.update_layer_params()

        self.params["output_dim_seq"]       = ['S', 'B', 'D']
        v = self.params["expect_input_dim_values"]
        self.params["output_dim_values"]    = {'S': v['S'], 'B': v['B'], 'D': self.params["size"]}

        input_dim  = self.params['expect_input_dim_values']['D']
        output_dim = self.params['output_dim_values']['D']
        num_channels = self.params["layer_config"]["num_channels"]
        time_len   = self.params['expect_input_dim_values']['T']
        self.layer_fn = SinenetLayer(time_len, output_dim, num_channels)
    
    def SinenetV1(self):
        self.params["expect_input_dim_seq"] = ['S','B','1','T']
        self.reshape_fn = Tensor_Reshape(self.params)
        self.params = self.reshape_fn.update_layer_params()

        self.params["output_dim_seq"]       = ['S', 'B', 'D']
        v = self.params["expect_input_dim_values"]
        self.params["output_dim_values"]    = {'S': v['S'], 'B': v['B'], 'D': self.params["size"]} 

        input_dim  = self.params['expect_input_dim_values']['D']
        output_dim = self.params['output_dim_values']['D']
        num_channels = self.params["layer_config"]["num_channels"]
        time_len   = self.params['expect_input_dim_values']['T']

        self.layer_fn = SinenetLayerV1(time_len, output_dim, num_channels)
        self.params["output_dim_values"]['D'] += 1 # +1 to append nlf F0 values

    def SinenetV2(self):
        self.params["expect_input_dim_seq"] = ['S','B','1','T']
        self.reshape_fn = Tensor_Reshape(self.params)
        self.params = self.reshape_fn.update_layer_params()

        self.params["output_dim_seq"]       = ['S', 'B', 'D']
        v = self.params["expect_input_dim_values"]
        self.params["output_dim_values"]    = {'S': v['S'], 'B': v['B'], 'D': self.params["size"]} 

        input_dim  = self.params['expect_input_dim_values']['D']
        output_dim = self.params['output_dim_values']['D']
        num_channels = self.params["layer_config"]["num_channels"]
        time_len   = self.params['expect_input_dim_values']['T']

        self.layer_fn = SinenetLayerV2(time_len, output_dim, num_channels)
        self.params["output_dim_values"]['D'] += 1 # +1 to append nlf F0 values

    def SinenetV3(self, activation_fn='ReLU'):
        self.params["expect_input_dim_seq"] = ['S','B','M','T']
        self.reshape_fn = Tensor_Reshape(self.params)
        self.params = self.reshape_fn.update_layer_params()

        self.params["output_dim_seq"] = ['S', 'B', 'D']
        v = self.params["expect_input_dim_values"]
        layer_config = self.params["layer_config"]
        assert layer_config['size'] == layer_config['sine_size'] + 1
        total_output_dim = (layer_config['size']) * layer_config['num_win']
        self.params["output_dim_values"] = {'S': v['S'], 'B': v['B'], 'D': total_output_dim} 

        output_dim = layer_config['size']
        sine_size  = layer_config['sine_size']
        num_freq   = layer_config['num_freq']
        win_len    = layer_config['win_len']
        num_win    = layer_config['num_win']
        batch_norm = layer_config['batch_norm']
        
        if activation_fn == 'ReLU':
            self.layer_fn = SinenetLayerV3(sine_size, num_freq, win_len, num_win, batch_norm)
        elif activation_fn == 'LReLU':
            self.layer_fn = SinenetLayerV3(sine_size, num_freq, win_len, num_win, batch_norm, activation_fn=torch.nn.LeakyReLU())

    def LSinenetV3(self):
        self.SinenetV3(activation_fn='LReLU')

    def SinenetV3_ST(self, activation_fn='ReLU'):
        '''
        Input ST
        ST --> SBT --> SBMT, 2 unfold ops
        '''
        self.params["expect_input_dim_seq"] = ['S','T']
        self.reshape_fn = Tensor_Reshape(self.params)
        self.params = self.reshape_fn.update_layer_params()

        self.params["output_dim_seq"] = ['S', 'B', 'D']
        v = self.params["expect_input_dim_values"]
        layer_config = self.params["layer_config"]
        T = layer_config['total_length']
        assert v['T'] == T
        batch_seq_len, batch_seq_shift = layer_config['win_len_shift_list'][0]
        B = int((T - batch_seq_len) / batch_seq_shift) + 1
        seq_win_len, seq_win_shift = layer_config['win_len_shift_list'][1]
        M = int((batch_seq_len - seq_win_len) / seq_win_shift) + 1
        D = layer_config['size']
        sine_size = layer_config['sine_size']
        assert D == (sine_size+1)
        total_output_dim = M * D
        self.params["output_dim_values"] = {'S': v['S'], 'B': B, 'D': total_output_dim} 
        
        win_len_shift_list = layer_config['win_len_shift_list']
        batch_norm = layer_config["batch_norm"]

        num_freq = layer_config['num_freq']

        if activation_fn == 'ReLU':
            self.layer_fn = SinenetLayerV3_ST(sine_size, num_freq, win_len_shift_list, T, M, batch_norm)
        elif activation_fn == 'LReLU':
            self.layer_fn = SinenetLayerV3_ST(sine_size, num_freq, win_len_shift_list, T, M, batch_norm, activation_fn=torch.nn.LeakyReLU())

    def SinenetV4(self):
        self.params["expect_input_dim_seq"] = ['S','B','M','T']
        self.reshape_fn = Tensor_Reshape(self.params)
        self.params = self.reshape_fn.update_layer_params()

        self.params["output_dim_seq"] = ['S', 'B', 'D']
        v = self.params["expect_input_dim_values"]
        layer_config = self.params["layer_config"]
        assert layer_config['size'] == layer_config['sine_size'] + layer_config['relu_size'] + 1
        total_output_dim = (layer_config['size']) * layer_config['num_win']
        self.params["output_dim_values"] = {'S': v['S'], 'B': v['B'], 'D': total_output_dim} 

        sine_size  = layer_config['sine_size']
        num_freq   = layer_config['num_freq']
        relu_size  = layer_config['relu_size']
        win_len    = layer_config['win_len']
        num_win    = layer_config['num_win']

        self.layer_fn = SinenetLayerV4(sine_size, num_freq, relu_size, win_len, num_win)

def batch_norm_D_tensor(input_tensor, bn_fn, index_D):
    # Move index_D to 1, to norm D
    h_SDB = torch.transpose(input_tensor, 1, index_D)
    h_SDB = bn_fn(h_SDB)
    # Reshape back, swap 1 and index_D again
    h_SBD = torch.transpose(h_SDB, 1, index_D)
    if index_D > 2:
        h_SBD = h_SBD.contiguous()
    return h_SBD

class ReLUDVLayer(torch.nn.Module):
    def __init__(self, input_dim, output_dim, batch_norm, activation_fn=torch.nn.ReLU()):
        super().__init__()
        self.input_dim  = input_dim
        self.output_dim = output_dim
        self.batch_norm = batch_norm

        self.fc_fn = torch.nn.Linear(input_dim, output_dim)
        # self.fc_fn = torch.nn.Linear(input_dim, output_dim, bias=(not self.batch_norm))
        self.activation_fn = activation_fn

        if self.batch_norm:
            self.bn_fn = torch.nn.BatchNorm1d(output_dim)

    def forward(self, x_dict):
        if 'h_reshape' in x_dict:
            x = x_dict['h_reshape']
        elif 'x' in x_dict:
            x = x_dict['x']
        # Linear
        h_i = self.fc_fn(x)
        # Batch Norm
        if self.batch_norm:
            h_i = batch_norm_D_tensor(h_i, self.bn_fn, index_D=2)
        # ReLU
        h = self.activation_fn(h_i)
        y_dict = {'h': h}
        return y_dict

class ReLUDVMaxLayer(torch.nn.Module):
    def __init__(self, input_dim, output_dim, num_channels, activation_fn=torch.nn.ReLU()):
        super().__init__()
        self.input_dim    = input_dim
        self.output_dim   = output_dim
        self.num_channels = num_channels

        self.fc_list = torch.nn.ModuleList([torch.nn.Linear(input_dim, output_dim) for i in range(self.num_channels)])
        self.activation_fn = activation_fn

    def forward(self, x_dict):
        if 'h_reshape' in x_dict:
            x = x_dict['h_reshape']
        elif 'x' in x_dict:
            x = x_dict['x']
        h_list = []
        for i in range(self.num_channels):
            # Linear
            h_i = self.fc_list[i](x)
            # ReLU
            h_i = self.activation_fn(h_i)
            h_list.append(h_i)

        h_stack = torch.stack(h_list, dim=0)
        # MaxOut
        h_max, _indices = torch.max(h_stack, dim=0, keepdim=False)
        y_dict = {'h': h_max, 'h_stack':h_stack}
        return y_dict

class LinearDVLayer(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim    = input_dim
        self.output_dim   = output_dim
        self.linear_fn = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x_dict):
        if 'h_reshape' in x_dict:
            x = x_dict['h_reshape']
        elif 'x' in x_dict:
            x = x_dict['x']

        h = self.linear_fn(x)
        y_dict = {'h': h}
        return y_dict

class ReLUSubWinLayer(torch.nn.Module):
    ''' Input dimensions
            x: S*B*M*T
            Output dimensions
            y: S*B*M*D
            h: S*B*(M*(D+1))
        Apply ReLU on each sub-window within 
        Then stack the outputs, S_B_M_D --> S_B_MD
    '''
    def __init__(self, output_dim, win_len, num_win, batch_norm, activation_fn=torch.nn.ReLU()):
        super().__init__()

        self.output_dim = output_dim
        self.batch_norm = batch_norm

        self.win_len = win_len
        self.num_win = num_win

        self.fc_fn   = torch.nn.Linear(win_len, output_dim, bias=(not self.batch_norm)) # Remove bias if batch_norm is applied
        self.activation_fn = activation_fn

        if self.batch_norm:
            self.bn_fn = torch.nn.BatchNorm2d(output_dim)

    def forward(self, x_dict):
        if 'h_reshape' in x_dict:
            x = x_dict['h_reshape']
        elif 'x' in x_dict:
            x = x_dict['x']
        # Linear
        y_SBMD = self.fc_fn(x)
        # Batch Norm
        if self.batch_norm:
            y_SBMD = batch_norm_D_tensor(y_SBMD, self.bn_fn, index_D=3)

        y_size = y_SBMD.size()
        y_SBD  = y_SBMD.view([y_size[0], y_size[1], -1]) # S*B*M*D -> S*B*MD
        # Batch Norm
        # if self.batch_norm:
        #     y_SBD = batch_norm_D_tensor(y_SBD, self.bn_fn, index_D=2)

        # ReLU
        h = self.activation_fn(y_SBD)
        y_dict = {'h': h}
        return y_dict

class ReLUSubWinLayer_ST(torch.nn.Module):
    ''' Input dimensions
            x: S*T
            Output dimensions
            y: S*B*M*D
            h: S*B*(M*D)
        Apply ReLU on each sub-window within 
        Then stack the outputs, S_B_M_D --> S_B_MD
    '''
    def __init__(self, output_dim, win_len_shift_list, total_length, num_win, batch_norm, activation_fn=torch.nn.ReLU()):
        super().__init__()

        assert len(win_len_shift_list) == 2
        self.win_len_shift_list = win_len_shift_list
        win_len = win_len_shift_list[1][0]
        self.relu_subwin_layer = ReLUSubWinLayer(output_dim, win_len, num_win, batch_norm, activation_fn)

    def forward(self, x_dict):
        if 'h_reshape' in x_dict:
            x = x_dict['h_reshape']
        elif 'x' in x_dict:
            x = x_dict['x']

        '''
        1. Unfold x twice; S*T --> S*B*M*T
        2. relu_subwin_function
        '''
        for i in range(2):
            win_len_shift = self.win_len_shift_list[i]
            win_len, win_shift = win_len_shift
            x = x.unfold(i+1, win_len, win_shift)

        x_new_dict = {'h_reshape': x}
        y_dict = self.relu_subwin_layer(x_new_dict)
        return y_dict

class SincNetLayer_ST(torch.nn.Module):
    ''' Input dimensions
            x: S*T
        1. unfold to make S*B*T
        2. view to make (SB)*T
        3. Apply SincNet to output (SB)*D
        4. view to make S*B*D
        Then stack the outputs, S_B_M_D --> S_B_MD
    '''
    def __init__(self, win_len_shift_list, batch_norm=False, activation_fn=torch.nn.ReLU()):
        super().__init__()

        assert len(win_len_shift_list) == 2
        self.win_len_shift_list = win_len_shift_list
        self.options = self.make_options()
        self.sincnet_layer = SincNet(self.options)
        self.out_dim = self.sincnet_layer.out_dim

    def forward(self, x_dict):
        if 'h_reshape' in x_dict:
            x = x_dict['h_reshape']
        elif 'x' in x_dict:
            x = x_dict['x']

        win_len_shift = self.win_len_shift_list[0]
        win_len, win_shift = win_len_shift
        x_SBT = x.unfold(1, win_len, win_shift)

        S = x_SBT.shape[0]
        B = x_SBT.shape[1]
        T = x_SBT.shape[2]

        x_SB_T = x_SBT.reshape(S*B, T)

        y_SB_D = self.sincnet_layer(x_SB_T)
        D = y_SB_D.shape[1]
        y_SBD = y_SB_D.view(S,B,D)

        y_dict = {'h': y_SBD}
        return y_dict

    def make_options(self):
        CNN_arch = {'input_dim': self.win_len_shift_list[0][0],
          'fs': 16000,
          'cnn_N_filt': [80,60,60],
          'cnn_len_filt': [251,5,5],
          'cnn_max_pool_len':[3,3,3],
          'cnn_use_laynorm_inp': True,
          'cnn_use_batchnorm_inp': False,
          'cnn_use_laynorm':[True,True,True],
          'cnn_use_batchnorm':[False,False,False],
          'cnn_act': ['leaky_relu','leaky_relu','leaky_relu'],
          'cnn_drop':[0.0,0.0,0.0],          
        }
        return CNN_arch

class ReLUSubWinLayer_f_tau_vuv_ST(torch.nn.Module):
    ''' Input dimensions
            x: S*T
            Output dimensions
            y: S*B*M*D
            h: S*B*(M*D)
        Apply ReLU on each sub-window within 
        Then stack the outputs, S_B_M_D --> S_B_MD
        Then, append nlf, tau, vuv
    '''
    def __init__(self, output_dim, win_len_shift_list, total_length, num_win, batch_norm, activation_fn=torch.nn.ReLU()):
        super().__init__()

        assert len(win_len_shift_list) == 2
        self.win_len_shift_list = win_len_shift_list
        win_len = win_len_shift_list[1][0]
        self.relu_subwin_layer = ReLUSubWinLayer(output_dim, win_len+3, num_win, batch_norm, activation_fn)

    def forward(self, x_dict):
        if 'h_reshape' in x_dict:
            x = x_dict['h_reshape']
        elif 'x' in x_dict:
            x = x_dict['x']
        nlf = x_dict['nlf']
        tau = x_dict['tau']
        vuv = x_dict['vuv']

        '''
        1. Unfold x twice; S*T --> S*B*M*T
        2. relu_subwin_function
        '''
        for i in range(2):
            win_len_shift = self.win_len_shift_list[i]
            win_len, win_shift = win_len_shift
            x = x.unfold(i+1, win_len, win_shift)

        '''
        3. Append nlf, tau, vuv
        '''
        nlf_1 = torch.unsqueeze(nlf, 3) # S*B*M --> # S*B*M*1
        tau_1 = torch.unsqueeze(tau, 3) # S*B*M --> # S*B*M*1
        vuv_1 = torch.unsqueeze(vuv, 3) # S*B*M --> # S*B*M*1
        x_f_tau_vuv = torch.cat([x, nlf_1, tau_1, vuv_1], 3)

        x_new_dict = {'h_reshape': x}
        y_dict = self.relu_subwin_layer(x_new_dict)
        return y_dict

class SinenetLayer(torch.nn.Module):
    ''' f tau dependent sine waves, convolve and stack '''
    ''' output doesn't contain f0 information, pad outside '''
    def __init__(self, time_len, output_dim, num_channels):
        super().__init__()
        self.time_len     = time_len
        self.output_dim   = output_dim   # Total output dimension
        self.num_channels = num_channels # Number of components per frequency
        self.num_freq     = int(output_dim / num_channels) # Number of frequency components

        self.t_wav = 1./16000

        self.log_f_mean = 5.04418
        self.log_f_std  = 0.358402

        self.i_2pi_tensor = self.make_i_2pi_tensor()   # D*1
        self.k_T_tensor   = self.make_k_T_tensor_t_1() # 1*T

        a_init_value   = numpy.random.normal(loc=0.1, scale=0.1, size=output_dim)
        phi_init_value = numpy.random.normal(loc=0.0, scale=1.0, size=output_dim)
        self.a   = torch.nn.Parameter(torch.tensor(a_init_value, dtype=torch.float), requires_grad=True) # D*1
        self.phi = torch.nn.Parameter(torch.tensor(phi_init_value, dtype=torch.float), requires_grad=True) # D

        self.relu_fn = torch.nn.ReLU()

    def forward(self, x_dict):
        ''' 
        Input dimensions
        x: S*B*1*T
        nlf, tau: S*B*1*1
        '''
        # Denorm and exp norm_log_f (S*B)
        # Norm: norm_features = (features - mean_matrix) / std_matrix
        # Denorm: features = norm_features * std_matrix + mean_matrix
        if 'h_reshape' in x_dict:
            x = x_dict['h_reshape']
        elif 'x' in x_dict:
            x = x_dict['x']
        nlf = x_dict['nlf']
        tau = x_dict['tau']

        lf = torch.add(torch.mul(nlf, self.log_f_std), self.log_f_mean) # S*B*1*1
        f  = torch.exp(lf)                                              # S*B*1*1

        # Time
        t = torch.add(self.k_T_tensor, torch.neg(tau)) # T*1 + S*B*1*1 -> S*B*T*1

        # Degree in radian
        f_t = torch.mul(f, t)                        # S*B*1*1 * S*B*T*1 -> S*B*T*1
        deg = torch.nn.functional.linear(f_t, self.i_2pi_tensor, bias=self.phi) # S*B*T*1, D*1, D -> S*B*T*D
        deg = torch.transpose(deg, 2, 3) # S*B*T*D -> S*B*D*T, but no additional storage

        # Sine
        s = torch.sin(deg)                    # S*B*D*T
        # Multiply sine with x first
        x_SBT = torch.squeeze(x, 2)  # S*B*1*T -> S*B*T
        sin_x = torch.einsum('sbdt,sbt->sbd', s, x_SBT) # S*B*D*T, S*B*T -> S*B*D

        h_SBD = torch.mul(self.a, sin_x)         # D * S*B*D -> S*B*D
        # ReLU
        h = self.relu_fn(h_SBD)
        y_dict = {'h': h}
        return y_dict

    def make_i_2pi_tensor(self):
        # indices of frequency components
        i_vec = numpy.zeros((self.output_dim, 1))
        for i in range(self.num_freq):
            for j in range(self.num_channels):
                d = int(i * self.num_channels + j)
                i_vec[d,0] = i + 1
        i_vec = i_vec * 2 * numpy.pi
        i_vec_tensor = torch.tensor(i_vec, dtype=torch.float, requires_grad=False)
        i_vec_tensor = torch.nn.Parameter(i_vec_tensor, requires_grad=False)
        return i_vec_tensor

    def make_k_T_tensor_t_1(self):
        # indices along time
        k_T_vec = numpy.zeros((self.time_len,1))
        for i in range(self.time_len):
            k_T_vec[i,0] = i
        k_T_vec = k_T_vec * self.t_wav
        k_T_tensor = torch.tensor(k_T_vec, dtype=torch.float, requires_grad=False)
        k_T_tensor = torch.nn.Parameter(k_T_tensor, requires_grad=False)
        return k_T_tensor

    def return_a_value(self):
        return self.a.data.cpu().detach().numpy()

    def return_phi_value(self):
        return self.phi.data.cpu().detach().numpy()

    def keep_phi_within_2pi(self, gpu_id):
        phi_val = self.return_phi_value()

        i = (phi_val / (2 * numpy.pi)).astype(int)
        if numpy.any(i!=0):
            print('Original phi_val')
            print(phi_val)
            print('i matrix')
            print(i)

            device_id = torch.device("cuda:%i" % gpu_id)
            i2pi = torch.tensor(i * (2 * numpy.pi), device=device_id)
            with torch.no_grad():
                self.phi -= i2pi
            phi_val = self.return_phi_value()
            print('New phi_val')
            print(phi_val)

class SinenetLayerV1(torch.nn.Module):
    ''' 3 Parts: f-prediction, tau-prediction, sinenet '''
    def __init__(self, time_len, output_dim, num_channels):
        super().__init__()
        self.time_len     = time_len
        self.output_dim   = output_dim   # Total output dimension
        self.num_channels = num_channels # Number of components per frequency
        self.num_freq     = int(output_dim / num_channels) # Number of frequency components

        self.nlf_pred_layer = torch.nn.Linear(time_len, 1)
        self.tau_pred_layer = torch.nn.Linear(time_len, 1)
        self.sinenet_layer  = SinenetLayer(time_len, output_dim, num_channels)

    def forward(self, x_dict):
        if 'h_reshape' in x_dict:
            x = x_dict['h_reshape']
        elif 'x' in x_dict:
            x = x_dict['x']
        nlf = self.nlf_pred_layer(x)
        tau = self.tau_pred_layer(x)
        sine_dict = {'h_reshape':x, 'nlf':nlf, 'tau':tau}

        h_SBD  = self.sinenet_layer(sine_dict)['h']

        # Append nlf
        nlf_SBD = torch.squeeze(nlf, 2)    # S*B*1*1 -> S*B*1
        h_SBD   = torch.cat((nlf_SBD, h_SBD), 2)
        y_dict = {'h': h_SBD}
        return y_dict

class SinenetLayerV2(torch.nn.Module):
    ''' 3 Parts: f-prediction, tau-prediction, sinenet '''
    def __init__(self, time_len, output_dim, num_channels):
        super().__init__()
        self.time_len     = time_len
        self.output_dim   = output_dim   # Total output dimension
        self.num_channels = num_channels # Number of components per frequency
        self.num_freq     = int(output_dim / num_channels) # Number of frequency components

        self.sinenet_layer  = SinenetLayer(time_len, output_dim, num_channels)

    def forward(self, x_dict):
        h_SBD = self.sinenet_layer(x_dict)['h']
        nlf   = x_dict['nlf']
        # Append nlf
        nlf_SBD = torch.squeeze(nlf, 2)    # S*B*1*1 -> S*B*1
        h_SBD   = torch.cat((nlf_SBD, h_SBD), 2)
        y_dict = {'h': h_SBD}
        return y_dict

class SinenetLayerV3(torch.nn.Module):
    ''' Predicted lf0 and tau values     
            Input dimensions
            x: S*B*M*T
            nlf, tau: S*B*M
            Output dimensions
            y: S*B*M*D
            h: S*B*(M*(D+1))
        1. Apply sinenet on each sub-window within 
        2. Stack the outputs, S_B_M_D --> S_B_MD
        3. Apply fc_relu
        4. Append f0
    '''
    def __init__(self, sine_size, num_freq, win_len, num_win, batch_norm, activation_fn=torch.nn.ReLU()):
        super().__init__()

        self.t_wav = 1./16000
        self.log_f_mean = 5.04418
        self.log_f_std  = 0.358402

        self.sine_size  = sine_size # Output size per sub-window; not including f0 yet
        self.num_freq   = num_freq  # Number of frequency components
        self.batch_norm = batch_norm

        self.win_len      = win_len
        self.num_win      = num_win

        self.k_2pi_tensor = self.make_k_2pi_tensor() # K
        self.n_T_tensor   = self.make_n_T_tensor()   # T
        # self.fc_fn = torch.nn.Linear(self.num_freq*2, sine_size, bias=(not self.batch_norm))
        self.fc_fn = torch.nn.Linear(self.num_freq*2, sine_size)
        self.activation_fn = activation_fn

        if self.batch_norm:
            self.bn_fn = torch.nn.BatchNorm2d(sine_size)

    def forward(self, x_dict):
        
        if 'h_reshape' in x_dict:
            x = x_dict['h_reshape']
        elif 'x' in x_dict:
            x = x_dict['x']
        
        tau = x_dict['tau']
        if 'nlf' in x_dict:
            nlf = x_dict['nlf']
            f = self.convert_nlf_2_f(nlf)
        elif 'f' in x_dict:
            f = x_dict['f']
            nlf = self.convert_f_2_nlf(f)
        
        sin_cos_matrix = self.construct_w_sin_cos_matrix(f, tau) # S*B*M*2K*T
        sin_cos_x = torch.einsum('sbmkt,sbmt->sbmk', sin_cos_matrix, x) # S*B*M*2K*T, # S*B*M*T -> S*B*M*2K
        y_SBMD = self.fc_fn(sin_cos_x)                             # S*B*M*2K -> S*B*M*D
        # Batch Norm
        if self.batch_norm:
            y_SBMD = batch_norm_D_tensor(y_SBMD, self.bn_fn, index_D=3)

        y_size = y_SBMD.size()
        y_SBD  = y_SBMD.view([y_size[0], y_size[1], -1])           # S*B*M*D -> S*B*MD
        y_f_SBD = torch.cat([y_SBD, nlf], 2)

        # ReLU
        h = self.activation_fn(y_f_SBD)
        y_dict = {'h': h}
        return y_dict

    def convert_nlf_2_f(self, nlf):
        lf = torch.add(torch.mul(nlf, self.log_f_std), self.log_f_mean) # S*B*M
        f  = torch.exp(lf)                                              # S*B*M
        return f

    def convert_f_2_nlf(self, f):
        lf = torch.torch.log(f)
        nlf = torch.mul(torch.add(lf, (-1)*self.log_f_mean), 1./self.log_f_std)
        return nlf

    def make_k_2pi_tensor(self):
        ''' indices of frequency components '''
        k_vec = numpy.zeros(self.num_freq)
        for k in range(self.num_freq):
            k_vec[k] = k + 1
        k_vec = k_vec * 2 * numpy.pi
        k_vec_tensor = torch.tensor(k_vec, dtype=torch.float, requires_grad=False)
        k_vec_tensor = torch.nn.Parameter(k_vec_tensor, requires_grad=False)
        return k_vec_tensor

    def make_n_T_tensor(self):
        ''' indices along time '''
        n_T_vec = numpy.zeros(self.win_len)
        for n in range(self.win_len):
            n_T_vec[n] = n * self.t_wav
        n_T_tensor = torch.tensor(n_T_vec, dtype=torch.float, requires_grad=False)
        n_T_tensor = torch.nn.Parameter(n_T_tensor, requires_grad=False)
        return n_T_tensor

    def compute_deg(self, f, tau):
        ''' Return degree in radian '''
        # Time
        tau_1 = torch.unsqueeze(tau, 3) # S*B*M --> # S*B*M*1
        t = torch.add(self.n_T_tensor, torch.neg(tau_1)) # T + S*B*M*1 -> S*B*M*T

        # Degree in radian
        f_1 = torch.unsqueeze(f, 3) # S*B*M --> # S*B*M*1
        k_2pi_f = torch.mul(self.k_2pi_tensor, f_1) # K + S*B*M*1 -> S*B*M*K
        k_2pi_f_1 = torch.unsqueeze(k_2pi_f, 4) # S*B*M*K -> S*B*M*K*1
        t_1 = torch.unsqueeze(t, 3) # S*B*M*T -> S*B*M*1*T
        deg = torch.mul(k_2pi_f_1, t_1) # S*B*M*K*1, S*B*M*1*T -> S*B*M*K*T
        return deg

    def construct_w_sin_cos_matrix(self, nlf, tau):
        deg = self.compute_deg(nlf, tau) # S*B*M*K*T
        s   = torch.sin(deg)             # S*B*M*K*T
        c   = torch.cos(deg)             # S*B*M*K*T
        s_c = torch.cat([s,c], dim=3)    # S*B*M*2K*T
        return s_c

    def return_w_mul_w_sin_cos(self, nlf, tau):
        sin_cos_matrix = self.construct_w_sin_cos_matrix(nlf, tau) # S*B*M*2K*T
        s_c_mat_t      = sin_cos_matrix.transpose(3,4)             # S*B*M*2K*T -> # S*B*M*T*2K
        w_w   = self.fc_fn(s_c_mat_t)                              # S*B*M*T*2K -> S*B*M*T*D
        w_w_t = w_w.transpose(3,4)                                 # S*B*M*T*D -> # S*B*M*D*T
        return w_w_t

    def return_w_mul_w_sin_cos_from_x(self, x_dict):
        nlf = x_dict['nlf']
        tau = x_dict['tau']
        return self.return_w_mul_w_sin_cos(nlf, tau)

    def return_w_sin_cos_from_x(self, x_dict):
        nlf = x_dict['nlf']
        tau = x_dict['tau']
        return self.construct_w_sin_cos_matrix(nlf, tau)

class SinenetLayerV3_ST(torch.nn.Module):
    ''' Input dimensions
            x: S*T
            Output dimensions
            y: S*B*M*D
            h: S*B*(M*(D+1))
        Apply SineNet on each sub-window within 
        Then stack the outputs, S_B_M_D --> S_B_MD
    '''
    def __init__(self, sine_size, num_freq, win_len_shift_list, total_length, num_win, activation_fn=torch.nn.ReLU()):
        super().__init__()

        assert len(win_len_shift_list) == 2
        self.win_len_shift_list = win_len_shift_list
        win_len = win_len_shift_list[1][0]
        self.sinenet_layer = SinenetLayerV3(sine_size, num_freq, win_len, num_win, activation_fn)

    def forward(self, x_dict):
        if 'h_reshape' in x_dict:
            x = x_dict['h_reshape']
        elif 'x' in x_dict:
            x = x_dict['x']
        tau = x_dict['tau']

        '''
        1. Unfold x twice; S*T --> S*B*M*T
        2. relu_subwin_function
        '''
        for i in range(2):
            win_len_shift = self.win_len_shift_list[i]
            win_len, win_shift = win_len_shift
            x = x.unfold(i+1, win_len, win_shift)

        x_new_dict = {'h_reshape': x, 'tau':tau}
        if 'nlf' in x_dict:
            nlf = x_dict['nlf']
            x_new_dict['nlf'] = nlf
        elif 'f' in x_dict:
            f = x_dict['f']
            x_new_dict['f'] = f
        y_dict = self.sinenet_layer(x_new_dict)
        return y_dict

class SinenetLayerV4(torch.nn.Module):
    
    def __init__(self, sine_size, num_freq, relu_size, win_len, num_win):
        super().__init__()
        self.sinenet_fn  = SinenetLayerV3(sine_size, num_freq, win_len, num_win)
        self.relu_win_fn = ReLUSubWinLayer(relu_size, win_len, num_win, batch_norm=False)

    def forward(self, x_dict):
        y_sin_f_SBD = self.sinenet_fn(x_dict)['h']
        y_relu_SBD  = self.relu_win_fn(x_dict)['h']

        # Concatenate
        h = torch.cat([y_sin_f_SBD, y_relu_SBD], 2)
        y_dict = {'h': h}
        return y_dict

########################
# PyTorch-based Models #
########################

class DV_Y_NN_model(torch.nn.Module):
    ''' S_B_D input, SB_D/S_D logit output if train_by_window '''
    def __init__(self, dv_y_cfg, input_layer=None):
        super().__init__()
        
        self.num_nn_layers = dv_y_cfg.num_nn_layers
        self.train_by_window = dv_y_cfg.train_by_window

        self.input_layer = input_layer(dv_y_cfg)
        self.input_dim   = self.input_layer.input_dim
        prev_layer = self.input_layer

        # Hidden layers
        # The last is bottleneck, output_dim is lambda_dim
        self.layer_list = torch.nn.ModuleList()
        for i in range(self.num_nn_layers):
            layer_config = dv_y_cfg.nn_layer_config_list[i]

            layer_temp = Build_NN_Layer(layer_config, prev_layer)
            prev_layer = layer_temp
            self.layer_list.append(layer_temp)

        # Expansion layer, from lambda to logit
        self.dv_dim  = dv_y_cfg.dv_dim
        self.output_dim = dv_y_cfg.num_speaker_dict['train']
        self.expansion_layer = torch.nn.Linear(self.dv_dim, self.output_dim)

    def gen_lambda_SBD(self, x_dict):
        ''' Simple sequential feed-forward '''
        for i in range(self.num_nn_layers):
            layer_temp = self.layer_list[i]
            x_dict = layer_temp(x_dict)
        return x_dict['h']

    def gen_logit_SBD(self, x_dict):
        lambda_SBD = self.gen_lambda_SBD(x_dict)
        logit_SBD  = self.expansion_layer(lambda_SBD)
        return logit_SBD

    def gen_p_SBD(self, x_dict):
        logit_SBD = self.gen_logit_SBD(x_dict)
        self.softmax_fn = torch.nn.Softmax(dim=2)
        p_SBD = self.softmax_fn(logit_SBD)
        return p_SBD

    def forward(self, x_dict):
        logit_SBD = self.gen_logit_SBD(x_dict)
        # Choose which logit to use for cross-entropy
        if self.train_by_window:
            # Flatten to 2D for cross-entropy
            logit_SB_D = logit_SBD.view(-1, self.output_dim)
            return logit_SB_D
        else:
            # Average over B
            logit_S_D = torch.mean(logit_SBD, dim=1, keepdim=False)
            return logit_S_D

    def lambda_to_logits_SBD(self, x_dict):
        ''' lambda_S_B_D to indices_S_B '''
        if 'lambda' in x_dict:
            logit_SBD = self.expansion_layer(x_dict['lambda'])
        elif 'h' in x_dict:
            logit_SBD = self.expansion_layer(x_dict['h'])
        else:
            print('No valid key found in x_dict, expect lambda or h !')
            raise
        return logit_SBD

class DV_Y_CMP_NN_model(DV_Y_NN_model):
    def __init__(self, dv_y_cfg):
        super().__init__(dv_y_cfg, input_layer=Build_S_B_TD_Input_Layer)

class DV_Y_SubWin_NN_model(DV_Y_NN_model):
    def __init__(self, dv_y_cfg):
        super().__init__(dv_y_cfg, input_layer=Build_S_B_M_T_Input_Layer)

class DV_Y_ST_NN_model(DV_Y_NN_model):
    def __init__(self, dv_y_cfg):
        super().__init__(dv_y_cfg, input_layer=Build_S_T_Input_Layer)
        
##############################################
# Model Wrappers, between Python and PyTorch #
##############################################

class General_Model(object):

    ###################
    # THings to build #
    ###################

    def __init__(self):
        self.nn_model = None

    def build_optimiser(self):
        pass

    def gen_loss(self, feed_dict):
        pass

    def gen_lambda_SBD(self, feed_dict):
        pass

    def cal_accuracy(self, feed_dict):
        pass

    def numpy_to_tensor(self, feed_dict):
        pass

    ###################
    # Things can stay #
    ###################

    def __call__(self, x):
        ''' Simulate PyTorch forward() method '''
        ''' Note that x could be feed_dict '''
        return self.nn_model(x)

    def eval(self):
        ''' Simulate PyTorch eval() method, change to eval mode '''
        self.nn_model.eval()

    def train(self):
        ''' Simulate PyTorch train() method, change to train mode '''
        self.nn_model.train()

    def to_device(self, device_id):
        self.device_id = device_id
        self.nn_model.to(device_id)

    def no_grad(self):
        ''' Simulate PyTorch no_grad() method, change to eval mode '''
        return torch.no_grad()

    def DataParallel(self):
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        self.nn_model = torch.nn.DataParallel(self.nn_model)

    def print_model_parameters(self, logger):
        logger.info('Print Parameter Sizes')
        size = 0
        for name, param in self.nn_model.named_parameters():
            print(str(name)+'  '+str(param.size())+'  '+str(param.type()))
            s = 1
            for n in param.size():
                s *= n
            size += s
        logger.info("Total model size is %i" % size)
        
    def detect_nan_model_parameters(self, logger):
        logger.info('Detect NaN in Parameters')
        detect_bool = False
        for name, param in self.nn_model.named_parameters():
            # Detect Inf
            if not torch.isfinite(param).all():
                detect_bool = True
                print(str(name)+' contains Inf')
                print(torch.isfinite(param))
                # Detect NaN; NaN is not finite in Torch
                if torch.isnan(param).any():
                    print(str(name)+' contains NaN')
                    print(torch.isnan(param))
        if not detect_bool:
            logger.info('No Inf or NaN found in Parameters')

    def update_parameters(self, feed_dict):
        self.loss = self.gen_loss(feed_dict)
        # perform a backward pass, and update the weights.
        self.loss.backward()
        self.optimiser.step()

    def update_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate
        # Re-build an optimiser, use new learning rate, and reset gradients
        self.build_optimiser()

    def gen_loss_value(self, feed_dict):
        ''' Return the numpy value of self.loss '''
        self.loss = self.gen_loss(feed_dict)
        return self.loss.item()

    def gen_SB_loss_value(self, feed_dict):
        ''' Return the numpy value of self.loss in SB form '''
        self.SB_loss = self.gen_SB_loss(feed_dict)
        return self.SB_loss.cpu().detach().numpy()

    def save_nn_model(self, nnets_file_name):
        ''' Model Only '''
        save_dict = {'model_state_dict': self.nn_model.state_dict()}
        torch.save(save_dict, nnets_file_name)

    def load_nn_model(self, nnets_file_name):
        ''' Model Only '''
        checkpoint = torch.load(nnets_file_name)
        self.nn_model.load_state_dict(checkpoint['model_state_dict'])

    def save_nn_model_optim(self, nnets_file_name):
        ''' Model and Optimiser '''
        save_dict = {'model_state_dict': self.nn_model.state_dict(), 'optimiser_state_dict': self.optimiser.state_dict()}
        torch.save(save_dict, nnets_file_name)

    def load_nn_model_optim(self, nnets_file_name):
        ''' Model and Optimiser '''
        checkpoint = torch.load(nnets_file_name)
        self.nn_model.load_state_dict(checkpoint['model_state_dict'])
        self.optimiser.load_state_dict(checkpoint['optimiser_state_dict'])

    def count_parameters(self):
        return sum(p.numel() for p in self.nn_model.parameters() if p.requires_grad)

class DV_Y_model(General_Model):
    ''' Acoustic data --> Speaker Code --> Classification '''
    def __init__(self, dv_y_cfg):
        super().__init__()
        self.learning_rate = dv_y_cfg.learning_rate
        if dv_y_cfg.use_voiced_only:
            ''' Use vuv for weighted mean '''
            self.criterion = torch.nn.CrossEntropyLoss(reduction='none')
            self.gen_loss = self.gen_loss_vuv_weight
        else:
            self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')
            self.gen_loss = self.gen_loss_no_weight

    def build_optimiser(self):
        self.optimiser = torch.optim.Adam(self.nn_model.parameters(), lr=self.learning_rate)
        # Zero gradients
        self.optimiser.zero_grad()

    def gen_loss_vuv_weight(self, feed_dict):
        ''' Use vuv as weight; weighted sum then normalise by sum of weights '''
        ''' Returns Tensor, not value! For value, use gen_loss_value '''
        x_dict, y = self.numpy_to_tensor(feed_dict)
        y_pred = self.nn_model(x_dict) # This is either logit_SB_D or logit_S_D, 2D matrix
        vuv_SB_weight = x_dict['vuv_SB']
        # Compute and print loss
        self.SB_loss = self.criterion(y_pred, y)
        self.loss = torch.sum(self.SB_loss * vuv_SB_weight) / torch.sum(vuv_SB_weight)
        return self.loss

    def gen_loss_no_weight(self, feed_dict):
        ''' Returns Tensor, not value! For value, use gen_loss_value '''
        x_dict, y = self.numpy_to_tensor(feed_dict)
        y_pred = self.nn_model(x_dict) # This is either logit_SB_D or logit_S_D, 2D matrix
        # Compute and print loss
        self.loss = self.criterion(y_pred, y)
        return self.loss

    def gen_SB_loss(self, feed_dict):
        ''' Returns Tensor, not value! For value, use gen_loss_value '''
        x_dict, y = self.numpy_to_tensor(feed_dict)
        y_pred = self.nn_model(x_dict) # This is either logit_SB_D or logit_S_D, 2D matrix
        # Compute and print loss
        self.SB_criterion = torch.nn.CrossEntropyLoss(reduction='none')
        self.SB_loss = self.SB_criterion(y_pred, y)
        return self.SB_loss

    def gen_lambda_SBD_value(self, feed_dict):
        x_dict, y = self.numpy_to_tensor(feed_dict)
        self.lambda_SBD = self.nn_model.gen_lambda_SBD(x_dict)
        return self.lambda_SBD.cpu().detach().numpy()

    def gen_logit_SBD_value(self, feed_dict):
        x_dict, y = self.numpy_to_tensor(feed_dict)
        self.logit_SBD = self.nn_model.gen_logit_SBD(x_dict)
        return self.logit_SBD.cpu().detach().numpy()

    def gen_p_SBD_value(self, feed_dict):
        x_dict, y = self.numpy_to_tensor(feed_dict)
        self.p_SBD = self.nn_model.gen_p_SBD(x_dict)
        return self.p_SBD.cpu().detach().numpy()

    def lambda_to_indices(self, feed_dict):
        ''' lambda_S_B_D to indices_S_B '''
        x_dict, _y = self.numpy_to_tensor(feed_dict) # x_dict['lambda'] lambda_S_B_D! _y is useless
        logit_SBD  = self.nn_model.lambda_to_logits_SBD(x_dict)
        _values, predict_idx_list = torch.max(logit_SBD.data, -1)
        return predict_idx_list.cpu().detach().numpy()

    def cal_accuracy(self, feed_dict):
        x_dict, y = self.numpy_to_tensor(feed_dict)
        outputs = self.nn_model(x_dict)
        _values, predict_idx_list = torch.max(outputs.data, 1)
        total = y.size(0)
        correct = (predict_idx_list == y).sum().item()
        accuracy = correct/total
        return correct, total, accuracy

    def gen_all_h_values(self, feed_dict):
        x_dict, y = self.numpy_to_tensor(feed_dict)
        h_list = []
        for nn_layer in self.nn_model.layer_list:
            x_dict = nn_layer(x_dict)
            if 'h_stack' in x_dict:
                h = x_dict['h_stack']
            else:
                h = x_dict['h']
            h_list.append(h.cpu().detach().numpy())
        logit_SBD = self.nn_model.lambda_to_logits_SBD(x_dict)
        h_list.append(logit_SBD.cpu().detach().numpy())
        return h_list

    def numpy_to_tensor(self, feed_dict):
        x_dict = {}
        y = None
        for k in feed_dict:
            k_val = feed_dict[k]
            if k == 'y':
                # 'y' is the one-hot speaker class
                # High precision for cross-entropy function
                k_dtype = torch.long
                y = torch.tensor(k_val, dtype=k_dtype, device=self.device_id)
            else:
                k_dtype = torch.float
            k_tensor = torch.tensor(k_val, dtype=k_dtype, device=self.device_id)
            x_dict[k] = k_tensor
        return x_dict, y

class DV_Y_CMP_model(DV_Y_model):
    ''' S_B_D input, SB_D logit output, classification, cross-entropy '''
    def __init__(self, dv_y_cfg):
        super().__init__(dv_y_cfg)
        self.nn_model = DV_Y_CMP_NN_model(dv_y_cfg)

class DV_Y_Wav_SubWin_model(DV_Y_model):
    ''' S_B_M_T input, SB_D logit output, classification, cross-entropy '''
    def __init__(self, dv_y_cfg):
        super().__init__(dv_y_cfg)
        self.nn_model = DV_Y_SubWin_NN_model(dv_y_cfg)

class DV_Y_Wav_SubWin_Sinenet_model(DV_Y_Wav_SubWin_model):
    ''' S_B_M_T input, SB_D logit output, classification, cross-entropy '''
    def __init__(self, dv_y_cfg):
        super().__init__(dv_y_cfg)

class DV_Y_ST_model(DV_Y_model):
    ''' S_T input, SB_D logit output, classification, cross-entropy '''
    def __init__(self, dv_y_cfg):
        super().__init__(dv_y_cfg)
        self.nn_model = DV_Y_ST_NN_model(dv_y_cfg)

    # TODO: move to the sinenet model later!!
    def gen_w_mul_w_sin_cos(self, feed_dict):
        x_dict, y = self.numpy_to_tensor(feed_dict)
        w = self.nn_model.layer_list[0].layer_fn.sinenet_layer.return_w_mul_w_sin_cos_from_x(x_dict)
        return w.cpu().detach().numpy()

    def gen_w_sin_cos(self, feed_dict):
        x_dict, y = self.numpy_to_tensor(feed_dict)
        s_c = self.nn_model.layer_list[0].layer_fn.sinenet_layer.return_w_sin_cos_from_x(x_dict)
        return s_c.cpu().detach().numpy()





def torch_initialisation(dv_y_cfg):
    logger = make_logger("torch initialisation")
    if dv_y_cfg.gpu_id == 'cpu':
        logger.info('Using CPU')
        device_id = torch.device("cpu")
    elif torch.cuda.is_available():
    # if False:
        logger.info('Using GPU cuda:%i' % dv_y_cfg.gpu_id)
        device_id = torch.device("cuda:%i" % dv_y_cfg.gpu_id)
    else:
        logger.info('Using CPU; No GPU')
        device_id = torch.device("cpu")

    dv_y_model_class = dv_y_cfg.dv_y_model_class
    model = dv_y_model_class(dv_y_cfg)
    model.to_device(device_id)
    # if torch.cuda.device_count() > 1:
    #     model.DataParallel()
    return model

#############################
# PyTorch-based Simple Test #
#############################

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
            loss = dv_y_model.gen_loss_value(feed_dict)
            logger.info('%i, %f' % (t, loss))



###########################
# Useless Tensorflow Code #
#   Remember to clean up  #
###########################

class build_tf_am_model(object):

    def __init__(self, am_cfg, dv_tensor=None):
        # am_input_dim = am_cfg.iv_dim + am_cfg.input_dim

        with tf.device('/device:GPU:'+str(am_cfg.gpu_id)):
            # This is mandatory for now; reshape shouldn't be too hard
            input_dim_seq      = ['T', 'S', 'D']
            input_dim_values   = {'S':am_cfg.batch_num_spk, 'T':am_cfg.batch_seq_len, 'D':am_cfg.x_feat_dim}
            input_shape_values = [input_dim_values['T'], input_dim_values['S'], input_dim_values['D']]
            self.am_x      = tf.placeholder(tf.float32, shape=input_shape_values)

            dv_dim_seq      = ['T', 'S', 'D']
            dv_dim_values   = {'S':am_cfg.batch_num_spk, 'T':am_cfg.batch_seq_len, 'D':am_cfg.dv_dim}
            dv_shape_values = [dv_dim_values['T'], dv_dim_values['S'], dv_dim_values['D']]
            if dv_tensor is None:
                self.am_dv  = tf.placeholder(tf.float32, shape=dv_shape_values)
            else:
                self.am_dv  = dv_tensor

            output_dim_seq      = ['T', 'S', 'D']
            output_dim_values   = {'S':am_cfg.batch_num_spk, 'T':am_cfg.batch_seq_len, 'D':am_cfg.y_feat_dim}
            output_shape_values = [output_dim_values['T'], output_dim_values['S'], output_dim_values['D']]
            self.am_y           = tf.placeholder(tf.float32, shape=output_shape_values)
            self.am_y_mask      = tf.placeholder(tf.float32, shape=output_shape_values)


            self.nn_layers   = []
            self.init_c_h    = []
            self.final_c_h   = []

            self.train_scope = []
            self.train_vars    = []
            self.learning_rate = am_cfg.learning_rate
            self.learning_rate_holder = tf.placeholder(dtype=tf.float32, name='am_learning_rate_holder')
            self.is_train_bool = tf.placeholder(tf.bool, name="is_train") # This controls the "mode" of the model, training or feed-forward; for drop-out and batch norm

            # Setting up for the first layer
            layer_input = tf.concat([self.am_x, self.am_dv], axis=2)
            input_dim_values['D'] += am_cfg.dv_dim
            prev_layer = None

            # Start of hidden layers
            for i in range(am_cfg.num_nn_layers):
                scope_name = am_cfg.tf_scope_name + '/am_layer_'+str(i)
                self.train_scope.append(scope_name)
                layer_config = am_cfg.nn_layer_config_list[i]
                self.nn_layers.append(build_nn_layer(self, am_cfg, scope_name, layer_input, layer_config, input_dim_seq, input_dim_values, prev_layer))

                # Setting up for the next layer
                prev_layer       = self.nn_layers[-1]
                input_dim_seq    = prev_layer.output_dim_seq
                input_dim_values = prev_layer.output_dim_values
                if am_cfg.dv_connect_layers == 'input':
                    layer_input = prev_layer.layer_output
                elif am_cfg.dv_connect_layers == 'all':
                    layer_input = tf.concat([prev_layer.layer_output, self.am_dv], axis=2)
                    input_dim_values['D'] += am_cfg.dv_dim

                # Setting up RNN/LSTM related tensors
                try:    
                    if prev_layer.contain_c:
                        self.init_c_h.append(prev_layer.rnn_init_c)
                        self.final_c_h.append(prev_layer.rnn_final_c)
                except: prev_layer.contain_c = False
                try:    
                    if prev_layer.contain_h:
                        self.init_c_h.append(prev_layer.rnn_init_h)
                        self.final_c_h.append(prev_layer.rnn_final_h)
                except: prev_layer.contain_h = False
            # End of hidden layers
                        
            with tf.variable_scope('am_final_layer'):
                self.final_layer_output = tf.contrib.layers.fully_connected(inputs=layer_input, num_outputs=am_cfg.y_feat_dim, activation_fn=None)
                self.loss = tf.losses.mean_squared_error(labels=self.am_y, predictions=self.final_layer_output, weights=self.am_y_mask)
                self.train_loss = self.loss

            scope_name = am_cfg.tf_scope_name + '/am_optimiser'
            with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
                for i in self.train_scope:
                    vars_i = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, i)
                    self.train_vars.extend(vars_i)
            self.train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate_holder,epsilon=1.e-03).minimize(self.train_loss, var_list=self.train_vars)

            # init = tf.initialize_all_variables()
            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()           

    def update_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate
        # self.train_step  = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)



'''
# Sinenet Maths Dimension test
import torch
S = 1
B = 20
T = 30
D = 40
f = torch.randn(S,B,1,1)
tau = torch.randn(S,B,1,1)
kt = torch.randn(1,T)
t = torch.add(kt, torch.neg(tau))
i_2pi_tensor = torch.randn(D,1)
deg = torch.mul(i_2pi_tensor, t)
deg = torch.mul(f, deg)
phi = torch.randn(D,1)
deg = torch.add(deg, phi)
s = torch.sin(deg)
a = torch.randn(D,1)
W = torch.mul(a, s)
x = torch.randn(S,B,1,T)
h_SBDT = torch.mul(W, x)
h_SBD  = torch.sum(h_SBDT, dim=-1, keepdim=False)
'''