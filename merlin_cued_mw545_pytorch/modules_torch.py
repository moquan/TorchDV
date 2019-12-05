# modules_torch.py

import os, sys, pickle, time, shutil, logging, copy
import math, numpy, scipy
numpy.random.seed(545)
import torch
torch.manual_seed(545)

from modules import make_logger

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

    def ReLUDV(self):
        self.params["expect_input_dim_seq"] = ['S','B','D']
        self.reshape_fn = Tensor_Reshape(self.params)
        self.params = self.reshape_fn.update_layer_params()

        self.params["output_dim_seq"]       = ['S', 'B', 'D']
        v = self.params["expect_input_dim_values"]
        self.params["output_dim_values"]    = {'S': v['S'], 'B': v['B'], 'D': self.params["size"]}

        input_dim  = self.params['expect_input_dim_values']['D']
        output_dim = self.params['output_dim_values']['D']
        batch_norm = self.params["layer_config"]["batch_norm"]
        self.layer_fn = ReLUDVLayer(input_dim, output_dim, batch_norm)

    def LReLUDV(self):
        self.params["expect_input_dim_seq"] = ['S','B','D']
        self.reshape_fn = Tensor_Reshape(self.params)
        self.params = self.reshape_fn.update_layer_params()

        self.params["output_dim_seq"]       = ['S', 'B', 'D']
        v = self.params["expect_input_dim_values"]
        self.params["output_dim_values"]    = {'S': v['S'], 'B': v['B'], 'D': self.params["size"]}

        input_dim  = self.params['expect_input_dim_values']['D']
        output_dim = self.params['output_dim_values']['D']
        self.layer_fn = LReLUDVLayer(input_dim, output_dim)

    def ReLUDVMax(self):
        self.params["expect_input_dim_seq"] = ['S','B','D']
        self.reshape_fn = Tensor_Reshape(self.params)
        self.params = self.reshape_fn.update_layer_params()

        self.params["output_dim_seq"]       = ['S', 'B', 'D']
        v = self.params["expect_input_dim_values"]
        self.params["output_dim_values"]    = {'S': v['S'], 'B': v['B'], 'D': self.params["size"]}

        input_dim  = self.params['expect_input_dim_values']['D']
        output_dim = self.params['output_dim_values']['D']
        num_channels = self.params["layer_config"]["num_channels"]
        self.layer_fn = ReLUDVMaxLayer(input_dim, output_dim, num_channels)

    def LReLUDVMax(self):
        self.params["expect_input_dim_seq"] = ['S','B','D']
        self.reshape_fn = Tensor_Reshape(self.params)
        self.params = self.reshape_fn.update_layer_params()

        self.params["output_dim_seq"]       = ['S', 'B', 'D']
        v = self.params["expect_input_dim_values"]
        self.params["output_dim_values"]    = {'S': v['S'], 'B': v['B'], 'D': self.params["size"]}

        input_dim  = self.params['expect_input_dim_values']['D']
        output_dim = self.params['output_dim_values']['D']
        num_channels = self.params["layer_config"]["num_channels"]
        self.layer_fn = LReLUDVMaxLayer(input_dim, output_dim, num_channels)

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

    def SinenetV3(self):
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
        win_len    = self.params["layer_config"]["win_len"]
        win_shift  = self.params["layer_config"]["win_shift"]

        self.layer_fn = SinenetLayerV3(time_len, output_dim, num_channels, win_len, win_shift)
        self.params["output_dim_values"]['D'] *= self.layer_fn.num_wins # Multiple windows
        self.params["output_dim_values"]['D'] += 1 # +1 to append nlf F0 values

class ReLUDVLayer(torch.nn.Module):
    def __init__(self, input_dim, output_dim, batch_norm):
        super().__init__()
        self.input_dim  = input_dim
        self.output_dim = output_dim
        self.batch_norm = batch_norm

        self.fc_fn   = torch.nn.Linear(input_dim, output_dim)
        self.relu_fn = torch.nn.ReLU()

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
            # Reshape S*B*D to S*D*B, to norm D
            h_i_SDB = torch.transpose(h_i, 1,2)
            h_i_SDB = self.bn_fn(h_i_SDB)
            # Reshape back to S*B*D
            h_i_SBD = torch.transpose(h_i_SDB, 1,2)
            h_i = h_i_SBD
        # ReLU
        h = self.relu_fn(h_i)
        y_dict = {'h': h}
        return y_dict

class LReLUDVLayer(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim    = input_dim
        self.output_dim   = output_dim

        self.fc_fn    = torch.nn.Linear(input_dim, output_dim)
        self.lrelu_fn = torch.nn.LeakyReLU()

    def forward(self, x_dict):
        if 'h_reshape' in x_dict:
            x = x_dict['h_reshape']
        elif 'x' in x_dict:
            x = x_dict['x']
        # Linear
        h_i = self.fc_fn(x)
        # ReLU
        h = self.lrelu_fn(h_i)
        y_dict = {'h': h}
        return y_dict

class ReLUDVMaxLayer(torch.nn.Module):
    def __init__(self, input_dim, output_dim, num_channels):
        super().__init__()
        self.input_dim    = input_dim
        self.output_dim   = output_dim
        self.num_channels = num_channels

        self.fc_list = torch.nn.ModuleList([torch.nn.Linear(input_dim, output_dim) for i in range(self.num_channels)])
        self.relu_fn = torch.nn.ReLU()

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
            h_i = self.relu_fn(h_i)
            h_list.append(h_i)

        h_stack = torch.stack(h_list, dim=0)
        # MaxOut
        h_max, _indices = torch.max(h_stack, dim=0, keepdim=False)
        y_dict = {'h': h_max, 'h_stack':h_stack}
        return y_dict

class LReLUDVMaxLayer(torch.nn.Module):
    ''' Leaky ReLU, then MaxOut '''
    def __init__(self, input_dim, output_dim, num_channels):
        super().__init__()
        self.input_dim    = input_dim
        self.output_dim   = output_dim
        self.num_channels = num_channels

        self.fc_list = torch.nn.ModuleList([torch.nn.Linear(input_dim, output_dim) for i in range(self.num_channels)])
        self.lrelu_fn = torch.nn.LeakyReLU()

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
            h_i = self.lrelu_fn(h_i)
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

        self.log_f_mean = 5.02654
        self.log_f_std  = 0.373288

        self.i_2pi_tensor = self.make_i_2pi_tensor()   # D*1
        self.k_T_tensor   = self.make_k_T_tensor_t_1() # 1*T

        a_init_value   = numpy.random.normal(loc=0.1, scale=0.1, size=output_dim)
        phi_init_value = numpy.random.normal(loc=0.0, scale=1.0, size=output_dim)
        self.a   = torch.nn.Parameter(torch.tensor(a_init_value, dtype=torch.float), requires_grad=True) # D*1
        self.phi = torch.nn.Parameter(torch.tensor(phi_init_value, dtype=torch.float), requires_grad=True) # D

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
        y_dict = {'h': h_SBD}
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
    ''' 3 Parts: f-prediction, tau-prediction, sinenet_list '''
    ''' Apply sinenet on each sub-window within '''
    ''' nlf is shared, but tau_i is sub-window-specific '''
    def __init__(self, time_len, output_dim, num_channels, win_len, win_shift):
        super().__init__()
        self.time_len     = time_len
        self.output_dim   = output_dim   # Total output dimension
        self.num_channels = num_channels # Number of components per frequency
        self.num_freq     = int(output_dim / num_channels) # Number of frequency components

        self.win_len      = win_len
        self.win_shift    = win_shift
        self.num_wins     = int((time_len-self.win_len)/self.win_shift) + 1

        self.nlf_pred_layer = torch.nn.Linear(time_len, 1)
        # self.tau_pred_layer = torch.nn.Linear(time_len, 1)
        self.tau_layer_list = torch.nn.ModuleList([torch.nn.Linear(time_len, 1) for i in range(self.num_wins)])
        self.sinenet_layer  = SinenetLayer(self.win_len, output_dim, num_channels)

    def forward(self, x_dict):
        if 'h_reshape' in x_dict:
            x = x_dict['h_reshape']
        elif 'x' in x_dict:
            x = x_dict['x']
        nlf = self.nlf_pred_layer(x)
        # tau = self.tau_pred_layer(x)

        h_SBD_i_list = []
        for i in range(self.num_wins):
            x_i = x_dict['x_win_list'][i]
            tau_i = self.tau_layer_list[i](x)
            sine_dict_i = {'h_reshape':x_i, 'nlf':nlf, 'tau':tau_i}
            h_SBD_i = self.sinenet_layer(sine_dict_i)['h']
            h_SBD_i_list.append(h_SBD_i)
        h_SBD   = torch.cat(h_SBD_i_list, 2)
        nlf_SBD = torch.squeeze(nlf, 2)    # S*B*1*1 -> S*B*1
        h_SBD   = torch.cat((nlf_SBD, h_SBD), 2)
        y_dict = {'h': h_SBD}
        return y_dict

    def return_nlf_value(self, x_dict):
        if 'h_reshape' in x_dict:
            x = x_dict['h_reshape']
        elif 'x' in x_dict:
            x = x_dict['x']
        nlf = self.nlf_pred_layer(x)
        return nlf.data.cpu().detach().numpy()

    def return_tau_value(self, x_dict):
        if 'h_reshape' in x_dict:
            x = x_dict['h_reshape']
        elif 'x' in x_dict:
            x = x_dict['x']
        tau = self.tau_pred_layer(x)
        return tau.data.cpu().detach().numpy()

    def return_tau_list_value(self, x_dict):
        if 'h_reshape' in x_dict:
            x = x_dict['h_reshape']
        elif 'x' in x_dict:
            x = x_dict['x']
        tau = self.tau_pred_layer(x)

        tau_list = []
        for i in range(self.num_wins):
            tau_i = self.tau_layer_list[i](tau)
            tau_list.append(tau_i.data.cpu().detach().numpy())
        return tau_list

########################
# PyTorch-based Models #
########################

class DV_Y_CMP_NN_model(torch.nn.Module):
    ''' S_B_D input, SB_D/S_D logit output if train_by_window '''
    def __init__(self, dv_y_cfg):
        super().__init__()
        
        self.num_nn_layers = dv_y_cfg.num_nn_layers
        self.train_by_window = dv_y_cfg.train_by_window

        self.input_layer = Build_S_B_TD_Input_Layer(dv_y_cfg)
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
            print('No valid key found in x_dict, expect lambda or h!')
            raise
        return logit_SBD

class DV_Y_F0_Tau_NN_model(DV_Y_CMP_NN_model):
    ''' Don't know what to put in this model yet '''
    ''' Most functions can be shared, no need to change '''
    ''' for sinenet, input becomes x_list '''
    def __init__(self, dv_y_cfg):
        super().__init__(dv_y_cfg)
        


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

class DV_Y_CMP_model(General_Model):
    ''' S_B_D input, SB_D logit output, classification, cross-entropy '''
    def __init__(self, dv_y_cfg):
        super().__init__()
        self.nn_model = DV_Y_CMP_NN_model(dv_y_cfg)
        self.learning_rate = dv_y_cfg.learning_rate

    def build_optimiser(self):
        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        self.optimiser = torch.optim.Adam(self.nn_model.parameters(), lr=self.learning_rate)
        # Zero gradients
        self.optimiser.zero_grad()

    def gen_loss(self, feed_dict):
        ''' Returns Tensor, not value! For value, use gen_loss_value '''
        x_dict, y = self.numpy_to_tensor(feed_dict)
        y_pred = self.nn_model(x_dict)
        # TODO: Add dimension check
        # Compute and print loss
        self.loss = self.criterion(y_pred, y)
        return self.loss

    def gen_lambda_SBD_value(self, feed_dict):
        x_dict, y = self.numpy_to_tensor(feed_dict)
        self.lambda_SBD = self.nn_model.gen_lambda_SBD(x_dict)
        return self.lambda_SBD.cpu().detach().numpy()

    def gen_logit_SBD_value(self, feed_dict):
        x_dict, y = self.numpy_to_tensor(feed_dict)
        self.logit_SBD = self.nn_model.gen_logit_SBD(x_dict)
        return self.logit_SBD.cpu().detach().numpy()

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

class DV_Y_F0_Tau_model(DV_Y_CMP_model):
    ''' S_B_D input, SB_D logit output, classification, cross-entropy '''
    ''' Most functions can be shared, no need to change '''
    ''' for sinenet, x becomes x_list '''
    def __init__(self, dv_y_cfg):
        super().__init__(dv_y_cfg)

class DV_Y_Wav_SubWin_model(DV_Y_CMP_model):
    ''' S_B_D input, SB_D logit output, classification, cross-entropy '''
    ''' Main difference: x  '''
    ''' Most functions can be shared, no need to change '''
    def __init__(self, dv_y_cfg):
        super().__init__(dv_y_cfg)
        layer_1_config = dv_y_cfg.nn_layer_config_list[0]
        self.win_len   = layer_1_config['win_len']
        self.win_shift = layer_1_config['win_shift']
        self.num_wins  = int((dv_y_cfg.batch_seq_len-self.win_len)/self.win_shift) + 1

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
            if k == 'x':
                # S,B,T to smaller S,B,Tw
                x_win_list = self.make_x_win_list(k_val)
                x_dict['x_win_list'] = x_win_list
        return x_dict, y

    def make_x_win_list(self, x_val):
        # S,B,T to smaller S,B,Tw
        x_win_list = []
        for i in range(self.num_wins):
            t_start = i * self.win_shift
            t_end   = t_start + self.win_len
            x_win   = x_val[:, :, t_start:t_end]
            # Make S,B,T to S,B,1,T
            x_win   = numpy.expand_dims(x_win, axis=2)
            x_win_tensor = torch.tensor(x_win, dtype=torch.float, device=self.device_id)
            x_win_list.append(x_win_tensor)
        return x_win_list

    def gen_nlf_tau_values(self, feed_dict):
        x_dict, y = self.numpy_to_tensor(feed_dict)
        sinenet_layer = self.nn_model.layer_list[0].layer_fn
        nlf = sinenet_layer.return_nlf_value(x_dict)
        tau = sinenet_layer.return_tau_value(x_dict)
        tau_list = sinenet_layer.return_tau_list_value(x_dict)
        return nlf, tau, tau_list

    def gen_sinenet_h_value(self, feed_dict):
        x_dict, y = self.numpy_to_tensor(feed_dict)
        sinenet_layer = self.nn_model.layer_list[0]
        y_dict = sinenet_layer(x_dict)
        h = y_dict['h']
        return h.data.cpu().detach().numpy()




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