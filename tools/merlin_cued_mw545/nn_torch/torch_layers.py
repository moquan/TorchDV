# torch_layers.py

import os, sys, pickle, time, shutil, logging, copy
import math, numpy, scipy
numpy.random.seed(545)
import torch
torch.manual_seed(545)

from frontend_mw545.modules import make_logger, copy_dict

#####################
# PyTorch functions #
#####################

def batch_norm_D_tensor(input_tensor, bn_fn, D_index):
    # Move D_index to 1, to norm D
    if D_index == 1:
        h_SDB = input_tensor
    else:
        h_SDB = torch.transpose(input_tensor, 1, D_index)
    h_SDB_norm = bn_fn(h_SDB)
    # Reshape back, swap 1 and D_index again
    if D_index == 1:
        return h_SDB_norm
    else:
        h_SBD_norm = torch.transpose(h_SDB_norm, 1, D_index)
        if D_index > 2:
            h_SBD_norm = h_SBD_norm.contiguous()
        return h_SBD_norm

def compute_f_nlf(x_dict):
    log_f_mean = 5.04418
    log_f_std  = 0.358402
    if 'nlf_SBM' in x_dict:
        nlf = x_dict['nlf_SBM']
        lf = torch.add(torch.mul(nlf, log_f_std), log_f_mean) # S*B*M
        f  = torch.exp(lf)                                              # S*B*M
    elif 'f_SBM' in x_dict:
        f = x_dict['f_SBM']
        lf = torch.log(f)
        nlf = torch.mul(torch.add(lf, (-1)*log_f_mean), 1./log_f_std)
    return f, nlf

########################
# PyTorch-based Layers #
########################

class Build_DV_Y_Input_Layer(object):
    ''' This layer has only parameters, no torch.nn.module '''
    ''' Mainly for the prev_layer argument '''
    def __init__(self, dv_y_cfg):
        self.dv_y_cfg = dv_y_cfg
        if dv_y_cfg.y_feat_name == 'wav':
            self.init_wav(dv_y_cfg)
        elif dv_y_cfg.y_feat_name == 'cmp':
            self.init_cmp(dv_y_cfg)
        elif dv_y_cfg.y_feat_name == 'mfcc':
            self.init_mfcc(dv_y_cfg)

    def init_wav(self, dv_y_cfg):
        self.params = {}
        self.params["output_dim_seq"] = ['S', 'B', 'T']
        if 'wav_SBT' in dv_y_cfg.out_feat_list:
            T = dv_y_cfg.input_data_dim['T_B']
        # if 'wav_ST' in dv_y_cfg.out_feat_list:
        #     T = dv_y_cfg.input_data_dim['T_S']
        # elif 'wav_SBMT' in dv_y_cfg.out_feat_list:
        #     T = dv_y_cfg.input_data_dim['T_M']
        self.params["output_dim_values"]   = {'S':dv_y_cfg.input_data_dim['S'], 'B':None, 'T':T, 'D':dv_y_cfg.input_data_dim['D']}
        # self.params["output_dim_values"]   = {'S':dv_y_cfg.input_data_dim['S'], 'B':None, 'M':dv_y_cfg.input_data_dim['M'], 'T':T}

    def init_cmp(self, dv_y_cfg):
        self.params = {}
        self.params["output_dim_seq"]      = ['S', 'B', 'D']
        self.params["output_dim_values"]   = {'S':dv_y_cfg.input_data_dim['S'], 'B':None, 'D':dv_y_cfg.input_data_dim['D']}

    def init_mfcc(self, dv_y_cfg):
        self.params = {}
        self.params["output_dim_seq"]      = ['S', 'B', 'D']
        self.params["output_dim_values"]   = {'S':dv_y_cfg.input_data_dim['S'], 'B':None, 'D':dv_y_cfg.input_data_dim['D']}

class Build_ATTEN_Input_Layer(object):
    ''' This layer has only parameters, no torch.nn.module '''
    ''' Mainly for the prev_layer argument '''
    def __init__(self, dv_attn_cfg):
        self.dv_attn_cfg = dv_attn_cfg
        if dv_attn_cfg.feat_name == 'lab':
            self.init_lab(dv_attn_cfg)

    def init_lab(self, dv_attn_cfg):
        self.params = {}
        self.params["output_dim_seq"]      = ['S', 'B', 'D']
        self.params["output_dim_values"]   = {'S':dv_attn_cfg.input_data_dim['S'], 'B':None, 'D':dv_attn_cfg.input_data_dim['D']}

class Build_SB1_Masked_Softmax_Layer(torch.nn.Module):
    '''
    Masked Soffmax Layer
    Input: S*B*1
    Mask: S*B
    Output: S*B*1
    '''
    def __init__(self, S):
        super().__init__()

    def forward(self, x_dict):
        if 'h' in x_dict:
            x = x_dict['h']
        elif 'x' in x_dict:
            x = x_dict['x']
        mask_SB1 = torch.unsqueeze(x_dict['output_mask_S_B'], 2)
        x_exp = torch.exp(x)
        x_exp_masked = torch.mul(mask_SB1, x_exp)
        x_sum_S_1_1 = torch.sum(x_exp_masked, dim=1, keepdim=True)
        y_norm = torch.true_divide(x_exp_masked, x_sum_S_1_1)

        y_dict = copy_dict(x_dict, except_List=['h','x', 'output_mask_S_B'])
        y_dict['h'] = y_norm
        return y_dict





class Build_FC_Layer(torch.nn.Module):
    """
    Fully-connected layer
    Operation to the last dimension
    Output last dimension is 'D'
    1. Linear transform
    2. Batch Norm or Layer Norm, if needed
    3. Activation function e.g. ReLU, LReLU; None for linear layer
    """
    def __init__(self, params):
        super().__init__()
        self.params = params
        layer_config = self.params["layer_config"]

        D_name = self.params["input_dim_seq"][-1]
        assert D_name == 'D'
        D_in   = self.params["input_dim_values"][D_name]
        D_out  = layer_config['size']

        self.params["output_dim_seq"] = self.params["input_dim_seq"]
        self.params["output_dim_seq"][-1] = 'D'
        self.params["output_dim_values"] = copy.deepcopy(self.params["input_dim_values"])
        self.params["output_dim_values"]['D'] = D_out

        self.linear_fn = torch.nn.Linear(D_in, D_out)

        self.batch_norm = layer_config["batch_norm"]
        if self.batch_norm:
            self.D_index = len(self.params["input_dim_seq"]) - 1
            self.bn_fn = torch.nn.BatchNorm1d(D_out)

        self.layer_norm = layer_config["layer_norm"]
        if self.layer_norm:
            self.ln_fn = torch.nn.LayerNorm(D_out)

        self.activation_fn = self.params['activation_fn']

    def forward(self, x_dict):
        if 'h' in x_dict:
            x = x_dict['h']
        elif 'x' in x_dict:
            x = x_dict['x']
        # Linear
        h_i = self.linear_fn(x)
        # Batch Norm
        if self.batch_norm:
            h_i = batch_norm_D_tensor(h_i, self.bn_fn, D_index=self.D_index) # Batch Norm on last index
        # Layer Norm
        if self.layer_norm:
            h_i = self.ln_fn(h_i)
        # Activation
        if self.activation_fn is None:
            h = h_i
        else:
            h = self.activation_fn(h_i)
        y_dict = {'h': h}
        if 'in_lens' in x_dict:
            y_dict['in_lens'] = x_dict['in_lens']
        return y_dict

        
class Build_Tensor_Reshape(torch.nn.Module):
    """
    Very specific reshape methods
    Each method requires 1 definition, 1 forward function
    """
    def __init__(self, params):
        super().__init__()
        self.params = params

        construct_layer = getattr(self, self.params["layer_config"]["io_name"])
        construct_layer()

    def forward(self, x_dict):
        ''' 
        To be defined in each function
        '''
        pass

    def compute_num_seq(self, T_total, T_win, shift_win):
        num_seq = int((T_total - T_win) / shift_win) + 1
        return num_seq

    def convert_f_2_nlf(self, f):
        neg_log_f_mean = -5.04418
        inv_log_f_std  = 2.79016 # 1./0.358402
        lf  = torch.log(f)
        nlf = torch.mul(torch.add(lf, neg_log_f_mean), inv_log_f_std)
        return nlf

    def copy_dict(self, x_dict, except_List=[]):
        '''
        Copy every key-value pair to the new dict, except keys in the list
        '''
        return copy_dict(x_dict, except_List)

    def wav_SBT_2_wav_SBMT(self):
        '''
        Compute SBMT from SBT
        '''
        win_len_shift = self.params["layer_config"]["win_len_shift"]
        S = self.params["input_dim_values"]['S']
        B = self.params["input_dim_values"]['B']
        T = self.params["input_dim_values"]['T']
        M = self.compute_num_seq(T, win_len_shift[0], win_len_shift[1])

        self.params["output_dim_seq"] = ['S', 'B', 'M', 'T']
        self.params['output_dim_values'] = {'S': S, 'B': B, 'M': M, 'T': win_len_shift[0]}

        self.forward = self.wav_SBT_2_wav_SBMT_fn

    def wav_SBT_2_wav_SBMT_fn(self, x_dict):
        '''
        Unfold wav once; S*B*T --> S*B*M*T
        '''
        wav = x_dict['wav_SBT']
        win_len_shift = self.params["layer_config"]["win_len_shift"]
        win_len, win_shift = win_len_shift
        wav = wav.unfold(2, win_len, win_shift)
        y_dict = self.copy_dict(x_dict, except_List=['wav_SBT'])
        y_dict['wav_SBMT'] = wav
        return y_dict

    def h_SBD_2_wav_SBT(self):
        self.params["output_dim_seq"] = ['S', 'B', 'T']
        self.params['output_dim_values'] = self.params["input_dim_values"]
        self.forward = self.h_SBD_2_wav_SBT_fn

    def h_SBD_2_wav_SBT_fn(self, x_dict):
        y_dict = self.copy_dict(x_dict, except_List=['h'])
        y_dict['wav_SBT'] = x_dict['h']
        return y_dict

    def wav_SBT_2_wav_SB_T(self):
        '''
        Reshape 3D into 2D
        '''
        self.params["output_dim_seq"] = ['SB', 'T']
        self.params['output_dim_values'] = self.params["input_dim_values"]

        self.forward = self.wav_SBT_2_wav_SB_T_fn

    def wav_SBT_2_wav_SB_T_fn(self, x_dict):
        wav = x_dict['wav_SBT']
        T = self.params["input_dim_values"]['T']
        wav_SB_T = wav.view([-1, T])
        y_dict = self.copy_dict(x_dict, except_List=['wav_SBT'])
        y_dict['wav_SB_T'] = wav_SB_T
        return y_dict

    def concat_wav_nlf_tau_vuv(self):
        '''
        Concatenate wav_SBMT, nlf_SBM, tau_SBM, vuv_SBM
        Output: h_SBMD; D <-- T+3
        Convert f to nlf
        '''
        S = self.params["input_dim_values"]['S']
        B = self.params["input_dim_values"]['B']
        M = self.params["input_dim_values"]['M']
        T = self.params["input_dim_values"]['T']

        self.params["output_dim_seq"] = ['S', 'B', 'M', 'D']
        self.params['output_dim_values'] = {'S': S, 'B': B, 'M': M, 'D': T+3}

        self.forward = self.concat_wav_nlf_tau_vuv_fn

    def concat_wav_nlf_tau_vuv_fn(self, x_dict):
        '''
        Concatenate wav_SBMT, nlf_SBM, tau_SBM
        Output: h_SBMD; D <-- T+3
        Convert f to nlf
        '''
        wav = x_dict['wav_SBMT']
        if 'nlf_SBM' in x_dict:
            nlf = x_dict['nlf_SBM']
        else:
            nlf = self.convert_f_2_nlf(x_dict['f_SBM'])
        tau = x_dict['tau_SBM']
        vuv = x_dict['vuv_SBM']

        nlf_1 = torch.unsqueeze(nlf, 3) # S*B*M --> # S*B*M*1
        tau_1 = torch.unsqueeze(tau, 3) # S*B*M --> # S*B*M*1
        vuv_1 = torch.unsqueeze(vuv, 3) # S*B*M --> # S*B*M*1

        h = torch.cat([wav, nlf_1, tau_1, vuv_1], 3)
        y_dict = {'h': h}
        return y_dict

    def h_SBMD_2_h_SBD(self):
        '''
        Reshape; D <-- M*D
        '''
        S = self.params["input_dim_values"]['S']
        B = self.params["input_dim_values"]['B']
        M = self.params["input_dim_values"]['M']
        D = self.params["input_dim_values"]['D']

        self.params["output_dim_seq"] = ['S', 'B', 'D']
        self.params['output_dim_values'] = {'S': S, 'B': B, 'D': M*D}

        self.forward = self.h_SBMD_2_h_SBD_fn

    def h_SBMD_2_h_SBD_fn(self, x_dict):
        '''
        h_SBMD --> h_SBD
        '''
        h_SBMD = x_dict['h']
        h_size = h_SBMD.size()
        h_SBD  = h_SBMD.view([h_size[0], h_size[1], -1])
        y_dict = {'h': h_SBD}
        return y_dict

    def h_SB_D_2_h_SBD(self):
        '''
        Reshape 2D into 3D
        '''
        self.params["output_dim_seq"] = ['S', 'B', 'D']
        self.params['output_dim_values'] = self.params["input_dim_values"]

        self.forward = self.h_SB_D_2_h_SBD_fn

    def h_SB_D_2_h_SBD_fn(self, x_dict):
        '''
        Reshape 2D into 3D
        '''
        h_SB_D = x_dict['h']
        if 'S_data' in x_dict:
            S = x_dict['S_data']
        else:
            S = self.params["input_dim_values"]['S']
        D = self.params["input_dim_values"]['D']
        h_SBD  = h_SB_D.view([S, -1, D])
        y_dict = self.copy_dict(x_dict, except_List=['h'])
        y_dict['h'] = h_SBD
        return y_dict

    def h_SBD_wav_feats_split(self):
        '''
        Split h_SBD into features
        possible values are ['wav_SBT', 'f_SBM', 'tau_SBM', 'vuv_SBM']
        '''
        self.params["output_dim_seq"] = ['S', 'B', 'T', 'M'] # This T is T_B
        self.params['output_dim_values'] = self.params["input_dim_values"]
        self.params['output_dim_values']['M'] = self.params["layer_config"]["input_data_dim"]['M']
        self.params['out_feat_list'] = self.params["layer_config"]["out_feat_list"]

        self.forward = self.h_SBD_wav_feats_split_fn

    def h_SBD_wav_feats_split_fn(self, x_dict):
        start_index = 0
        T = self.params['output_dim_values']['T']
        M = self.params['output_dim_values']['M']

        for feat_name in self.params['out_feat_list']:
            if feat_name == 'wav_SBT':
                wav_SBT = x_dict['h'][:,:,start_index:start_index+T]
                y_dict = {'wav_SBT': wav_SBT}
                start_index += T
            else:
                feat_tensor = x_dict['h'][:,:,start_index:start_index+M]
                y_dict[feat_name] = feat_tensor
                start_index += M

        return y_dict

class Build_NN_Layer(torch.nn.Module):
    def __init__(self, layer_config, prev_layer=None):
        super().__init__()
        self.params = {}
        self.params["layer_config"] = layer_config
        self.params["type"] = layer_config['type']
        self.set_param_default_values(layer_config)

        # Extract dimension information from previous layer, or specify in params
        if prev_layer is not None:
            self.params["input_dim_seq"]    = prev_layer.params["output_dim_seq"]
            self.params["input_dim_values"] = prev_layer.params["output_dim_values"]

        construct_layer = getattr(self, self.params["layer_config"]["type"])
        construct_layer()
        self.params   = self.layer_fn.params

        ''' Dropout '''
        if self.params["layer_config"]["dropout_p"] > 0:
            self.dropout_fn = torch.nn.Dropout(p=self.params["layer_config"]["dropout_p"])

    def set_param_default_values(self, layer_config):
        if 'dropout_p' not in layer_config:
            self.params["layer_config"]["dropout_p"] = 0.
        if 'batch_norm' not in layer_config:
            self.params["layer_config"]["batch_norm"] = False
        if 'layer_norm' not in layer_config:
            self.params["layer_config"]["layer_norm"] = False

    def forward(self, x_dict):
        y_dict = self.layer_fn(x_dict)
        if self.params["layer_config"]["dropout_p"] > 0:
            y_dict['h'] = self.dropout_fn(y_dict['h'])
        return y_dict

    def Linear(self, activation_fn=None):
        self.params['activation_fn'] = None
        self.layer_fn = Build_FC_Layer(self.params)

    def ReLU(self):
        self.params['activation_fn'] = torch.nn.ReLU()
        self.layer_fn = Build_FC_Layer(self.params)

    def LReLU(self):
        self.params['activation_fn'] = torch.nn.LeakyReLU()
        self.layer_fn = Build_FC_Layer(self.params)

    def Tensor_Reshape(self):
        self.layer_fn = Build_Tensor_Reshape(self.params)

    def DW3(self, activation_fn=torch.nn.LeakyReLU()):
        '''
        DNN: wav and 3 features
        '''
        self.params['activation_fn'] = activation_fn
        self.layer_fn = Build_DNN_wav_3_nlf_tau_vuv(self.params)

    def Sinenet_V0(self, activation_fn=torch.nn.LeakyReLU()):
        self.params['activation_fn'] = activation_fn
        self.layer_fn = Build_Sinenet_V0(self.params)

    def Sinenet_V1(self, activation_fn=torch.nn.LeakyReLU()):
        self.params['activation_fn'] = activation_fn
        self.layer_fn = Build_Sinenet_V1(self.params)

    def Sinenet_V2(self, activation_fn=torch.nn.LeakyReLU()):
        self.params['activation_fn'] = activation_fn
        self.layer_fn = Build_Sinenet_V2(self.params)

    def Sinenet_V1_Residual(self, activation_fn=torch.nn.LeakyReLU()):
        self.params['activation_fn'] = activation_fn
        self.layer_fn = Build_Sinenet_V1_Residual(self.params)

    def SincNet(self):
        self.layer_fn = Build_SincNet(self.params)

    def X_vector(self):
        self.layer_fn = Build_X_vector(self.params)        

class Build_SincNet(torch.nn.Module):
    ''' 
        Inputs: wav_SB_T
        Output: h: SB_D
        1. Use the imported SincNet layer
        2. Use imported configuration
    '''
    def __init__(self, params):
        super().__init__()
        self.params = params

        from nn_torch.sincnet_models import SincNet
        self.sincnet_config = self.make_sincnet_config()
        self.sincnet_fn = SincNet(self.sincnet_config)

        self.params["output_dim_seq"] = ['SB', 'D']
        self.params["output_dim_values"] = {'S': self.params["input_dim_values"]['S'], 'B': self.params["input_dim_values"]['B'], 'D': self.sincnet_fn.out_dim}

    def forward(self, x_dict):
        
        if 'wav_SB_T' in x_dict:
            x = x_dict['wav_SB_T']
        elif 'h' in x_dict:
            x = x_dict['h']

        h_SB_D = self.sincnet_fn(x)

        y_dict = copy_dict(x_dict, except_List=['h','wav_SB_T'])
        y_dict['h'] = h_SB_D
        return y_dict

    def make_sincnet_config(self):
        T = self.params["input_dim_values"]['T']
        D = self.params["layer_config"]['size']
        sincnet_config = {'input_dim': T,
          'fs': 24000,
          'cnn_N_filt': [80,D,D],
          'cnn_len_filt': [251,5,5],
          'cnn_max_pool_len':[3,3,3],
          'cnn_use_laynorm_inp': True,
          'cnn_use_batchnorm_inp': False,
          'cnn_use_laynorm':[True,True,True],
          'cnn_use_batchnorm':[False,False,False],
          'cnn_act': ['leaky_relu','leaky_relu','leaky_relu'],
          'cnn_drop':[0.0,0.0,0.0],          
        }
        return sincnet_config

class Build_X_vector(torch.nn.Module):
    def __init__(self, params):
        super(X_vector, self).__init__()
        self.params = params

        from nn_torch.x_vector import TDNN
        D_in  = self.params['input_dim_values']['D']
        D_out = self.params["layer_config"]['size']

        self.params["output_dim_seq"] = ['S','B', 'D']
        self.params["output_dim_values"] = {'S': self.params["input_dim_values"]['S'], 'B': None, 'D': D_out}

        self.tdnn1 = TDNN(input_dim=input_dim, output_dim=512, context_size=5, dilation=1,dropout_p=0.5)
        self.tdnn2 = TDNN(input_dim=512, output_dim=512, context_size=3, dilation=1,dropout_p=0.5)
        self.tdnn3 = TDNN(input_dim=512, output_dim=512, context_size=2, dilation=2,dropout_p=0.5)
        self.tdnn4 = TDNN(input_dim=512, output_dim=512, context_size=1, dilation=1,dropout_p=0.5)
        self.tdnn5 = TDNN(input_dim=512, output_dim=D_out, context_size=1, dilation=3,dropout_p=0.5)
        #### Frame levelPooling
        # self.segment6 = nn.Linear(1024, 512)
        # self.segment7 = nn.Linear(512, 512)
        # self.output = nn.Linear(512, num_classes)
        # self.softmax = nn.Softmax(dim=1)
    def forward(self, x_dict):
        if 'h' in x_dict:
            x = x_dict['h']
        elif 'x' in x_dict:
            x = x_dict['x']
        tdnn1_out = self.tdnn1(x)
        tdnn2_out = self.tdnn2(tdnn1_out)
        tdnn3_out = self.tdnn3(tdnn2_out)
        tdnn4_out = self.tdnn4(tdnn3_out)
        tdnn5_out = self.tdnn5(tdnn4_out)
        ### Stat Pool
        # mean = torch.mean(tdnn5_out,1)
        # std = torch.std(tdnn5_out,1)
        # stat_pooling = torch.cat((mean,std),1)
        # segment6_out = self.segment6(stat_pooling)
        # x_vec = self.segment7(segment6_out)
        # predictions = self.softmax(self.output(x_vec))
        # return predictions,x_vec
        y_dict = {'h': tdnn5_out}
        return y_dict

    def modify_x_dict(self, x_dict):
        # Modify lens and mask
        pass



class Build_DNN_wav_3_nlf_tau_vuv(torch.nn.Module):
    ''' 
        Inputs: wav_SBMT, f0_SBM, tau_SBM, vuv_SBM
        Output: h: S*B*M*D
        1. Use 2 separate Linear_fn for x, and nlf_SBM + tau_SBM + vuv_SBM
        2. Add the 2; batch_norm, activation_fn
    '''
    def __init__(self, params):
        super().__init__()
        self.params = params
        layer_config = self.params["layer_config"]
        D_out  = layer_config['size']

        self.params["output_dim_seq"] = ['S', 'B', 'M', 'D']
        self.params["output_dim_values"] = {'S': self.params["input_dim_values"]['S'], 'B': self.params["input_dim_values"]['B'], 'M': self.params["input_dim_values"]['M'], 'D': D_out}

        self.linear_fn_1 = torch.nn.Linear(self.params["input_dim_values"]['T'], D_out)
        self.linear_fn_2 = torch.nn.Linear(3, D_out)

        self.batch_norm = layer_config["batch_norm"]
        if self.batch_norm:
            self.D_index = len(self.params["input_dim_seq"]) - 1
            self.bn_fn = torch.nn.BatchNorm2d(D_out)

        self.layer_norm = layer_config["layer_norm"]
        if self.layer_norm:
            self.ln_fn = torch.nn.LayerNorm(D_out)

        self.activation_fn = self.params['activation_fn']

    def forward(self, x_dict):
        
        if 'wav_SBMT' in x_dict:
            x = x_dict['wav_SBMT']
        elif 'h' in x_dict:
            x = x_dict['h']
        
        f, nlf = compute_f_nlf(x_dict)
        tau = x_dict['tau_SBM']
        vuv = x_dict['vuv_SBM']
        
        y_SBMD_1  = self.linear_fn_1(x) # S*B*M*T -> S*B*M*D
        # nlf, tau, vuv
        nlf_1 = torch.unsqueeze(nlf, 3)
        tau_1 = torch.unsqueeze(tau, 3)
        vuv_1 = torch.unsqueeze(vuv, 3)
        nlf_tau_vuv = torch.cat([nlf_1, tau_1, vuv_1], 3)
        y_SBMD_2 = self.linear_fn_2(nlf_tau_vuv) # S*B*M*3 -> S*B*M*D

        y_SBMD = y_SBMD_1 + y_SBMD_2
        # Batch Norm
        if self.batch_norm:
            y_SBMD = batch_norm_D_tensor(y_SBMD, self.bn_fn, index_D=self.D_index)
        # Layer Norm
        if self.layer_norm:
            y_SBMD = self.ln_fn(y_SBMD)

        # ReLU
        h_SBMD = self.activation_fn(y_SBMD)

        y_dict = {'h': h_SBMD}
        return y_dict


################################
# SineNet component and layers #
################################

class Build_Sinenet(torch.nn.Module):
    ''' 
    Inputs: wav_SBMT, f0_SBM, tau_SBM
    Output: sin_cos_x: S*B*M*2K, K=num_freq
    (Optional) output: w_sin_cos_matrix
    '''
    def __init__(self, params):
        super().__init__()
        self.params = params

        self.num_freq = self.params["layer_config"]['num_freq']
        self.win_len  = self.params["input_dim_values"]['T']
        self.k_space = self.params["layer_config"]['k_space']
        self.k_train = self.params["layer_config"]['k_train']

        self.t_wav = 1./24000

        self.k_2pi_tensor = self.make_k_2pi_tensor(self.num_freq, self.k_space) # K
        self.n_T_tensor   = self.make_n_T_tensor(self.win_len, self.t_wav)   # T

    def forward(self, x, f, tau):
        sin_cos_matrix = self.construct_w_sin_cos_matrix(f, tau) # S*B*M*2K*T
        sin_cos_x = torch.einsum('sbmkt,sbmt->sbmk', sin_cos_matrix, x) 
        return sin_cos_x

    def make_k_2pi_tensor(self, num_freq, k_space):
        '''
        This is actually gamma_k in the paper
        indices of frequency components
        k_space between consecutive components
        '''
        # k_vec = numpy.zeros(num_freq)
        # for k in range(num_freq):
        #     k_vec[k] = k + 1
        k_vec = numpy.arange(num_freq)+1

        k_vec = k_vec * 2 * numpy.pi * k_space
        if self.k_train:
            k_vec_tensor = torch.tensor(k_vec, dtype=torch.float, requires_grad=True)
            k_vec_tensor = torch.nn.Parameter(k_vec_tensor, requires_grad=True)
        else:
            k_vec_tensor = torch.tensor(k_vec, dtype=torch.float, requires_grad=False)
            k_vec_tensor = torch.nn.Parameter(k_vec_tensor, requires_grad=False)
        return k_vec_tensor

    def make_n_T_tensor(self, win_len, t_wav):
        ''' indices along time '''
        n_T_vec = numpy.zeros(win_len)
        for n in range(win_len):
            n_T_vec[n] = float(n) * t_wav
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

    def construct_w_sin_cos_matrix(self, f, tau):
        deg = self.compute_deg(f, tau) # S*B*M*K*T
        s   = torch.sin(deg)             # S*B*M*K*T
        c   = torch.cos(deg)             # S*B*M*K*T
        s_c = torch.cat([s,c], dim=3)    # S*B*M*2K*T
        return s_c

class Build_Sinenet_V0(torch.nn.Module):
    ''' 
        Inputs: wav_SBMT, f0_SBM, tau_SBM
        Output: h: S*B*M*D
        1. Apply sinenet on each sub-window within
            1.1 f and tau may come from data, or model parameter
            1.2 f and tau cannot be both model parameters; use Sinenet_V2 instead
        2. Stack nlf_SBM, tau_SBM, vuv_SBM (if inc_a)
        3. Apply fc, add, batch_norm, relu
    '''
    def __init__(self, params):
        super().__init__()
        self.params = params
        layer_config = self.params["layer_config"]
        D_out = layer_config['size']

        self.params["output_dim_seq"] = ['S', 'B', 'M', 'D']
        self.num_freq = layer_config['num_freq']
        # Options; D: from data, f_SBM; P: model parameter
        self.use_f   = layer_config['use_f']
        self.use_tau = layer_config['use_tau']
        self.inc_a   = layer_config['inc_a']
        # assert layer_config['size'] == sine_size + 3
        # D_out = (layer_config['size']) * self.params["input_dim_values"]['M']
        self.params["output_dim_values"] = {'S': self.params["input_dim_values"]['S'], 'B': self.params["input_dim_values"]['B'], 'M': self.params["input_dim_values"]['M'], 'D': D_out}
        self.make_f_tau()

        self.sinenet_fn = Build_Sinenet(params)

        self.linear_fn_1 = torch.nn.Linear(self.num_freq*2, D_out)
        if self.inc_a:
            self.linear_fn_2 = torch.nn.Linear(3, D_out)

        self.activation_fn = self.params['activation_fn']

    def make_f_tau(self):
        # Make f and tau if they are model parameters
        if self.use_f == 'P':
            # gamma_k is already in sinenet_fn
            # feed f_bar of data
            self.f_init_value = 155.
            S = self.params["output_dim_values"]['S']
            B = self.params["output_dim_values"]['B']
            M = self.params["output_dim_values"]['M']
            f_value  = numpy.ones((S,B,M)) * self.f_init_value
            f_tensor = torch.tensor(f_value, dtype=torch.float, requires_grad=False)
            self.f_use = torch.nn.Parameter(f_tensor, requires_grad=False) # K

        if self.use_tau == 'P':
            # TODO: this is slightly tricky; data tau_SBM; param tau_K
            pass
            # tau_vec = numpy.zeros(self.num_freq)
            # tau_tensor = torch.tensor(tau_vec, dtype=torch.float, requires_grad=True)
            # tau_tensor = torch.nn.Parameter(tau_tensor, requires_grad=True) # K

    def read_f_tau(self, f, tau):
        # Use data or parameter f and tau
        if self.use_f == 'D':
            f_use = f
        else:
            f_use = self.f_use

        if self.use_tau == 'D':
            tau_use = tau
        else:
            # Expand from K to SBMK
            tau_use = self.tau_use

        return f_use, tau_use


    def forward(self, x_dict):
        
        if 'wav_SBMT' in x_dict:
            x = x_dict['wav_SBMT']
        elif 'h' in x_dict:
            x = x_dict['h']
        
        f, nlf = compute_f_nlf(x_dict)
        tau = x_dict['tau_SBM']
        vuv = x_dict['vuv_SBM']
        
        f_use, tau_use = self.read_f_tau(f, tau)
        # sinenet
        sin_cos_x = self.sinenet_fn(x, f_use, tau_use)  # S*B*M*2K
        y_SBMD = self.linear_fn_1(sin_cos_x) # S*B*M*2K -> S*B*M*D

        # nlf, tau, vuv
        if self.inc_a:
            nlf_1 = torch.unsqueeze(nlf, 3)
            tau_1 = torch.unsqueeze(tau, 3)
            vuv_1 = torch.unsqueeze(vuv, 3)
            nlf_tau_vuv = torch.cat([nlf_1, tau_1, vuv_1], 3)
            y_SBMD_2 = self.linear_fn_2(nlf_tau_vuv) # S*B*M*3 -> S*B*M*D

            y_SBMD = y_SBMD + y_SBMD_2

        # ReLU
        h_SBMD = self.activation_fn(y_SBMD)

        y_dict = {'h': h_SBMD}
        return y_dict

class Build_Sinenet_V1(torch.nn.Module):
    ''' 
        Inputs: wav_SBMT, f0_SBM, tau_SBM
        Output: h: S*B*M*D
        1. Apply sinenet on each sub-window within
            1.1 f and tau come from data
        2. Apply fc, add, batch_norm, relu
        3. Stack nlf_SBM, tau_SBM, vuv_SBM
    '''
    def __init__(self, params):
        super().__init__()
        self.params = params
        layer_config = self.params["layer_config"]
        D_out = layer_config['size']

        self.params["output_dim_seq"] = ['S', 'B', 'M', 'D']
        self.num_freq = layer_config['num_freq']
        self.params["output_dim_values"] = {'S': self.params["input_dim_values"]['S'], 'B': self.params["input_dim_values"]['B'], 'M': self.params["input_dim_values"]['M'], 'D': D_out+3}

        self.sinenet_fn = Build_Sinenet(params)
        self.linear_fn_1 = torch.nn.Linear(self.num_freq*2, D_out)
        self.activation_fn = self.params['activation_fn']


    def read_f_tau(self, f, tau):
        # Use data or parameter f and tau
        if self.use_f == 'D':
            f_use = f
        else:
            f_use = self.f_use

        if self.use_tau == 'D':
            tau_use = tau
        else:
            # Expand from K to SBMK
            tau_use = self.tau_use

        return f_use, tau_use


    def forward(self, x_dict):
        
        if 'wav_SBMT' in x_dict:
            x = x_dict['wav_SBMT']
        elif 'h' in x_dict:
            x = x_dict['h']
        
        f, nlf = compute_f_nlf(x_dict)
        tau = x_dict['tau_SBM']
        vuv = x_dict['vuv_SBM']
        
        # sinenet
        sin_cos_x = self.sinenet_fn(x, f, tau)  # S*B*M*2K
        y_SBMD = self.linear_fn_1(sin_cos_x) # S*B*M*2K -> S*B*M*D
        # ReLU
        y_SBMD_1 = self.activation_fn(y_SBMD)

        # nlf, tau, vuv
        nlf_1 = torch.unsqueeze(nlf, 3)
        tau_1 = torch.unsqueeze(tau, 3)
        vuv_1 = torch.unsqueeze(vuv, 3)
        h_SBMD = torch.cat([y_SBMD_1, nlf_1, tau_1, vuv_1], 3)

        y_dict = {'h': h_SBMD}
        return y_dict

class Build_Sinenet_V2(torch.nn.Module):
    ''' 
        Inputs: wav_SBMT, f0_SBM, tau_SBM
        Output: h: S*B*M*D
        1. Apply sinenet on each sub-window within
            1.1 f and tau come from data
        2. Apply fc, add, batch_norm, relu
        3. Stack nlf_SBM, tau_SBM, vuv_SBM
    '''
    def __init__(self, params):
        super().__init__()
        self.params = params
        layer_config = self.params["layer_config"]
        D_out = layer_config['size']

        self.params["output_dim_seq"] = ['S', 'B', 'M', 'D']
        self.num_freq = layer_config['num_freq']
        self.params["output_dim_values"] = {'S': self.params["input_dim_values"]['S'], 'B': self.params["input_dim_values"]['B'], 'M': self.params["input_dim_values"]['M'], 'D': D_out+3}

        self.sinenet_fn = Build_Sinenet(params)
        self.linear_fn_1 = torch.nn.Linear(self.num_freq*2, D_out)
        self.uv_net = self.build_uv_net(params, layer_config)
        self.activation_fn = self.params['activation_fn']

    def build_uv_net(self, params, layer_config):
        if layer_config['uv_net'] == 'DNN':
            D_in = self.params["input_dim_values"]['T']
            D_out = layer_config['size']
            return torch.nn.Linear(D_in, D_out)

    def forward(self, x_dict):
        
        if 'wav_SBMT' in x_dict:
            x = x_dict['wav_SBMT']
        elif 'h' in x_dict:
            x = x_dict['h']
        
        f, nlf = compute_f_nlf(x_dict)
        tau = x_dict['tau_SBM']
        vuv = x_dict['vuv_SBM']
        
        # sinenet
        sin_cos_x = self.sinenet_fn(x, f, tau)  # S*B*M*2K
        y_SBMD_v = self.linear_fn_1(sin_cos_x) # S*B*M*2K -> S*B*M*D

        # uv_net
        y_SBMD_uv = self.uv_net(x)

        # gating
        vuv_1 = torch.unsqueeze(vuv, 3)
        y_SBMD = torch.mul(vuv_1, y_SBMD_v) + torch.mul((1-vuv_1), y_SBMD_uv)

        # ReLU
        y_SBMD_1 = self.activation_fn(y_SBMD)

        # nlf, tau, vuv
        nlf_1 = torch.unsqueeze(nlf, 3)
        tau_1 = torch.unsqueeze(tau, 3)
        vuv_1 = torch.unsqueeze(vuv, 3)
        h_SBMD = torch.cat([y_SBMD_1, nlf_1, tau_1, vuv_1], 3)

        y_dict = {'h': h_SBMD}
        return y_dict

###########################
# SineNet old and useless #
###########################

class Build_Sinenet_V1_old(torch.nn.Module):
    ''' 
        Inputs: wav_SBMT, f0_SBM, tau_SBM
        Output: h: S*B*M*D
        1. Apply sinenet on each sub-window within
        2. Stack nlf_SBM, tau_SBM, vuv_SBM
        3. Apply 2 fc, add, batch_norm, relu
    '''
    def __init__(self, params):
        super().__init__()
        self.params = params
        layer_config = self.params["layer_config"]
        D_out = layer_config['size']

        self.params["output_dim_seq"] = ['S', 'B', 'M', 'D']
        num_freq  = layer_config['num_freq']
        # assert layer_config['size'] == sine_size + 3
        # D_out = (layer_config['size']) * self.params["input_dim_values"]['M']
        self.params["output_dim_values"] = {'S': self.params["input_dim_values"]['S'], 'B': self.params["input_dim_values"]['B'], 'M': self.params["input_dim_values"]['M'], 'D': D_out}

        self.sinenet_fn = Build_Sinenet(params)

        self.linear_fn_1 = torch.nn.Linear(num_freq*2, D_out)
        self.linear_fn_2 = torch.nn.Linear(3, D_out)

        self.batch_norm = layer_config["batch_norm"]

        self.activation_fn = self.params['activation_fn']

    def forward(self, x_dict):
        
        if 'wav_SBMT' in x_dict:
            x = x_dict['wav_SBMT']
        elif 'h' in x_dict:
            x = x_dict['h']
        
        f, nlf = compute_f_nlf(x_dict)
        tau = x_dict['tau_SBM']
        vuv = x_dict['vuv_SBM']
        
        # sinenet
        sin_cos_x = self.sinenet_fn(x, f, tau)  # S*B*M*2K
        y_SBMD_1  = self.linear_fn_1(sin_cos_x) # S*B*M*2K -> S*B*M*D
        # nlf, tau, vuv
        nlf_1 = torch.unsqueeze(nlf, 3)
        tau_1 = torch.unsqueeze(tau, 3)
        vuv_1 = torch.unsqueeze(vuv, 3)
        nlf_tau_vuv = torch.cat([nlf_1, tau_1, vuv_1], 3)
        y_SBMD_2 = self.linear_fn_2(nlf_tau_vuv) # S*B*M*3 -> S*B*M*D

        y_SBMD = y_SBMD_1 + y_SBMD_2
        # Batch Norm
        if self.batch_norm:
            y_SBMD = batch_norm_D_tensor(y_SBMD, self.bn_fn, index_D=self.D_index)

        # ReLU
        h_SBMD = self.activation_fn(y_SBMD)

        y_dict = {'h': h_SBMD}
        return y_dict

class Build_Sinenet_V2_old(torch.nn.Module):
    ''' 
        Model Parameters lf0 and tau values
        Input:  wav_SBMT
        Output: y: S*B*M*D, h: S*B*D, D <-- (M*(D+1))
        Predicted lf0 and tau values
            Inputs: wav_SBMT, f0_SBM, tau_SBM
            Output: h: S*B*M*D; D=K
        1. Construct matrix, K*T
        2. Apply fc, batch_norm, relu
    '''
    def __init__(self, params):
        super().__init__()
        self.params = params
        layer_config = self.params["layer_config"]
        D_out = layer_config['size']

        self.params["output_dim_seq"] = ['S', 'B', 'M', 'D']
        self.num_freq = layer_config['num_freq']
        self.inc_a    = layer_config['inc_a']
        # assert layer_config['size'] == sine_size + 3
        # D_out = (layer_config['size']) * self.params["input_dim_values"]['M']
        self.params["output_dim_values"] = {'S': self.params["input_dim_values"]['S'], 'B': self.params["input_dim_values"]['B'], 'M': self.params["input_dim_values"]['M'], 'D': D_out}


        self.sinenet_fn = Build_Sinenet(params)
        self.k_2pi_tensor = self.sinenet_fn.k_2pi_tensor
        self.n_T_tensor   = self.sinenet_fn.n_T_tensor
        self.tau_tensor   = self.make_tau_tensor()
        self.f_init_value = 155.

        self.linear_fn = torch.nn.Linear(self.num_freq*2, D_out)
        if self.inc_a:
            self.linear_fn_2 = torch.nn.Linear(3, D_out)

        self.activation_fn = self.params['activation_fn']

    def make_tau_tensor(self):
        tau_vec = numpy.zeros(self.num_freq)
        tau_tensor = torch.tensor(tau_vec, dtype=torch.float, requires_grad=True)
        tau_tensor = torch.nn.Parameter(tau_tensor, requires_grad=True) # K
        return tau_tensor

    def compute_deg(self):
        ''' Return degree in radian '''
        # Time
        tau_1 = torch.unsqueeze(self.tau_tensor, 1) # K --> # K*1
        t = torch.add(self.n_T_tensor, torch.neg(tau_1)) # T + K*1 -> K*T

        # Degree in radian
        k_2pi_f = torch.mul(self.k_2pi_tensor, self.f_init_value) # K
        k_2pi_f_1 = torch.unsqueeze(k_2pi_f, 1) # K -> K*1
        deg = torch.mul(k_2pi_f_1, t) # K*1, K*T -> K*T
        return deg

    def construct_w_sin_cos_matrix(self):
        deg = self.compute_deg()      # K*T
        s   = torch.sin(deg)          # K*T
        c   = torch.cos(deg)          # K*T
        s_c = torch.cat([s,c], dim=0) # 2K*T
        s_c_t = torch.t(s_c)  # T*2K
        return s_c_t

    def forward(self, x_dict):
        if 'wav_SBMT' in x_dict:
            x = x_dict['wav_SBMT']
        elif 'h' in x_dict:
            x = x_dict['h']

        self.w_sin_cos_matrix = self.construct_w_sin_cos_matrix() # T*2K
        sin_cos_x = torch.matmul(x, self.w_sin_cos_matrix) # S*B*M*T, T*2K -> S*B*M*2K
        y_SBMD = self.linear_fn(sin_cos_x)                 # S*B*M*2K -> S*B*M*D

        # nlf, tau, vuv
        if self.inc_a:
            f, nlf = compute_f_nlf(x_dict)
            tau = x_dict['tau_SBM']
            vuv = x_dict['vuv_SBM']
            nlf_1 = torch.unsqueeze(nlf, 3)
            tau_1 = torch.unsqueeze(tau, 3)
            vuv_1 = torch.unsqueeze(vuv, 3)
            nlf_tau_vuv = torch.cat([nlf_1, tau_1, vuv_1], 3)
            y_SBMD_2 = self.linear_fn_2(nlf_tau_vuv) # S*B*M*3 -> S*B*M*D

            y_SBMD = y_SBMD + y_SBMD_2

        # ReLU
        h_SBMD = self.activation_fn(y_SBMD)

        y_dict = {'h': h_SBMD}
        return y_dict

class Build_Sinenet_V1_Residual(torch.nn.Module):
    ''' 
        Inputs: wav_SBMT, f0_SBM, tau_SBM
        Output: h: S*B*M*D
        1. Use sinenet to compute sine/cosine matrix
        2. Extract residual x
        2. Append nlf_SBM, tau_SBM, vuv_SBM
        3. Apply fc, batch_norm, relu
    '''
    def __init__(self, params):
        super().__init__()
        self.params = params
        layer_config = self.params["layer_config"]

        self.params["output_dim_seq"] = ['S', 'B', 'M', 'D']
        num_freq  = layer_config['num_freq']
        # assert layer_config['size'] == sine_size + 3
        # D_out = (layer_config['size']) * self.params["input_dim_values"]['M']
        self.params["output_dim_values"] = {'S': self.params["input_dim_values"]['S'], 'B': self.params["input_dim_values"]['B'], 'M': self.params["input_dim_values"]['M'], 'D': layer_config['size']}

        self.sinenet_fn = Build_Sinenet(params)

        self.linear_fn_1 = torch.nn.Linear(self.params["input_dim_values"]['T'], layer_config['size'])
        self.linear_fn_2 = torch.nn.Linear(3, layer_config['size'])
        self.batch_norm = layer_config["batch_norm"]
        if self.batch_norm:
            self.D_index = len(self.params["input_dim_seq"]) - 1
            self.bn_fn = torch.nn.BatchNorm2d(layer_config['size'])
        self.activation_fn = self.params['activation_fn']

    def compute_x_residual(self, x, w_sc):
        '''
        Inputs:
            x: S*B*M*T
            w_sc: # S*B*M*2K*T
        '''
        w_sc_T = torch.transpose(w_sc, 3, 4)      # S*B*M*T*2K
        a_sc_inv = torch.matmul(w_sc, w_sc_T)     # S*B*M*2K*2K
        a_sc = torch.inverse(a_sc_inv)            # S*B*M*2K*2K

        w_sc_x = torch.einsum('sbmkt,sbmt->sbmk', w_sc, x)
        a_sc_w_sc_x = torch.einsum('sbmk,sbmjk->sbmj', w_sc_x, a_sc) # Use j as another k, since a_sc is k*k
        x_sc = torch.einsum('sbmtk,sbmk->sbmt', w_sc_T, a_sc_w_sc_x)

        x_res = x - x_sc
        return x_res

    def forward(self, x_dict):
        
        if 'wav_SBMT' in x_dict:
            x = x_dict['wav_SBMT']
        elif 'h' in x_dict:
            x = x_dict['h']
        
        f, nlf = compute_f_nlf(x_dict)
        tau = x_dict['tau_SBM']
        vuv = x_dict['vuv_SBM']
        
        # sinenet
        w_sc = self.sinenet_fn.construct_w_sin_cos_matrix(f, tau) # S*B*M*2K*T
        x_res = self.compute_x_residual(x, w_sc)
        
        y_SBMD_1  = self.linear_fn_1(x_res) # S*B*M*T -> S*B*M*D
        # nlf, tau, vuv
        nlf_1 = torch.unsqueeze(nlf, 3)
        tau_1 = torch.unsqueeze(tau, 3)
        vuv_1 = torch.unsqueeze(vuv, 3)
        nlf_tau_vuv = torch.cat([nlf_1, tau_1, vuv_1], 3)
        y_SBMD_2 = self.linear_fn_2(nlf_tau_vuv) # S*B*M*3 -> S*B*M*D

        y_SBMD = y_SBMD_1 + y_SBMD_2
        # Batch Norm
        if self.batch_norm:
            y_SBMD = batch_norm_D_tensor(y_SBMD, self.bn_fn, index_D=self.D_index)

        # ReLU
        h_SBMD = self.activation_fn(y_SBMD)

        y_dict = {'h': h_SBMD}
        return y_dict
