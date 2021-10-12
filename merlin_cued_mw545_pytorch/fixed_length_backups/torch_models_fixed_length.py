# torch_models.py

import os, sys, pickle, time, shutil, logging, copy
import math, numpy, scipy
import torch

from frontend_mw545.modules import make_logger
# from nn_torch.torch_layers  import Build_DV_Y_Input_Layer, Build_NN_Layer
from fixed_length_backups.torch_layers_fixed_length import Build_DV_Y_Input_Layer, Build_NN_Layer

########################
# PyTorch-based Models #
########################

class Build_DV_Y_NN_model(torch.nn.Module):
    ''' S_B_D input, SB_D/S_D logit output if train_by_window '''
    def __init__(self, dv_y_cfg):
        super().__init__()
        
        self.num_nn_layers = dv_y_cfg.num_nn_layers
        self.train_by_window = dv_y_cfg.train_by_window

        self.input_layer = Build_DV_Y_Input_Layer(dv_y_cfg)
        self.input_dim_values = self.input_layer.params["output_dim_values"]
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
        self.output_dim = dv_y_cfg.cfg.num_speaker_dict['train']
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

    def gen_lambda_SD(self, x_dict):
        ''' 
        Average over B
        For 1. better estimation of lambda; and 2. classification
        '''
        lambda_SBD = self.gen_lambda_SBD(x_dict)
        lambda_SD  = torch.mean(lambda_SBD, dim=1, keepdim=False)
        return lambda_SD

    def gen_logit_SD(self, x_dict):
        lambda_SD = self.gen_lambda_SD(x_dict)
        logit_SD  = self.expansion_layer(lambda_SD)
        return logit_SD

    def gen_p_SD(self, x_dict): 
        logit_SD = self.gen_logit_SD(x_dict)
        self.softmax_fn = torch.nn.Softmax(dim=2)
        p_SD = self.softmax_fn(logit_SD)
        return p_SD

    def forward(self, x_dict):
        # Choose which logit to use for cross-entropy
        if self.train_by_window:
            # Flatten to 2D for cross-entropy
            logit_SBD  = self.gen_logit_SBD(x_dict)
            logit_SB_D = logit_SBD.view(-1, self.output_dim)
            return logit_SB_D
        else:
            # Already 2D (S*D), do nothing
            logit_S_D  = self.gen_logit_SD(x_dict)
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

       
##############################################
# Model Wrappers, between Python and PyTorch #
# I/O are numpy, not tensors
##############################################

class General_Model(object):

    ###################
    # THings to build #
    ###################

    def __init__(self):
        self.nn_model = None
        self.model_config = None

    def build_optimiser(self):
        pass

    def gen_loss(self, feed_dict):
        pass

    def gen_lambda_SBD(self, feed_dict):
        pass

    def cal_accuracy(self, feed_dict):
        pass

    def numpy_to_tensor(self, feed_dict):
        ''' Maybe need different precisions '''
        pass

    def print_output_dim_values(self, logger):
        pass

    ###################
    # Things can stay #
    ###################

    def __call__(self, x):
        ''' Simulate PyTorch forward() method '''
        ''' Note that x should be feed_dict '''
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

    def print_model_parameter_sizes(self, logger):
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
        # Reset gradient, otherwise equivalent to momentum>1
        self.loss.backward()
        self.optimiser.step()
        self.optimiser.zero_grad()

    def update_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate
        # Re-build an optimiser, use new learning rate, and reset gradients
        self.build_optimiser()

    def gen_loss_value(self, feed_dict):
        ''' Return the numpy value of self.loss '''
        self.loss = self.gen_loss(feed_dict)
        return self.loss.item()

    def torch_initialisation(self, model_config=None, nn_model=None):
        logger = make_logger("torch init")
        torch.manual_seed(554455)

        if nn_model is None:
            nn_model = self.nn_model
        if model_config is None:
            model_config = self.model_config
        
        if model_config.gpu_id == 'cpu':
            logger.info('Using CPU')
            device_id = torch.device("cpu")
        elif torch.cuda.is_available():
        # if False:
            logger.info('Using GPU cuda:%i' % model_config.gpu_id)
            device_id = torch.device("cuda:%i" % model_config.gpu_id)
        else:
            logger.info('Using CPU; No GPU')
            device_id = torch.device("cpu")

        self.device_id = device_id
        nn_model.to(device_id)

    def save_nn_model(self, nnets_file_name):
        ''' Model Only '''
        save_dict = {'model_state_dict': self.nn_model.state_dict()}
        torch.save(save_dict, nnets_file_name)

    def load_nn_model(self, nnets_file_name):
        ''' Model Only '''
        checkpoint = torch.load(nnets_file_name)
        self.nn_model.load_state_dict(checkpoint['model_state_dict'])

    def save_nn_model_optim(self, nnets_file_name):
        '''Save both Model and Optimiser '''
        save_dict = {'model_state_dict': self.nn_model.state_dict(), 'optimiser_state_dict': self.optimiser.state_dict()}
        torch.save(save_dict, nnets_file_name)

    def load_nn_model_optim(self, nnets_file_name):
        ''' Load both Model and Optimiser '''
        checkpoint = torch.load(nnets_file_name)
        self.nn_model.load_state_dict(checkpoint['model_state_dict'])
        self.optimiser.load_state_dict(checkpoint['optimiser_state_dict'])

    def count_parameters(self):
        return sum(p.numel() for p in self.nn_model.parameters() if p.requires_grad)

class Build_DV_Y_model(General_Model):
    ''' 
    Acoustic data --> Speaker Code --> Classification 
    For this model, feed_dict['one_hot] --> y (tensor)
    '''
    def __init__(self, dv_y_cfg):
        super().__init__()
        self.nn_model = Build_DV_Y_NN_model(dv_y_cfg)
        
        self.dv_y_cfg = dv_y_cfg
        self.model_config = dv_y_cfg
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

    def gen_SB_loss_value(self, feed_dict):
        ''' Return the numpy value of self.loss in SB form '''
        SB_loss = self.gen_SB_loss(feed_dict)
        return self.tensor_to_numpy(SB_loss)

    def gen_lambda_SBD_value(self, feed_dict):
        x_dict, y = self.numpy_to_tensor(feed_dict)
        lambda_SBD = self.nn_model.gen_lambda_SBD(x_dict)
        return self.tensor_to_numpy(lambda_SBD)

    def gen_logit_SBD_value(self, feed_dict):
        x_dict, y = self.numpy_to_tensor(feed_dict)
        logit_SBD = self.nn_model.gen_logit_SBD(x_dict)
        return self.tensor_to_numpy(logit_SBD)

    def gen_logit_SD_value(self, feed_dict):
        x_dict, y = self.numpy_to_tensor(feed_dict)
        logit_SD = self.nn_model.gen_logit_SD(x_dict)
        return self.tensor_to_numpy(logit_SD)

    def gen_p_SBD_value(self, feed_dict):
        x_dict, y = self.numpy_to_tensor(feed_dict)
        p_SBD = self.nn_model.gen_p_SBD(x_dict)
        return self.tensor_to_numpy(p_SBD)

    def lambda_to_indices(self, feed_dict):
        ''' lambda_S_B_D to indices_S_B '''
        x_dict, _y = self.numpy_to_tensor(feed_dict) # x_dict['lambda'] lambda_S_B_D! _y is useless
        logit_SBD  = self.nn_model.lambda_to_logits_SBD(x_dict)
        _values, predict_idx_list = torch.max(logit_SBD.data, -1)
        return self.tensor_to_numpy(predict_idx_list)

    def cal_accuracy(self, feed_dict):
        '''
        1. Generate logit_SD
        2. Check and convert y to S?
        3. Compute accuracy
        '''
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
            h = x_dict['h']
            h_list.append(self.tensor_to_numpy(h))
        logit_SBD = self.nn_model.lambda_to_logits_SBD(x_dict)
        h_list.append(self.tensor_to_numpy(logit_SBD))
        return h_list

    def numpy_to_tensor(self, feed_dict):
        x_dict = {}
        y = None
        for k in feed_dict:
            k_val = feed_dict[k]
            if k == 'one_hot':
                # 'y' is the one-hot speaker class
                # High precision for cross-entropy function
                k_dtype = torch.long
                y = torch.tensor(k_val, dtype=k_dtype, device=self.device_id)
            else:
                k_dtype = torch.float
            k_tensor = torch.tensor(k_val, dtype=k_dtype, device=self.device_id)
            x_dict[k] = k_tensor
        return x_dict, y

    def tensor_to_numpy(self, x_tensor):
        return x_tensor.cpu().detach().numpy()

    def print_output_dim_values(self, logger):
        logger.info('Print Output Dim Values')
        print(self.nn_model.input_layer.params["output_dim_values"])
        for nn_layer in self.nn_model.layer_list:
            print(nn_layer.params["output_dim_values"])