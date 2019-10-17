import os, sys, pickle, time, shutil, logging, copy
import math, numpy, scipy
numpy.random.seed(545)
import torch
torch.manual_seed(545)



from modules import make_logger

'''
This file contains handy modules of using PyTorch
'''










class dv_y_model(object):
    """general dv_y_model"""
    def __init__(self, dv_y_cfg):
        self.dv_y_cfg = dv_y_cfg
        self.nn_layers     = []
        self.train_scope   = []
        self.learning_rate = dv_y_cfg.learning_rate
        # This is necessary for tensorflow session operations
        # self.sess = None
        self.CE_SB_cost = None
        self.CE_S_cost  = None
        self.logger = make_logger("dv_y_model")

    def build_layers(self):
        pass

    def build_train_step(self, train_scope):
        dv_y_cfg = self.dv_y_cfg
        # if train_scope is None:
            # train_scope = self.train_scope
        # Select optimisation criterion
        if dv_y_cfg.train_by_window:
            self.train_loss = self.CE_SB_cost
        else:
            self.train_loss = self.CE_S_cost
        # Collect all trainable parameters
        self.train_vars    = []
        for i in train_scope:
            vars_i = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, i)
            self.train_vars.extend(vars_i)
        scope_name = dv_y_cfg.tf_scope_name + '/dv_y_optimiser'
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            self.train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate_holder,epsilon=1.e-06).minimize(self.train_loss, var_list=self.train_vars)

    def update_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    def train_model_param(self, feed_dict):
        self.sess.run(self.train_step, feed_dict=feed_dict)

    def return_train_loss(self, feed_dict):
        return self.sess.run(self.train_loss, feed_dict=feed_dict)

    def gen_logit_SBD(self, feed_dict):
        return self.sess.run(self.logit_SBD, feed_dict=feed_dict)

    def gen_logit_SD(self, feed_dict):
        return self.sess.run(self.logit_SD, feed_dict=feed_dict)

    def gen_lambda_SBD(self, feed_dict):
        return self.sess.run(self.lambda_SBD, feed_dict=feed_dict)

    def gen_lambda_SD(self, feed_dict):
        return self.sess.run(self.lambda_SD, feed_dict=feed_dict)

    def load_prev_model(self, previous_model_name=None):
        if previous_model_name is None: previous_model_name = self.dv_y_cfg.previous_model_name
        self.logger.info('restore previous dv_y_model, '+previous_model_name)
        try:
            self.saver.restore(self.sess, previous_model_name)
            # logger.info('use TF saver')
        except:
            self.logger.info('cannot use TF saver, use cPickle')
            self = cPickle.load(open(nnets_file_name, 'rb'))

    def save_current_model(self, nnets_file_name=None):
        if nnets_file_name is None: nnets_file_name=self.dv_y_cfg.nnets_file_name
        self.logger.info('saving model, '+nnets_file_name)
        try:
            # Use TF saver
            save_path = self.saver.save(self.sess, nnets_file_name)
        except:
            self.logger.info('cannot use TF saver, use cPickle')
            cPickle.dump(self, open(nnets_file_name, 'wb'))

    def close_tf_session_and_reset(self):
        self.sess.close()
        tf.reset_default_graph() 
            

class dv_y_cmp_model(dv_y_model):
    """ baseline, SBTD input """

    def __init__(self, dv_y_cfg):
        super(dv_y_cmp_model, self).__init__(dv_y_cfg)

        with tf.device('/device:GPU:'+str(dv_y_cfg.gpu_id)):
            self.learning_rate_holder = tf.placeholder(dtype=tf.float32, name='dv_y_learning_rate_holder')
            self.is_train_bool = tf.placeholder(dtype=tf.bool, name="is_train")
            self.train_scope = self.build_layers()
            self.build_train_step(self.train_scope)

            self.init  = tf.global_variables_initializer()
            self.saver = tf.train.Saver()

    def build_layers(self):
        train_scope = []
        dv_y_cfg = self.dv_y_cfg
        # Input Layer
        scope_name = dv_y_cfg.tf_scope_name+'/SBTD_input_layer'
        train_scope.append(scope_name)
        self.input_layer = build_SBTD_input_layer(scope_name, dv_y_cfg)
        prev_layer = self.input_layer
        # Hidden Layers
        for i in range(dv_y_cfg.num_nn_layers):
            scope_name = dv_y_cfg.tf_scope_name + '/dv_y_layer_'+str(i)
            self.train_scope.append(scope_name)
            new_layer = build_nn_layer(scope_name, dv_y_cfg.nn_layer_config_list[i], prev_layer, self)
            # Setting up for the next layer
            prev_layer = new_layer
            self.nn_layers.append(new_layer)
        # Output Layer
        scope_name = dv_y_cfg.tf_scope_name+'/SBD_output_layer'
        train_scope.append(scope_name)
        self.output_layer = build_SBD_output_layer(scope_name, dv_y_cfg, prev_layer)
        self.lambda_SBD = self.output_layer.tensor_outputs['h_input_reshape_SBD']
        self.lambda_SD  = self.output_layer.tensor_outputs['h_input_reshape_SD']
        self.logit_SBD  = self.output_layer.tensor_outputs['h_SBD']
        self.logit_SD   = self.output_layer.tensor_outputs['h_SD']
        self.CE_SB_cost = self.output_layer.tensor_outputs['loss_SB']
        self.CE_S_cost  = self.output_layer.tensor_outputs['loss_S']

        self.train_scope = train_scope
        return train_scope

def config_torch(dv_y_cfg):
    pass
    # tf_config = tf.ConfigProto()
    # tf_config.gpu_options.allow_growth = True
    # tf_config.gpu_options.per_process_gpu_memory_fraction = dv_y_cfg.gpu_per_process_gpu_memory_fraction
    # tf_config.allow_soft_placement = True
    # tf_config.log_device_placement = False
    # return tf_config

def tensor_reshape(layer_input, input_dim_seq, input_dim_values, expect_input_dim_seq, expect_input_dim_values=None, downsample=1):
    # First, check if change is needed at all; pass on if not
    if (input_dim_seq == expect_input_dim_seq) and (downsample == 1):
        return layer_input, input_dim_values
    else:
        # Make anything into ['S', 'B', 'T', 'D']
        if input_dim_seq == ['S', 'B', 'T', 'D']:
            # Do nothing, pass on
            temp_input_dim_values = input_dim_values
            temp_input = layer_input
        elif input_dim_seq == ['S', 'B', 'D']:
            # Expand to 4D tensor and T=1
            temp_input_dim_values = {'S':input_dim_values['S'], 'B':input_dim_values['B'], 'T':1, 'D':input_dim_values['D'] }
            temp_input_shape_values = [temp_input_dim_values['S'], temp_input_dim_values['B'], temp_input_dim_values['T'], temp_input_dim_values['D']]
            temp_input = layer_input.view(temp_input_shape_values)
        elif input_dim_seq == ['T', 'SB', 'D']:
            # 1. Transpose to make ['SB', 'T', 'D']
            temp_input = torch.transpose(layer_input, 0, 1)
            # 2. Reshape to make ['S', 'B', 'T', 'D']
            temp_input_dim_values = input_dim_values
            temp_input_shape_values = [temp_input_dim_values['S'], temp_input_dim_values['B'], temp_input_dim_values['T'], temp_input_dim_values['D']]
            temp_input = temp_input.view(temp_input_shape_values)
        else:
            print "Input dimension sequence not recognised"

        # Downsampling by stacking
        # TODO: Upsampling by?
        if downsample > 1:
            temp_input_dim_values['T'] = temp_input_dim_values['T'] / downsample
            temp_input_dim_values['D'] = temp_input_dim_values['D'] * downsample
            temp_input_shape_values = [temp_input_dim_values['S'], temp_input_dim_values['B'], temp_input_dim_values['T'], temp_input_dim_values['D']]
            temp_input = temp_input.view(temp_input_shape_values)
        elif downsample < 1:
            print "Upsampling not implemented yet"

        # Then, make from ['S', 'B', 'T', 'D']
        if expect_input_dim_seq == ['S', 'B', 'D']:
            # So basically, stack and remove T; last dimension D -> T * D
            expect_input_shape_values = [temp_input_dim_values['S'], temp_input_dim_values['B'], temp_input_dim_values['T']*temp_input_dim_values['D']]
            expect_input = temp_input.view(expect_input_shape_values)
            expect_input_dim_values = {'S':temp_input_dim_values['S'], 'B':temp_input_dim_values['B'], 'T':0, 'D':temp_input_dim_values['T']*temp_input_dim_values['D'] }
        elif expect_input_dim_seq == ['SB', 'D']:
            # So basically, stack and remove T; last dimension D -> T * D
            expect_input_shape_values = [temp_input_dim_values['S']*temp_input_dim_values['B'], temp_input_dim_values['T']*temp_input_dim_values['D']]
            expect_input = temp_input.view(expect_input_shape_values)
            expect_input_dim_values = {'S':temp_input_dim_values['S'], 'B':temp_input_dim_values['B'], 'T':0, 'D':temp_input_dim_values['T']*temp_input_dim_values['D'] }
        elif expect_input_dim_seq == ['S', 'B', 'T', '1']:
            # So basically, stack and remove D; second-last dimension T -> T * D
            expect_input_shape_values = [temp_input_dim_values['S'], temp_input_dim_values['B'], temp_input_dim_values['T']*temp_input_dim_values['D'], 1]
            expect_input = temp_input.view(expect_input_shape_values)
            expect_input_dim_values = {'S':temp_input_dim_values['S'], 'B':temp_input_dim_values['B'], 'T':temp_input_dim_values['T']*temp_input_dim_values['D'] , 'D':1}
        elif expect_input_dim_seq == ['TD', 'SB', '1']:
            # 1. make ['SB', 'TD', 1]
            expect_input_shape_values = [temp_input_dim_values['S']*temp_input_dim_values['B'], temp_input_dim_values['T']*temp_input_dim_values['D'], 1]
            expect_input = temp_input.view(expect_input_shape_values)
            # 2. Transpose to make ['TD', 'SB', 1]
            expect_input = torch.transpose(expect_input, 0, 1)
            expect_input_dim_values = temp_input_dim_values
        elif expect_input_dim_seq == ['T', 'SB', 'D']:
            # 1. make ['SB', 'T', 'D']
            expect_input_shape_values = [temp_input_dim_values['S']*temp_input_dim_values['B'], temp_input_dim_values['T'], temp_input_dim_values['D']]
            expect_input = temp_input.view(expect_input_shape_values)
            # 2. Transpose to make ['T', 'SB', 'D']
            expect_input = torch.transpose(expect_input, 0, 1)
            expect_input_dim_values = temp_input_dim_values

        return expect_input, expect_input_dim_values

class build_SBTD_input_layer(object):
    ''' This layer has only "output" to the next layer, no input '''
    def __init__(self, scope_name, dv_y_cfg, tensor_input_h=None):
        self.params = {}
        self.params["scope_name"] = scope_name
        self.params["output_dim_seq"]      = ['S', 'B', 'T', 'D']
        self.params["output_dim_values"]   = {'S':dv_y_cfg.batch_num_spk, 'B':dv_y_cfg.spk_num_seq, 'T':dv_y_cfg.batch_seq_len, 'D':dv_y_cfg.feat_dim}
        v = self.params["output_dim_values"]
        self.params["output_shape_values"] = [v['S'], v['B'], v['T'], v['D']]
        
        if tensor_input_h is None:
            self.tensor_outputs = {'h': tf.placeholder(tf.float32, shape=self.params["output_shape_values"])}
        else:
            self.tensor_outputs = {'h': tensor_input_h}

class build_SBD_output_layer(object):
    ''' This layer gets input from previous layer '''
    ''' It also includes training loss '''
    def __init__(self, scope_name, dv_y_cfg, prev_layer):
        self.params = {}
        self.params["scope_name"] = scope_name
        self.params["input_dim_seq"]           = prev_layer.params["output_dim_seq"]
        self.params["input_dim_values"]        = prev_layer.params["output_dim_values"]
        self.params["expect_input_dim_seq"]    = ['S', 'B', 'D']
        self.params["expect_input_dim_values"] = {'S':dv_y_cfg.batch_num_spk, 'B':dv_y_cfg.spk_num_seq, 'D':dv_y_cfg.dv_dim}
        self.params["output_dim_seq"]          = ['S', 'B', 'D']
        self.params["output_dim_values"]       = {'S':dv_y_cfg.batch_num_spk, 'B':dv_y_cfg.spk_num_seq, 'D':dv_y_cfg.num_train_speakers}
        self.params["target_shape_values"]     = [self.params["output_dim_values"]['S'], self.params["output_dim_values"]['D']]

        with tf.variable_scope(self.params["scope_name"], reuse=tf.AUTO_REUSE):
            self.tensor_inputs  = {'h': prev_layer.tensor_outputs['h'], 'target_SD': tf.placeholder(tf.float32, shape=self.params["target_shape_values"])}
            self.tensor_outputs = {'h_SBD': None, 'h_SD': None} # These are logits, need to run softmax over them; still can use argmax on them for classification
            self.tensor_outputs['h_input_reshape_SBD'], self.params["expect_input_dim_values"] = tensor_reshape(self.tensor_inputs['h'], self.params["input_dim_seq"], self.params["input_dim_values"], self.params["expect_input_dim_seq"])
            self.make_SBD_output()
            self.make_SD_output(batch_output_form=dv_y_cfg.batch_output_form)

    def make_SBD_output(self):
        self.tensor_outputs['target_SBD'] = tf.tile(tf.expand_dims(self.tensor_inputs['target_SD'], axis=1), [1, self.params["output_dim_values"]['B'], 1])
        self.tensor_outputs['h_SBD']      = tf.contrib.layers.fully_connected(self.tensor_outputs['h_input_reshape_SBD'], self.params["output_dim_values"]['D'], activation_fn=None)
        # This part needs reshape, as tf.losses.softmax_cross_entropy handles 3D tensors wrong, use [1] rather than [-1]
        # so reshape to 2D by stacking [S,B,D] to [SB,D]
        expect_SB_D_dim_seq = ['SB', 'D']
        self.tensor_outputs['h_SB_D'], _ = tensor_reshape(self.tensor_outputs['h_SBD'], self.params["output_dim_seq"], self.params["output_dim_values"], expect_SB_D_dim_seq)
        self.tensor_outputs['target_SB_D'], _ = tensor_reshape(self.tensor_outputs['target_SBD'], self.params["output_dim_seq"], self.params["output_dim_values"], expect_SB_D_dim_seq)
        self.tensor_outputs['loss_SB']    = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=self.tensor_outputs['target_SB_D'], logits=self.tensor_outputs['h_SB_D']))#, reduction=MEAN)

    def make_SD_output(self, batch_output_form):
        B = self.params["output_dim_values"]['B']
        if B > 1:
            if batch_output_form == 'mean':
                self.tensor_outputs['h_input_reshape_SD'] = tf.scalar_mul(tf.constant(float(1./B), name="1/batch", dtype=tf.float32), tf.reduce_sum(self.tensor_outputs['h_input_reshape_SBD'], 1))
        else:
            self.tensor_outputs['h_input_reshape_SD'] = tf.squeeze(self.tensor_outputs['h_input_reshape_SBD'], 1)
        self.tensor_outputs['h_SD'] = tf.contrib.layers.fully_connected(self.tensor_outputs['h_input_reshape_SD'], self.params["output_dim_values"]['D'], activation_fn=None)
        self.tensor_outputs['loss_S'] = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=self.tensor_inputs['target_SD'], logits=self.tensor_outputs['h_SD']))#, reduction=MEAN)
       
      
class build_nn_layer(object):
    '''
    Just a general layer, with expected attribute and method names
    '''
    def __init__(self, scope_name, layer_config, prev_layer, model):
        self.params = {}
        self.params["scope_name"] = scope_name
        self.params["layer_config"] = layer_config
        self.params["type"]   = layer_config['type']
        self.params["size"]   = layer_config['size']

        self.params["input_dim_seq"]           = prev_layer.params["output_dim_seq"]
        self.params["input_dim_values"]        = prev_layer.params["output_dim_values"]

        # To be set per layer type; mostly for definition of h
        self.params["expect_input_dim_seq"]    = []
        self.params["expect_input_dim_values"] = {}
        self.params["output_dim_seq"]          = []
        self.params["output_dim_values"]       = {}
        
        with tf.variable_scope(self.params["scope_name"], reuse=tf.AUTO_REUSE):
            self.tensor_inputs  = {'h': prev_layer.tensor_outputs['h'], 'is_train': model.is_train_bool}
            self.tensor_outputs = {'h': None}
            self.tensor_params  = {} # Not necessary for most layers

            construct_layer = getattr(self, self.params["layer_config"]["type"])
            construct_layer()

            self.tensor_outputs['h'] = self.apply_dropout(self.tensor_outputs['h'])

    def apply_dropout(self, dropout_input):
        # Apply dropout to self.tensor_outputs['h']
        try: 
            self.params["dropout_p"] = self.params["layer_config"]['dropout_p']
        except KeyError: 
            self.params["dropout_p"] = 0.
            return dropout_input

        if self.params["dropout_p"] > 0:
            layer_output = tf.layers.dropout(dropout_input, rate=1.-self.params["dropout_p"], seed=numpy.random.randint(0, 545), training=self.tensor_inputs['is_train'])
            return layer_output
        else:
            return dropout_input

    def input_tensor_reshape(self):
        # returns tensor_outputs['h_input_reshape'] and params["expect_input_dim_values"]
        self.tensor_outputs['h_input_reshape'], self.params["expect_input_dim_values"] = tensor_reshape(self.tensor_inputs['h'], self.params["input_dim_seq"], self.params["input_dim_values"], self.params["expect_input_dim_seq"])

    def ReLUDVMax(self, layer_config=None):
        if layer_config is None: layer_config=self.params["layer_config"]
        self.params["expect_input_dim_seq"] = ['S', 'B', 'D']
        self.params["output_dim_seq"]       = ['S', 'B', 'D']
        self.input_tensor_reshape()
        v = self.params["expect_input_dim_values"]
        self.params["output_dim_values"]    = {'S': v['S'], 'B': v['B'], 'D': self.params["size"]}

        # 1. ReLU 
        self.tensor_outputs['h_list'] = []
        for i in range(self.params["layer_config"]['num_channels']):
            # No need to give variable_scope for each channel; they are auto named fully_connected and fully_connected_1 in tensorflow
            h = tf.contrib.layers.fully_connected(self.tensor_outputs['h_input_reshape'], self.params["size"], activation_fn=tf.nn.relu)
            self.tensor_outputs['h_list'].append(h)
        # 2. Maxout 
        self.tensor_outputs['h'] = tf.squeeze(tf.contrib.layers.maxout(self.tensor_outputs["h_list"], 1, axis=0, name='maxout'), axis=0)


    def ReLUDV(self, layer_config=None):
        if layer_config is None: layer_config=self.params["layer_config"]
        self.params["expect_input_dim_seq"] = ['S', 'B', 'D']
        self.params["output_dim_seq"]       = ['S', 'B', 'D']
        self.input_tensor_reshape()
        v = self.params["expect_input_dim_values"]
        self.params["output_dim_values"]    = {'S': v['S'], 'B': v['B'], 'D': self.params["size"]}
        self.tensor_outputs['h'] = tf.contrib.layers.fully_connected(self.tensor_outputs['h_input_reshape'], self.params["size"], activation_fn=tf.nn.relu)

    def LinDV(self, layer_config=None):
        if layer_config is None: layer_config=self.params["layer_config"]
        self.params["expect_input_dim_seq"] = ['S', 'B', 'D']
        self.params["output_dim_seq"]       = ['S', 'B', 'D']
        self.input_tensor_reshape()
        v = self.params["expect_input_dim_values"]
        self.params["output_dim_values"]    = {'S': v['S'], 'B': v['B'], 'D': self.params["size"]}
        self.tensor_outputs['h'] = tf.contrib.layers.fully_connected(self.tensor_outputs['h_input_reshape'], self.params["size"], activation_fn=None)

    def Wav1DCNN(self, layer_config=None):
        if layer_config is None: layer_config=self.params["layer_config"]
        self.params["expect_input_dim_seq"] = ['S', 'B', 'T', 'D']
        self.params["output_dim_seq"]       = ['S', 'B', 'T', 'D']
        self.input_tensor_reshape()
        v = self.params["expect_input_dim_values"]
        self.params["output_dim_values"]    = {'S': v['S'], 'B': v['B'], 'T':int((self.expect_input_dim_values['T']-layer_config['CNN_kernel_size'][1])/layer_config['CNN_stride'][1])+1, 'D': self.params["size"]} # No padding, shorter output!

        if layer_config['CNN_activation'] == 'ReLU':
            self.activation_fn = tf.nn.relu
        else:
            self.activation_fn = None
        # ['S', 'B', 'T', '1'] --> ['S', 'B', 'T', 'D']; T_new = T_old
        self.tensor_outputs['h'] = tf.layers.conv2d(self.tensor_outputs['h_input_reshape'], filters=layer_config['size'], kernel_size=layer_config['CNN_kernel_size'], strides=layer_config['CNN_stride'], activation=self.activation_fn) # No padding, shorter output!

    def Wav1DSineNet(self, layer_config=None):
        if layer_config is None: layer_config=self.params["layer_config"]
        # TODO: a lot to be done
        self.params["expect_input_dim_seq"] = ['S', 'B', 'T', 'D']
        self.params["output_dim_seq"]       = ['S', 'B', 'D']
        self.input_tensor_reshape()
        v = self.params["expect_input_dim_values"]
        self.params["output_dim_values"]    = {'S': v['S'], 'B': v['B'], 'T':int((self.expect_input_dim_values['T']-layer_config['CNN_kernel_size'][1])/layer_config['CNN_stride'][1])+1, 'D': self.params["size"]}
        self.params["f_tau_shape"] = [v['S'], v['B'], v['T'], 1]  # S,B,T,1
        self.tensor_inputs['f']   = tf.placeholder(tf.float32, shape = self.params["f_tau_shape"])   # S,B,T,1
        self.tensor_inputs['tau'] = tf.placeholder(tf.float32, shape = self.params["f_tau_shape"])   # S,B,T,1


        self.tensor_params = {}
        



class build_am_model(object):

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


