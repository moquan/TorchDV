# exp_dv_y.py

# This file uses dv_cmp experiments to slowly progress with pytorch

import os, sys, pickle, time, shutil, logging, copy
import math, numpy

from frontend_mw545.modules import make_logger, log_class_attri
from frontend_mw545.frontend_tests import Graph_Plotting

from nn_torch.torch_models import Build_DV_Y_model
from frontend_mw545.data_loader import Build_dv_y_train_data_loader

from exp_mw545.exp_base import Build_Model_Trainer_Base

#############
# Processes #
#############

class Build_DV_Y_Model_Trainer(Build_Model_Trainer_Base):
    '''
    Tests specific for acoustic-part of speaker representation model
    Main change: additional accuracy tests during eval_action_epoch
    '''
    def __init__(self, cfg, dv_y_cfg):
        super().__init__(cfg, dv_y_cfg)

        self.model = Build_DV_Y_model(dv_y_cfg)
        self.model.torch_initialisation()
        self.model.build_optimiser()

        if dv_y_cfg.finetune_model:
            self.logger.info('Loading %s for finetune' % dv_y_cfg.prev_nnets_file_name)
            self.model.load_nn_model_optim(dv_y_cfg.prev_nnets_file_name)
        self.model.print_model_parameter_sizes(self.logger)
        self.model.print_output_dim_values(self.logger)

    def build_data_loader(self):
        self.data_loader = Build_dv_y_train_data_loader(self.cfg, self.train_cfg)

    def eval_action_epoch(self, epoch_num):
        '''
        Additional Accuracy Actions
        '''
        output_string = {'loss':'epoch %i train & valid & test loss:' % epoch_num, 'accuracy':'epoch %i train & valid & test accu:' % epoch_num,}
        epoch_loss = {}
        epoch_valid_load_time  = 0.
        epoch_valid_model_time = 0.
        epoch_num_batch = self.train_cfg.epoch_num_batch['valid']

        for utter_tvt_name in ['train', 'valid', 'test']:
            total_batch_size = 0.
            total_loss       = 0.
            total_accuracy   = 0.

            for batch_idx in range(epoch_num_batch):
                batch_load_time, batch_model_time, batch_mean_loss, batch_size, feed_dict = self.eval_action_batch(utter_tvt_name)
                total_loss += batch_mean_loss
                epoch_valid_load_time  += batch_load_time
                epoch_valid_model_time += batch_model_time

                if self.train_cfg.classify_in_training:
                    logit_SBD  = self.model.gen_logit_SBD_value(feed_dict=feed_dict)
                    batch_mean_accuracy = self.cal_accuracy(logit_SBD, feed_dict['one_hot'])
                    total_accuracy += batch_mean_accuracy

            mean_loss = total_loss/float(epoch_num_batch)
            epoch_loss[utter_tvt_name] = mean_loss
            output_string['loss'] = output_string['loss'] + ' & %.4f' % (mean_loss)

            if self.train_cfg.classify_in_training:
                mean_accuracy = total_accuracy/float(epoch_num_batch)
                output_string['accuracy'] = output_string['accuracy'] + ' & %.4f' % (mean_accuracy)

        self.logger.info('valid load & model time: %.2f & %.2f' % (epoch_valid_load_time, epoch_valid_model_time))
        self.logger.info(output_string['loss'])
        if self.train_cfg.classify_in_training:
            self.logger.info(output_string['accuracy'])
        return epoch_loss

    def cal_accuracy(self, logit_SBD, one_hot_SB):
        pred_S_B = numpy.argmax(logit_SBD, axis=2)
        pred_SB  = numpy.reshape(pred_S_B, (-1))
        # print(pred_SB)
        # print(one_hot_SB)
        total_num = pred_SB.size
        correct_num = numpy.sum(pred_SB==one_hot_SB)
        return correct_num/float(total_num)

    def train_single_batch(self):
        '''
        Train with a single batch, repeatedly
        '''
        self.logger.info('Training with single batch!')
        numpy.random.seed(545)
        self.logger.info('Creating data loader')
        self.build_data_loader()

        feed_dict, batch_size = self.data_loader.make_feed_dict(utter_tvt_name='train')

        nnets_file_name = self.train_cfg.nnets_file_name
        epoch_num = 0
        
        while (epoch_num < self.train_cfg.num_train_epoch):
            epoch_num = epoch_num + 1
            epoch_start_time = time.time()

            self.logger.info('start Training Epoch '+str(epoch_num))
            self.model.train()
            for batch_idx in range(self.train_cfg.epoch_num_batch['train']):
                self.model.update_parameters(feed_dict=feed_dict)
            epoch_train_time = time.time()

            self.logger.info('start Evaluating Epoch '+str(epoch_num))
            self.model.eval()
            with self.model.no_grad():
                batch_mean_loss = self.model.gen_loss_value(feed_dict=feed_dict)
                logit_SBD  = self.model.gen_logit_SBD_value(feed_dict=feed_dict)
                batch_mean_accuracy = self.cal_accuracy(logit_SBD, feed_dict['one_hot'])
            self.logger.info('epoch %i loss: %.4f' %(epoch_num, batch_mean_loss))
            self.logger.info('epoch %i accu: %.4f' %(epoch_num, batch_mean_accuracy))
            is_finish = self.validation_action(batch_mean_loss, epoch_num)
            epoch_valid_time = time.time()

            self.logger.info('epoch %i train & valid time %.2f & %.2f' %(epoch_num, (epoch_train_time - epoch_start_time), (epoch_valid_time - epoch_train_time)))
            self.train_cfg.additional_action_epoch(self.logger, self.model)

        self.logger.info('Reach num_train_epoch, best epoch %i, best loss %.4f' % (self.best_epoch_num, self.best_epoch_loss))
        self.logger.info('Model: %s' % (nnets_file_name))



class Build_DV_Y_Testing(object):
    """ Wrapper class for various tests of dv_y models """
    def __init__(self, cfg, dv_y_cfg):
        super().__init__()
        self.cfg = cfg
        self.test_cfg = dv_y_cfg

    def vuv_loss_test(self, fig_file_name='/home/dawna/tts/mw545/Export_Temp/PNG_out/vuv_loss.png'):
        if self.test_cfg.y_feat_name == 'wav':
            test_fn = Build_Wav_VUV_Loss_Test(self.cfg, self.test_cfg, fig_file_name)
        elif self.test_cfg.y_feat_name == 'cmp':
            test_fn = Build_CMP_VUV_Loss_Test(self.cfg, self.test_cfg, fig_file_name)
        test_fn.test()

    def positional_test(self, fig_file_name='/home/dawna/tts/mw545/Export_Temp/PNG_out/positional.png'):
        # test_fn = Build_Positional_Wav_Test(self.cfg, self.test_cfg, fig_file_name)
        # test_fn.test()
        pass

class Build_DV_Y_Testing_Base(object):
    """Base class of tests of dv_y models"""
    def __init__(self, cfg, dv_y_cfg):
        super().__init__()
        self.logger = make_logger("test_model")
        self.cfg = cfg
        self.dv_y_cfg = dv_y_cfg
        log_class_attri(dv_y_cfg, self.logger, except_list=dv_y_cfg.log_except_list)

        self.load_model()
        numpy.random.seed(546)
        self.logger.info('Creating data loader')
        self.data_loader = Build_dv_y_train_data_loader(self.cfg, dv_y_cfg)

    def load_model(self):
        dv_y_cfg = self.dv_y_cfg

        self.model = Build_DV_Y_model(dv_y_cfg)
        self.model.torch_initialisation()
        if dv_y_cfg.prev_nnets_file_name is None:
            prev_nnets_file_name = dv_y_cfg.nnets_file_name
        else:
            prev_nnets_file_name = dv_y_cfg.prev_nnets_file_name
        self.logger.info('Loading %s for testing' % prev_nnets_file_name)
        self.model.load_nn_model(prev_nnets_file_name)
        self.model.print_model_parameter_sizes(self.logger)
        self.model.print_output_dim_values(self.logger)

        self.model.eval()

    def test(self, plot_loss=True):
        for utter_tvt_name in ['train', 'valid', 'test']:
            for batch_idx in range(self.total_num_batch):
                self.action_per_batch(utter_tvt_name)

        if plot_loss:
            self.plot()

    def action_per_batch(self, utter_tvt_name):
        pass

    def plot(self):
        pass

class Build_Wav_VUV_Loss_Test(Build_DV_Y_Testing_Base):
    '''
    For waveform-based models only; Based on vuv_SBM
    Compare the average loss in voiced or unvoiced region
    Return a dict: key is train/valid/test, value is a list for plotting
        loss_mean_dict[utter_tvt_name] = loss_mean_list
        in loss_mean_list: loss_mean vs voicing (increasing)
    '''
    def __init__(self, cfg, dv_y_cfg, fig_file_name):
        super().__init__(cfg, dv_y_cfg)
        self.total_num_batch = 1000
        self.fig_file_name = fig_file_name
        self.tvt_list = ['train', 'valid', 'test']

        self.loss_dict = {}
        self.accu_dict = {}
        for utter_tvt_name in self.tvt_list:
            self.loss_dict[utter_tvt_name] = {i:[] for i in range(dv_y_cfg.input_data_dim['M']+1)}
            self.accu_dict[utter_tvt_name] = {i:[] for i in range(dv_y_cfg.input_data_dim['M']+1)}

    def action_per_batch(self, utter_tvt_name):
        # Make feed_dict for evaluation
        feed_dict, batch_size = self.data_loader.make_feed_dict(utter_tvt_name=utter_tvt_name)
        batch_loss_SB = self.model.gen_SB_loss_value(feed_dict=feed_dict)
        logit_SBD = self.model.gen_logit_SBD_value(feed_dict=feed_dict)
        batch_accu_SB = self.cal_accuracy(logit_SBD, feed_dict['one_hot_S'])
        vuv_SBM = feed_dict['vuv_SBM']

        vuv_sum_SB = numpy.sum(vuv_SBM, axis=2)

        for s in range(self.dv_y_cfg.input_data_dim['S']):
            for b in range(self.dv_y_cfg.input_data_dim['B']):
                self.loss_dict[utter_tvt_name][vuv_sum_SB[s,b]].append(batch_loss_SB[s*self.dv_y_cfg.input_data_dim['B']+b])
                self.accu_dict[utter_tvt_name][vuv_sum_SB[s,b]].append(batch_accu_SB[s,b])

    def cal_accuracy(self, logit_SBD, one_hot_S):
        # Return binary results of classification
        # Return a matrix of S*B, contains 1. and 0.
        pred_SB = numpy.argmax(logit_SBD, axis=2)
        S, B = pred_SB.shape
        one_hot_SB = numpy.repeat(one_hot_S, B).reshape([S,B])
        batch_accu_SB = (pred_SB == one_hot_SB).astype(float)
        return batch_accu_SB

    def plot(self):
        '''
        x-axis is percentage of vuv; [num of voiced]/M
        Easy to plot together with different M
        '''
        M = self.dv_y_cfg.input_data_dim['M']
        
        loss_mean_dict = {}
        accu_mean_dict = {}
        # Make x-axis
        x_list = numpy.arange(M+1) / float(M)
        loss_mean_dict['x'] = x_list
        accu_mean_dict['x'] = x_list

        for utter_tvt_name in self.tvt_list:
            loss_mean_dict[utter_tvt_name] = []
            accu_mean_dict[utter_tvt_name] = []

        for utter_tvt_name in self.tvt_list:
            for i in range(M+1):
                loss_mean_dict[utter_tvt_name].append(numpy.mean(self.loss_dict[utter_tvt_name][i]))
                accu_mean_dict[utter_tvt_name].append(numpy.mean(self.accu_dict[utter_tvt_name][i]))

        graph_plotter = Graph_Plotting()
        graph_plotter.single_plot(self.fig_file_name, [loss_mean_dict['x']]*3,[loss_mean_dict['train'], loss_mean_dict['valid'],loss_mean_dict['test']], ['train', 'valid', 'test'],title='Cross Entropy against VUV', x_label='VUV %', y_label='CE')
        accu_fig_file_name = self.fig_file_name.replace('loss','accu')
        graph_plotter.single_plot(accu_fig_file_name, [loss_mean_dict['x']]*3,[accu_mean_dict['train'], accu_mean_dict['valid'],accu_mean_dict['test']], ['train', 'valid', 'test'],title='Accuracy against VUV', x_label='VUV %', y_label='Accuracy')

        self.loss_mean_dict = loss_mean_dict
        self.accu_mean_dict = accu_mean_dict
        self.logger.info('Print vuv and loss_test')
        print(loss_mean_dict['x'])
        print(loss_mean_dict['test'])
        self.logger.info('Print vuv and accu_test')
        print(accu_mean_dict['x'])
        print(accu_mean_dict['test'])

class DV_Y_Config_CMP_2_Wav(object):
    """
    A temporary class for cmp_vuv_test
    Generate a wav_cfg from cmp_cfg

    For reference:
    self.input_data_dim['T_S'] = 200 # Number of frames at 200Hz
    self.input_data_dim['T_B'] = 40 # T
    self.input_data_dim['B_shift'] = 1
    self.input_data_dim['D'] =  self.input_data_dim['T_B'] * self.cfg.nn_feature_dims['cmp']
    self.input_data_dim['B'] = int((self.input_data_dim['T_S'] - self.input_data_dim['T_B']) / self.input_data_dim['B_shift']) + 1
    """
    def __init__(self, dv_y_cfg):
        super().__init__()
        self.out_feat_list = ['vuv_SBM']
        self.input_data_dim = {}
        for k in ['S','B','M']:
            self.input_data_dim[k] = dv_y_cfg.input_data_dim[k]

        self.input_data_dim['S'] = dv_y_cfg.input_data_dim['S']
        self.input_data_dim['B'] = dv_y_cfg.input_data_dim['B']
        self.input_data_dim['M'] = dv_y_cfg.input_data_dim['T_B']

        self.input_data_dim['B_shift'] = 80
        self.input_data_dim['T_M'] = 640
        self.input_data_dim['M_shift'] = 80

        self.input_data_dim['T_B'] = self.input_data_dim['T_M'] + (self.input_data_dim['M']-1) * self.input_data_dim['M_shift']
        self.input_data_dim['T_S'] = self.input_data_dim['T_B'] + (self.input_data_dim['B']-1) * self.input_data_dim['B_shift']

        self.log_except_list = []

class Build_CMP_VUV_Loss_Test(Build_Wav_VUV_Loss_Test):
    '''
    For vocoder-based models only; 
    Load vuv_SBM based on position of h_SBD
    Based on vuv_SBM
    Compare the average loss in voiced or unvoiced region
    Return a dict: key is train/valid/test, value is a list for plotting
        loss_mean_dict[utter_tvt_name] = loss_mean_list
        in loss_mean_list: loss_mean vs voicing (increasing)
    '''
    def __init__(self, cfg, dv_y_cfg, fig_file_name):
        dv_y_cfg.input_data_dim['M'] = dv_y_cfg.input_data_dim['T_B']
        super().__init__(cfg, dv_y_cfg, fig_file_name)

        self.wav_cfg = DV_Y_Config_CMP_2_Wav(dv_y_cfg)
        self.cmp_cfg = dv_y_cfg

        self.cmp_data_loader = self.data_loader.dv_y_data_loader
        self.wav_data_loader = self.build_wav_data_loader() # This is a single file loader!

    def build_wav_data_loader(self):
        '''
        This is a single file loader!
        '''
        from frontend_mw545.data_loader import Build_dv_y_wav_data_loader_Single_File
        wav_data_loader = Build_dv_y_wav_data_loader_Single_File(self.cfg, self.wav_cfg)
        return wav_data_loader

    def frame_2_sample_number(self, frame_number):
        '''
        Convert start_frame_no_sil to start_sample_no_sil
        '''
        wav_data_dim = self.wav_cfg.input_data_dim
        sample_number = frame_number * wav_data_dim['M_shift'] - wav_data_dim['T_M'] * 0.5
        return int(sample_number)

    def action_per_batch(self, utter_tvt_name):
        '''
        1. Draw file list
        2. For each file
            2.1 Make cmp_BD, while get start_frame_no_sil
            2.2 Convert to start_sample_no_sil
                2.2.1 Each cmp frame is at centre of sample window
                2.2.2 could be negative; got silence padding in data so no problem
            2.3 Extract vuv_BM
        '''
        dv_y_cfg = self.wav_cfg
        batch_speaker_id_list = self.data_loader.dv_selecter.draw_n_speakers(dv_y_cfg.input_data_dim['S'])
        file_id_list = self.data_loader.draw_n_files(batch_speaker_id_list, utter_tvt_name)

        one_hot, one_hot_S, batch_size = self.data_loader.dv_selecter.make_one_hot(batch_speaker_id_list)
        feed_dict = {}
        feed_dict['one_hot'] = one_hot
        feed_dict['one_hot_S'] = one_hot_S

        extra_file_len_ratio_list = numpy.random.rand(dv_y_cfg.input_data_dim['S'])

        cmp_SBD = numpy.zeros((dv_y_cfg.input_data_dim['S'], dv_y_cfg.input_data_dim['B'], self.cmp_cfg.input_data_dim['D']))
        vuv_SBM = numpy.zeros((dv_y_cfg.input_data_dim['S'], dv_y_cfg.input_data_dim['B'], self.wav_cfg.input_data_dim['M']))

        for i, file_id in enumerate(file_id_list):
            extra_file_len_ratio = extra_file_len_ratio_list[i]
            cmp_resil_norm_file_name = os.path.join(self.cmp_data_loader.cmp_dir, file_id+'.cmp')
            cmp_BD, start_frame_no_sil = self.cmp_data_loader.make_cmp_BD(cmp_resil_norm_file_name, None, extra_file_len_ratio)
            cmp_SBD[i] = cmp_BD

            start_sample_no_sil = self.frame_2_sample_number(start_frame_no_sil)
            pitch_resil_norm_file_name = os.path.join(self.cfg.nn_feat_resil_dirs['pitch'], file_id+'.pitch')
            tau_BM, vuv_BM = self.wav_data_loader.make_tau_BM(pitch_resil_norm_file_name, start_sample_no_sil)
            vuv_SBM[i] = vuv_BM

        feed_dict['h'] = cmp_SBD

        batch_loss_SB = self.model.gen_SB_loss_value(feed_dict=feed_dict)
        logit_SBD = self.model.gen_logit_SBD_value(feed_dict=feed_dict)
        batch_accu_SB = self.cal_accuracy(logit_SBD, feed_dict['one_hot_S'])

        vuv_sum_SB = numpy.sum(vuv_SBM, axis=2)

        for s in range(self.dv_y_cfg.input_data_dim['S']):
            for b in range(self.dv_y_cfg.input_data_dim['B']):
                self.loss_dict[utter_tvt_name][vuv_sum_SB[s,b]].append(batch_loss_SB[s*self.dv_y_cfg.input_data_dim['B']+b])
                self.accu_dict[utter_tvt_name][vuv_sum_SB[s,b]].append(batch_accu_SB[s,b])
