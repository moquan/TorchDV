# exp_dv_y.py

# This file uses dv_cmp experiments to slowly progress with pytorch

import os, sys, pickle, time, shutil, logging, copy
import math, numpy, scipy, scipy.stats

from frontend_mw545.modules import make_logger, read_file_list, log_class_attri, Graph_Plotting, List_Random_Loader, File_List_Selecter
from frontend_mw545.data_io import Data_File_IO, Data_List_File_IO, Data_Meta_List_File_IO

from nn_torch.torch_models import Build_DV_Y_model
from frontend_mw545.data_loader import Build_dv_y_train_data_loader, Build_dv_TTS_selecter

from exp_mw545.exp_base import Build_Model_Trainer_Base

#########
# Tools #
#########

class DV_Calculator(object):
    """ handy calculator functions for dv """
    def __init__(self):
        super().__init__()
        pass

    def cal_accuracy_S(self, logit_SD, one_hot_S):
        pred_S = numpy.argmax(logit_SD, axis=1)
        total_num = pred_S.size
        correct_num = numpy.sum(pred_S==one_hot_S)
        return correct_num/float(total_num)

    def cal_accuracy_SB(self, logit_SBD, one_hot_SB, mask_SB=None):
        pred_SB = numpy.argmax(logit_SBD, axis=2)
        if mask_SB is None:
            total_num = pred_SB.size
            correct_num = numpy.sum(pred_SB==one_hot_SB)
        else:
            total_num = numpy.sum(mask_SB)
            correct_num = numpy.sum(numpy.multiply(mask_SB, (pred_SB==one_hot_SB)))
        return correct_num/float(total_num)

    def compute_SBD_distance(self, p_SBD_1, p_SBD_2, distance_name='entropy'):
        '''
        Compute total distance between 2 sets of vectors
        '''
        S = p_SBD_1.shape[0]
        B = p_SBD_1.shape[1]

        distance_sum = 0.
        for s in range(S):
            for b in range(B):
                if distance_name == 'entropy':
                    distance_sum += scipy.stats.entropy(p_SBD_1[s,b], p_SBD_2[s,b])
        # ce_mean = ce_sum / float(S*B)
        return distance_sum

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

        if dv_y_cfg.prev_nnets_file_name is not None:
            self.logger.info('Loading %s for retraining' % dv_y_cfg.prev_nnets_file_name)
            self.model.load_nn_model_optim(dv_y_cfg.prev_nnets_file_name)
        self.model.print_model_parameter_sizes(self.logger)
        self.model.print_output_dim_values(self.logger)

        self.dv_calculator = DV_Calculator()

    def build_data_loader(self):
        self.data_loader = Build_dv_y_train_data_loader(self.cfg, self.train_cfg)

    def eval_action_epoch(self, epoch_num):
        '''
        Additional Accuracy Actions
        '''
        output_string = {'loss':'epoch %i train & valid & test loss:' % epoch_num, 'accuracy':'epoch %i train & valid & test accu:' % epoch_num, 'win_accuracy':'epoch %i train & valid & test win accu:' % epoch_num,}
        epoch_loss = {}
        epoch_valid_load_time  = 0.
        epoch_valid_model_time = 0.
        epoch_num_batch = self.train_cfg.epoch_num_batch['valid']

        for utter_tvt_name in ['train', 'valid', 'test']:
            total_batch_size = 0.
            total_loss       = 0.
            total_utter_accuracy = 0.
            total_win_accuracy = 0.

            for batch_idx in range(epoch_num_batch):
                batch_load_time, batch_model_time, batch_mean_loss, batch_size, feed_dict = self.eval_action_batch(utter_tvt_name)
                total_loss += batch_mean_loss * float(batch_size)
                total_batch_size += batch_size
                epoch_valid_load_time  += batch_load_time
                epoch_valid_model_time += batch_model_time

                if self.train_cfg.classify_in_training:
                    logit_SD  = self.model.gen_logit_SD_value(feed_dict=feed_dict)
                    batch_mean_utter_accuracy = self.dv_calculator.cal_accuracy_S(logit_SD, feed_dict['one_hot_S'])
                    total_utter_accuracy += batch_mean_utter_accuracy
                    logit_SBD  = self.model.gen_logit_SBD_value(feed_dict=feed_dict)
                    batch_mean_win_accuracy = self.dv_calculator.cal_accuracy_SB(logit_SBD, feed_dict['one_hot_S_B'], feed_dict['output_mask_S_B'])
                    total_win_accuracy += batch_mean_win_accuracy * float(batch_size)

            mean_loss = total_loss/float(total_batch_size)
            epoch_loss[utter_tvt_name] = mean_loss
            output_string['loss'] = output_string['loss'] + ' & %.4f' % (mean_loss)

            if self.train_cfg.classify_in_training:
                mean_utter_accuracy = total_utter_accuracy/float(epoch_num_batch)
                output_string['accuracy'] = output_string['accuracy'] + ' & %.4f' % (mean_utter_accuracy)
                mean_win_accuracy = total_win_accuracy/float(total_batch_size)
                output_string['win_accuracy'] = output_string['win_accuracy'] + ' & %.4f' % (mean_win_accuracy)

        self.logger.info('valid load & model time: %.2f & %.2f' % (epoch_valid_load_time, epoch_valid_model_time))
        self.logger.info(output_string['loss'])
        if self.train_cfg.classify_in_training:
            self.logger.info(output_string['accuracy'])
            self.logger.info(output_string['win_accuracy'])
        return epoch_loss

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

            if epoch_num > 1:
                self.logger.info('start Training Epoch '+str(epoch_num))
                self.model.train()
                for batch_idx in range(self.train_cfg.epoch_num_batch['train']):
                    self.model.update_parameters(feed_dict=feed_dict)
                epoch_train_time = time.time()
            else:
                epoch_train_time = epoch_start_time

            self.logger.info('start Evaluating Epoch '+str(epoch_num))
            self.model.eval()
            with self.model.no_grad():
                batch_mean_loss = self.model.gen_loss_value(feed_dict=feed_dict)
                logit_SD  = self.model.gen_logit_SD_value(feed_dict=feed_dict)
                batch_mean_utter_accuracy = self.dv_calculator.cal_accuracy_S(logit_SD, feed_dict['one_hot_S'])
                logit_SBD  = self.model.gen_logit_SBD_value(feed_dict=feed_dict)
                batch_mean_win_accuracy = self.dv_calculator.cal_accuracy_SB(logit_SBD, feed_dict['one_hot_S_B'], feed_dict['output_mask_S_B'])

            self.debug_test(feed_dict)

            self.logger.info('epoch %i loss: %.4f' %(epoch_num, batch_mean_loss))
            self.logger.info('epoch %i utter accu: %.4f' %(epoch_num, batch_mean_utter_accuracy))
            self.logger.info('epoch %i win accu: %.4f' %(epoch_num, batch_mean_win_accuracy))
            is_finish = self.validation_action(batch_mean_loss, epoch_num)
            epoch_valid_time = time.time()

            self.logger.info('epoch %i train & valid time %.2f & %.2f' %(epoch_num, (epoch_train_time - epoch_start_time), (epoch_valid_time - epoch_train_time)))
            self.train_cfg.additional_action_epoch(self.logger, self.model)

            if batch_mean_loss == 0:
                self.logger.info('Reach 0 loss, best epoch %i, best loss %.4f' % (self.best_epoch_num, self.best_epoch_loss))
                self.logger.info('Model: %s' % (nnets_file_name))
                return None

            if numpy.isnan(batch_mean_loss):
                self.logger.info('Reach NaN loss, best epoch %i, best loss %.4f' % (self.best_epoch_num, self.best_epoch_loss))
                self.logger.info('Model: %s' % (nnets_file_name))
                self.model.detect_nan_model_parameters(self.logger)
                return None

        self.logger.info('Reach num_train_epoch, best epoch %i, best loss %.4f' % (self.best_epoch_num, self.best_epoch_loss))
        self.logger.info('Model: %s' % (nnets_file_name))

    def debug_test(self, feed_dict):
        '''
        Temporaty place for debug tests
        running per epoch in self.train_single_batch
        '''
        # for name, param in self.model.nn_model.named_parameters():
        #     print(name)
        #     print(param.grad)
        pass

class Build_DV_Y_Testing(object):
    """ Wrapper class for various tests of dv_y models """
    def __init__(self, cfg, dv_y_cfg):
        super().__init__()
        self.cfg = cfg
        self.test_cfg = dv_y_cfg

    def cross_entropy_accuracy_test(self):
        test_fn = Build_DV_Y_CE_Accu_Test(self.cfg, self.test_cfg)
        test_fn.test()

    def gen_dv(self, output_dir):
        test_fn = Build_DV_Generator(self.cfg, self.test_cfg, output_dir)
        test_fn.test()

    def vuv_loss_test(self, fig_file_name='/home/dawna/tts/mw545/Export_Temp/PNG_out/vuv_loss.png'):
        if self.test_cfg.y_feat_name == 'wav':
            test_fn = Build_Wav_VUV_Loss_Test(self.cfg, self.test_cfg, fig_file_name)
        elif self.test_cfg.y_feat_name == 'cmp':
            test_fn = Build_CMP_VUV_Loss_Test(self.cfg, self.test_cfg, fig_file_name)
        test_fn.test()

    def positional_test(self, fig_file_name='/home/dawna/tts/mw545/Export_Temp/PNG_out/positional.png'):
        if self.test_cfg.y_feat_name == 'wav':
            test_fn = Build_Positional_Wav_Test(self.cfg, self.test_cfg, fig_file_name)
        elif self.test_cfg.y_feat_name == 'cmp':
            test_fn = Build_Positional_CMP_Test(self.cfg, self.test_cfg, fig_file_name)
        test_fn.test()

    def number_secs_accu_test(self, output_dir='/home/dawna/tts/mw545/Export_Temp'):
        # if self.test_cfg.y_feat_name == 'cmp':
        test_fn = Build_DV_Y_Number_Seconds_Accu_Test(self.cfg, self.test_cfg, output_dir)
        test_fn.test()

class Build_DV_Y_Testing_Base(object):
    """Base class of tests of dv_y models"""
    def __init__(self, cfg, dv_y_cfg):
        super().__init__()
        self.logger = make_logger("test_model")
        self.cfg = cfg
        self.dv_y_cfg = dv_y_cfg
        self.dv_y_cfg.input_data_dim['S'] = 1
        log_class_attri(self.dv_y_cfg, self.logger, except_list=self.dv_y_cfg.log_except_list)

        self.load_model()
        numpy.random.seed(546)
        self.logger.info('Creating data loader')
        self.data_loader = Build_dv_y_train_data_loader(self.cfg, self.dv_y_cfg)

    def load_model(self):
        dv_y_cfg = self.dv_y_cfg

        self.model = Build_DV_Y_model(dv_y_cfg)
        self.model.torch_initialisation()
        # if dv_y_cfg.prev_nnets_file_name is None:
        #     prev_nnets_file_name = dv_y_cfg.nnets_file_name
        # else:
        #     prev_nnets_file_name = dv_y_cfg.prev_nnets_file_name
        nnets_file_name = dv_y_cfg.nnets_file_name
        self.logger.info('Loading %s for testing' % nnets_file_name)
        self.model.load_nn_model(nnets_file_name)
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

class Build_DV_Y_CE_Accu_Test(Build_DV_Y_Testing_Base):
    """ 
    use test files only
    compute win_ce, utter_accuracy, win_accuracy 
    """
    def __init__(self, cfg, dv_y_cfg):
        super().__init__(cfg, dv_y_cfg)
        self.logger.info('DV_Y_CE_Accu_Test')
        # use test files only
        self.speaker_id_list = self.cfg.speaker_id_list_dict['train']
        # compute win_ce, utter_accuracy, win_accuracy
        self.model.build_loss_function(train_by_window=True, use_voiced_only=False)

        self.dv_calculator = DV_Calculator()

    def test(self):
        total_batch_size = 0.
        total_num_files = 0.
        total_win_ce = 0.
        total_utter_accuracy = 0.
        total_win_accuracy = 0.

        for speaker_id in self.speaker_id_list:
            file_id_list = self.data_loader.dv_selecter.file_list_dict[(speaker_id, 'test')]
            for file_id in file_id_list:
                feed_dict, batch_size = self.data_loader.make_feed_dict(file_id_list=[file_id],start_sample_list=[0])
                mean_win_ce = self.compute_win_ce(feed_dict)
                utter_accuracy = self.compute_utter_accuracy(feed_dict)
                mean_win_accuracy = self.compute_win_accuracy(feed_dict)

                total_batch_size += batch_size
                total_num_files += 1.
                total_win_ce += mean_win_ce * float(batch_size)
                total_utter_accuracy += utter_accuracy
                total_win_accuracy += mean_win_accuracy * float(batch_size)

        self.logger.info('Results of model %s' % self.dv_y_cfg.nnets_file_name)
        self.logger.info('batch size & win_ce & win_accuracy & utter_accuracy')
        self.logger.info('%f & %.4f & %.4f & %.4f' %(total_batch_size, total_win_ce/total_batch_size, total_win_accuracy/total_batch_size, total_utter_accuracy/total_num_files))

    def compute_win_ce(self, feed_dict):
        SB_loss = self.model.gen_SB_loss_value(feed_dict)
        return numpy.mean(SB_loss)

    def compute_utter_accuracy(self, feed_dict):
        logit_SD = self.model.gen_logit_SD_value(feed_dict)
        one_hot_S = feed_dict['one_hot_S']
        return self.dv_calculator.cal_accuracy_S(logit_SD, one_hot_S)

    def compute_win_accuracy(self, feed_dict):
        logit_SBD = self.model.gen_logit_SBD_value(feed_dict)
        one_hot_SB = feed_dict['one_hot_S_B']
        return self.dv_calculator.cal_accuracy_SB(logit_SBD, one_hot_SB)

class Build_DV_Y_Number_Seconds_Accu_Test(Build_DV_Y_Testing_Base):
    """
    Build_DV_Y_Number_Seconds_Accu_Test
    Test the influence of number of utterances
    1. Accuracy test
        Plot mean and variance of accuracy, against number of utterances
    2. Speaker embedding test
        Plot mean cosine distance from speaker's average, against number of utterances
    Note:
        time is proportional to num_seconds_to_test
        e.g. cmp model, 5s test, num_draw_per_speaker=1, 2min; 50s, 20min
    """
    def __init__(self, cfg, dv_y_cfg, output_dir):
        super().__init__(cfg, dv_y_cfg)
        self.output_dir = output_dir
        self.logger.info('DV_Y_Number_Seconds_Accuracy_Test')

        self.speaker_id_list = self.cfg.speaker_id_list_dict['train']
        self.dv_calculator = DV_Calculator()

        self.list_num_seconds_to_test = [5,10,15,20,25,30,35,40,45,50,55]
        self.num_accuracies_mean_std = 30
        self.num_draw_per_speaker = 5

    def test(self):
        mean_list = []
        std_list  = []
        for num_secs in self.list_num_seconds_to_test:
            self.logger.info('Testing %i seconds' % num_secs)
            m, s = self.accuracy_test(num_secs)
            self.logger.info('Results are %s %s' % (str(m), str(s)))
            mean_list.append(m)
            std_list.append(s)
        print(mean_list)
        print(std_list)

    def accuracy_test(self, num_secs):
        '''
        find mean and std of multiple accuracy values
        '''
        mean_accuracy_list = []
        for i in range(self.num_accuracies_mean_std):
            mean_accuracy = self.accuracy_single_test(num_secs)
            mean_accuracy_list.append(mean_accuracy)
        m = numpy.mean(mean_accuracy_list)
        s = numpy.std(mean_accuracy_list,ddof=1)
        return m,s

    def accuracy_single_test(self, num_secs):
        '''
        find mean accuracy of multiple draws, up to num_secs
        '''
        total_accuracy = 0.
        total_batch_size = 0.

        for speaker_id in self.speaker_id_list:
            for i in range(self.num_draw_per_speaker):
                file_id_str = self.data_loader.dv_selecter.draw_n_seconds(speaker_id, 'test', num_secs)
                utter_accuracy = self.handle_file_id_str(file_id_str)
                # file_id_list = file_id_str.split('|')
                # n = len(file_id_list)
                # feed_dict, batch_size = self.data_loader.make_feed_dict(file_id_list=[file_id_str],start_sample_list=[0])
                # utter_accuracy = self.compute_utter_accuracy(feed_dict)

                total_accuracy += utter_accuracy
                total_batch_size += 1

        return (total_accuracy / total_batch_size)

    def handle_file_id_str(self, file_id_str):
        '''
        1. find average lambda_D of multiple files
        2. get logit and accuracy
        '''
        file_id_list = file_id_str.split('|')
        n = len(file_id_list)

        if n == 1:
            feed_dict, batch_size = self.data_loader.make_feed_dict(file_id_list=[file_id_str],start_sample_list=[0])
            logit_SD = self.model.gen_logit_SD_value(feed_dict)
        else:
            lambda_SBD_list = []
            for file_id in file_id_list:
                feed_dict, batch_size = self.data_loader.make_feed_dict(file_id_list=[file_id],start_sample_list=[0])
                lambda_SBD = self.model.gen_lambda_SBD_value(feed_dict)
                lambda_SBD_list.append(lambda_SBD)
            lambda_SBD_long = numpy.concatenate(lambda_SBD_list, axis=1)
            lambda_SD = numpy.mean(lambda_SBD_long, axis=1)
            feed_dict['lambda_SD'] = lambda_SD
            logit_SD = self.model.gen_logit_SD_value_from_lambda_SD_value(feed_dict)

        one_hot_S = feed_dict['one_hot_S']
        utter_accuracy = self.dv_calculator.cal_accuracy_S(logit_SD, one_hot_S)
        return utter_accuracy

        


class Build_DV_Generator(Build_DV_Y_Testing_Base):
    """
    Build_CMP_Generator
    For vocoder-based system only
    Generate a dict of speaker representations:
        dv_file_dict[file_id] = (dv, num_windows)
        dv_spk_dict[file_id] = dv
        dv is a 1D numpy array
    """
    def __init__(self, cfg, dv_y_cfg, output_dir=None):
        super().__init__(cfg, dv_y_cfg)
        self.output_dir = output_dir
        self.logger.info('Build_CMP_Generator')

        # change S to 1; re-build data_loader
        dv_y_cfg.input_data_dim['S'] = 1
        self.dv_tts_selector = Build_dv_TTS_selecter(self.cfg, dv_y_cfg)
        self.speaker_id_list = self.cfg.speaker_id_list_dict['all']

        self.dv_file_dict = {}
        self.dv_spk_dict = {}
        self.DMLFIO = Data_Meta_List_File_IO(cfg)


    def test(self):
        for k in self.dv_tts_selector.file_list_dict:
            # print("%s:%i" %(k,len((self.dv_tts_selector.file_list_dict[k]))))
            if self.dv_y_cfg.run_mode == 'debug':
                # Keep 10 files only in debug mode
                self.dv_tts_selector.file_list_dict[k] = self.dv_tts_selector.file_list_dict[k][:10]

        for speaker_id in self.speaker_id_list:
            file_id_list = self.dv_tts_selector.file_list_dict[(speaker_id, 'SR')]
            for file_id in file_id_list:
                self.logger.info('Processing %s' % file_id)
                feed_dict, batch_size = self.data_loader.make_feed_dict(file_id_list=[file_id],start_sample_list=[0])
                lambda_SD = self.model.gen_lambda_SD_value(feed_dict)
                dv_file = lambda_SD[0]
                self.dv_file_dict[file_id] = (batch_size, dv_file)

        for speaker_id in self.speaker_id_list:
            self.logger.info('Processing %s' % speaker_id)
            dv_speaker = numpy.zeros(self.dv_y_cfg.dv_dim)
            total_num_frames = 0.
            file_id_list = self.dv_tts_selector.file_list_dict[(speaker_id, 'SR')]
            for file_id in file_id_list:
                num_frames = self.dv_file_dict[file_id][0]
                total_num_frames += num_frames
                dv_speaker += self.dv_file_dict[file_id][1] * float(num_frames)

            self.dv_spk_dict[speaker_id] = dv_speaker / total_num_frames

        self.save_dv_files(self.dv_y_cfg.exp_dir)
        if self.output_dir is not None:
            self.save_dv_files(self.output_dir)

    def action_per_file(self, file_id):
        self.logger.info('Processing %s' % file_id)
        speaker_id = file_id.split('_')[0]

        feed_dict, batch_size = self.data_loader.make_feed_dict(file_id_list=[file_id])
        lambda_SD = self.model.gen_lambda_SD_value(feed_dict=feed_dict)

        self.dv_file_dict[file_id] = (batch_size, lambda_SD[0])

    def save_dv_files(self, output_dir):

        dv_spk_dict_file  = os.path.join(output_dir, 'dv_spk_dict.dat')
        dv_spk_dict_text  = os.path.join(output_dir, 'dv_spk_dict.txt')
        dv_file_dict_file = os.path.join(output_dir, 'dv_file_dict.dat')
        dv_file_dict_text = os.path.join(output_dir, 'dv_file_dict.txt')
        self.logger.info('Saving to files:')
        print(dv_spk_dict_file)
        print(dv_spk_dict_text)
        print(dv_file_dict_file)
        print(dv_file_dict_text)
        self.DMLFIO.write_dv_spk_values_to_file(self.dv_spk_dict, dv_spk_dict_file, file_type='pickle')
        self.DMLFIO.write_dv_spk_values_to_file(self.dv_spk_dict, dv_spk_dict_text, file_type='text')
        self.DMLFIO.write_dv_file_values_to_file(self.dv_file_dict, dv_file_dict_file, file_type='pickle')
        self.DMLFIO.write_dv_file_values_to_file(self.dv_file_dict, dv_file_dict_text, file_type='text')

class Build_Positional_CMP_Test(Build_DV_Y_Testing_Base):
    """ 
    use test files only
    compute win_ce, utter_accuracy, win_accuracy 
    """
    def __init__(self, cfg, dv_y_cfg, fig_file_name):
        
        super().__init__(cfg, dv_y_cfg)
        self.logger.info('Build_Positional_CMP_Test')
        # use test files only
        self.file_id_list = read_file_list(cfg.file_id_list_file['dv_pos_test']) # 186^5 files
        self.cmp_dir = '/data/mifs_scratch/mjfg/mw545/dv_pos_test/cmp_shift_resil_norm'

        self.max_distance   = 50
        # self.distance_space = 1
        self.dv_calculator = DV_Calculator()

    def test(self):
        total_batch_size = 0.
        total_distance_list = numpy.zeros(self.max_distance)

        for file_id in self.file_id_list:
            distance_list, B = self.compute_distance_list(file_id)
            total_batch_size += B
            total_distance_list += distance_list

        mean_distance_list = total_distance_list/total_batch_size

        self.logger.info('Results of model %s' % self.dv_y_cfg.nnets_file_name)
        self.logger.info('batch size & mean_distance')
        self.logger.info('%f & %s' %(total_batch_size, mean_distance_list))
        print(mean_distance_list)

    def compute_distance_list(self, file_id):
        distance_list = numpy.zeros(self.max_distance)
        speaker_id = file_id.split('_')[0]
        feed_dict, batch_size = self.data_loader.make_feed_dict(file_id_list=[file_id])
        p_SBD_0 = self.model.gen_p_SBD_value(feed_dict)

        for i in range(0, self.max_distance):
            cmp_resil_norm_file_name = '%s/%s/%s.cmp.%i' % (self.cmp_dir, speaker_id, file_id, i+1)
            cmp_BD, B, start_sample_number = self.data_loader.dv_y_data_loader.make_cmp_BD_data_single_file(cmp_resil_norm_file_name)
            cmp_SBD = numpy.expand_dims(cmp_BD, 0)
            feed_dict['h'] = cmp_SBD
            p_SBD_i = self.model.gen_p_SBD_value(feed_dict)
            distance_list[i] = self.dv_calculator.compute_SBD_distance(p_SBD_0, p_SBD_i, 'entropy') # This is sum, not mean

        return distance_list, batch_size





class Build_Wav_VUV_Loss_Test(Build_DV_Y_Testing_Base):
    '''
    For waveform-based models only; Based on vuv_SBM
    Compare the average loss in voiced or unvoiced region
    Return a dict: key is train/valid/test, value is a list for plotting
        loss_mean_dict[utter_tvt_name] = loss_mean_list
        in loss_mean_list: loss_mean vs voicing (increasing)
    '''
    def __init__(self, cfg, dv_y_cfg, fig_file_name):
        if 'vuv_SBM' not in dv_y_cfg.out_feat_list:
            dv_y_cfg.out_feat_list.append('vuv_SBM')
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
        batch_loss_SB = self.model.gen_SB_loss_value(feed_dict=feed_dict) # This is a 1D vector!!
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
        # This is a single wav file loader!
        # Use this to load vuv_BM and form vuv_SBM
        self.wav_data_loader = self.build_wav_data_loader() 

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
            speaker_id = file_id.split('_')[0]
            extra_file_len_ratio = extra_file_len_ratio_list[i]
            cmp_resil_norm_file_name = os.path.join(self.cmp_data_loader.cmp_dir, speaker_id, file_id+'.cmp')
            cmp_BD, start_frame_no_sil = self.cmp_data_loader.make_cmp_BD(cmp_resil_norm_file_name, None, extra_file_len_ratio)
            cmp_SBD[i] = cmp_BD

            start_sample_no_sil = self.frame_2_sample_number(start_frame_no_sil)
            pitch_resil_norm_file_name = os.path.join(self.cfg.nn_feat_resil_dirs['pitch'], speaker_id, file_id+'.pitch')
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

class Build_Positional_Wav_Test(Build_DV_Y_Testing_Base):
    '''
    For waveform-based models only
    Compare the average loss in voiced or unvoiced region
    Return a dict: key is train/valid/test, value is a list for plotting
        loss_mean_dict[utter_tvt_name] = loss_mean_list
        in loss_mean_list: loss_mean vs voicing (increasing)
    '''
    def __init__(self, cfg, dv_y_cfg, fig_file_name):
        super().__init__(cfg, dv_y_cfg)
        self.logger.info('Build_Positional_Wav_Test')
        self.total_num_batch = 100
        self.fig_file_name = fig_file_name
        self.tvt_list = ['train', 'valid', 'test']

        self.max_distance   = 50
        self.distance_space = 1
        self.num_to_plot = int(self.max_distance / self.distance_space)

        self.loss_dict = {}
        for utter_tvt_name in self.tvt_list:
            self.loss_dict[utter_tvt_name] = {i:[] for i in range(self.num_to_plot)}

        self.wav_dir   = self.cfg.nn_feat_scratch_dirs['wav']

        self.test_file_list = read_file_list(cfg.file_id_list_file['dv_pos_test']) # 186^5 files
        self.test_file_selecter = List_Random_Loader(self.test_file_list)
        self.file_num_silence_samples_dict = self.read_file_list_num_silence_samples()

    def write_file_list_num_silence_samples(self):
        '''
        Call this once only
        Write a Data_Meta_List_File
        1. Read silence data meta
        2. Get file_id_list, self.test_file_list
        3. For file_id in file_id_list, compute num_extra_samples
        4. Write a new extra data meta
        '''
        from frontend_mw545.data_io import Data_Meta_List_File_IO
        DMLF_IO = Data_Meta_List_File_IO(self.cfg)

        file_id_list_dir = self.cfg.file_id_list_dir
        in_file_name  = os.path.join(file_id_list_dir, 'data_meta/file_id_list_num_sil_frame.scp')
        out_file_name = os.path.join(file_id_list_dir, 'data_meta/file_id_list_num_extra_wav_pos_test.scp')
        file_id_list = self.test_file_list

        file_frame_dict = DMLF_IO.read_file_list_num_silence_frame(in_file_name)

        wav_cmp_ratio = int(self.cfg.wav_sr / self.cfg.frame_sr)
        with open(out_file_name, 'w') as f_1:
            for file_id in file_id_list:
                l, x, y = file_frame_dict[file_id]
                num_no_sil_frames  = y-x+1
                num_no_sil_samples = num_no_sil_frames * wav_cmp_ratio
                num_extra_samples = num_no_sil_samples - self.dv_y_cfg.input_data_dim['T_S'] - self.max_distance

                l = '%s %i' %(file_id, num_extra_samples)
                f_1.write(l+'\n')

    def read_file_list_num_silence_samples(self):
        file_id_list_dir = self.cfg.file_id_list_dir
        in_file_name  = os.path.join(file_id_list_dir, 'data_meta/file_id_list_num_sil_frame.scp')
        file_frame_dict = {}

        fid = open(in_file_name)
        for line in fid.readlines():
            line = line.strip()
            if len(line) < 1:
                continue
            x_list = line.split(' ')

            file_id = x_list[0]
            l = int(x_list[1])

            file_frame_dict[file_id] = l
        return file_frame_dict

    def action_per_batch(self, utter_tvt_name):
        '''
        Make feed_dict_0 and shifted feed, generate probability
        Compute CE between p_0 and p_shift
        '''
        dv_y_cfg = self.dv_y_cfg
        S = dv_y_cfg.input_data_dim['S']
        file_id_list = self.test_file_selecter.draw_n_samples(S)
        extra_file_len_ratio_list = numpy.random.rand(S)

        start_sample_list_0 = numpy.zeros(S).astype(int)
        for s in range(S):
            file_id = file_id_list[s]
            extra_file_len_ratio = extra_file_len_ratio_list[s]
            extra_file_len = self.file_num_silence_samples_dict[file_id]
            start_sample_no_sil = int(extra_file_len_ratio * (extra_file_len+1))
            start_sample_list_0[s] = start_sample_no_sil

        feed_dict_0, batch_size = self.data_loader.make_feed_dict(utter_tvt_name=utter_tvt_name,file_id_list=file_id_list, start_sample_list=start_sample_list_0)
        p_SBD_0 = self.model.gen_p_SBD_value(feed_dict=feed_dict_0)

        for i in range(self.num_to_plot):
            start_sample_list = start_sample_list_0 + (i+1) * self.distance_space
            feed_dict, batch_size = self.data_loader.make_feed_dict(utter_tvt_name=utter_tvt_name,file_id_list=file_id_list, start_sample_list=start_sample_list)

            p_SBD_i = self.model.gen_p_SBD_value(feed_dict=feed_dict)
            dist_i = self.compute_distance(p_SBD_0, p_SBD_i)
            self.loss_dict[utter_tvt_name][i].append(dist_i)

    def compute_distance(self, p_SBD_1, p_SBD_2):
        '''
        Compute distance between 2 probability
        '''
        dv_y_cfg = self.dv_y_cfg
        S = dv_y_cfg.input_data_dim['S']
        B = dv_y_cfg.input_data_dim['B']

        ce_sum = 0.
        for s in range(S):
            for b in range(B):
                ce_sum += scipy.stats.entropy(p_SBD_1[s,b], p_SBD_2[s,b])
        ce_mean = ce_sum / float(S*B)
        return ce_mean

    def test(self, plot_loss=True):
        '''
        Shift the input by a few samples
        Compute difference in lambda
        '''
        # self.write_file_list_num_silence_samples()
        numpy.random.seed(546)

        for utter_tvt_name in ['train', 'valid', 'test']:
            for batch_idx in range(self.total_num_batch):
                self.action_per_batch(utter_tvt_name)

        if plot_loss:
            self.plot()

    def plot(self):
        loss_mean_dict = {}
        # Make x-axis
        x_list = numpy.arange(self.distance_space, self.max_distance+1, self.distance_space)
        loss_mean_dict['x'] = x_list

        for utter_tvt_name in self.tvt_list:
            loss_mean_dict[utter_tvt_name] = []

        for utter_tvt_name in self.tvt_list:
            for i in range(self.num_to_plot):
                loss_mean_dict[utter_tvt_name].append(numpy.mean(self.loss_dict[utter_tvt_name][i]))

        graph_plotter = Graph_Plotting()
        graph_plotter.single_plot(self.fig_file_name, [loss_mean_dict['x']]*3,[loss_mean_dict['train'], loss_mean_dict['valid'],loss_mean_dict['test']], ['train', 'valid', 'test'],title='KL against distance', x_label='Distance', y_label='KL')

        self.loss_mean_dict = loss_mean_dict
        self.logger.info('Print distance and loss_test')
        print(loss_mean_dict['x'])
        print(loss_mean_dict['test'])



