# exp_dv_attention.py

import os, sys, pickle, time, shutil, logging, copy
import math, numpy, scipy, scipy.stats

from frontend_mw545.modules import make_logger, read_file_list, log_class_attri, Graph_Plotting, List_Random_Loader, File_List_Selecter
from frontend_mw545.data_io import Data_File_IO, Data_List_File_IO, Data_Meta_List_File_IO

from nn_torch.torch_models import Build_DV_Attention_model
from frontend_mw545.data_loader import Build_dv_TTS_selecter, Build_dv_atten_train_data_loader

from exp_mw545.exp_base import Build_Model_Trainer_Base, DV_Calculator

#############
# Processes #
#############

class Build_DV_Attention_Model_Trainer(Build_Model_Trainer_Base):
    '''
    Tests specific for acoustic-part of speaker representation model
    Main change: additional accuracy tests during eval_action_epoch
    '''
    def __init__(self, cfg, dv_attn_cfg):
        super().__init__(cfg, dv_attn_cfg)

        self.logger.info("--------- Log dv_attn_cfg ---------")
        log_class_attri(dv_attn_cfg, self.logger, except_list=dv_attn_cfg.log_except_list)
        self.logger.info("--------- Log dv_attn_cfg.dv_y_cfg ---------")
        log_class_attri(dv_attn_cfg.dv_y_cfg, self.logger, except_list=dv_attn_cfg.dv_y_cfg.log_except_list)

        self.model = Build_DV_Attention_model(dv_attn_cfg)
        self.model.torch_initialisation()
        self.model.build_optimiser()

        if dv_attn_cfg.retrain_model:
            self.logger.info('Loading %s for retraining' % dv_attn_cfg.prev_nnets_file_name)
            self.model.load_nn_model_optim(dv_attn_cfg.prev_nnets_file_name)
        else:
            if dv_attn_cfg.load_y_model:
                self.logger.info('Loading %s for initialisation' % dv_attn_cfg.prev_y_model_name)
                self.model.load_y_nn_model(dv_attn_cfg.prev_y_model_name)
            if dv_attn_cfg.load_attention_model:
                self.logger.info('Loading %s for initialisation' % dv_attn_cfg.prev_attention_model_name)
                self.model.load_a_nn_model(dv_attn_cfg.prev_attention_model_name)

        self.model.print_model_parameter_sizes(self.logger)
        self.model.print_output_dim_values(self.logger)

        self.dv_calculator = DV_Calculator()

    def build_data_loader(self):
        self.data_loader = Build_dv_atten_train_data_loader(self.cfg, self.train_cfg)

    def eval_action_epoch(self, epoch_num):
        '''
        Additional Accuracy Actions
        '''
        output_string = {'loss':'epoch %i train & valid & test loss:' % epoch_num, 'accuracy':'epoch %i train & valid & test accu:' % epoch_num}
        epoch_loss = {}
        epoch_valid_load_time  = 0.
        epoch_valid_model_time = 0.
        epoch_num_batch = self.train_cfg.epoch_num_batch['valid']

        for utter_tvt_name in ['train', 'valid', 'test']:
            total_batch_size = 0.
            total_loss       = 0.
            total_utter_accuracy = 0.

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

            mean_loss = total_loss/float(total_batch_size)
            epoch_loss[utter_tvt_name] = mean_loss
            output_string['loss'] = output_string['loss'] + ' & %.4f' % (mean_loss)

            if self.train_cfg.classify_in_training:
                mean_utter_accuracy = total_utter_accuracy/float(epoch_num_batch)
                output_string['accuracy'] = output_string['accuracy'] + ' & %.4f' % (mean_utter_accuracy)

        self.logger.info('valid load & model time: %.2f & %.2f' % (epoch_valid_load_time, epoch_valid_model_time))
        self.logger.info(output_string['loss'])
        if self.train_cfg.classify_in_training:
            self.logger.info(output_string['accuracy'])
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

            self.debug_test(feed_dict)

            self.logger.info('epoch %i loss: %.4f' %(epoch_num, batch_mean_loss))
            self.logger.info('epoch %i utter accu: %.4f' %(epoch_num, batch_mean_utter_accuracy))
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
                # lambda_SBD  = self.model.gen_lambda_SBD_value(feed_dict=feed_dict)
                # for s in range(self.train_cfg.input_data_dim['S']):
                #     for b in range(numpy.max(feed_dict['in_lens'])):
                #         print(feed_dict['output_mask_S_B'][s,b])
                #         print(lambda_SBD[s,b])
                # return None

        self.logger.info('Reach num_train_epoch, best epoch %i, best loss %.4f' % (self.best_epoch_num, self.best_epoch_loss))
        self.logger.info('Model: %s' % (nnets_file_name))

    def debug_test(self, feed_dict):
        pass

class Build_DV_Attention_Testing(object):
    """ Wrapper class for various tests of dv_y models """
    def __init__(self, cfg, dv_attn_cfg):
        super().__init__()
        self.cfg = cfg
        self.test_cfg = dv_attn_cfg

    '''
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
    '''

    def number_secs_accu_test(self, output_dir='/home/dawna/tts/mw545/Export_Temp', list_num_seconds_to_test=None):
        # if self.test_cfg.y_feat_name == 'cmp':
        test_fn = Build_DV_Attention_Number_Seconds_Accu_Test(self.cfg, self.test_cfg, output_dir, list_num_seconds_to_test)
        test_fn.test()

    def cross_entropy_accuracy_test(self):
        test_fn = Build_DV_Attention_Accu_Test(self.cfg, self.test_cfg)
        test_fn.test()

    def gen_dv(self, output_dir):
        test_fn = Build_DV_Generator(self.cfg, self.test_cfg, output_dir)
        test_fn.test()


class Build_DV_Attention_Testing_Base(object):
    """Base class of tests of dv_y models"""
    def __init__(self, cfg, dv_attn_cfg):
        super().__init__()
        self.logger = make_logger("test_model")
        self.cfg = cfg
        self.dv_attn_cfg = dv_attn_cfg
        self.dv_attn_cfg.input_data_dim['S'] = 1
        self.dv_attn_cfg.dv_y_cfg.input_data_dim['S'] = 1
        self.logger.info("--------- Log dv_attn_cfg ---------")
        log_class_attri(dv_attn_cfg, self.logger, except_list=dv_attn_cfg.log_except_list)
        self.logger.info("--------- Log dv_attn_cfg.dv_y_cfg ---------")
        log_class_attri(dv_attn_cfg.dv_y_cfg, self.logger, except_list=dv_attn_cfg.dv_y_cfg.log_except_list)

        self.load_model()
        numpy.random.seed(546)
        self.logger.info('Creating data loader')
        self.data_loader = Build_dv_atten_train_data_loader(self.cfg, self.dv_attn_cfg)

    def load_model(self):
        dv_attn_cfg = self.dv_attn_cfg

        self.model = Build_DV_Attention_model(dv_attn_cfg)
        self.model.torch_initialisation()
        # if dv_y_cfg.prev_nnets_file_name is None:
        #     prev_nnets_file_name = dv_y_cfg.nnets_file_name
        # else:
        #     prev_nnets_file_name = dv_y_cfg.prev_nnets_file_name
        nnets_file_name = dv_attn_cfg.nnets_file_name
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


class Build_DV_Attention_Number_Seconds_Accu_Test(Build_DV_Attention_Testing_Base):
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
    def __init__(self, cfg, dv_attn_cfg, output_dir, list_num_seconds_to_test=None):
        super().__init__(cfg, dv_attn_cfg)
        self.output_dir = output_dir
        self.logger.info('DV_Attention_Number_Seconds_Accuracy_Test')

        self.speaker_id_list = self.cfg.speaker_id_list_dict['train']
        self.dv_calculator = DV_Calculator()

        if list_num_seconds_to_test is None:
            self.list_num_seconds_to_test = [5,10,15,20,25,30,35,40,45,50,55]
        else:
            self.list_num_seconds_to_test = list_num_seconds_to_test
        self.num_accuracies_mean_std = 30
        self.num_draw_per_speaker = 5

    def test(self):
        mean_list = []
        std_list  = []
        for num_secs in self.list_num_seconds_to_test:
            self.logger.info('Testing %i seconds' % num_secs)
            m, s = self.accuracy_test(num_secs)
            self.logger.info('Results are %.5f %.5f' % (m, s))
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
            beta_SB1_list = []
            for file_id in file_id_list:
                feed_dict, batch_size = self.data_loader.make_feed_dict(file_id_list=[file_id],start_sample_list=[0])
                lambda_SBD = self.model.gen_lambda_SBD_value(feed_dict)
                lambda_SBD_list.append(lambda_SBD)
                beta_SB1 = self.model.gen_beta_SB1_value(feed_dict)
                beta_SB1_list.append(beta_SB1)
            lambda_SBD_long = numpy.concatenate(lambda_SBD_list, axis=1)
            beta_SB1_long = numpy.concatenate(beta_SB1_list, axis=1)
            beta_exp = numpy.exp(beta_SB1_long)
            alpha_SB1_long = numpy.true_divide(beta_exp, numpy.sum(beta_exp))
            # print(beta_SB1_long[:30])
            # print(alpha_SB1_long[:30])

            lambda_SD = numpy.sum(numpy.multiply(lambda_SBD_long, alpha_SB1_long), axis=1)

            feed_dict['lambda_SD'] = lambda_SD
            logit_SD = self.model.gen_logit_SD_value_from_lambda_SD_value(feed_dict)

        one_hot_S = feed_dict['one_hot_S']
        utter_accuracy = self.dv_calculator.cal_accuracy_S(logit_SD, one_hot_S)
        return utter_accuracy


class Build_DV_Attention_Accu_Test(Build_DV_Attention_Testing_Base):
    """ 
    use test files only
    compute utter_accuracy
    cannot compute win_ce, win_accuracy; ignoring attention is wrong
    """
    def __init__(self, cfg, dv_attn_cfg):
        super().__init__(cfg, dv_attn_cfg)
        self.logger.info('DV_Attention_Accu_Test')
        # use test files only
        self.speaker_id_list = self.cfg.speaker_id_list_dict['train']
        self.dv_calculator = DV_Calculator()

    def test(self):
        self.compute_accuracy_only()

    def compute_accuracy_only(self):
        total_num_files = 0.
        total_utter_accuracy = 0.

        for speaker_id in self.speaker_id_list:
            file_id_list = self.data_loader.dv_selecter.file_list_dict[(speaker_id, 'test')]
            for file_id in file_id_list:
                feed_dict, batch_size = self.data_loader.make_feed_dict(file_id_list=[file_id],start_sample_list=[0])
                utter_accuracy = self.compute_utter_accuracy(feed_dict)

                total_num_files += 1.
                total_utter_accuracy += utter_accuracy

        self.logger.info('Results of model %s' % self.dv_attn_cfg.nnets_file_name)
        self.logger.info('Number of files & utter_accuracy')
        self.logger.info('%f & %.4f' %(total_num_files, total_utter_accuracy/total_num_files))

    def compute_utter_accuracy(self, feed_dict):
        logit_SD = self.model.gen_logit_SD_value(feed_dict)
        one_hot_S = feed_dict['one_hot_S']
        return self.dv_calculator.cal_accuracy_S(logit_SD, one_hot_S)

class Build_DV_Generator(Build_DV_Attention_Testing_Base):
    """
    Build_DV_Generator
    Generate a dict of speaker representations:
        dv_file_dict[file_id] = (dv, num_windows)
        dv_spk_dict[file_id] = dv
        dv is a 1D numpy array
    """
    def __init__(self, cfg, dv_attn_cfg, output_dir=None):
        super().__init__(cfg, dv_attn_cfg)
        self.output_dir = output_dir
        self.logger.info('Build_DV_Generator')

        # change S to 1; re-build data_loader
        dv_attn_cfg.input_data_dim['S'] = 1
        dv_attn_cfg.dv_y_cfg.input_data_dim['S'] = 1
        self.dv_tts_selector = Build_dv_TTS_selecter(self.cfg, dv_attn_cfg.dv_y_cfg)
        self.speaker_id_list = self.cfg.speaker_id_list_dict['all']

        self.dv_file_dict = {}
        self.dv_spk_dict = {}
        self.DMLFIO = Data_Meta_List_File_IO(cfg)


    def test(self):
        for k in self.dv_tts_selector.file_list_dict:
            if self.dv_attn_cfg.run_mode == 'debug':
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
            dv_speaker = numpy.zeros(self.dv_attn_cfg.dv_y_cfg.dv_dim)
            total_num_frames = 0.
            file_id_list = self.dv_tts_selector.file_list_dict[(speaker_id, 'SR')]
            for file_id in file_id_list:
                num_frames = self.dv_file_dict[file_id][0]
                total_num_frames += num_frames
                dv_speaker += self.dv_file_dict[file_id][1] * float(num_frames)

            self.dv_spk_dict[speaker_id] = dv_speaker / total_num_frames

        self.save_dv_files(self.dv_attn_cfg.exp_dir)
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



class Build_DV_Attention_Plotter(Build_DV_Attention_Testing_Base):
    """
    Build_DV_Attention_Plotter
    Generate a dict of speaker representations:
        Plot attention against text/label
    """
    def __init__(self, cfg, dv_attn_cfg, output_dir=None):
        super().__init__(cfg, dv_attn_cfg)
        self.output_dir = output_dir
        self.logger.info('Build_DV_Atten_Plot')

        # change S to 1; re-build data_loader
        dv_attn_cfg.input_data_dim['S'] = 1
        dv_attn_cfg.dv_y_cfg.input_data_dim['S'] = 1
        self.dv_tts_selector = Build_dv_TTS_selecter(self.cfg, dv_attn_cfg.dv_y_cfg)
        self.speaker_id_list = self.cfg.speaker_id_list_dict['all']

        self.dv_file_dict = {}
        self.dv_spk_dict = {}
        self.DMLFIO = Data_Meta_List_File_IO(cfg)


    def test(self):
        for k in self.dv_tts_selector.file_list_dict:
            if self.dv_attn_cfg.run_mode == 'debug':
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
            dv_speaker = numpy.zeros(self.dv_attn_cfg.dv_y_cfg.dv_dim)
            total_num_frames = 0.
            file_id_list = self.dv_tts_selector.file_list_dict[(speaker_id, 'SR')]
            for file_id in file_id_list:
                num_frames = self.dv_file_dict[file_id][0]
                total_num_frames += num_frames
                dv_speaker += self.dv_file_dict[file_id][1] * float(num_frames)

            self.dv_spk_dict[speaker_id] = dv_speaker / total_num_frames

        self.save_dv_files(self.dv_attn_cfg.exp_dir)
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

