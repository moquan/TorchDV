# exp_dv_y.py

# This file uses dv_cmp experiments to slowly progress with pytorch

import os, sys, pickle, time, shutil, logging, copy
import math, numpy

from frontend_mw545.modules import make_logger, log_class_attri
from frontend_mw545.frontend_tests import Graph_Plotting

from nn_torch.torch_models import Build_DV_Y_model
from frontend_mw545.data_loader import Build_dv_y_train_data_loader

from exp_mw545.exp_base import Build_Model_Trainer

#############
# Processes #
#############

class Build_DV_Y_Model_Trainer(Build_Model_Trainer):
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

    def build_data_loader(self):
        self.data_loader = Build_dv_y_train_data_loader(self.cfg, self.train_cfg)

    def eval_action_epoch(self, epoch_num):
        '''
        Additional Accuracy Actions
        '''
        output_string = {'loss':'epoch %i loss:' % epoch_num, 'accuracy':'epoch %i accu:' % epoch_num}
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
                    logit_SD  = self.model.gen_logit_SD_value(feed_dict=feed_dict)
                    batch_mean_accuracy = self.cal_accuracy(logit_SD, feed_dict['one_hot_S'])
                    total_accuracy += batch_mean_accuracy

            mean_loss = total_loss/float(epoch_num_batch)
            output_string['loss'] = output_string['loss'] + ' %s %.4f;' % (utter_tvt_name, mean_loss)

            if utter_tvt_name == 'valid':
                valid_loss = mean_loss

            if self.train_cfg.classify_in_training:
                mean_accuracy = total_accuracy/float(epoch_num_batch)
                output_string['accuracy'] = output_string['accuracy'] + ' %s %.4f;' % (utter_tvt_name, mean_accuracy)

        self.logger.info('valid load time %.2f, valid model time %.2f' % (epoch_valid_load_time, epoch_valid_model_time))
        self.logger.info(output_string['loss'])
        if self.train_cfg.classify_in_training:
            self.logger.info(output_string['accuracy'])
        return valid_loss

    def cal_accuracy(self, logit_SD, one_hot_S):
        pred_S = numpy.argmax(logit_SD, axis=1)
        # print(pred_S)
        # print(one_hot_S)
        total_num = pred_S.size
        correct_num = numpy.sum(pred_S==one_hot_S)
        return correct_num/float(total_num)



class Build_DV_Y_Testing(object):
    """docstring for Build_DV_Y_Testing"""
    def __init__(self, cfg, dv_y_cfg):
        super().__init__()
        
        self.logger = make_logger("test_model")
        self.cfg = cfg
        self.test_cfg = dv_y_cfg
        log_class_attri(dv_y_cfg, self.logger, except_list=dv_y_cfg.log_except_list)

        self.load_model()

    def load_model(self):
        dv_y_cfg = self.test_cfg

        self.model = Build_DV_Y_model(dv_y_cfg)
        self.model.torch_initialisation()
        if dv_y_cfg.prev_nnets_file_name is None:
            prev_nnets_file_name = dv_y_cfg.nnets_file_name
        else:
            prev_nnets_file_name = dv_y_cfg.prev_nnets_file_name
        self.logger.info('Loading %s for testing' % prev_nnets_file_name)
        self.model.load_nn_model(prev_nnets_file_name)
        self.model.print_model_parameter_sizes(self.logger)

        self.model.eval()

    def vuv_loss_test(self, plot_loss=True):
        '''
        For waveform-based models only; Based on vuv_SBM
        Compare the average loss in voiced or unvoiced region
        Return a dict: key is train/valid/test, value is a list for plotting
            loss_dict[utter_tvt_name] = loss_list
            in loss_list: loss vs voicing (increasing)
        '''
        numpy.random.seed(546)

        dv_y_cfg = self.test_cfg
        assert 'vuv_SBM' in dv_y_cfg.out_feat_list
        self.logger.info('Creating data loader')
        dv_y_train_data_loader = Build_dv_y_train_data_loader(self.cfg, dv_y_cfg)

        total_num_batch = 100
        loss_dict = {}

        for utter_tvt_name in ['train', 'valid', 'test']:
            loss_list_dict = {i:[] for i in range(dv_y_cfg.input_data_dim['M']+1)}
            for batch_idx in range(total_num_batch):
                # Make feed_dict for evaluation
                feed_dict, batch_size = dv_y_train_data_loader.make_feed_dict(utter_tvt_name=utter_tvt_name)
                batch_loss_SB = self.model.gen_SB_loss_value(feed_dict=feed_dict)

                vuv_SBM = feed_dict['vuv_SBM']
                vuv_sum_SB = numpy.sum(vuv_SBM, axis=2)

                for s in range(dv_y_cfg.input_data_dim['S']):
                    for b in range(dv_y_cfg.input_data_dim['B']):
                        loss_list_dict[vuv_sum_SB[s,b]].append(batch_loss_SB[s*dv_y_cfg.input_data_dim['B']+b])

            loss_list = []
            for i in range(dv_y_cfg.input_data_dim['M']+1):
                loss_mean = numpy.mean(loss_list_dict[i])
                loss_list.append(loss_mean)

            loss_dict[utter_tvt_name] = loss_list

        if plot_loss:
            graph_plotter = Graph_Plotting()
            fig_file_name = '/home/dawna/tts/mw545/Export_Temp/PNG_out/sinenet_test/vuv_loss.png'
            graph_plotter.single_plot(fig_file_name, None, [loss_dict['train'],loss_dict['valid'],loss_dict['test']], ['train', 'valid', 'test'])

        return(loss_dict)

    def positional_sensitivity_test(self, plot_loss=True):
        '''
        Shift the input by a few samples
        Compute difference in lambda
        '''
        pass

    def generate_lambda_u_dict(self, lambda_u_dict_file_name=None):
        '''
        Generate 1 lambda for each utterance
        lambda_u_dict[file_name] = [lambda_speaker, total_batch_size]
        Store in file; also return the dict
        '''
        cfg = self.cfg
        dv_y_cfg = self.test_cfg

        if lambda_u_dict_file_name is None:
            lambda_u_dict_file_name = os.path.join(dv_y_cfg.exp_dir, 'lambda_u_class_test.dat')

        self.logger.info('Creating data loader')
        dv_y_test_data_loader = Build_dv_y_class_test_data_loader(cfg, dv_y_cfg)



    def classification_test(self):
        '''
        Classify based on averaged lambda
        Generate
        1. Load / generate all lambda of all test files files
        2. Draw spk_num_utter files, weighted average lambda, classify
        '''
        numpy.random.seed(546)

        spk_num_utter_list = [1,2,5,10]
        total_num_batch = 100

        dv_y_cfg = self.test_cfg
        lambda_u_dict_file_name = os.path.join(dv_y_cfg.exp_dir, 'lambda_u_class_test.dat')

        try: 
            lambda_u_dict = pickle.load(open(lambda_u_dict_file_name, 'rb'))
            logger.info('Loaded lambda_u_dict from %s' % lambda_u_dict_file_name)
        # Generate
        except:
            logger.info('Cannot load from %s, generate instead' % lambda_u_dict_file_name)
            lambda_u_dict = self.generate_lambda_u_dict(lambda_u_dict_file_name)






        

    def sinenet_weight_test(self):
        '''
        Quick test: print weights in W_A
        '''

        W_A = self.model.nn_model.layer_list[1].layer_fn.fc_fn.weight

        W_A_value = W_A.cpu().detach().numpy()

        print(W_A_value.shape)

        sum_0 = numpy.sum(numpy.abs(W_A_value), axis=0)
        sum_1 = numpy.sum(numpy.abs(W_A_value), axis=1)

        print(sum_0)
        print(sum_1)

        return (sum_0, sum_1)



        
def class_test_dv_y_model_v2(cfg, dv_y_cfg):
    numpy.random.seed(546)
    # Classification test
    # Also generates lambda_u per utterance; store in lambda_u_dict[file_name]
    # Use test utterances only
    # Make or load lambda_u_dict
    # lambda_u_dict[file_name] = [lambda_u, B_u]
    # Draw random groups of files, weighted average lambda, then classify

    logger = make_logger("dv_y_config")
    dv_y_cfg = copy.deepcopy(dv_y_cfg)
    dv_y_cfg.change_to_class_test_mode()
    log_class_attri(dv_y_cfg, logger, except_list=dv_y_cfg.log_except_list)

    logger = make_logger("class_dvy")
    logger.info('Creating data lists')
    dv_y_data_loader = Build_dv_y_test_data_loader(cfg, dv_y_cfg)

    dv_y_model = torch_initialisation(dv_y_cfg)
    dv_y_model.load_nn_model(dv_y_cfg.nnets_file_name)
    dv_y_model.eval()

    try: 
        lambda_u_dict = pickle.load(open(dv_y_cfg.lambda_u_dict_file_name, 'rb'))
        logger.info('Loaded lambda_u_dict from %s' % dv_y_cfg.lambda_u_dict_file_name)
    # Generate
    except:
        logger.info('Cannot load from %s, generate instead' % dv_y_cfg.lambda_u_dict_file_name)
        lambda_u_dict = {}   # lambda_u[file_name] = [lambda_speaker, total_batch_size]



        for speaker_id in dv_y_data_loader.speaker_id_list:
            logger.info('Generating %s' % speaker_id)
            for file_id in dv_y_data_loader.file_list_dict[(speaker_id, 'test')]:
                # logger.info('Generating %s' % file_id)
                lambda_temp_list = []
                batch_size_list  = []
                dv_y_data_loader.file_gen_finish = False
                dv_y_data_loader.load_new_file_bool = True
                while not (dv_y_data_loader.file_gen_finish):
                    feed_dict, batch_size = dv_y_data_loader.make_feed_dict_method_test(speaker_id, file_id)
                    with dv_y_model.no_grad():
                        lambda_temp = dv_y_model.gen_lambda_SBD_value(feed_dict=feed_dict)
                    lambda_temp_list.append(lambda_temp)
                    batch_size_list.append(batch_size)
                B_u = numpy.sum(batch_size_list)
                lambda_u = numpy.zeros(dv_y_cfg.dv_dim)
                for lambda_temp, batch_size in zip(lambda_temp_list, batch_size_list):
                    for b in range(batch_size):
                        lambda_u += lambda_temp[0,b]
                lambda_u /= float(B_u)
                lambda_u_dict[file_id] = [lambda_u, B_u]
        logger.info('Saving lambda_u_dict to %s' % dv_y_cfg.lambda_u_dict_file_name)
        pickle.dump(lambda_u_dict, open(dv_y_cfg.lambda_u_dict_file_name, 'wb'))

    # for k in lambda_u_dict:
    #     print(k)
    #     print(lambda_u_dict[k])      # [array([nan, nan, nan, nan, nan, nan, nan, nan]), 0.0]

    # Classify
    for spk_num_utter in dv_y_cfg.spk_num_utter_list:
        logger.info('Testing with %i utterances per speaker' % spk_num_utter)
        accuracy_list = []
        for speaker_id in dv_y_data_loader.speaker_id_list:
            logger.info('testing speaker %s' % speaker_id)
            speaker_lambda_list = []
            for batch_idx in range(dv_y_cfg.epoch_num_batch['test']):
                logger.info('batch %i' % batch_idx)
                batch_file_list = dv_y_data_loader.file_loader_dict[(speaker_id, 'test')].draw_n_samples(spk_num_utter)

                # Weighted average of lambda_u
                batch_lambda = numpy.zeros(dv_y_cfg.dv_dim)
                B_total = 0.
                for file_id in batch_file_list:
                    lambda_u, B_u = lambda_u_dict[file_id]
                    batch_lambda += lambda_u * B_u
                    B_total += B_u
                batch_lambda /= B_total
                speaker_lambda_list.append(batch_lambda)

            true_speaker_index = dv_y_cfg.speaker_id_list_dict['train'].index(speaker_id)
            B_remain = dv_y_cfg.epoch_num_batch['test']
            b_index = 0 # Track counter, instead of removing elements
            correct_counter = 0.
            while B_remain > 0:
                lambda_val = numpy.zeros((dv_y_cfg.batch_num_spk, dv_y_cfg.spk_num_seq, dv_y_cfg.dv_dim))
                if B_remain > dv_y_cfg.spk_num_seq:
                    # Fill all dv_y_cfg.spk_num_seq, keep remain for later
                    B_actual = dv_y_cfg.spk_num_seq
                    B_remain -= dv_y_cfg.spk_num_seq
                else:
                    # No more remain
                    B_actual = B_remain
                    B_remain = 0

                for b in range(B_actual):
                    lambda_val[0, b] = speaker_lambda_list[b_index + b]

                # Set up for next round (if dv_y_cfg.spk_num_seq)
                b_index += B_actual

                feed_dict = {'lambda': lambda_val}
                with dv_y_model.no_grad():
                    idx_list_S_B = dv_y_model.lambda_to_indices(feed_dict=feed_dict)
                for b in range(B_actual):
                    if idx_list_S_B[0, b] == true_speaker_index: 
                        correct_counter += 1.
            speaker_accuracy = correct_counter/float(dv_y_cfg.epoch_num_batch['test'])
            logger.info('speaker %s accuracy is %f' % (speaker_id, speaker_accuracy))
            accuracy_list.append(speaker_accuracy)
        mean_accuracy = numpy.mean(accuracy_list)
        logger.info('Accuracy with %i utterances per speaker is %.4f' % (spk_num_utter, mean_accuracy))