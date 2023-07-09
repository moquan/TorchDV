# exp_dv_cmp_pytorch.py

# This file uses dv_cmp experiments to slowly progress with pytorch

import os, sys, pickle, time, shutil, logging, copy
import math, numpy

from frontend_mw545.modules import make_logger, log_class_attri
from frontend_mw545.frontend_tests import Graph_Plotting

from nn_torch.torch_models import Build_DV_Y_model
from frontend_mw545.data_loader import Build_dv_y_train_data_loader

#############
# Processes #
#############

class Build_Model_Trainer(object):
    """docstring for Build_Model_Trainer"""
    def __init__(self, train_cfg):
        super().__init__()
        numpy.random.seed(545)
        self.logger = make_logger("train_model")
        log_class_attri(train_cfg, self.logger, except_list=train_cfg.log_except_list)

        # Initialise optimisation process
        self.best_valid_loss = sys.float_info.max
        self.prev_valid_loss = sys.float_info.max
        self.warmup_epoch = train_cfg.warmup_epoch
        self.early_stop = 0
        self.early_stop_epoch = train_cfg.early_stop_epoch
        self.num_decay  = 0
        self.max_num_decay = train_cfg.max_num_decay

    def train(self):
        self.logger.info('Creating data loader')
        self.build_data_loader()

        nnets_file_name = self.train_cfg.nnets_file_name
        epoch_num = 0
        while (epoch_num < self.train_cfg.num_train_epoch):
            epoch_num = epoch_num + 1
            epoch_start_time = time.time()

            self.logger.info('start Training Epoch '+str(epoch_num))
            self.train_action_epoch()
            epoch_train_time = time.time()

            self.logger.info('start Evaluating Epoch '+str(epoch_num))
            valid_loss = self.eval_action_epoch(epoch_num)
            epoch_valid_time = time.time()

            is_finish = self.optimisation_action(valid_loss, epoch_num)
            if is_finish:
                self.logger.info('Best model, %s, best valid error %.4f' % (nnets_file_name, self.best_valid_loss))
                return self.best_valid_loss

            self.logger.info('epoch %i; train time is %.2f, valid time is %.2f' %(epoch_num, (epoch_train_time - epoch_start_time), (epoch_valid_time - epoch_train_time)))
            self.train_cfg.additional_action_epoch(self.logger, self.model)

        self.logger.info('Reach num_train_epoch, best model, %s, best valid error %.4f' % (nnets_file_name, self.best_valid_loss))
        return self.best_valid_loss

    def build_data_loader(self):
        '''
        Need to define a self.data_loader
        '''
        pass

    def train_action_epoch(self):
        epoch_start_time = time.time()
        epoch_train_load_time = 0.
        epoch_train_model_time = 0.
        epoch_num_batch = self.train_cfg.epoch_num_batch['train']

        for batch_idx in range(epoch_num_batch):
            batch_load_time, batch_model_time = self.train_action_batch()
            epoch_train_load_time  += batch_load_time
            epoch_train_model_time += batch_model_time
        self.logger.info('train load time %.2f, train model time %.2f' % (epoch_train_load_time, epoch_train_model_time))

    def train_action_batch(self):
        batch_start_time = time.time()
        # Make feed_dict for training
        feed_dict, batch_size = self.data_loader.make_feed_dict(utter_tvt_name='train')
        batch_load_end_time = time.time()
        # Run Model
        self.model.train()
        self.model.update_parameters(feed_dict=feed_dict)
        batch_model_end_time = time.time()

        batch_load_time = batch_load_end_time - batch_start_time
        batch_model_time = batch_model_end_time - batch_load_end_time
        return (batch_load_time, batch_model_time)

    def eval_action_epoch(self, epoch_num):
        '''
        Additional Accuracy Actions
        '''
        output_string = {'loss':'epoch %i loss:' % epoch_num}
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

            mean_loss = total_loss/float(epoch_num_batch)
            output_string['loss'] = output_string['loss'] + ' %s %.4f;' % (utter_tvt_name, mean_loss)

            if utter_tvt_name == 'valid':
                valid_loss = mean_loss

        self.logger.info('valid load time %.2f, valid model time %.2f' % (epoch_valid_load_time, epoch_valid_model_time))
        self.logger.info(output_string['loss'])
        return valid_loss

    def eval_action_batch(self, utter_tvt_name):
        batch_start_time = time.time()
        # Make feed_dict for evaluation
        feed_dict, batch_size = self.data_loader.make_feed_dict(utter_tvt_name=utter_tvt_name)
        batch_load_end_time = time.time()
        # Run Model
        self.model.eval()
        with self.model.no_grad():
            batch_mean_loss = self.model.gen_loss_value(feed_dict=feed_dict)
        batch_model_end_time = time.time()

        batch_load_time = batch_load_end_time - batch_start_time
        batch_model_time = batch_model_end_time - batch_load_end_time
        return (batch_load_time, batch_model_time, batch_mean_loss, batch_size, feed_dict)

    def optimisation_action(self, valid_loss, epoch_num):
        nnets_file_name = self.train_cfg.nnets_file_name
        if valid_loss < self.best_valid_loss:
            self.early_stop = 0
            self.logger.info('valid error reduced, saving model, %s' % nnets_file_name)
            self.model.save_nn_model_optim(nnets_file_name)
            self.best_valid_loss = valid_loss
        elif valid_loss > self.previous_valid_loss:
            self.early_stop += 1
            self.logger.info('valid error increased, early stop %i' % self.early_stop)
        if (self.early_stop > self.early_stop_epoch) and (epoch_num > self.warmup_epoch):
            self.early_stop = 0
            self.num_decay = self.num_decay + 1
            if self.num_decay > self.max_num_decay:
                # End of training
                self.logger.info('Stopping early')
                return True
            else:
                new_learning_rate = self.model.learning_rate*0.5
                self.logger.info('reduce learning rate to '+str(new_learning_rate)) # Use str(lr) for full length
                self.model.update_learning_rate(new_learning_rate)
            self.logger.info('loading previous best model, %s ' % nnets_file_name)
            self.model.load_nn_model_optim(nnets_file_name)
        self.previous_valid_loss = valid_loss
        return False

class Build_DV_Y_Model_Trainer(Build_Model_Trainer):
    def __init__(self, cfg, dv_y_cfg):
        super().__init__(dv_y_cfg)
        self.cfg = cfg
        self.train_cfg = dv_y_cfg

        self.model = Build_DV_Y_model(dv_y_cfg)
        self.model.torch_initialisation()
        self.model.build_optimiser()

        if dv_y_cfg.finetune_model:
            self.logger.info('Loading %s for finetune' % dv_y_cfg.prev_nnets_file_name)
            self.model.load_nn_model_optim(dv_y_cfg.prev_nnets_file_name)
        self.model.print_model_parameter_sizes(self.logger)

    def build_data_loader(self):
        from frontend_mw545.data_loader import Build_dv_y_train_data_loader
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
                    one_hot_S = feed_dict['one_hot_S']
                    batch_mean_accuracy = self.cal_accuracy(logit_SD, one_hot_S)
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



def train_dv_y_model_ref(cfg, dv_y_cfg):
    ''' New version: data_loader is a class, not a function '''
    numpy.random.seed(545)
    # Feed data use feed_dict style

    logger = make_logger("dv_y_config")
    log_class_attri(dv_y_cfg, logger, except_list=dv_y_cfg.log_except_list)

    logger = make_logger("train_dvy")
    logger.info('Creating data loader')
    
    dv_y_train_data_loader = Build_dv_y_train_data_loader(cfg, dv_y_cfg)

    dv_y_model = Build_DV_Y_model(dv_y_cfg)
    dv_y_model.torch_initialisation()
    dv_y_model.build_optimiser()

    if dv_y_cfg.finetune_model:
        logger.info('Loading %s for finetune' % dv_y_cfg.prev_nnets_file_name)
        dv_y_model.load_nn_model_optim(dv_y_cfg.prev_nnets_file_name)
    dv_y_model.print_model_parameter_sizes(logger)

    epoch      = 0
    early_stop = 0
    num_decay  = 0    
    best_valid_loss  = sys.float_info.max
    num_train_epoch  = dv_y_cfg.num_train_epoch
    early_stop_epoch = dv_y_cfg.early_stop_epoch
    max_num_decay    = dv_y_cfg.max_num_decay
    previous_valid_loss = sys.float_info.max

    while (epoch < num_train_epoch):
        epoch = epoch + 1

        logger.info('start training Epoch '+str(epoch))
        epoch_start_time = time.time()
        epoch_train_load_time = 0.
        epoch_train_model_time = 0.

        for batch_idx in range(dv_y_cfg.epoch_num_batch['train']):
            batch_start_time = time.time()
            # logger.info('start loading batch '+str(batch_idx))
            # Make feed_dict for training
            feed_dict, batch_size = dv_y_train_data_loader.make_feed_dict(utter_tvt_name='train')
            batch_load_time = time.time()
            epoch_train_load_time += (batch_load_time - batch_start_time)
            # logger.info('start training batch '+str(batch_idx))
            dv_y_model.nn_model.train()
            dv_y_model.update_parameters(feed_dict=feed_dict)
            batch_train_time = time.time()
            epoch_train_model_time += (batch_train_time - batch_load_time)
        epoch_train_time = time.time()

        logger.info('epoch train load time is %s, train model time is %s' % (str(epoch_train_load_time), str(epoch_train_model_time)))

        logger.info('start evaluating Epoch '+str(epoch))
        output_string = {'loss':'epoch %i' % epoch, 'accuracy':'epoch %i' % epoch, 'time':'epoch %i' % epoch}
        epoch_valid_load_time = 0.
        epoch_valid_model_time = 0.
        for utter_tvt_name in ['train', 'valid', 'test']:
            total_batch_size = 0.
            total_loss       = 0.
            total_accuracy   = 0.
            for batch_idx in range(dv_y_cfg.epoch_num_batch['valid']):
                batch_start_time = time.time()
                # Make feed_dict for evaluation
                feed_dict, batch_size = dv_y_train_data_loader.make_feed_dict(utter_tvt_name=utter_tvt_name)
                batch_load_time = time.time()
                epoch_valid_load_time += (batch_load_time - batch_start_time)
                dv_y_model.eval()
                with dv_y_model.no_grad():
                    batch_mean_loss = dv_y_model.gen_loss_value(feed_dict=feed_dict)
                    batch_train_time = time.time()
                    epoch_valid_model_time += (batch_train_time - batch_load_time)
                    total_batch_size += batch_size
                    total_loss       += batch_mean_loss
                    if dv_y_cfg.classify_in_training:
                        _c, _t, accuracy = dv_y_model.cal_accuracy(feed_dict=feed_dict)
                        total_accuracy   += accuracy
            average_loss = total_loss/float(dv_y_cfg.epoch_num_batch['valid'])
            output_string['loss'] = output_string['loss'] + ';  %s loss %.4f' % (utter_tvt_name, average_loss)

            if dv_y_cfg.classify_in_training:
                average_accu = total_accuracy/float(dv_y_cfg.epoch_num_batch['valid'])
                output_string['accuracy'] = output_string['accuracy'] + '; %s accuracy %.4f' % (utter_tvt_name, average_accu)

            if utter_tvt_name == 'valid':
                nnets_file_name = dv_y_cfg.nnets_file_name
                # Compare validation error
                valid_error = average_loss
                if valid_error < best_valid_loss:
                    early_stop = 0
                    logger.info('valid error reduced, saving model, %s' % nnets_file_name)
                    dv_y_model.save_nn_model_optim(nnets_file_name)
                    best_valid_loss = valid_error
                elif valid_error > previous_valid_loss:
                    early_stop = early_stop + 1
                    logger.info('valid error increased, early stop %i' % early_stop)
                if (early_stop > early_stop_epoch) and (epoch > dv_y_cfg.warmup_epoch):
                    early_stop = 0
                    num_decay = num_decay + 1
                    if num_decay > max_num_decay:
                        logger.info('stopping early, best model, %s, best valid error %.4f' % (nnets_file_name, best_valid_loss))
                        return best_valid_loss
                    else:
                        new_learning_rate = dv_y_model.learning_rate*0.5
                        logger.info('reduce learning rate to '+str(new_learning_rate)) # Use str(lr) for full length
                        dv_y_model.update_learning_rate(new_learning_rate)
                    logger.info('loading previous best model, %s ' % nnets_file_name)
                    dv_y_model.load_nn_model_optim(nnets_file_name)
                    # logger.info('reduce learning rate to '+str(new_learning_rate))
                    # dv_y_model.update_learning_rate(new_learning_rate)
                previous_valid_loss = valid_error

        logger.info('epoch valid load time is %s, train model time is %s' % (str(epoch_valid_load_time), str(epoch_valid_model_time)))

        epoch_valid_time = time.time()
        output_string['time'] = output_string['time'] + '; train time is %.2f, valid time is %.2f' %((epoch_train_time - epoch_start_time), (epoch_valid_time - epoch_train_time))
        logger.info(output_string['loss'])
        if dv_y_cfg.classify_in_training:
            logger.info(output_string['accuracy'])
        logger.info(output_string['time'])
        

        dv_y_cfg.additional_action_epoch(logger, dv_y_model)

    logger.info('Reach num_train_epoch, best model, %s, best valid error %.4f' % (nnets_file_name, best_valid_loss))
    return best_valid_loss





def vuv_loss_test(cfg, dv_y_cfg):
    '''
    For waveform-based models only; Based on vuv_SBM
    Compare the average loss in voiced or unvoiced region
    Return a dict: key is amount of voiced, value is mean 
    '''
    numpy.random.seed(545)
    # Feed data use feed_dict style

    logger = make_logger("dv_y_config")
    log_class_attri(dv_y_cfg, logger, except_list=dv_y_cfg.log_except_list)

    logger = make_logger("test_dvy")
    logger.info('Creating data loader')
    dv_y_train_data_loader = Build_dv_y_train_data_loader(cfg, dv_y_cfg)

    dv_y_model = DV_Y_model(dv_y_cfg)
    dv_y_model.torch_initialisation()

    if dv_y_cfg.prev_nnets_file_name is None:
        prev_nnets_file_name = dv_y_cfg.nnets_file_name
    else:
        prev_nnets_file_name = dv_y_cfg.prev_nnets_file_name
    logger.info('Loading %s for testing' % prev_nnets_file_name)
    dv_y_model.load_nn_model(prev_nnets_file_name)
    dv_y_model.print_model_parameter_sizes(logger)

    total_num_batch = 100
    loss_list_dict = {i:[] for i in range(dv_y_cfg.seq_num_win+1)}

    dv_y_model.eval()
    for utter_tvt_name in ['train', 'valid', 'test']:
        for batch_idx in range(total_num_batch):
            # Make feed_dict for evaluation
            feed_dict, batch_size = dv_y_train_data_loader.make_feed_dict(utter_tvt_name=utter_tvt_name)
            batch_loss_SB = dv_y_model.gen_SB_loss_value(feed_dict=feed_dict)

            vuv_SBM = feed_dict['vuv_SBM']
            vuv_SB = numpy.sum(vuv_SBM, axis=2)

            for s in range(dv_y_cfg.batch_num_spk):
                for b in range(dv_y_cfg.spk_num_seq):
                    loss_list_dict[vuv_SB[s,b]].append(batch_loss_SB[s*dv_y_cfg.spk_num_seq+b])

    loss_list = []
    for i in range(dv_y_cfg.seq_num_win+1):
        loss_mean = numpy.mean(loss_list_dict[i])
        loss_list.append(loss_mean)

    print(loss_list)
    graph_plotter = Graph_Plotting()
    fig_file_name = '/home/dawna/tts/mw545/Export_Temp/PNG_out/sinenet_test/vuv_loss.png'
    graph_plotter.single_plot(fig_file_name, [None], [loss_list], [None])

    return(loss_list)