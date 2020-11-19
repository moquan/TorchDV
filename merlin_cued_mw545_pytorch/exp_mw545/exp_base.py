# exp_base.py

# This file uses dv_cmp experiments to slowly progress with pytorch

import os, sys, pickle, time, shutil, logging, copy
import math, numpy

from frontend_mw545.modules import make_logger, log_class_attri

#############
# Processes #
#############

class Build_Model_Trainer(object):
    """
    Base class of a trainer
    Need to change: __init__ (to build model), build_data_loader
    Could change:   eval_action_epoch, eval_action_batch; e.g. additional evaluations
    Note: eval_action_batch returns feed_dict, for additional tests
    """
    def __init__(self, cfg, train_cfg):
        super().__init__()
        self.logger = make_logger("train_model")
        self.cfg = cfg
        self.train_cfg = train_cfg

        log_class_attri(train_cfg, self.logger, except_list=train_cfg.log_except_list)

        # Initialise optimisation process
        self.best_valid_loss = sys.float_info.max
        self.prev_valid_loss = sys.float_info.max
        self.best_epoch_num  = 0
        self.warmup_epoch = train_cfg.warmup_epoch
        self.early_stop = 0
        self.early_stop_epoch = train_cfg.early_stop_epoch
        self.num_decay  = 0
        self.max_num_decay = train_cfg.max_num_decay

    def train(self):
        numpy.random.seed(545)
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

            is_finish = self.validation_action(valid_loss, epoch_num)
            if is_finish:
                self.logger.info('Best model, %s, best valid error %.4f' % (nnets_file_name, self.best_valid_loss))
                return self.best_valid_loss

            self.logger.info('epoch %i; train time is %.2f, valid time is %.2f' %(epoch_num, (epoch_train_time - epoch_start_time), (epoch_valid_time - epoch_train_time)))
            self.train_cfg.additional_action_epoch(self.logger, self.model)

        self.logger.info('Reach num_train_epoch, best epoch %i, best valid error %.4f' % (self.best_epoch_num, self.best_valid_loss))
        self.logger.info('Model: %s' % (nnets_file_name))
        return self.best_valid_loss

    def train_no_validation(self):
        '''
        Train all the way, no early stop, no roll back
        '''
        self.logger.info('Training without validation!')
        numpy.random.seed(545)
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

            self.logger.info('epoch %i; train time is %.2f, valid time is %.2f' %(epoch_num, (epoch_train_time - epoch_start_time), (epoch_valid_time - epoch_train_time)))
            self.train_cfg.additional_action_epoch(self.logger, self.model)

        self.logger.info('Reach num_train_epoch, best model, %s, best valid error %.4f' % (nnets_file_name, self.best_valid_loss))
        return self.best_valid_loss


    def build_data_loader(self):
        '''
        Need to define a self.data_loader
        '''
        self.data_loader = None

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
        Generate all 3 losses
        Optimisation decision based on valid_loss, and epoch_num to check warmup
        '''
        output_string = {'loss':'epoch %i train & valid & test loss:' % epoch_num}
        epoch_valid_load_time  = 0.
        epoch_valid_model_time = 0.
        epoch_num_batch = self.train_cfg.epoch_num_batch['valid']

        for utter_tvt_name in ['train', 'valid', 'test']:
            total_batch_size = 0.
            total_loss       = 0.

            for batch_idx in range(epoch_num_batch):
                batch_load_time, batch_model_time, batch_mean_loss, batch_size, feed_dict = self.eval_action_batch(utter_tvt_name)
                total_loss += batch_mean_loss
                epoch_valid_load_time  += batch_load_time
                epoch_valid_model_time += batch_model_time

            mean_loss = total_loss/float(epoch_num_batch)
            output_string['loss'] = output_string['loss'] + ' & %.4f' % (mean_loss)

            if utter_tvt_name == 'valid':
                valid_loss = mean_loss

        self.logger.info('valid load & model time: %.2f & %.2f' % (epoch_valid_load_time, epoch_valid_model_time))
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

    def validation_action(self, valid_loss, epoch_num):
        '''
        Save Model, Early-stop, Load previous model, etc.
        '''
        nnets_file_name = self.train_cfg.nnets_file_name
        if valid_loss < self.best_valid_loss:
            self.early_stop = 0
            self.logger.info('valid error reduced, saving model, %s' % nnets_file_name)
            self.model.save_nn_model_optim(nnets_file_name)
            self.best_valid_loss = valid_loss
            self.best_epoch_num = epoch_num
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
