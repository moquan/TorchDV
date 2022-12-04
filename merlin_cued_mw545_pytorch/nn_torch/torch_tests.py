# torch_tests.py

import os, sys, pickle, time, shutil, logging, copy
import math, numpy, scipy
numpy.random.seed(545)
import torch
torch.manual_seed(545)

from frontend_mw545.modules import make_logger, log_class_attri
from frontend_mw545.frontend_tests import Graph_Plotting

from exp_mw545.exp_dv_config import dv_y_configuration
from nn_torch.torch_models import Build_DV_Y_model
from frontend_mw545.data_loader import Build_dv_y_train_data_loader

from exp_mw545.exp_dv_wav_subwin import dv_y_wav_subwin_configuration

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

class Numpy_Pytorch_Unfold_Test(object):
    """ Tests for unfolding operation using numpy """
    def __init__(self):
        super(Numpy_Pytorch_Unfold_Test, self).__init__()
        self.batch_num_spk = 40
        self.batch_seq_total_len = 6400 # Number of frames at 16kHz; 32000 for 2s
        self.batch_seq_len   = 3200 # T
        self.batch_seq_shift = 80
        self.seq_win_len   = 640
        self.seq_win_shift = 80

        self.spk_num_seq = int((self.batch_seq_total_len - self.batch_seq_len) / self.batch_seq_shift) + 1
        self.seq_num_win = int((self.batch_seq_len - self.seq_win_len) / self.seq_win_shift) + 1

        self.device_id = torch.device("cuda:0")
        self.win_len_shift_list = [[self.batch_seq_len, self.batch_seq_shift], [self.seq_win_len, self.seq_win_shift]]
        
    def consistent_test(self):
        '''
        Test 2 methods to make S*T into S*B*M*D
        Check if the output matrices are consistent
        '''
        x_ST = numpy.random.rand(self.batch_num_spk, self.batch_seq_total_len)
        x_1 = self.numpy_unfold_method_1(x_ST, return_value=True)
        x_2 = self.numpy_unfold_method_2(x_ST, return_value=True)
        print(x_1.shape)
        print(x_2.shape)
        print(x_ST[0,:10])
        print(x_1[0,0,0,:10])
        print(x_2[0,0,0,:10])
        assert (x_1 == x_2).all()

    def speed_test(self):
        '''
        Test 2 methods to make S*T into S*B*M*D
        Measure the time taken
        '''
        test_start_time = time.time()
        for i in range(1000):
            x_ST = numpy.random.rand(self.batch_num_spk, self.batch_seq_total_len)
            self.numpy_unfold_method_1(x_ST, return_value=False)
        test_end_time = time.time()
        print('%s' % str(test_end_time-test_start_time))

        test_start_time = time.time()
        for i in range(1000):
            x_ST = numpy.random.rand(self.batch_num_spk, self.batch_seq_total_len)
            self.numpy_unfold_method_2(x_ST, return_value=False)
        test_end_time = time.time()
        print('%s' % str(test_end_time-test_start_time))

        '''
        Results:
        263.42780780792236
        3.795353651046753
        PyTorch method is much faster
        '''

    def numpy_unfold_method_1(self, x_ST, return_value=False):
        '''
        Slow method: Find indices
        '''
        x_SBMD = numpy.zeros((self.batch_num_spk, self.spk_num_seq, self.seq_num_win, self.seq_win_len), dtype=numpy.float)
        for b in range(self.spk_num_seq):
            for m in range(self.seq_num_win):
                w_start = b * self.batch_seq_shift + m * self.seq_win_shift
                w_end = w_start + self.seq_win_len
                x_SBMD[:,b,m,:] = x_ST[:,w_start:w_end]

        x = torch.tensor(x_SBMD, dtype=torch.float, device=self.device_id)
        if return_value:
            return x.cpu().detach().numpy()

    def numpy_unfold_method_2(self, x_ST, return_value=False):
        x = torch.tensor(x_ST, dtype=torch.float, device=self.device_id)
        for i in range(2):
            win_len_shift = self.win_len_shift_list[i]
            win_len, win_shift = win_len_shift
            x = x.unfold(i+1, win_len, win_shift)

        if return_value:
            return x.cpu().detach().numpy()

class DV_Y_Build_Test(object):
    def __init__(self, cfg):
        super(DV_Y_Build_Test, self).__init__()
        self.logger = make_logger("Test")
        

        self.cfg = cfg
        self.dv_y_cfg = dv_y_wav_subwin_configuration(cfg)
        # log_class_attri(dv_y_cfg, self.logger)

    def model_build_test(self):
        dv_y_model = Build_DV_Y_model(self.dv_y_cfg)
        dv_y_model.torch_initialisation()
        dv_y_model.build_optimiser()

        dv_y_model.print_model_parameter_sizes(self.logger)
        return dv_y_model

    def single_batch_train_test(self):
        '''
        1. See if the batch data is passed and trained properly
            Result: Good, after one iteration, both improves
        2. See if this batch can be perfectly matched
            Result: Good enough; loss 1.0467; accuracy 0.9146; although slowly
        '''
        
        dv_y_data_loader = Build_dv_y_train_data_loader(self.cfg, self.dv_y_cfg)
        feed_dict, batch_size = dv_y_data_loader.make_feed_dict(utter_tvt_name='train')

        dv_y_model = self.model_build_test()

        dv_y_model.eval()
        batch_mean_loss = dv_y_model.gen_loss_value(feed_dict=feed_dict)
        _c, _t, accuracy = dv_y_model.cal_accuracy(feed_dict=feed_dict)

        self.logger.info('loss %.4f; accuracy %.4f' % (batch_mean_loss, accuracy))

        for i in range(10000):
            dv_y_model.nn_model.train()
            dv_y_model.update_parameters(feed_dict=feed_dict)

            dv_y_model.eval()
            batch_mean_loss = dv_y_model.gen_loss_value(feed_dict=feed_dict)
            _c, _t, accuracy = dv_y_model.cal_accuracy(feed_dict=feed_dict)

            self.logger.info('loss %.4f; accuracy %.4f' % (batch_mean_loss, accuracy))

    def new_forward_test(self):
        '''
        The forward method in dv_y_model is changed; check it
        Average of logit_SB_D may not be same as logit_S_D; only same if Linear expansion layer
        Result: Good; max(abs(diff)) almost zero, e-7, precision
        '''
        dv_y_data_loader = Build_dv_y_train_data_loader(self.cfg, self.dv_y_cfg)
        feed_dict, batch_size = dv_y_data_loader.make_feed_dict(utter_tvt_name='train')

        dv_y_model = self.model_build_test()

        dv_y_model.eval()

        assert self.dv_y_cfg.train_by_window == True
        logit_SB_D = dv_y_model.gen_logit_SBD_value(feed_dict)
        print(logit_SB_D.shape)
        logit_S_D  = dv_y_model.gen_logit_SD_value(feed_dict)
        print(logit_S_D.shape)

        logit_SB_D_SD = numpy.mean(logit_SB_D, axis=1, keepdims=False)
        print((logit_SB_D_SD == logit_S_D).all())
        print(numpy.max(numpy.abs(logit_SB_D_SD-logit_S_D)))

    def speed_test(self):
        '''
        Test speed difference of calling model.train() and model.eval()
        Decide where to wrap, on batch or epoch level
        '''
        dv_y_data_loader = Build_dv_y_train_data_loader(self.cfg, self.dv_y_cfg)
        feed_dict, batch_size = dv_y_data_loader.make_feed_dict(utter_tvt_name='train')
        self.model = self.model_build_test()

        num_epoch = 50
        num_batch = 1000
        '''
        1. Train test
            result: 703.61 801.66
        '''
        if False:
            start_time = time.time()
            self.model.train()
            for epoch_idx in range(num_epoch):
                for batch_idx in range(num_batch):
                    self.model.update_parameters(feed_dict=feed_dict)
            time_1 = time.time() - start_time

            start_time = time.time()
            for epoch_idx in range(num_epoch):
                for batch_idx in range(num_batch):
                    self.model.train()
                    self.model.update_parameters(feed_dict=feed_dict)
            time_2 = time.time() - start_time
            print('%.2f %.2f' % (time_1, time_2))

        '''
        2. Valid test
            result: 416.80 391.71
        '''
        if True:
            start_time = time.time()
            self.model.eval()
            with self.model.no_grad():
                for epoch_idx in range(num_epoch):
                    for batch_idx in range(num_batch):
                        batch_mean_loss = self.model.gen_loss_value(feed_dict=feed_dict)
            time_1 = time.time() - start_time

            start_time = time.time()
            for epoch_idx in range(num_epoch):
                for batch_idx in range(num_batch):
                    self.model.eval()
                    with self.model.no_grad():
                        batch_mean_loss = self.model.gen_loss_value(feed_dict=feed_dict)
            time_2 = time.time() - start_time
            print('%.2f %.2f' % (time_1, time_2))

def return_gpu_memory(gpu_id):
    device_id = torch.device("cuda:%i" % gpu_id)
    return torch.cuda.memory_summary(device_id)

class Sinenet_Test(object):
    """
    Testing Sinenet
    This tests the sine/cosine part of the weight matrix only
    i.e. the sinenet_fn inside the sinenet layer; before (linear+activation)
    """
    def __init__(self, cfg):
        super(Sinenet_Test, self).__init__()

        self.cfg = cfg
        self.graph_plotter = Graph_Plotting()
        self.device_id = torch.device("cuda:0")

        layer_config = {'type': 'Sinenet_V1', 'size':86, 'num_freq':16, 'dropout_p':0, 'batch_norm':False}
        self.params = {'layer_config': layer_config, "input_dim_values": {'T': 640}}

        from nn_torch.torch_layers import Build_Sinenet
        self.sinenet_layer = Build_Sinenet(self.params)
        self.sinenet_layer.to(self.device_id)
        
    def gen_w_f_tau(self, f_val, tau_val, input_dim_values=None):
        '''
        Generate w_sin_cos matrix based on a pair of f and tau values
        Return a numpy matrix w
        '''
        if input_dim_values is None:
            input_dim_values = {'S':1, 'B':1, 'M':1, 'T': 640}
        self.params['input_dim_values'] = input_dim_values
        f   = numpy.ones((input_dim_values['S'], input_dim_values['B'], input_dim_values['M']))
        tau = numpy.ones((input_dim_values['S'], input_dim_values['B'], input_dim_values['M']))

        f   = f * f_val
        tau = tau * tau_val

        f_SBM, tau_SBM = self.numpy_to_tensor([f, tau])
        w_sin_cos = self.sinenet_layer.construct_w_sin_cos_matrix(f_SBM, tau_SBM) # S*B*M*2K*T

        w = w_sin_cos.cpu().detach().numpy()
        return w

    def numpy_to_tensor(self, x_list):
        t_list = []
        for x in x_list:
            t = torch.tensor(x, dtype=torch.float, device=self.device_id)
            t_list.append(t)
        return t_list

    def test_1(self):
        '''
        f = 25, tau = 0; plot all sine and cosine
        Result: Good, plots show (multiples of) full periods, as expected
        '''
        tau = 0

        plot_dir = '/home/dawna/tts/mw545/Export_Temp/PNG_out/sinenet_test'
        K = self.params['layer_config']['num_freq']
        x_index = range(self.params['input_dim_values']['T'])

        for f in [25, 50, 125]:
            w = self.gen_w_f_tau(f, tau)
            for k in range(K * 2):
                w_i = w[0,0,0,k,:]
                fig_file_name = os.path.join(plot_dir, '%s_%03d.png' % (f, k+1))
                self.graph_plotter.single_plot(fig_file_name, [x_index], [w_i], ['%s' % (k+1)])

    def test_2(self):
        '''
        tau shift experiment
        shift integer multiples of T_wav, check if filter values are equal
        Result: Good; filter values are slightly off due to precision
        '''
        f = 100
        num_test = 3

        w_dict = {}
        T_wav = 1./16000.
        K = self.params['layer_config']['num_freq']
        T = self.params['input_dim_values']['T']

        x_index = range(T)
        plot_dir = '/home/dawna/tts/mw545/Export_Temp/PNG_out/sinenet_test'

        for i in range(num_test):
            tau = i * T_wav * 10
            w = self.gen_w_f_tau(f, tau)
            w_dict[i] = w

        # for k in range(K * 2):
        #     for i in range(1,num_test):
        #     # compare w_dict[0] and w_dict[i+1]
        #         w_k_0 = w_dict[0][0,0,0,k,:]
        #         w_k_i = w_dict[i][0,0,0,k,:]

        #         print((w_k_0 == w_k_i).all())

        fig_file_name = os.path.join(plot_dir, 'tau.png')
        w_k_0 = w_dict[0][0,0,0,0,:]
        w_k_2 = w_dict[2][0,0,0,0,:]

        # self.graph_plotter.single_plot(fig_file_name, [x_index]*2, [w_k_0, w_k_2], ['0', '20'])

        '''
        Check which shift gives the smallest MSE of filter difference
        Result: smallest when shift = tau, as expected
        '''
        tau_diff_list = []
        for i in range(40):
            w_k_0_i = w_k_0[:T-i]
            w_k_2_i = w_k_2[i:]
            tau_diff = numpy.sum(numpy.abs(w_k_0_i-w_k_2_i))
            tau_diff_list.append(tau_diff)

        fig_file_name = os.path.join(plot_dir, 'tau_diff.png')
        x_index = range(40)
        self.graph_plotter.single_plot(fig_file_name, [x_index], [tau_diff_list], ['dif'])

    def test_3(self):
        '''
        Residual experiment
        More details see report
        '''
        tau_val = 0.013
        f_val   = 150
        h_sq_list = []
        f_list = range(20,220)
        for f_val in f_list:
            w = self.gen_w_f_tau(f_val=f_val,tau_val=tau_val)  # S*B*M*2K*T
            w_2D = w[0,0,0]
            # print(w_2D.shape)  # (32,640)
            x_sc_ref = numpy.zeros(640)
            for i in range(32):
                x_sc_temp = numpy.random.randint(1,4) * w_2D[i]
                x_sc_ref += x_sc_temp
            x = numpy.random.rand(640) + x_sc_ref
            # print(x)
            h = numpy.dot(w_2D, x)
            # print(h)
            w_2D_T = numpy.transpose(w_2D)
            a = numpy.diag(1./numpy.diag(numpy.dot(w_2D, w_2D_T)))
            # print(a) # ~1/320 for all
            w_a_w = numpy.dot(w_2D_T, numpy.dot(a, w_2D))
            # print(numpy.diag(w_a_w))  # all 0.05
            w_w_a = numpy.dot(w_2D, numpy.dot(w_2D_T, a))
            # print(numpy.diag(w_w_a))  # all 1
            x_sc = numpy.dot(w_a_w, x)
            x_res = x - x_sc
            h_res = numpy.dot(w_2D, x_res)
            # print(h_res)
            h_sq_list.append(numpy.sum(numpy.square(h_res)))
        fig_file_name = '/home/dawna/tts/mw545/Export_Temp/PNG_out/sinenet_test/h_sq_vs_f.png'
        self.graph_plotter.single_plot(fig_file_name, [f_list], [h_sq_list], [' '])

    def test_4(self):
        '''
        Residual experiment
        More details see report
        '''
        from nn_torch.torch_layers import Build_Sinenet

        tau_val = 0.013
        f_val   = 150
        h_sq_list = []
        T_list = range(640,64000,10)
        for T in T_list:
            self.params["input_dim_values"]['T'] = T
            self.sinenet_layer = Build_Sinenet(self.params)
            self.sinenet_layer.to(self.device_id)

            input_dim_values = {'S':1, 'B':1, 'M':1, 'T': T}
            w = self.gen_w_f_tau(f_val=f_val,tau_val=tau_val)  # S*B*M*2K*T
            w_2D = w[0,0,0]
            # print(w_2D.shape)  # (32,640)
            x_sc_ref = numpy.zeros(T)
            for i in range(32):
                x_sc_temp = numpy.random.randint(1,4) * w_2D[i]
                x_sc_ref += x_sc_temp
            x = numpy.random.rand(T) + x_sc_ref
            # print(x)
            h = numpy.dot(w_2D, x)
            # print(h)
            w_2D_T = numpy.transpose(w_2D)
            a = numpy.diag(1./numpy.diag(numpy.dot(w_2D, w_2D_T)))
            # print(a) # ~1/320 for all
            w_a_w = numpy.dot(w_2D_T, numpy.dot(a, w_2D))
            # print(numpy.diag(w_a_w))  # all 0.05
            w_w_a = numpy.dot(w_2D, numpy.dot(w_2D_T, a))
            # print(numpy.diag(w_w_a))  # all 1
            x_sc = numpy.dot(w_a_w, x)
            x_res = x - x_sc
            h_res = numpy.dot(w_2D, x_res)
            # print(h_res)
            h_sq_list.append(numpy.sum(numpy.square(h_res)))
        fig_file_name = '/home/dawna/tts/mw545/Export_Temp/PNG_out/sinenet_test/h_sq_vs_T.png'
        self.graph_plotter.single_plot(fig_file_name, [T_list], [h_sq_list], [' '])

    def test_5(self):
        '''
        Residual experiment
        More details see report
        '''
        tau_val = 0.013
        f_val   = 150
        h_sq_list = []
        f_list = range(1,220)
        y_max_abs_list = []
        a_a_off_list = []
        for f_val in f_list:
            w = self.gen_w_f_tau(f_val=f_val,tau_val=tau_val)  # S*B*M*2K*T
            w_sc = w[0,0,0]
            # print(w_sc.shape)  # (32,640)
            x_sc_ref = numpy.zeros(640)
            for i in range(32):
                x_sc_temp = numpy.random.randint(-2,2) * w_sc[i]
                x_sc_ref += x_sc_temp
            x = numpy.random.rand(640) + x_sc_ref

            w_sc = w[0,0,0]
            # print(w_sc.shape)
            w_sc_T = numpy.transpose(w_sc)
            # print(w_sc_T.shape)
            a_sc_inv = numpy.matmul(w_sc, w_sc_T)
            # print(a_sc_inv.shape)
            a_sc = numpy.linalg.inv(a_sc_inv)
            # print(a_sc.shape)
            # print(numpy.diag(a_sc))
            # for i in range(32):
            #     print(a_sc[i])

            w_sc_x = numpy.matmul(w_sc, x)
            # print(w_sc_x.shape)
            a_sc_w_sc_x = numpy.matmul(a_sc, w_sc_x)
            # print(a_sc_w_sc_x.shape)
            x_sc = numpy.matmul(w_sc_T, a_sc_w_sc_x)
            # print(x_sc.shape)

            x_res = x - x_sc
            y_res = numpy.matmul(w_sc, x_res)
            # print(y_res.shape)
            y_max_abs = numpy.max(numpy.abs(y_res))
            a_a = numpy.matmul(a_sc_inv, a_sc)
            a_a_off = numpy.sum(numpy.abs(a_a - numpy.eye(32)))
            print('%s; %s'% (y_max_abs, a_a_off))
            y_max_abs_list.append(y_max_abs)
            a_a_off_list.append(a_a_off)

        # fig_file_name = '/home/dawna/tts/mw545/Export_Temp/PNG_out/sinenet_test/y_max_abs_vs_f_25+.png'
        fig_file_name = '/home/dawna/tts/mw545/Export_Temp/PNG_out/sinenet_test/y_max_abs_vs_f.png'
        self.graph_plotter.single_plot(fig_file_name, [f_list], [y_max_abs_list], ['max(abs(Wx))'])
        fig_file_name = '/home/dawna/tts/mw545/Export_Temp/PNG_out/sinenet_test/inv_e.png'
        self.graph_plotter.single_plot(fig_file_name, [f_list], [a_a_off_list], ['inversion error'])
        fig_file_name = '/home/dawna/tts/mw545/Export_Temp/PNG_out/sinenet_test/y_max_abs_vs_f_25.png'
        self.graph_plotter.single_plot(fig_file_name, [f_list[25:]], [y_max_abs_list[25:]], ['max(abs(Wx))'])
        fig_file_name = '/home/dawna/tts/mw545/Export_Temp/PNG_out/sinenet_test/inv_e_25.png'
        self.graph_plotter.single_plot(fig_file_name, [f_list[25:]], [a_a_off_list[25:]], ['inversion error'])

    def test_6(self):
        '''
        Check SineNet_Numpy is the same as the Pytorch version
        Result: w_sin_cos are slightly different on the plots; 
            Difference is larger in high frequency
            Possible due to different sine / cosine methods? Maybe check later
            h = w * x is consistent within each method
        '''
        from frontend_mw545.data_loader import Build_Sinenet_Numpy
        sinenet_layer_numpy = Build_Sinenet_Numpy(self.params)

        tau_val = 0.013
        f_val   = 150

        for f_val in range(100,101):
            tau_val = numpy.random.rand(1).astype(float)
            # w_1 = self.gen_w_f_tau(f_val=f_val,tau_val=tau_val)

            input_dim_values = {'S':1, 'B':1, 'M':1, 'T': 640}
            self.params['input_dim_values'] = input_dim_values
            f   = numpy.ones((input_dim_values['S'], input_dim_values['B'], input_dim_values['M'])).astype(float)
            tau = numpy.ones((input_dim_values['S'], input_dim_values['B'], input_dim_values['M'])).astype(float)
            x = numpy.random.rand(input_dim_values['S'], input_dim_values['B'], input_dim_values['M'], input_dim_values['T']).astype(float)
            f   = f * f_val
            tau = tau * tau_val
            f_SBM, tau_SBM, x_SBMT = self.numpy_to_tensor([f, tau, x])
            w_1 = self.sinenet_layer.construct_w_sin_cos_matrix(f_SBM, tau_SBM).cpu().detach().numpy()
            w_2 = sinenet_layer_numpy.construct_w_sin_cos_matrix(f, tau)
            print(numpy.max(numpy.abs(w_1 - w_2)))

            h_1 = self.sinenet_layer(x_SBMT, f_SBM, tau_SBM).cpu().detach().numpy()
            h_2 = sinenet_layer_numpy(x, f, tau)

            s_1 = 0
            s_2 = 0

            for t in range(input_dim_values['T']):
                s_1 += w_1[0,0,0,0,t] * x[0,0,0,t]
                s_2 += w_2[0,0,0,0,t] * x[0,0,0,t]

            print(h_1[0,0,0,0]-s_1)
            print(h_2[0,0,0,0]-s_2)

            print((h_1-h_2)[0,0,0])
            print(numpy.max(numpy.abs(h_1 - h_2)))

            # for k in range(self.params["layer_config"]['num_freq']):
            #     fig_file_name = '/home/dawna/tts/mw545/Export_Temp/PNG_out/sinenet_test/w_sc_%i.png' % k
            #     self.graph_plotter.single_plot(fig_file_name, [range(input_dim_values['T'])]*2, [w_1[0,0,0,k]+0.5, w_2[0,0,0,k]], ['w_1', 'w_2'])

            
class dv_y_test_configuration(dv_y_configuration):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.use_voiced_only = False   # Use voiced regions only
        self.use_voiced_threshold = 1. # Percentage of voiced required
        self.retrain_model = False
        self.learning_rate  = 0.00001
        # self.prev_nnets_file_name = ''
        self.python_script_name = os.path.realpath(__file__)
        self.data_dir_mode = 'data' # Use scratch for speed up

        # Waveform-level input configuration
        self.y_feat_name   = 'wav'
        self.init_wav_data()
        # self.out_feat_list = ['wav_ST', 'f_SBM', 'tau_SBM', 'vuv_SBM']
        self.out_feat_list = ['wav_SBT', 'f_SBM', 'tau_SBM', 'vuv_SBM']

        self.exp_dir = cfg.work_dir + '/exp_test'

        self.input_data_dim['S'] = 1
        self.feed_per_update = 2
        self.learning_rate = self.learning_rate / self.feed_per_update
        S_per_update = self.input_data_dim['S'] * self.feed_per_update
        self.epoch_num_batch  = {'train': int(52000/S_per_update), 'valid': int(8000/self.input_data_dim['S'])}
        self.input_data_dim['T_M'] = 160
        self.input_data_dim['M_shift'] = 40
        
        self.dv_dim = 2048
        self.nn_layer_config_list = [
            # {'type': 'Tensor_Reshape', 'io_name': 'wav_ST_2_wav_SBMT', 'win_len_shift_list':[[self.input_data_dim['T_B'], self.input_data_dim['B_shift']], [self.input_data_dim['T_M'], self.input_data_dim['M_shift']]]},
            {'type': 'Tensor_Reshape', 'io_name': 'wav_SBT_2_wav_SBMT', 'win_len_shift':[self.input_data_dim['T_M'], self.input_data_dim['M_shift']]},
            {'type': 'DW3', 'size':80, 'dropout_p':0, 'batch_norm':False},
            {'type': 'Tensor_Reshape', 'io_name': 'h_SBMD_2_h_SBD'},     # h_SBD
            # {'type': 'LReLU', 'size':256*8, 'dropout_p':0, 'batch_norm':True},
            # {'type': 'LReLU', 'size':256*8, 'dropout_p':0, 'batch_norm':True},
            {'type': 'Linear', 'size':self.dv_dim, 'dropout_p':0, 'batch_norm':True}
        ]

        # self.gpu_id = 'cpu'
        self.gpu_id = 0
        self.auto_complete(cfg)

class Pytorch_Batch_Test(object):
    """
    Pytorch_Batch_Test
    Test the effect of multiple loss.backward(), then one optimizer.step()
    """
    def __init__(self, cfg):
        super(Pytorch_Batch_Test, self).__init__()
        self.logger = make_logger("Test")
        self.cfg = cfg
        self.dv_y_cfg = dv_y_test_configuration(self.cfg)
        self.data_S = 60
        self.make_data()

    def test(self):
        self.test_2()

    def make_data(self):
        # Make a feed_dict for usage
        # S = 60 for all situations
        numpy.random.seed(545)
        dv_y_cfg = self.dv_y_cfg
        dv_y_cfg.input_data_dim['S'] = self.data_S
        self.data_loader = Build_dv_y_train_data_loader(self.cfg, dv_y_cfg)

        self.feed_dict, batch_size = self.data_loader.make_feed_dict(utter_tvt_name='train')
        # Remember to set this
        self.s_index = 0

    def load_s_data(self, s, s_index=None):
        '''
        Return s samples from self.feed_dict
        Add s to self.s_index
        Reset if self.data_S is reached
        '''
        if s_index is None:
            s_index = self.s_index

        new_feed_dict = {k: self.feed_dict[k][s_index:s_index+s] for k in self.feed_dict}
        # one_hot is 1-D S*B, need special care
        new_feed_dict['one_hot'] = self.feed_dict['one_hot'][s_index*self.dv_y_cfg.input_data_dim['B']:(s_index+s)*self.dv_y_cfg.input_data_dim['B']]

        self.s_index = s_index + s
        if self.s_index == self.data_S:
            self.s_index = 0

        return new_feed_dict       

    def build_model(self, dv_y_cfg):
        model = Build_DV_Y_model(dv_y_cfg)
        model.torch_initialisation()
        model.build_optimiser()

        return model

    def tensor_to_numpy(self, x_tensor):
        return x_tensor.cpu().detach().numpy()

    def print_model_parameters(self, model):
        self.logger.info('Print Parameters')
        for name, param in model.nn_model.named_parameters():
            print(str(name)+'  '+str(param.size())+'  '+str(param.type()))
            print(str(param.grad))

    def test_1(self):
        '''
        Debug run
        1. model.optimiser.state_dict() remains the same after zero_grad, unaffected
        2. param.grad remains the same after model.optimiser.step()
        3. param.grad becomes all 0 after model.optimiser.zero_grad()
        '''
        S = 1
        dv_y_cfg = self.dv_y_cfg
        dv_y_cfg.input_data_dim['S'] = S
        model = self.build_model(dv_y_cfg)

        feed_dict = self.load_s_data(S)
        loss = model.gen_loss(feed_dict)
        loss.backward()

        self.print_model_parameters(model)

        model.optimiser.step()
        # self.logger.info("print optimiser state_dict")
        # print(model.optimiser.state_dict())
        self.print_model_parameters(model)


        model.optimiser.zero_grad()
        # self.logger.info("print optimiser state_dict")
        # print(model.optimiser.state_dict())
        self.print_model_parameters(model)

    def test_2(self):
        '''
        Debug run
        1. (Done) loss.backward() twice, gradient double
            True for 2 different batches
        2. 2 different batches, gradient add?
            Not equal yet... Precision?
        '''

        S = 1
        dv_y_cfg = self.dv_y_cfg
        dv_y_cfg.input_data_dim['S'] = S
        model = self.build_model(dv_y_cfg)
        model.optimiser.zero_grad()

        feed_dict_1 = self.load_s_data(S)
        feed_dict_2 = self.load_s_data(S)
        

        grad_dict_1 = {}
        loss = model.gen_loss(feed_dict_1)
        loss.backward()
        for name, param in model.nn_model.named_parameters():
            grad_dict_1[name] = self.tensor_to_numpy(param.grad)
        print(grad_dict_1)

        model.optimiser.zero_grad()
        grad_dict_2 = {}
        loss = model.gen_loss(feed_dict_2)
        loss.backward()
        for name, param in model.nn_model.named_parameters():
            grad_dict_2[name] = self.tensor_to_numpy(param.grad)
        print(grad_dict_2)

        grad_dict_3 = {}
        loss = model.gen_loss(feed_dict_1)
        loss.backward()
        for name, param in model.nn_model.named_parameters():
            grad_dict_3[name] = self.tensor_to_numpy(param.grad)
        print(grad_dict_3)

        # Expect: grad_dict_3 = grad_dict_1 + grad_dict_2
        for k in grad_dict_1:
            diff = grad_dict_3[k] - grad_dict_1[k] - grad_dict_2[k]
            print(k)
            print((diff==0).all())

    def test_3(self):
        '''
        batch_norm test
        1. test train()
        '''
        pass


class SBD_to_SD_mask_Test(object):
    """docstring for SBD_to_SD_mask_Test"""
    def __init__(self, arg):
        super(SBD_to_SD_mask_Test, self).__init__()
        self.arg = arg

    def test_1(self):
        '''
        Test the current method of lambda_SBD+mask_SB1 --> lambda_SD
        '''
        S = 10
        B = 20
        D = 30
        

    def gen_lambda_SD(self, x_dict):
        ''' 
        Average over B
        For 1. better estimation of lambda; and 2. classification
        '''
        lambda_SBD = self.gen_lambda_SBD(x_dict)
        mask_SB1 = torch.unsqueeze(x_dict['output_mask_SB'], 2)
        lambda_SBD_zero_pad = torch.mul(lambda_SBD, mask_SB1)
        lambda_SD_sum  = torch.sum(lambda_SBD_zero_pad, dim=1, keepdim=False)
        out_lens_S1 = torch.unsqueeze(x_dict['out_lens'],1)
        lambda_SD = torch.true_divide(lambda_SD_sum, out_lens_S1)
        return lambda_SD

    def make_pad_mask(self, lengths, xs=None, length_dim=-1):
        """Make mask tensor containing indices of padded part.

        Args:
            lengths (LongTensor or List): Batch of lengths (B,).
            xs (Tensor, optional): The reference tensor.
                If set, masks will be the same shape as this tensor.
            length_dim (int, optional): Dimension indicator of the above tensor.
                See the example.

        Returns:
            Tensor: Mask tensor containing indices of padded part.
                    dtype=torch.uint8 in PyTorch 1.2-
                    dtype=torch.bool in PyTorch 1.2+ (including 1.2)

        Examples:
            With only lengths.

            >>> lengths = [5, 3, 2]
            >>> make_non_pad_mask(lengths)
            masks = [[0, 0, 0, 0 ,0],
                     [0, 0, 0, 1, 1],
                     [0, 0, 1, 1, 1]]

            With the reference tensor.

            >>> xs = torch.zeros((3, 2, 4))
            >>> make_pad_mask(lengths, xs)
            tensor([[[0, 0, 0, 0],
                     [0, 0, 0, 0]],
                    [[0, 0, 0, 1],
                     [0, 0, 0, 1]],
                    [[0, 0, 1, 1],
                     [0, 0, 1, 1]]], dtype=torch.uint8)
            >>> xs = torch.zeros((3, 2, 6))
            >>> make_pad_mask(lengths, xs)
            tensor([[[0, 0, 0, 0, 0, 1],
                     [0, 0, 0, 0, 0, 1]],
                    [[0, 0, 0, 1, 1, 1],
                     [0, 0, 0, 1, 1, 1]],
                    [[0, 0, 1, 1, 1, 1],
                     [0, 0, 1, 1, 1, 1]]], dtype=torch.uint8)

            With the reference tensor and dimension indicator.

            >>> xs = torch.zeros((3, 6, 6))
            >>> make_pad_mask(lengths, xs, 1)
            tensor([[[0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0],
                     [1, 1, 1, 1, 1, 1]],
                    [[0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0],
                     [1, 1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1, 1]],
                    [[0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0],
                     [1, 1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1, 1]]], dtype=torch.uint8)
            >>> make_pad_mask(lengths, xs, 2)
            tensor([[[0, 0, 0, 0, 0, 1],
                     [0, 0, 0, 0, 0, 1],
                     [0, 0, 0, 0, 0, 1],
                     [0, 0, 0, 0, 0, 1],
                     [0, 0, 0, 0, 0, 1],
                     [0, 0, 0, 0, 0, 1]],
                    [[0, 0, 0, 1, 1, 1],
                     [0, 0, 0, 1, 1, 1],
                     [0, 0, 0, 1, 1, 1],
                     [0, 0, 0, 1, 1, 1],
                     [0, 0, 0, 1, 1, 1],
                     [0, 0, 0, 1, 1, 1]],
                    [[0, 0, 1, 1, 1, 1],
                     [0, 0, 1, 1, 1, 1],
                     [0, 0, 1, 1, 1, 1],
                     [0, 0, 1, 1, 1, 1],
                     [0, 0, 1, 1, 1, 1],
                     [0, 0, 1, 1, 1, 1]]], dtype=torch.uint8)

        """
        if length_dim == 0:
            raise ValueError("length_dim cannot be 0: {}".format(length_dim))

        if not isinstance(lengths, list):
            lengths = lengths.tolist()
        bs = int(len(lengths))
        if xs is None:
            maxlen = int(max(lengths))
        else:
            maxlen = xs.size(length_dim)

        seq_range = torch.arange(0, maxlen, dtype=torch.int64)
        seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
        seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
        mask = seq_range_expand >= seq_length_expand

        if xs is not None:
            assert xs.size(0) == bs, (xs.size(0), bs)

            if length_dim < 0:
                length_dim = xs.dim() + length_dim
            # ind = (:, None, ..., None, :, , None, ..., None)
            ind = tuple(
                slice(None) if i in (0, length_dim) else None for i in range(xs.dim())
            )
            mask = mask[ind].expand_as(xs).to(xs.device)
        return mask

def make_pad_mask(lengths, xs=None, length_dim=-1):
    """Make mask tensor containing indices of padded part.

    Args:
        lengths (LongTensor or List): Batch of lengths (B,).
        xs (Tensor, optional): The reference tensor.
            If set, masks will be the same shape as this tensor.
        length_dim (int, optional): Dimension indicator of the above tensor.
            See the example.

    Returns:
        Tensor: Mask tensor containing indices of padded part.
                dtype=torch.uint8 in PyTorch 1.2-
                dtype=torch.bool in PyTorch 1.2+ (including 1.2)

    Examples:
        With only lengths.

        >>> lengths = [5, 3, 2]
        >>> make_non_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]

        With the reference tensor.

        >>> xs = torch.zeros((3, 2, 4))
        >>> make_pad_mask(lengths, xs)
        tensor([[[0, 0, 0, 0],
                 [0, 0, 0, 0]],
                [[0, 0, 0, 1],
                 [0, 0, 0, 1]],
                [[0, 0, 1, 1],
                 [0, 0, 1, 1]]], dtype=torch.uint8)
        >>> xs = torch.zeros((3, 2, 6))
        >>> make_pad_mask(lengths, xs)
        tensor([[[0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1]],
                [[0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1]],
                [[0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1]]], dtype=torch.uint8)

        With the reference tensor and dimension indicator.

        >>> xs = torch.zeros((3, 6, 6))
        >>> make_pad_mask(lengths, xs, 1)
        tensor([[[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1]],
                [[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1]],
                [[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1]]], dtype=torch.uint8)
        >>> make_pad_mask(lengths, xs, 2)
        tensor([[[0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1]],
                [[0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1]],
                [[0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1]]], dtype=torch.uint8)

    """
    if length_dim == 0:
        raise ValueError("length_dim cannot be 0: {}".format(length_dim))

    if not isinstance(lengths, list):
        lengths = lengths.tolist()
    bs = int(len(lengths))
    if xs is None:
        maxlen = int(max(lengths))
    else:
        maxlen = xs.size(length_dim)

    seq_range = torch.arange(0, maxlen, dtype=torch.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
    seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand

    if xs is not None:
        assert xs.size(0) == bs, (xs.size(0), bs)

        if length_dim < 0:
            length_dim = xs.dim() + length_dim
        # ind = (:, None, ..., None, :, , None, ..., None)
        ind = tuple(
            slice(None) if i in (0, length_dim) else None for i in range(xs.dim())
        )
        mask = mask[ind].expand_as(xs).to(xs.device)
    return mask
        
def make_pad_mask_test():
    S = 3
    B = 4
    lengths_SB_n = numpy.random.randint(1,10,(S,B))
    print(lengths_SB_n)
    lengths_SB_t = torch.tensor(lengths_SB_n)
    mask_SBT = make_pad_mask(lengths_SB_t)
    print(mask_SBT)
    print(mask_SBT.numpy())

def TDNN_test():
    # Test this new TDNN module
    from nn_torch.x_vector import TDNN
    input_dim = 23
    x_numpy = numpy.random.rand(10,300,input_dim)
    x = torch.tensor(x_numpy)
    tdnn = TDNN(input_dim=input_dim, output_dim=512, context_size=3, dilation=1,dropout_p=0.5)
    y = tdnn(x)
    print(y)
    print(y.size())

class Build_Layer(torch.nn.Module):
    '''
    Masked Soffmax Layer
    Input: S*B*1
    Mask: S*B
    Output: S*B*1
    '''
    def __init__(self):
        super().__init__()

    def forward(self, x_dict):
        
        x = x_dict['h'] +1
        y_dict = {'h':x}
        x_dict['l'] += 2
        return y_dict

def dict_modify_test():
    x = numpy.random.rand(10,300,23)
    l = numpy.random.randint(low=0,high=300,size=10)
    x = torch.tensor(x)
    l = torch.tensor(l)
    x_dict = {'h':x, 'l':l}

    print(x_dict['l'])
    layer_fn = Build_Layer()
    y_dict = layer_fn(x_dict)
    print(x_dict['l'])


def run_Torch_Test(cfg):
    # torch_test = Pytorch_Batch_Test(cfg)
    # torch_test.test()

    # dvy_test = DV_Y_Build_Test(cfg)
    # dvy_test.speed_test()

    # make_pad_mask_test()
    # TDNN_test()
    # dict_modify_test()
    from  nn_torch.sinenet_f_tau_test import run_sinenet_f_tau_test
    run_sinenet_f_tau_test(cfg)



