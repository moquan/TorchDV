# torch_tests.py

import os, sys, pickle, time, shutil, logging, copy
import math, numpy, scipy
numpy.random.seed(545)
import torch
torch.manual_seed(545)

from frontend_mw545.modules import make_logger, log_class_attri
from frontend_mw545.frontend_tests import Graph_Plotting

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

        import torch
        global torch
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
        from exp_mw545.exp_dv_wav_subwin import dv_y_wav_subwin_configuration

        self.cfg = cfg
        self.dv_y_cfg = dv_y_wav_subwin_configuration(cfg)
        # log_class_attri(dv_y_cfg, self.logger)

    def model_build_test(self):
        from nn_torch.torch_models import Build_DV_Y_model
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
        from frontend_mw545.data_loader import Build_dv_y_train_data_loader
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
        from frontend_mw545.data_loader import Build_dv_y_train_data_loader
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
        from frontend_mw545.data_loader import Build_dv_y_train_data_loader
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


def vuv_loss_multiple(cfg):
    '''
    plot loss vs vuv for different experiments
    each experiment has 3 curves, train/valid/test
    '''
    from exp_mw545.exp_dv_y import Build_DV_Y_Testing

    fig_file_name = '/home/dawna/tts/mw545/Export_Temp/PNG_out/sinenet_test/vuv_loss.png'

    dv_y_cfg_list = []
    legend_name_list = ['sine16','DNN']
    from exp_mw545.exp_dv_wav_sinenet_v1 import dv_y_wav_sinenet_configuration
    dv_y_cfg_1 = dv_y_wav_sinenet_configuration(cfg)
    dv_y_cfg_1.prev_nnets_file_name = '/home/dawna/tts/mw545/TorchDV/dv_wav_sinenet_v1/dv_y_wav_lr_0.000100_Sin83f16_LRe256B_LRe256B_Lin8BD_DV8S10B36M33T640/Model'
    dv_y_cfg_list.append(dv_y_cfg_1)

    from exp_mw545.exp_dv_wav_subwin import dv_y_wav_subwin_configuration
    dv_y_cfg_2 = dv_y_wav_subwin_configuration(cfg)
    dv_y_cfg_2.prev_nnets_file_name = '/home/dawna/tts/mw545/TorchDV/dv_wav_subwin/dv_y_wav_lr_0.000100_LRe80_LRe256B_LRe256B_Lin8BD_DV8S10B36M33T640/Model'
    dv_y_cfg_list.append(dv_y_cfg_2)

    

    error_list  = []
    legend_list = []

    for dv_y_cfg, legend_name in zip(dv_y_cfg_list, legend_name_list):
        dv_y_model_test = Build_DV_Y_Testing(cfg, dv_y_cfg)
        loss_dict = dv_y_model_test.vuv_loss_test(plot_loss=False)

        error_list.extend([loss_dict['train'], loss_dict['valid'], loss_dict['test']])
        legend_list.extend([legend_name+'_train', legend_name+'_valid', legend_name+'_test'])


    graph_plotter = Graph_Plotting()
    graph_plotter.single_plot(fig_file_name, None, error_list, legend_list)

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
        self.device_id = torch.device("cuda:2")

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

def run_Torch_Test(cfg):
    # sinenet_test = Sinenet_Test(cfg)
    # sinenet_test.test_5()

    dvy_test = DV_Y_Build_Test(cfg)
    dvy_test.speed_test()

    # vuv_loss_multiple(cfg)

