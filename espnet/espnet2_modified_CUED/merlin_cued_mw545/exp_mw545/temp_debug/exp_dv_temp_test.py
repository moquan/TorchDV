# exp_dv_temp_test.py

import os, sys, pickle, time, shutil, logging, copy
import math, numpy, scipy
numpy.random.seed(545)
from modules import make_logger, read_file_list, prepare_file_path, prepare_file_path_list, make_held_out_file_number, copy_to_scratch
from modules import keep_by_speaker, remove_by_speaker, keep_by_file_number, remove_by_file_number, keep_by_min_max_file_number, check_and_change_to_list
from modules_2 import compute_feat_dim, log_class_attri, resil_nn_file_list, norm_nn_file_list, get_utters_from_binary_dict, get_one_utter_by_name, count_male_female_class_errors
# from modules_torch import torch_initialisation

from io_funcs.binary_io import BinaryIOCollection
io_fun = BinaryIOCollection()



def temporary_test(cfg):
    pass


def plot_sinenet(cfg, dv_y_cfg):
    numpy.random.seed(548)
    '''
    Plot all filters in sinenet
    If use real data, plot real data too
    '''
    logger = make_logger("dv_y_config")
    dv_y_cfg = copy.deepcopy(dv_y_cfg)
    dv_y_cfg.batch_num_spk = 1
    dv_y_cfg.spk_num_seq   = 20 # Use this for different frequency
    dv_y_cfg.seq_num_win   = 40 # Use this for different pitch locations    
    log_class_attri(dv_y_cfg, logger, except_list=dv_y_cfg.log_except_list)

    dv_y_model = torch_initialisation(dv_y_cfg)
    dv_y_model.load_nn_model(dv_y_cfg.nnets_file_name)
    dv_y_model.eval()

    BTD_feat_remain = None
    start_frame_index = int(51000)
    speaker_id = 'p153'
    file_name  = 'p153_003'
    make_feed_dict_method_test = dv_y_cfg.make_feed_dict_method_test
    feed_dict, gen_finish, batch_size, BTD_feat_remain = make_feed_dict_method_test(dv_y_cfg, cfg.nn_feat_scratch_dirs, speaker_id, file_name, start_frame_index, BTD_feat_remain)

    W_a_b = dv_y_model.nn_model.layer_list[0].layer_fn.sinenet_layer.fc_fn.weight.cpu().detach().numpy()
    W_s_c = dv_y_model.gen_w_sin_cos(feed_dict)
    x = feed_dict['x']

    print(W_a_b.shape) # (80, 32)
    print(W_s_c.shape) # (1, 20, 40, 32, 640)
    print(x.shape)     # (1, 12000)

    num_freq = 16
    seq_win_len = 640
    sine_size = 80

    W_s_c = W_s_c[0,0,0]
    x = x[0,start_frame_index:start_frame_index+seq_win_len]

    print(W_a_b.shape) # (80, 32)
    print(W_s_c.shape) # (32, 640)
    print(x.shape)     # (640,)

    # Heatmap of W_a_b
    fig, ax = plt.subplots()
    im = ax.imshow(W_a_b)
    fig_name = '/home/dawna/tts/mw545/Export_Temp/PNG_out/' + "heatmap.png"
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('mag', rotation=-90, va="bottom")
    logger.info('Saving heatmap to %s' % fig_name)
    fig.savefig(fig_name)
    plt.close(fig)

    W_ab_combine = numpy.zeros((sine_size, num_freq))
    for i in range(sine_size):
        for j in range(num_freq):
            a = W_a_b[i,j]
            b = W_a_b[i,j+num_freq]
            W_ab_combine[i,j] = numpy.sqrt(numpy.square(a)+numpy.square(b))

    # Heatmap of W_ab_combine
    fig, ax = plt.subplots()
    im = ax.imshow(W_ab_combine)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('', rotation=-90, va="bottom")
    ax.set_title('Amplitude; 80 filters, 16 frequencies')
    fig_name = '/home/dawna/tts/mw545/Export_Temp/PNG_out/' + "heatmap_combine.png"
    logger.info('Saving heatmap to %s' % fig_name)
    fig.savefig(fig_name)
    plt.close(fig)

    phase_ab_combine = numpy.zeros((sine_size, num_freq))
    for i in range(sine_size):
        for j in range(num_freq):
            a = W_a_b[i,j]
            b = W_a_b[i,j+num_freq]
            phase_ab_combine[i,j] = numpy.arctan(a/b)

    # Heatmap of phase_ab_combine
    fig, ax = plt.subplots()
    im = ax.imshow(phase_ab_combine)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('', rotation=-90, va="bottom")
    ax.set_title('Phase; 80 filters, 16 frequencies')
    fig_name = '/home/dawna/tts/mw545/Export_Temp/PNG_out/' + "heatmap_phase.png"
    logger.info('Saving heatmap to %s' % fig_name)
    fig.savefig(fig_name)
    plt.close(fig)

    # Combine the filters by frequency?

    W_absc = numpy.zeros((sine_size, seq_win_len))
    for i in range(sine_size):
        for j in range(seq_win_len):
            a = W_a_b[i,0]
            b = W_a_b[i,num_freq]
            s = W_s_c[0,j]
            c = W_s_c[num_freq,j]

            W_absc[i,j] = a * s + b * c

    # Plot the lowest frequency only, for all filters
    D_size = 5
    D_tot  = int(sine_size/D_size)
    for d_1 in range(D_tot):
        fig, ax_list = plt.subplots(D_size+1)        

        for d_2 in range(D_size):
            d = d_1 * D_size + d_2
            ax_list[d_2].plot(W_absc[d])
        # Plot x as well
        ax_list[D_size].plot(x)

        fig_name = '/home/dawna/tts/mw545/Export_Temp/PNG_out/' + "sinenet_filter_%i.png" % d_1
        logger.info('Saving h to %s' % fig_name)
        fig.savefig(fig_name)
        plt.close(fig)







    # W = dv_y_model.gen_w_mul_w_sin_cos(feed_dict)

    # S = dv_y_cfg.batch_num_spk
    # B = dv_y_cfg.spk_num_seq
    # M = dv_y_cfg.seq_num_win
    # D = 80
    
    # D_size = 5
    # D_tot  = int(D/D_size)

    # for d_1 in range(D_tot):
    #     fig, ax_list = plt.subplots(D_size+1)        

    #     for d_2 in range(D_size):
    #         d = d_1 * D_size + d_2
    #         ax_list[d_2].plot(W[0,0,0,d])
    #     if is_use_true_data:
    #         # Plot x as well
    #         x = feed_dict['x']
    #         ax_list[D_size].plot(x[0,0,0])

    #     fig_name = '/home/dawna/tts/mw545/Export_Temp/PNG_out/' + "sinenet_filter_%i.png" % d_1
    #     logger.info('Saving h to %s' % fig_name)
    #     fig.savefig(fig_name)
    #     plt.close(fig)

def plot_sinenet_old(cfg, dv_y_cfg):
    numpy.random.seed(548)
    '''
    Plot all filters in sinenet
    If use real data, plot real data too
    '''
    logger = make_logger("dv_y_config")
    dv_y_cfg = copy.deepcopy(dv_y_cfg)
    dv_y_cfg.batch_num_spk = 1
    dv_y_cfg.spk_num_seq   = 20 # Use this for different frequency
    dv_y_cfg.seq_num_win   = 40 # Use this for different pitch locations    
    log_class_attri(dv_y_cfg, logger, except_list=dv_y_cfg.log_except_list)

    dv_y_model = torch_initialisation(dv_y_cfg)
    dv_y_model.load_nn_model(dv_y_cfg.nnets_file_name)
    dv_y_model.eval()

    is_use_true_data = True
    if is_use_true_data:
        BTD_feat_remain = None
        start_frame_index = int(51000)
        speaker_id = 'p15'
        file_name  = 'p15_003'
        make_feed_dict_method_test = dv_y_cfg.make_feed_dict_method_test
        feed_dict, gen_finish, batch_size, BTD_feat_remain = make_feed_dict_method_test(dv_y_cfg, cfg.nn_feat_scratch_dirs, speaker_id, file_name, start_frame_index, BTD_feat_remain)
    else:
        f_0     = 120.
        f_inc   = 10.
        tau_0   = 0.
        tau_inc = 10./16000
        f = numpy.zeros((dv_y_cfg.batch_num_spk, dv_y_cfg.spk_num_seq, dv_y_cfg.seq_num_win))
        for i in range(dv_y_cfg.spk_num_seq):
            f[0,i] = f_0 + i * f_inc
        lf = numpy.log(f)
        log_f_mean = 5.04418
        log_f_std  = 0.358402
        nlf = (lf - log_f_mean) / log_f_std
        # lf = torch.add(torch.mul(nlf, self.log_f_std), self.log_f_mean) # S*B*M
        tau = numpy.zeros((dv_y_cfg.batch_num_spk, dv_y_cfg.spk_num_seq, dv_y_cfg.seq_num_win))
        for i in range(dv_y_cfg.seq_num_win):
            tau[0,:,i] = tau_0 + i * tau_inc
        feed_dict = {'nlf': nlf, 'tau': tau}

    W = dv_y_model.gen_w_mul_w_sin_cos(feed_dict)

    S = dv_y_cfg.batch_num_spk
    B = dv_y_cfg.spk_num_seq
    M = dv_y_cfg.seq_num_win
    D = 80
    
    D_size = 5
    D_tot  = int(D/D_size)

    for d_1 in range(D_tot):
        fig, ax_list = plt.subplots(D_size+1)        

        for d_2 in range(D_size):
            d = d_1 * D_size + d_2
            ax_list[d_2].plot(W[0,0,0,d])
        if is_use_true_data:
            # Plot x as well
            x = feed_dict['x']
            ax_list[D_size].plot(x[0,0,0])

        fig_name = '/home/dawna/tts/mw545/Export_Temp/PNG_out/' + "sinenet_filter_%i.png" % d_1
        logger.info('Saving h to %s' % fig_name)
        fig.savefig(fig_name)
        plt.close(fig)

def vuv_test_dv_y_model(cfg, dv_y_cfg):
    numpy.random.seed(549)
    '''
    Run the evaluation part of the training procedure
    Store the results based on v/uv
    Print: amount of data, and CE, vs amount of v/uv
    '''
    logger = make_logger("dv_y_config")
    dv_y_cfg = copy.deepcopy(dv_y_cfg)
    log_class_attri(dv_y_cfg, logger, except_list=dv_y_cfg.log_except_list)

    # Need to extract vuv information
    if 'vuv' not in dv_y_cfg.out_feat_list:
        dv_y_cfg.out_feat_list.append('vuv')

    logger = make_logger("vuv_test_dvy")
    logger.info('Creating data lists')
    speaker_id_list = dv_y_cfg.speaker_id_list_dict['train'] # For DV training and evaluation, use train speakers only
    speaker_loader  = list_random_loader(speaker_id_list)
    file_id_list    = read_file_list(cfg.file_id_list_file)
    file_list_dict  = make_dv_file_list(file_id_list, speaker_id_list, dv_y_cfg.data_split_file_number) # In the form of: file_list[(speaker_id, 'train')]
    make_feed_dict_method_vuv_test = dv_y_cfg.make_feed_dict_method_vuv_test

    dv_y_model = torch_initialisation(dv_y_cfg)
    dv_y_model.load_nn_model(dv_y_cfg.nnets_file_name)
    dv_y_model.eval()

    for utter_tvt_name in ['train', 'valid', 'test']:
        ce_holders = [[] for i in range(dv_y_cfg.seq_num_win+1)]
        # vuv = numpy.zeros((dv_y_cfg.batch_num_spk, dv_y_cfg.spk_num_seq, dv_y_cfg.seq_num_win))
        num_batch = dv_y_cfg.epoch_num_batch['valid'] * 10
        # num_batch = 1
        for batch_idx in range(num_batch):
            # Draw random speakers
            batch_speaker_list = speaker_loader.draw_n_samples(dv_y_cfg.batch_num_spk)
            # Make feed_dict for evaluation
            feed_dict, batch_size = make_feed_dict_method_vuv_test(dv_y_cfg, file_list_dict, cfg.nn_feat_scratch_dirs, batch_speaker_list, utter_tvt=utter_tvt_name)
            vuv_SBM = feed_dict['vuv']
            with dv_y_model.no_grad():
                ce_SB = dv_y_model.gen_SB_loss_value(feed_dict=feed_dict) # 1D vector
                vuv_SB = numpy.sum(vuv_SBM, axis=2).reshape(-1)
                s_b = dv_y_cfg.batch_num_spk * dv_y_cfg.spk_num_seq
                for i in range(s_b):
                    vuv = int(vuv_SB[i])
                    ce  = int(ce_SB[i])
                    ce_holders[vuv].append(ce)

        len_list = [len(ce_list) for ce_list in ce_holders]
        mean_list = [numpy.mean(ce_list) for ce_list in ce_holders]
        print(len_list)
        print(mean_list)
        ce_sum = 0.
        num_sum = 0
        for (l,m) in zip(len_list, mean_list):
            ce_sum += l*m
            num_sum += l
        ce_mean = ce_sum / float(num_sum)
        logger.info('Mean Cross Entropy Results of %s Dataset is %.4f' % (utter_tvt_name, ce_mean))

def ce_vs_var_nlf_test(cfg, dv_y_cfg):
    numpy.random.seed(550)
    '''
    Run the evaluation part of the training procedure
    Print: amount of data, and CE, vs amount of var(nlf); only use voiced region data
    '''
    logger = make_logger("dv_y_config")
    dv_y_cfg = copy.deepcopy(dv_y_cfg)
    log_class_attri(dv_y_cfg, logger, except_list=dv_y_cfg.log_except_list)
    if 'vuv' not in dv_y_cfg.out_feat_list:
        dv_y_cfg.out_feat_list.append('vuv')
    if 'nlf' not in dv_y_cfg.out_feat_list:
        dv_y_cfg.out_feat_list.append('nlf')

    logger = make_logger("vuv_test_dvy")
    logger.info('Creating data lists')
    speaker_id_list = dv_y_cfg.speaker_id_list_dict['train'] # For DV training and evaluation, use train speakers only
    speaker_loader  = list_random_loader(speaker_id_list)
    file_id_list    = read_file_list(cfg.file_id_list_file)
    file_list_dict  = make_dv_file_list(file_id_list, speaker_id_list, dv_y_cfg.data_split_file_number) # In the form of: file_list[(speaker_id, 'train')]
    make_feed_dict_method_train = dv_y_cfg.make_feed_dict_method_train

    dv_y_model = torch_initialisation(dv_y_cfg)
    dv_y_model.load_nn_model(dv_y_cfg.nnets_file_name)
    dv_y_model.eval()

    dv_y_cfg.out_feat_list.append('nlf_var')
    nlf_var_list = {}
    ce_list = {}

    # DNN-based model part
    from exp_mw545.exp_dv_wav_subwin import dv_y_wav_subwin_configuration
    cfg_dnn = copy.deepcopy(cfg)
    cfg_dnn.work_dir = "/home/dawna/tts/mw545/TorchDV/dv_wav_subwin"
    dnn_cfg = dv_y_wav_subwin_configuration(cfg_dnn)
    dnn_model = torch_initialisation(dnn_cfg)
    dnn_model.load_nn_model(dnn_cfg.nnets_file_name)
    dnn_model.eval()

    for utter_tvt_name in ['train', 'valid', 'test']:
        nlf_var_list[utter_tvt_name] = []
        ce_list[utter_tvt_name] = []
        # vuv = numpy.zeros((dv_y_cfg.batch_num_spk, dv_y_cfg.spk_num_seq, dv_y_cfg.seq_num_win))
        num_batch = dv_y_cfg.epoch_num_batch['valid']
        # num_batch = 1
        for batch_idx in range(num_batch):
            # Draw random speakers
            batch_speaker_list = speaker_loader.draw_n_samples(dv_y_cfg.batch_num_spk)
            # Make feed_dict for evaluation
            feed_dict, batch_size = make_feed_dict_method_train(dv_y_cfg, file_list_dict, cfg.nn_feat_scratch_dirs, batch_speaker_list, utter_tvt=utter_tvt_name)
            nlf_var = feed_dict['nlf_var']
            vuv_SBM = feed_dict['vuv']
            with dv_y_model.no_grad():
                ce_SB = dv_y_model.gen_SB_loss_value(feed_dict=feed_dict) # 1D vector
            with dnn_model.no_grad():
                ce_SB_dnn = dnn_model.gen_SB_loss_value(feed_dict=feed_dict) # 1D vector
                vuv_SB = numpy.sum(vuv_SBM, axis=2).reshape(-1)
                nlf_var_SB = numpy.sum(nlf_var, axis=2).reshape(-1)
                s_b = dv_y_cfg.batch_num_spk * dv_y_cfg.spk_num_seq
                for i in range(s_b):
                    if vuv_SB[i] == dv_y_cfg.seq_num_win:
                        nlf_var_list[utter_tvt_name].append(nlf_var_SB[i])
                        ce_list[utter_tvt_name].append(ce_SB_dnn[i]-ce_SB[i])
        from scipy.stats.stats import pearsonr
        corr_coef = pearsonr(nlf_var_list[utter_tvt_name], ce_list[utter_tvt_name])
        print(corr_coef)
        # logger.info('Corr coef is %4.f for %s' % (corr_coef, utter_tvt_name))
        # fig, ax = plt.subplots()
        # ax.plot(nlf_var_list, ce_list, '.')
        # ax.set_title('ce vs var(nlf) %s' % utter_tvt_name)
        # fig_name = '/home/dawna/tts/mw545/Export_Temp/PNG_out/' + "ce_var_nlf_%s.png" % utter_tvt_name
        # logger.info('Saving to %s' % fig_name)
        # fig.savefig(fig_name)
        # plt.close(fig)
    # Dump results
    pickle.dump(nlf_var_list, open(os.path.join(dv_y_cfg.exp_dir, 'nlf_var_list.data'), 'wb'))
    pickle.dump(ce_list, open(os.path.join(dv_y_cfg.exp_dir, 'ce_list.data'), 'wb'))






# from exp_mw545.exp_dv_cmp_pytorch import list_random_loader, dv_y_configuration, make_dv_y_exp_dir_name, make_dv_file_list, train_dv_y_model, class_test_dv_y_model, distance_test_dv_y_model, vuv_test_dv_y_model, ce_vs_var_nlf_test
# from exp_mw545.exp_dv_cmp_pytorch import make_feed_dict_y_wav_subwin_train, make_feed_dict_y_wav_subwin_test

# class dv_y_wav_temp_test_configuration(dv_y_configuration):
    
#     def __init__(self, cfg):
#         super().__init__(cfg)
#         self.use_voiced_only = False # Use voiced regions only
#         self.use_voiced_threshold = 1. # Percentage of voiced required
#         self.finetune_model = False
#         # self.learning_rate  = 0.0001
#         # self.prev_nnets_file_name = '/home/dawna/tts/mw545/TorchDV/dv_wav_sinenet_v3/dv_y_wav_lr_0.000100_Sin80f10_ReL256BN_ReL256BN_ReL8DR_DV8S100B10T3200D1/Model'
#         self.python_script_name = os.path.realpath(__file__)

#         # Waveform-level input configuration
#         self.y_feat_name   = 'wav'
#         self.out_feat_list = ['wav', 'nlf', 'tau', 'vuv']
#         self.batch_seq_total_len = 12000 # Number of frames at 16kHz; 32000 for 2s
#         self.batch_seq_len   = 3200 # T
#         self.batch_seq_shift = 10*80
#         self.seq_win_len   = 640
#         self.seq_win_shift = 80
#         self.seq_num_win   = int((self.batch_seq_len - self.seq_win_len) / self.seq_win_shift) + 1

#         # self.batch_num_spk = 100
#         # self.dv_dim = 8
#         self.nn_layer_config_list = [
#             # Must contain: type, size; num_channels, dropout_p are optional, default 0, 1
#             {'type':'SinenetV3_ST', 'size':81, 'sine_size':80, 'num_freq':16, 'win_len_shift_list':[[self.batch_seq_len, self.batch_seq_shift], [self.seq_win_len, self.seq_win_shift]], 'total_length':self.batch_seq_total_len, 'dropout_p':0, 'batch_norm':False},
#             {'type':'LReLUDV', 'size':256, 'dropout_p':0, 'batch_norm':True},
#             {'type':'LReLUDV', 'size':256, 'dropout_p':0, 'batch_norm':True},
#             {'type':'LReLUDV', 'size':self.dv_dim, 'dropout_p':0.2, 'batch_norm':False}
#         ]

#         # self.gpu_id = 'cpu'
#         self.gpu_id = 2

#         from modules_torch import DV_Y_ST_model
#         self.dv_y_model_class = DV_Y_ST_model

#         self.make_feed_dict_method_train = make_feed_dict_y_wav_subwin_train
#         self.make_feed_dict_method_test  = make_feed_dict_y_wav_subwin_test
#         # if self.use_voiced_only:
#         #     self.make_feed_dict_method_train = make_feed_dict_y_wav_sinenet_train_voiced_only
#         #     self.make_feed_dict_method_test  = make_feed_dict_y_wav_sinenet_test_voiced_only
#         # else:
#         #     self.make_feed_dict_method_train = make_feed_dict_y_wav_sinenet_train
#         #     self.make_feed_dict_method_test  = make_feed_dict_y_wav_sinenet_test
#         #     self.make_feed_dict_method_distance  = make_feed_dict_y_wav_sinenet_distance
#         # self.make_feed_dict_method_vuv_test = make_feed_dict_y_wav_sinenet_train
#         self.make_feed_dict_method_vuv_test = make_feed_dict_y_wav_subwin_train
#         self.auto_complete(cfg)

#     def reload_model_param(self):
#         self.nn_layer_config_list[0]['win_len_shift_list'] = [[self.batch_seq_len, self.batch_seq_shift], [self.seq_win_len, self.seq_win_shift]]
#         self.nn_layer_config_list[0]['total_length'] = self.batch_seq_total_len

# def wav_ST_2_SBMD(wav_ST, dv_y_cfg):
#     # Reshape from S*T to S*B*M*D
#     wav_SBMD = numpy.zeros((dv_y_cfg.batch_num_spk, dv_y_cfg.spk_num_seq, dv_y_cfg.seq_num_win, dv_y_cfg.seq_win_len))
#     for speaker_idx in range(dv_y_cfg.batch_num_spk):
#         for seq_idx in range(dv_y_cfg.spk_num_seq):
#             spk_seq_index = seq_idx
#             seq_start = seq_idx * dv_y_cfg.batch_seq_shift
#             for win_idx in range(dv_y_cfg.seq_num_win):
#                 win_start = seq_start + win_idx * dv_y_cfg.seq_win_shift
#                 win_end   = win_start + dv_y_cfg.seq_win_len - 1 # Inclusive index
#                 wav_SBMD[speaker_idx, spk_seq_index, win_idx, :] = wav_ST[speaker_idx,win_start:win_end+1]
#     return wav_SBMD

# class Build_filter_test_class(object):
#     """docstring for filter_test_class"""
#     def __init__(self, dv_y_cfg):

#         self.dim_list = {'S': dv_y_cfg.batch_num_spk, 'B':dv_y_cfg.spk_num_seq, 'M':dv_y_cfg.seq_num_win, 'D':dv_y_cfg.seq_win_len}

#         self.t_wav = 1./16000
#         self.log_f_mean = 5.04418
#         self.log_f_std  = 0.358402

#         self.tot_voice_counter = 0
#         self.tot_max_counter = 0

#         self.f_list = numpy.arange(-30,31,1)
#         self.phi_list = numpy.arange(-180,181,5) / 180. * numpy.pi

#         self.f_result_holder = {}
#         self.phi_result_holder = {}
#         for f in self.f_list:
#             self.f_result_holder[f] = 0.
#         for phi in self.phi_list:
#             self.phi_result_holder[phi] = 0.

#         self.n_T_vec = self.make_n_T_vec()

#     def eval_x_nlf_tau(self, x, vuv, nlf, tau):

#         self.tot_voice_counter += numpy.sum(vuv)

#         lf = (nlf * self.log_f_std) + self.log_f_mean
#         f0  = numpy.exp(lf)

#         s_0 = self.filter_sum(x, f=f0, tau=tau, phi=0, vuv=vuv)

#         for f in self.f_list:
#             s_f = self.filter_sum(x, f=f0+f, tau=tau, phi=0, vuv=vuv)
#             self.f_result_holder[f] += (s_f - s_0)

#         for phi in self.phi_list:
#             s_phi = self.filter_sum(x, f=f0, tau=tau, phi=phi, vuv=vuv)
#             self.phi_result_holder[phi]  += (s_phi - s_0)

#         # self.check_if_max( x, vuv, nlf, tau)

#     def check_if_max(self, x, vuv, nlf, tau):
#         self.tot_voice_counter += numpy.sum(vuv)

#         lf = (nlf * self.log_f_std) + self.log_f_mean
#         f0  = numpy.exp(lf)

#         s_0 = self.filter_out_SBM(x, f=f0, tau=tau, phi=0)
#         max_bool = vuv
#         for f in self.f_list:
#             s_f = self.filter_out_SBM(x, f=f0+f, tau=tau, phi=0)
#             max_bool = max_bool * (s_0 > s_f)

#         for phi in self.phi_list:
#             s_phi = self.filter_out_SBM(x, f=f0, tau=tau, phi=phi)
#             max_bool = max_bool * (s_0 > s_phi)

#         self.tot_max_counter += numpy.sum(max_bool)

#     def make_n_T_vec(self):
#         win_len = self.dim_list['D']
#         n_T_vec = numpy.zeros(win_len)
#         for n in range(win_len):
#             n_T_vec[n] = n * self.t_wav
#         return n_T_vec

#     def filter_out_SBM(self, x, f, tau, phi):
#         tau_1 = numpy.expand_dims(tau, 3) # S*B*M --> # S*B*M*1
#         t = self.n_T_vec - tau_1 # T + S*B*M*1 -> S*B*M*T

#         # Degree in radian
#         f_1 = numpy.expand_dims(f, 3) # S*B*M --> # S*B*M*1
#         deg_SBMD = 2 * numpy.pi * f_1 * t + phi

#         s_SBM = numpy.sum(numpy.cos(deg_SBMD) * x, axis=3)
#         return s_SBM

#     def filter_sum(self, x, f, tau, phi, vuv):
#         s_SBM = self.filter_out_SBM(x, f, tau, phi)
#         return numpy.sum(s_SBM * vuv)

#     def print_results(self):
#         print(self.tot_voice_counter)
#         print(self.tot_max_counter)

#         print(self.f_list)
#         f_print_list = [self.f_result_holder[f] / self.tot_voice_counter for f in self.f_list]
#         print(f_print_list)

#         print(self.phi_list)
#         phi_print_list = [self.phi_result_holder[phi] / self.tot_voice_counter for phi in self.phi_list]
#         print(phi_print_list)
     
# def filter_test(cfg):
#     numpy.random.seed(545)

#     logger = make_logger("dv_y_config")
#     dv_y_cfg = dv_y_wav_temp_test_configuration(cfg)
#     log_class_attri(dv_y_cfg, logger, except_list=dv_y_cfg.log_except_list)

#     logger = make_logger("train_dvy")
#     logger.info('Creating data lists')
#     speaker_id_list = dv_y_cfg.speaker_id_list_dict['train'] # For DV training and evaluation, use train speakers only
#     speaker_loader  = list_random_loader(speaker_id_list)
#     file_id_list    = read_file_list(cfg.file_id_list_file)
#     file_list_dict  = make_dv_file_list(file_id_list, speaker_id_list, dv_y_cfg.data_split_file_number) # In the form of: file_list[(speaker_id, 'train')]
#     make_feed_dict_method_train = dv_y_cfg.make_feed_dict_method_train

#     epoch_num_batch = 40
#     filter_test_class = Build_filter_test_class(dv_y_cfg)
    

#     for batch_idx in range(epoch_num_batch):
#         # Draw random speakers
#         batch_speaker_list = speaker_loader.draw_n_samples(dv_y_cfg.batch_num_spk)
#         # Make feed_dict for training
#         feed_dict, batch_size = make_feed_dict_method_train(dv_y_cfg, file_list_dict, cfg.nn_feat_scratch_dirs, batch_speaker_list, utter_tvt='train')

#         x_ST = feed_dict['x']
#         x_SBMD = wav_ST_2_SBMD(x_ST, dv_y_cfg)

#         nlf_SBM = feed_dict['nlf']
#         tau_SBM = feed_dict['tau']
#         vuv_SBM = feed_dict['vuv']

#         filter_test_class.eval_x_nlf_tau(x_SBMD, vuv_SBM, nlf_SBM, tau_SBM)

#         # for speaker_idx in range(dv_y_cfg.batch_num_spk):
#         #     for seq_idx in range(dv_y_cfg.utter_num_seq):
#         #         for win_idx in range(dv_y_cfg.seq_num_win):
#         #             vuv = vuv_SBM[speaker_idx, seq_idx, win_idx]
#         #             if vuv > 0:
#         #                 tot_voice_counter += 1
#         #                 nlf = nlf_SBM[speaker_idx, seq_idx, win_idx]
#         #                 tau = tau_SBM[speaker_idx, seq_idx, win_idx]
#         #                 x = x_SBMD[speaker_idx, seq_idx, win_idx]
                        

#     filter_test_class.print_results()

# def vuv_tau_read_test(cfg):

#     logger = make_logger("dv_y_config")
#     dv_y_cfg = dv_y_wav_temp_test_configuration(cfg)
#     log_class_attri(dv_y_cfg, logger, except_list=dv_y_cfg.log_except_list)

#     file_id = 'p290_028'

# class Build_file_length_stats_class(object):
#     def __init__(self, cfg):
#         self.num_files_to_store = 10
#         self.init_check_len_min()
#         self.init_check_len_max()
#         self.init_check_len_diff()
#         self.init_store_len_bin()
#         self.init_check_file_used(cfg)

#     def load_new_file(self, new_file_name, file_length_dict):
#         file_length_list = file_length_dict.values()
#         len_min  = min(file_length_list)
#         len_max  = max(file_length_list)
#         len_diff = len_max - len_min

#         self.check_len_min(new_file_name, len_min)
#         self.check_len_max(new_file_name, len_max)
#         self.check_len_diff(new_file_name, len_diff)
#         self.store_len_bin(len_min)
#         pass

#     def init_check_file_used(self, cfg):
#         self.speaker_id_list_dict = cfg.speaker_id_list_dict
#         self.held_out_file_number = cfg.held_out_file_number

#     def check_file_used(self, file_id):
#         speaker_id = file_id.split('_')[0]
#         file_id    = file_id.split('_')[1]
#         if speaker_id in self.speaker_id_list_dict['train']:
#             if file_id in self.held_out_file_number:
#                 return False
#             else:
#                 return True
#         elif (speaker_id in self.speaker_id_list_dict['valid']) or (speaker_id in self.speaker_id_list_dict['test']) :
#             if file_id in self.held_out_file_number:
#                 return True
#             else:
#                 return False
#         else:
#             return False


#     def init_check_len_min(self):
#         self.len_min_max_value = 1000000
#         self.file_dict_len_min = {str(x): self.len_min_max_value for x in range(self.num_files_to_store)}
#         print(self.file_dict_len_min)

#     def check_len_min(self, new_file_name, len_min):
#         # Store 10 shortest files; lengths and names
#         # Replace the longest with the current, if current is shorter
#         if len_min < self.len_min_max_value:
#             for k in self.file_dict_len_min:
#                 if self.file_dict_len_min[k] == self.len_min_max_value:
#                     del self.file_dict_len_min[k]
#                     self.file_dict_len_min[new_file_name] = len_min
#                     break
#             self.len_min_max_value = max(self.file_dict_len_min.values())
#         pass

#     def init_check_len_max(self):
#         self.len_max_min_value = 0
#         self.file_dict_len_max = {str(x): self.len_max_min_value for x in range(self.num_files_to_store)}
#         print(self.file_dict_len_max)

#     def check_len_max(self, new_file_name, len_max):
#         # Store 10 longest files; lengths and names
#         # Replace the shortest with the current, if current is longer
#         if len_max > self.len_max_min_value:
#             for k in self.file_dict_len_max:
#                 if self.file_dict_len_max[k] == self.len_max_min_value:
#                     del self.file_dict_len_max[k]
#                     self.file_dict_len_max[new_file_name] = len_max
#                     break
#             self.len_max_min_value = min(self.file_dict_len_max.values())
#         pass

#     def init_check_len_diff(self):
#         self.len_diff_min_value = 0
#         self.file_dict_len_diff = {str(x): self.len_diff_min_value for x in range(self.num_files_to_store)}
#         print(self.file_dict_len_diff)

#     def check_len_diff(self, new_file_name, len_diff):
#         # Store 10 most different files; lengths and names
#         # Replace the smallest with the current, if current is larger
#         if len_diff > self.len_diff_min_value:
#             for k in self.file_dict_len_diff:
#                 if self.file_dict_len_diff[k] == self.len_diff_min_value:
#                     del self.file_dict_len_diff[k]
#                     self.file_dict_len_diff[new_file_name] = len_diff
#                     break
#             self.len_diff_min_value = min(self.file_dict_len_diff.values())
#         pass

#     def init_store_len_bin(self):
#         self.file_dict_len_bin = {}

#     def store_len_bin(self, len_min):
#         x = int(len_min / 100)
#         try:
#             self.file_dict_len_bin[x] += 1
#         except:
#             self.file_dict_len_bin[x] = 1
#         pass


#     def print_results(self):
#         print('Min:')
#         self.file_dict_len_min = sorted(self.file_dict_len_min.items(), key=lambda kv: kv[1])
#         print(self.file_dict_len_min)
#         print('Max:')
#         self.file_dict_len_max = sorted(self.file_dict_len_max.items(), key=lambda kv: kv[1])
#         print(self.file_dict_len_max)
#         print('Diff:')
#         self.file_dict_len_diff = sorted(self.file_dict_len_diff.items(), key=lambda kv: kv[1])
#         print(self.file_dict_len_diff)
#         print('Bin:')
#         self.file_dict_len_bin = sorted(self.file_dict_len_bin.items(), key=lambda kv: kv[0])
#         print(self.file_dict_len_bin)

# def file_length_stats_test(cfg):
#     '''
#     1. before silence reduction
#         1.1 10 shorted file lengths, and file names
#         1.2 all file lengths, grouped into bins of 100 frames
#         1.3 10 largest file length differences, and file names

#     2. after silence reduction
#     '''

#     # before silence reduction
#     file_id_list = read_file_list(cfg.file_id_list_file)

#     file_length_stats_class = Build_file_length_stats_class(cfg)

#     for file_id in file_id_list:
#         if file_length_stats_class.check_file_used(file_id):
#             file_length_dict = {}
#             for nn_feat in ['lab', 'cmp', 'wav']:
#                 dir_name = cfg.nn_feat_resil_dirs[nn_feat]
#                 feat_dim = cfg.nn_feature_dims[nn_feat]

#                 full_file_name = os.path.join(dir_name, file_id+'.'+nn_feat)
#                 features, frame_number = io_fun.load_binary_file_frame(full_file_name, feat_dim)
#                 file_length_dict[nn_feat] = frame_number

#             file_length_stats_class.load_new_file(file_id, file_length_dict)

#     file_length_stats_class.print_results()




class Build_Classification_Wav_Test(Build_DV_Y_Testing_Base):
    '''
    For waveform-based models only
    Compare the average loss in voiced or unvoiced region
    Return a dict: key is train/valid/test, value is a list for plotting
        loss_mean_dict[utter_tvt_name] = loss_mean_list
        in loss_mean_list: loss_mean vs voicing (increasing)
    '''
    def __init__(self, cfg, dv_y_cfg):
        super().__init__(cfg, dv_y_cfg)
        self.loss_dict = {}

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



    def test(self):
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


    def test_old_version(cfg, dv_y_cfg):
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



class Build_Sinenet_Weight_Test(Build_DV_Y_Testing_Base):
    '''
    For waveform-based models only
    Compare the average loss in voiced or unvoiced region
    Return a dict: key is train/valid/test, value is a list for plotting
        loss_mean_dict[utter_tvt_name] = loss_mean_list
        in loss_mean_list: loss_mean vs voicing (increasing)
    '''
    def __init__(self, cfg, dv_y_cfg):
        super().__init__(cfg, dv_y_cfg)      

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



        
