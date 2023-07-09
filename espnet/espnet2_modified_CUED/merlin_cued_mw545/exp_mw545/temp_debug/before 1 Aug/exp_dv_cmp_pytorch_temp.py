# Some modules no longer useful, but may re-visit later
import numpy

def lambda_0_test_dv_y_wav_model(cfg, dv_y_cfg):

    logger = make_logger("dv_y_config")
    dv_y_cfg = copy.deepcopy(dv_y_cfg)
    log_class_attri(dv_y_cfg, logger, except_list=dv_y_cfg.log_except_list)

    logger = make_logger("lamda_0_dvy")
    logger.info('Creating data lists')
    speaker_id_list = dv_y_cfg.speaker_id_list_dict['train'] # For DV training and evaluation, use train speakers only
    speaker_loader  = list_random_loader(speaker_id_list)
    file_id_list    = read_file_list(cfg.file_id_list_file)
    file_list_dict  = make_dv_file_list(file_id_list, speaker_id_list, dv_y_cfg.data_split_file_number) # In the form of: file_list[(speaker_id, 'train')]
    make_feed_dict_method_train = dv_y_cfg.make_feed_dict_method_train

    dv_y_model = torch_initialisation(dv_y_cfg)
    dv_y_model.load_nn_model(dv_y_cfg.nnets_file_name)
    dv_y_model.eval()
    dv_y_model.detect_nan_model_parameters(logger)

    logger.info('Printing bias of expansion layer')
    b = dv_y_model.nn_model.expansion_layer.bias
    print(b)

    num_batch = dv_y_cfg.epoch_num_batch['valid']
    # Collect x that produce 0 lambda
    x_list = []
    speaker_counter = {}
    for batch_idx in range(num_batch):
        batch_idx += 1
        logger.info('start generating Batch '+str(batch_idx))
        # Draw random speakers
        batch_speaker_list = speaker_loader.draw_n_samples(dv_y_cfg.batch_num_spk)
        # Make feed_dict for training
        feed_dict, batch_size = make_feed_dict_method_train(dv_y_cfg, file_list_dict, cfg.nn_feat_scratch_dirs, batch_speaker_list,  utter_tvt='test')
        with dv_y_model.no_grad():
            lambda_SBD = dv_y_model.gen_lambda_SBD_value(feed_dict=feed_dict)

        S,B,D = lambda_SBD.shape

        for s in range(S):
            for b in range(B):
                lambda_D = lambda_SBD[s,b]
                n = numpy.count_nonzero(lambda_D)
                if n == 0:
                    x = feed_dict['x'][s,b]
                    x_list.append(x)
                    speaker_id = batch_speaker_list[s]
                    try: speaker_counter[speaker_id] += 1
                    except: speaker_counter[speaker_id] = 1

    logger.info('Number of windows give 0 lambda are %i out of %i ' % (len(x_list), batch_size*num_batch))
    print(speaker_counter)

    # Plot these waveforms
    num_to_print = 5
    if len(x_list) > num_to_print:
        logger.info('PLot waveforms that give 0 lambda')
        fig, ax_list = plt.subplots(num_to_print)
        fig.suptitle('%i waveforms that give 0 lambda' % (num_to_print))
        for i in range(num_to_print):
            x = x_list[i]
            # Plot x, waveform
            ax_list[i].plot(x)
        fig_name = '/home/dawna/tts/mw545/Export_Temp' + "/wav_0_lambda.png"
        logger.info('Saving Waveform to %s' % fig_name)
        fig.savefig(fig_name)

    # Feed in waveforms that produce 0 lambda
    feed_dict_0 = feed_dict
    i = 0
    assert len(x_list) > (S*B)
    for s in range(S):
        for b in range(B):
            feed_dict_0['x'][s,b,:] = x_list[i]
            i += 1

    with dv_y_model.no_grad():
        h_list = dv_y_model.gen_all_h_values(feed_dict=feed_dict_0)

    # Insert x in h_list for plotting as well
    h_list.insert(0, feed_dict['x'])
    h_list_file_name = os.path.join(dv_y_cfg.exp_dir, "h_0_list.dat")
    pickle.dump(h_list, open(h_list_file_name, "wb" ))
    return h_list

def generate_all_h_dv_y_model(cfg, dv_y_cfg):

    # Generate h of all layers
    # File names: see dv_y_cfg.change_to_gen_h_mode

    logger = make_logger("dv_y_config")
    dv_y_cfg = copy.deepcopy(dv_y_cfg)
    dv_y_cfg.change_to_gen_h_mode()
    log_class_attri(dv_y_cfg, logger, except_list=dv_y_cfg.log_except_list)

    logger = make_logger("gen_h_dvy")
    make_feed_dict_method_train = dv_y_cfg.make_feed_dict_method_train
    
    batch_speaker_list = dv_y_cfg.batch_speaker_list
    file_list_dict = {}
    for speaker_name in batch_speaker_list:
        file_list_dict[(speaker_name, 'eval')] = ['%s_%s' % (speaker_name, dv_y_cfg.utter_name)]

    dv_y_model = torch_initialisation(dv_y_cfg)
    dv_y_model.load_nn_model(dv_y_cfg.nnets_file_name)
    dv_y_model.eval()

    # Make feed_dict for training
    feed_dict, batch_size = make_feed_dict_method_train(dv_y_cfg, file_list_dict, cfg.nn_feat_scratch_dirs, batch_speaker_list, all_utt_start_frame_index=0, utter_tvt='eval')
    with dv_y_model.no_grad():
        h_list = dv_y_model.gen_all_h_values(feed_dict=feed_dict)

    # Insert x in h_list for plotting as well
    h_list.insert(0, feed_dict['x'])
    h_list_file_name = dv_y_cfg.h_list_file_name
    pickle.dump(h_list, open(h_list_file_name, "wb" ))
    for h in h_list:
        print(h.shape)
    return h_list

def plot_all_h_dv_y_model(cfg, dv_y_cfg):
    logger = make_logger("dv_y_config")
    dv_y_cfg = copy.deepcopy(dv_y_cfg)
    dv_y_cfg.change_to_gen_h_mode()
    log_class_attri(dv_y_cfg, logger, except_list=dv_y_cfg.log_except_list)

    logger = make_logger("plot_h_dvy")
    h_list_file_name = dv_y_cfg.h_list_file_name
    try:
        h_list = pickle.load(open(h_list_file_name, "rb" ))
        logger.info('Loaded %s' % h_list_file_name)
    except:
        h_list = generate_all_h_dv_y_model(cfg, dv_y_cfg)

    S = dv_y_cfg.batch_num_spk
    B = dv_y_cfg.spk_num_seq

    for s in range(S):
        for b in range(B):
            fig, ax_list = plt.subplots(len(h_list))
            for i,h in enumerate(h_list):
                # logger.info('Layer %i ' % (i))
                # Print first row
                if len(h.shape) > 3:
                    for h_i in h:
                        ax_list[i].plot(h_i[s,b])
                else:
                    ax_list[i].plot(h[s,b])

            b_str = '0'*(3-len(str(b)))+str(b)
            fig_name = '/home/dawna/tts/mw545/Export_Temp/PNG_out/' + "h_spk_%i_seq_%s.png" % (s,b_str)
            logger.info('Saving h to %s' % fig_name)
            fig.savefig(fig_name)
            plt.close(fig)

def eval_logit_dv_y_model(cfg, dv_y_cfg):
    logger = make_logger("dv_y_config")
    dv_y_cfg = copy.deepcopy(dv_y_cfg)
    log_class_attri(dv_y_cfg, logger, except_list=dv_y_cfg.log_except_list)

    logger = make_logger("eval_logit")
    logger.info('Creating data lists')
    speaker_id_list = dv_y_cfg.speaker_id_list_dict['train'] # For DV training and evaluation, use train speakers only

    make_feed_dict_method_train = dv_y_cfg.make_feed_dict_method_train

    dv_y_cfg.batch_num_spk = 4
    batch_speaker_list = ['p15', 'p28', 'p122', 'p68'] # Males 2, Females 2
    file_list_dict = {}
    for speaker_name in batch_speaker_list:
        file_list_dict[(speaker_name, 'eval')] = ['%s_003' % speaker_name]

    dv_y_model = torch_initialisation(dv_y_cfg)
    dv_y_model.load_nn_model(dv_y_cfg.nnets_file_name)
    dv_y_model.eval()

    fig = plt.figure(figsize=(200,100))
    num_spk = dv_y_cfg.batch_num_spk
    num_win = 5
    # fig.set_size_inches(185, 105)
    fig, ax_list = plt.subplots(num_spk, num_win)
    fig.suptitle('%i speakers, %i windows' % (num_spk, num_win))
    # Make feed_dict for training
    feed_dict, batch_size = make_feed_dict_method_train(dv_y_cfg, file_list_dict, cfg.nn_feat_scratch_dirs, batch_speaker_list, all_utt_start_frame_index=4000, utter_tvt='eval')
    with dv_y_model.no_grad():
        logit_SBD = dv_y_model.gen_logit_SBD_value(feed_dict=feed_dict)
    for i in range(num_spk):
        for j in range(num_win):
            logit_D = logit_SBD[i,j]
            ax_list[i,j].plot(logit_D)

    fig_name = '/home/dawna/tts/mw545/Export_Temp' + "/gen_logit.png"
    logger.info('Saving logits to %s' % fig_name)
    fig.savefig(fig_name)


    # SinenetV3 specific
    plot_f0_tau = False
    plot_h      = False
    for nn_layer_config in dv_y_cfg.nn_layer_config_list:
        if nn_layer_config['type'] == 'SinenetV3':
            plot_f0_tau = True
            plot_h = True
            break
    if plot_f0_tau:
    # if False:
        with dv_y_model.no_grad():
            nlf, tau, tau_list = dv_y_model.gen_nlf_tau_values(feed_dict=feed_dict)
        # Plot f0
        num_spk = dv_y_cfg.batch_num_spk
        num_win = 5
        fig, ax = plt.subplots()
        fig.suptitle('F0, %i speakers, %i windows' % (num_spk, num_win))
        ax.plot(numpy.squeeze(nlf).T)
        fig_name = '/home/dawna/tts/mw545/Export_Temp' + "/nlf.png"
        logger.info('Saving NLF to %s' % fig_name)
        fig.savefig(fig_name)
        # Plot tau
        num_spk = dv_y_cfg.batch_num_spk
        num_win = 5
        fig, ax = plt.subplots()
        fig.suptitle('Tau, %i speakers, %i windows' % (num_spk, num_win))
        ax.plot(numpy.squeeze(tau).T)
        fig_name = '/home/dawna/tts/mw545/Export_Temp' + "/tau.png"
        logger.info('Saving Tau to %s' % fig_name)
        fig.savefig(fig_name)
        # Plot tau trajectories
        tau_SBT = numpy.stack(tau_list, axis=-1)
        num_spk = dv_y_cfg.batch_num_spk
        num_win = 5
        fig, ax_list = plt.subplots(num_spk, num_win)
        fig.suptitle('Tau trajectory, %i speakers, %i windows' % (num_spk, num_win))
        for i in range(num_spk):
            for j in range(num_win):
                tau_T = tau_SBT[i,j]
                ax_list[i,j].plot(numpy.squeeze(tau_T))
        fig_name = '/home/dawna/tts/mw545/Export_Temp' + "/tau_list.png"
        logger.info('Saving Tau trajectory to %s' % fig_name)
        fig.savefig(fig_name)

    if plot_h:
        with dv_y_model.no_grad():
            h = dv_y_model.gen_sinenet_h_value(feed_dict=feed_dict)
        # Plot different speaker
        fig, ax_list = plt.subplots(num_spk)
        fig.suptitle('h, %i speakers' % (num_spk))
        for i in range(num_spk):
            h_BD = h[i]
            h_D  = h_BD[0]
            ax_list[i].plot(h_D)
        fig_name = '/home/dawna/tts/mw545/Export_Temp' + "/h_speaker.png"
        logger.info('Saving h_speaker to %s' % fig_name)
        fig.savefig(fig_name)
        # Plot different window
        fig, ax_list = plt.subplots(num_win)
        fig.suptitle('h, %i windows' % (num_win))
        h_BD = h[0]
        for i in range(num_win):
            h_D  = h_BD[i]
            ax_list[i].plot(h_D)
        fig_name = '/home/dawna/tts/mw545/Export_Temp' + "/h_window.png"
        logger.info('Saving h_window to %s' % fig_name)
        fig.savefig(fig_name)

def relu_0_stats(cfg, dv_y_cfg):

    # Generate a lot of all_h
    # For each layer, each dimension, compute:
    # zero/non-zero ratio

    # Use all train speakers, train utterances first
    logger = make_logger("dv_y_config")
    dv_y_cfg = copy.deepcopy(dv_y_cfg)
    log_class_attri(dv_y_cfg, logger, except_list=dv_y_cfg.log_except_list)

    logger = make_logger("relu_0_stats")
    logger.info('Creating data lists')

    num_batch  = dv_y_cfg.epoch_num_batch['valid']
    if dv_y_cfg.train_by_window:
        batch_size = dv_y_cfg.batch_num_spk * dv_y_cfg.spk_num_seq
    else:
        batch_size = dv_y_cfg.batch_num_spk

    all_h_list_file_name = os.path.join(dv_y_cfg.exp_dir, 'all_h_list.dat')
    try:
        all_h_list = pickle.load(open(all_h_list_file_name, "rb" ))
        logger.info('Loaded %s' % all_h_list_file_name)
    except:
        speaker_id_list = dv_y_cfg.speaker_id_list_dict['train']
        speaker_loader  = list_random_loader(speaker_id_list)
        file_id_list    = read_file_list(cfg.file_id_list_file)
        file_list_dict  = make_dv_file_list(file_id_list, speaker_id_list, dv_y_cfg.data_split_file_number) # In the form of: file_list[(speaker_id, 'train')]
        make_feed_dict_method_train = dv_y_cfg.make_feed_dict_method_train

        dv_y_model = torch_initialisation(dv_y_cfg)
        dv_y_model.load_nn_model(dv_y_cfg.nnets_file_name)
        dv_y_model.eval()

        all_h_list = []
        for batch_idx in range(num_batch):
            batch_idx += 1
            if batch_idx % 10 == 0:
                logger.info('start generating Batch '+str(batch_idx))
            # Draw random speakers
            batch_speaker_list = speaker_loader.draw_n_samples(dv_y_cfg.batch_num_spk)
            # Make feed_dict for training
            feed_dict, batch_size = make_feed_dict_method_train(dv_y_cfg, file_list_dict, cfg.nn_feat_scratch_dirs, batch_speaker_list,  utter_tvt='train')
            with dv_y_model.no_grad():
                h_list = dv_y_model.gen_all_h_values(feed_dict=feed_dict)
            all_h_list.append(h_list)
        logger.info('Saving all_h_list to %s' % all_h_list_file_name)
        pickle.dump(all_h_list, open(all_h_list_file_name, "wb" ))

    # Create holders for stats
    h_list = all_h_list[0]
    h_stats = {}
    for k in ['non_zero_count']:
        h_stats[k] = []
        for h in h_list:
            h_stats[k].append(numpy.zeros(h.shape[-1]))

    for h_list in all_h_list:
        for i,h in enumerate(h_list):
            l = len(h.shape)
            # Detect non-zero values, change to 1
            h_temp = (h != 0).astype(int)
            # Sum over all dimensions except the last one
            for j in range(l-1):
                h_temp = numpy.sum(h_temp, axis=0)
            h_stats['non_zero_count'][i] += h_temp

    h_stats['non_zero_count'] = [h / (num_batch * batch_size) for h in h_stats['non_zero_count']]
    logger.info('Printing non-zero ratios')
    for h in h_stats['non_zero_count']:
        print(h)

def plot_sinenet(cfg, dv_y_cfg):
    logger = make_logger("dv_y_config")
    dv_y_cfg = copy.deepcopy(dv_y_cfg)
    dv_y_cfg.batch_num_spk = 1
    dv_y_cfg.spk_num_seq   = 20 # Use this for different frequency
    dv_y_cfg.seq_num_win   = 40 # Use this for different pitch locations
    f_0     = 120.
    f_inc   = 10.
    tau_0   = 0.
    tau_inc = 10./16000
    log_class_attri(dv_y_cfg, logger, except_list=dv_y_cfg.log_except_list)

    dv_y_model = torch_initialisation(dv_y_cfg)
    dv_y_model.load_nn_model(dv_y_cfg.nnets_file_name)
    dv_y_model.eval()

    f   = numpy.zeros((dv_y_cfg.batch_num_spk, dv_y_cfg.spk_num_seq, dv_y_cfg.seq_num_win))
    for i in range(dv_y_cfg.spk_num_seq):
        f[0,i] = f_0 + i * f_inc
    nlf = numpy.log(f)
    tau = numpy.zeros((dv_y_cfg.batch_num_spk, dv_y_cfg.spk_num_seq, dv_y_cfg.seq_num_win))
    for i in range(dv_y_cfg.seq_num_win):
        tau[0,i] = tau_0 + i * tau_inc

    feed_dict = {'nlf': nlf, 'tau': tau}
    W = dv_y_model.gen_w_mul_w_sin_cos(feed_dict)

    S = dv_y_cfg.batch_num_spk
    B = dv_y_cfg.spk_num_seq
    M = dv_y_cfg.seq_num_win
    D = 80
    
    D_size = 5
    D_tot  = int(D/D_size)

    for d_1 in range(D_tot):
        fig, ax_list = plt.subplots(D_size)        

        for d_2 in range(D_size):
            d = d_1 * D_size + d_2
            ax_list[d_2].plot(W[0,0,0,d]) 

        fig_name = '/home/dawna/tts/mw545/Export_Temp/PNG_out/' + "sinenet_filter_%i.png" % d_1
        logger.info('Saving h to %s' % fig_name)
        fig.savefig(fig_name)
        plt.close(fig)



def compute_Euclidean_distance(lambda_1, lambda_2):
    d = 0.
    S = lambda_1.shape[0]
    B = lambda_1.shape[1]
    D = lambda_1.shape[2]
    for i in range(S):
        for j in range(B):
            d_ij = numpy.linalg.norm(lambda_1[i,j]-lambda_2[i,j])
            d += d_ij
    d = d / (S*B)
    return d

def compute_Euclidean_distance(lambda_1, lambda_0, norm_by_0=False):
    d = 0.
    S = lambda_1.shape[0]
    B = lambda_1.shape[1]
    D = lambda_1.shape[2]
    for i in range(S):
        for j in range(B):
            if norm_by_0:
                d_ij = numpy.linalg.norm((lambda_1[i,j]-lambda_0[i,j]) / lambda_0[i,j])
            else:
                d_ij = numpy.linalg.norm(lambda_1[i,j]-lambda_0[i,j])
            d += d_ij
    d = d / (S*B)
    return d

    # dim_list = [5,5,10]
    # lambda_1 = numpy.random.rand(*dim_list)
    # lambda_0 = numpy.random.rand(*dim_list)

    # dist_1 = compute_Euclidean_distance(lambda_1, lambda_0, norm_by_0=True)
    # dist_2 = 0.
    # for i in range(dim_list[0]):
    #     for j in range(dim_list[1]):
    #         dist_ij = 0.
    #         for k in range(dim_list[2]):
    #             dist_ij += numpy.square((lambda_1[i,j,k] - lambda_0[i,j,k]) / lambda_0[i,j,k])
    #         dist_2 += numpy.sqrt(dist_ij)


    # dist_2 = dist_2 / (dim_list[0] * dim_list[1])
    # print(dist_1 - dist_2)

def test_sinenet(cfg, dv_y_cfg):
    # 1. Check nlf and f0 conversion; fabricate data or real data
    from exp_mw545.exp_dv_wav_sinenet_v3 import load_cmp_file
    file_id = 'p15_003'
    if True:
        cmp_file_name = os.path.join(cfg.nn_feat_scratch_dirs['cmp'], file_id+'.cmp')
        lf0_index     = cfg.acoustic_start_index['lf0']
        cmp_dim       = cfg.nn_feature_dims['cmp']
        lf0_scra_data = load_cmp_file(cmp_file_name, cmp_dim=cmp_dim, feat_dim_index=lf0_index)
        print(lf0_scra_data[:20])
    if False:
        # Checked; scratch and resil_norm are the same
        cmp_file_name = os.path.join(cfg.nn_feat_resil_norm_dirs['cmp'], file_id+'.cmp')
        lf0_index     = cfg.acoustic_start_index['lf0']
        cmp_dim       = cfg.nn_feature_dims['cmp']
        lf0_norm_data = load_cmp_file(cmp_file_name, cmp_dim=cmp_dim, feat_dim_index=lf0_index)
        print(lf0_norm_data[:20])
    if False:
        # Checked; denorm function is correct
        # Also, exp restores sensible f
        cmp_file_name = os.path.join(cfg.nn_feat_resil_dirs['cmp'], file_id+'.cmp')
        lf0_index     = cfg.acoustic_start_index['lf0']
        cmp_dim       = cfg.nn_feature_dims['cmp']
        lf0_data      = load_cmp_file(cmp_file_name, cmp_dim=cmp_dim, feat_dim_index=lf0_index)
        print(lf0_data[:20])
        '''
        self.log_f_mean = 5.04418
        self.log_f_std  = 0.358402
        lf = torch.add(torch.mul(nlf, self.log_f_std), self.log_f_mean) # S*B*M
        f  = torch.exp(lf)                                              # S*B*M
        '''
        log_f_mean = 5.04418
        log_f_std  = 0.358402
        lf = lf0_norm_data * log_f_std + log_f_mean
        print(lf[:20])
        f = numpy.exp(lf)
        print(f)
    if False:
        # Checked; pytorch denorm function is same
        import torch
        nlf = torch.tensor(lf0_scra_data, dtype=torch.float)
        log_f_mean = 5.04418
        log_f_std  = 0.358402
        lf = torch.add(torch.mul(nlf, log_f_std), log_f_mean)
        f  = torch.exp(lf)
        print(f)
    if True:
        # Need to check construct_w_sin_cos_matrix
        # Test 1: f=25Hz, sine and cosine should show 1 period in 640 samples
        log_f_mean = 5.04418
        log_f_std  = 0.358402
        f = numpy.ones((1,1,1)) * 25.
        lf = numpy.log(f)
        nlf = (lf - log_f_mean) / log_f_std
        tau = numpy.zeros((1,1,1))
        import torch
        nlf = torch.tensor(nlf, dtype=torch.float)
        tau = torch.tensor(tau, dtype=torch.float)
        from modules_torch import SinenetLayerV3
        sinenet = SinenetLayerV3(output_dim=1, num_freq=1, win_len=640, num_win=1)
        sin_cos_matrix = sinenet.construct_w_sin_cos_matrix(nlf, tau) # S*B*M*2K*T

        pass


##########################
#  exp_dv_wav_subwin.py  #
##########################

def make_feed_dict_y_wav_subwin_train_single_thread(pool_input):
    # Unzip input
    i, dv_y_cfg, file_list_dict, file_dir_dict, batch_speaker_list, utter_tvt, all_utt_start_frame_index = pool_input

    '''
    Draw Utterances; Load Data
    Draw starting frame; Slice; Fit into numpy holders
    '''
    feat_name_list = ['wav'] # Load wav
    feat_dim_list  = [1]

    # Make i/o shape arrays
    wav = numpy.zeros((dv_y_cfg.spk_num_seq, dv_y_cfg.seq_num_win, dv_y_cfg.seq_win_len))

    wav_sr  = dv_y_cfg.cfg.wav_sr
    cmp_sr  = dv_y_cfg.cfg.frame_sr
    wav_cmp_ratio = int(wav_sr / cmp_sr)
    # Do not use silence frames at the beginning or the end
    total_sil_one_side_cmp = dv_y_cfg.frames_silence_to_keep + dv_y_cfg.sil_pad  # This is at 200Hz
    total_sil_one_side_wav = total_sil_one_side_cmp * wav_cmp_ratio              # This is at 16kHz
    min_file_len = dv_y_cfg.batch_seq_total_len + 2 * total_sil_one_side_wav # This is at 16kHz

    start_frame_index_list = []
    
    speaker_id = batch_speaker_list[i]
    # Draw multiple utterances per speaker: dv_y_cfg.spk_num_utter
    # Draw multiple windows per utterance:  dv_y_cfg.utter_num_seq
    # Stack them along B
    speaker_file_name_list, speaker_utter_len_list, speaker_utter_list = get_utters_from_binary_dict(dv_y_cfg.spk_num_utter, file_list_dict[(speaker_id, utter_tvt)], file_dir_dict, feat_name_list=feat_name_list, feat_dim_list=feat_dim_list, min_file_len=min_file_len, random_seed=None)

    for utter_idx in range(dv_y_cfg.spk_num_utter):
        file_name = speaker_file_name_list[utter_idx]
        wav_file  = speaker_utter_list['wav'][utter_idx] # T * 1; 16kHz
        wav_file  = numpy.squeeze(wav_file, axis=1)      # T*1 -> T
        wav_file_len = speaker_utter_len_list[utter_idx]

        # Find start frame index, random if None
        if all_utt_start_frame_index is None:
            extra_file_len = wav_file_len - min_file_len
            utter_start_frame_index = numpy.random.randint(low=total_sil_one_side_wav, high=total_sil_one_side_wav+extra_file_len+1)
        else:
            utter_start_frame_index = total_sil_one_side_wav + all_utt_start_frame_index
        start_frame_index_list.append(utter_start_frame_index)

        # Slice data into seq and win
        for seq_idx in range(dv_y_cfg.utter_num_seq):
            spk_seq_index = utter_idx * dv_y_cfg.utter_num_seq + seq_idx
            seq_start = utter_start_frame_index + seq_idx * dv_y_cfg.batch_seq_shift
            for win_idx in range(dv_y_cfg.seq_num_win):
                win_start = seq_start + win_idx * dv_y_cfg.seq_win_shift
                win_end   = win_start + dv_y_cfg.seq_win_len - 1 # Inclusive index
                wav[spk_seq_index, win_idx, :] = wav_file[win_start:win_end+1]

    return(wav, speaker_file_name_list, start_frame_index_list)

def make_feed_dict_y_wav_subwin_train_parallel(dv_y_cfg, file_list_dict, file_dir_dict, batch_speaker_list, utter_tvt, all_utt_start_frame_index=None, return_dv=False, return_y=False, return_frame_index=False, return_file_name=False):
    logger = make_logger("make_dict")

    
    # Make i/o shape arrays
    # This is numpy shape, not Tensor shape!
    wav = numpy.zeros((dv_y_cfg.batch_num_spk, dv_y_cfg.spk_num_seq, dv_y_cfg.seq_num_win, dv_y_cfg.seq_win_len))
    dv  = numpy.zeros((dv_y_cfg.batch_num_spk))

    for speaker_idx in range(dv_y_cfg.batch_num_spk):
        speaker_id = batch_speaker_list[speaker_idx]
        # Make classification targets, index sequence
        true_speaker_index = dv_y_cfg.speaker_id_list_dict['train'].index(speaker_id)
        dv[speaker_idx] = true_speaker_index

    pool_input_list = [[i, dv_y_cfg, file_list_dict, file_dir_dict, batch_speaker_list, utter_tvt, all_utt_start_frame_index] for i in range(dv_y_cfg.batch_num_spk)]

    with Pool(dv_y_cfg.batch_num_spk) as p:
        pool_return_list = p.map(make_feed_dict_y_wav_subwin_train_single_thread, pool_input_list)

    file_name_list = []
    start_frame_index_list = []
    for speaker_idx in range(dv_y_cfg.batch_num_spk):
        wav_speaker, speaker_file_name_list, speaker_start_frame_index_list = pool_return_list[speaker_idx]
        wav[speaker_idx] = wav_speaker
        file_name_list.append(speaker_file_name_list)
        start_frame_index_list.append(speaker_start_frame_index_list)

    # S,B,M,D
    x_val = wav
    if dv_y_cfg.train_by_window:
        # S --> S*B
        y_val = numpy.repeat(dv, dv_y_cfg.spk_num_seq)
        batch_size = dv_y_cfg.batch_num_spk * dv_y_cfg.spk_num_seq
    else:
        y_val = one_hot
        batch_size = dv_y_cfg.batch_num_spk

    feed_dict = {'x':x_val, 'y':y_val}
    return_list = [feed_dict, batch_size]
    
    if return_dv:
        return_list.append(dv)
    if return_y:
        return_list.append(wav)
    if return_frame_index:
        return_list.append(start_frame_index_list)
    if return_file_name:
        return_list.append(file_name_list)
    return return_list

def make_feed_dict_y_wav_subwin_train_stacked(dv_y_cfg, file_list_dict, file_dir_dict, batch_speaker_list, utter_tvt, all_utt_start_frame_index=None, return_dv=False, return_y=False, return_frame_index=False, return_file_name=False):
    logger = make_logger("make_dict")

    '''
    Draw Utterances; Load Data
    Draw starting frame; Slice; Fit into numpy holders
    '''
    load_time  = 0.
    slice_time = 0.
    feat_name_list = ['wav'] # Load wav
    feat_dim_list  = [1]
    # Make i/o shape arrays
    # This is numpy shape, not Tensor shape!
    wav = numpy.zeros((dv_y_cfg.batch_num_spk, dv_y_cfg.spk_num_seq, dv_y_cfg.seq_num_win, dv_y_cfg.seq_win_len))
    dv  = numpy.zeros((dv_y_cfg.batch_num_spk))
    # Stack first, then slice in batches
    wav_SUT = numpy.zeros((dv_y_cfg.batch_num_spk, dv_y_cfg.spk_num_utter, dv_y_cfg.batch_seq_total_len))

    wav_sr  = dv_y_cfg.cfg.wav_sr
    cmp_sr  = dv_y_cfg.cfg.frame_sr
    wav_cmp_ratio = int(wav_sr / cmp_sr)
    # Do not use silence frames at the beginning or the end
    total_sil_one_side_cmp = dv_y_cfg.frames_silence_to_keep + dv_y_cfg.sil_pad  # This is at 200Hz
    total_sil_one_side_wav = total_sil_one_side_cmp * wav_cmp_ratio              # This is at 16kHz
    min_file_len = dv_y_cfg.batch_seq_total_len + 2 * total_sil_one_side_wav # This is at 16kHz

    file_name_list = []
    start_frame_index_list = [[] for i in range(dv_y_cfg.batch_num_spk)]
    
    speaker_start_time = time.time()
    for speaker_idx in range(dv_y_cfg.batch_num_spk):

        speaker_id = batch_speaker_list[speaker_idx]
        # Make classification targets, index sequence
        true_speaker_index = dv_y_cfg.speaker_id_list_dict['train'].index(speaker_id)
        dv[speaker_idx] = true_speaker_index

        # Draw multiple utterances per speaker: dv_y_cfg.spk_num_utter
        # Draw multiple windows per utterance:  dv_y_cfg.utter_num_seq
        # Stack them along B
        
        speaker_file_name_list, speaker_utter_len_list, speaker_utter_list = get_utters_from_binary_dict(dv_y_cfg.spk_num_utter, file_list_dict[(speaker_id, utter_tvt)], file_dir_dict, feat_name_list=feat_name_list, feat_dim_list=feat_dim_list, min_file_len=min_file_len, random_seed=None)
        file_name_list.append(speaker_file_name_list)

        for utter_idx in range(dv_y_cfg.spk_num_utter):
            file_name = speaker_file_name_list[utter_idx]
            wav_file  = speaker_utter_list['wav'][utter_idx] # T * 1; 16kHz
            wav_file  = numpy.squeeze(wav_file, axis=1)      # T*1 -> T
            wav_file_len = speaker_utter_len_list[utter_idx]

            # Find start frame index, random if None
            if all_utt_start_frame_index is None:
                extra_file_len = wav_file_len - min_file_len
                utter_start_frame_index = numpy.random.randint(low=total_sil_one_side_wav, high=total_sil_one_side_wav+extra_file_len+1)
            else:
                utter_start_frame_index = total_sil_one_side_wav + all_utt_start_frame_index
            start_frame_index_list[speaker_idx].append(utter_start_frame_index)
            wav_SUT[speaker_idx, utter_idx] = wav_file[utter_start_frame_index:utter_start_frame_index+dv_y_cfg.batch_seq_total_len]

    speaker_load_time = time.time()
    load_time = (speaker_load_time-speaker_start_time)

    # Slice data into seq and win
    for seq_idx in range(dv_y_cfg.utter_num_seq):
        spk_seq_index = utter_idx * dv_y_cfg.utter_num_seq + seq_idx
        seq_start = seq_idx * dv_y_cfg.batch_seq_shift
        for win_idx in range(dv_y_cfg.seq_num_win):
            win_start = seq_start + win_idx * dv_y_cfg.seq_win_shift
            win_end   = win_start + dv_y_cfg.seq_win_len - 1 # Inclusive index
            wav[:, spk_seq_index, win_idx, :] = wav_SUT[:,0, win_start:win_end+1]
    speaker_slice_time = time.time()
    slice_time = (speaker_slice_time-speaker_load_time)
    print('Load time is %s, Slice time is %s' %(str(load_time), str(slice_time)))

    # S,B,M,D
    x_val = wav
    if dv_y_cfg.train_by_window:
        # S --> S*B
        y_val = numpy.repeat(dv, dv_y_cfg.spk_num_seq)
        batch_size = dv_y_cfg.batch_num_spk * dv_y_cfg.spk_num_seq
    else:
        y_val = one_hot
        batch_size = dv_y_cfg.batch_num_spk

    feed_dict = {'x':x_val, 'y':y_val}
    return_list = [feed_dict, batch_size]
    
    if return_dv:
        return_list.append(dv)
    if return_y:
        return_list.append(y)
    if return_frame_index:
        return_list.append(start_frame_index_list)
    if return_file_name:
        return_list.append(file_name_list)
    return return_list


#####################################
#  exp_dv_wav_sinenet_v3_unfold.py  #
#####################################

def lf0_test(dv_y_cfg):
    # To confirm the new lf0 method is correct; delete later
    wav_sr  = dv_y_cfg.cfg.wav_sr
    cmp_sr  = dv_y_cfg.cfg.frame_sr
    wav_cmp_ratio = int(wav_sr / cmp_sr)

    lf0_norm_data = numpy.array(range(30))
    print(lf0_norm_data)
    utter_start_frame_index = 3300

    nlf = numpy.zeros((dv_y_cfg.spk_num_seq, dv_y_cfg.seq_num_win))
    # Slice data into seq and win
    for seq_idx in range(dv_y_cfg.spk_num_seq):
        spk_seq_index = seq_idx
        seq_start = utter_start_frame_index + seq_idx * dv_y_cfg.batch_seq_shift
        for win_idx in range(dv_y_cfg.seq_num_win):
            win_start = seq_start + win_idx * dv_y_cfg.seq_win_shift
            t_start = (win_start) / float(wav_sr)
            t_end   = (win_start+dv_y_cfg.seq_win_len) / float(wav_sr)

            t_mid = (t_start + t_end) / 2.
            n_mid = t_mid * float(cmp_sr)
            # e.g. 1.3 is between 0.5, 1.5; n_l=0, n_r=1
            n_l = int(n_mid-0.5)
            n_r = n_l + 1
            l = lf0_norm_data.shape[0]
            if n_r >= l:
                lf0_mid = lf0_norm_data[-1]
            else:
                lf0_l = lf0_norm_data[n_l]
                lf0_r = lf0_norm_data[n_r]
                r = n_mid - ( n_l + 0.5 )
                lf0_mid = (r * lf0_r) + ((1-r) * lf0_l)

            nlf[spk_seq_index, win_idx] = lf0_mid

    print(nlf[0])
    nlf = cal_seq_win_lf0_mid(lf0_norm_data, utter_start_frame_index, dv_y_cfg, wav_cmp_ratio)
    print(nlf[0])

def tau_test(dv_y_cfg):
    # To confirm the new tau method is correct; delete later
    wav_sr  = dv_y_cfg.cfg.wav_sr
    cmp_sr  = dv_y_cfg.cfg.frame_sr
    wav_cmp_ratio = int(wav_sr / cmp_sr)

    pitch_loc_data = numpy.array([0.1025625, 0.10775, 0.110625, 0.1218125, 0.1425625, 0.1635625, 0.197, 0.2194375, 0.274125, 0.320375, 0.3388125, 0.376125, 0.387, 0.5785, 0.60625, 0.63225, 0.6603125, 0.7064375, 0.7086875, 0.7242])
    # print(pitch_loc_data)
    for utter_start_frame_index in range(5000):
        tau, vuv = cal_seq_win_tau_vuv_old(pitch_loc_data, utter_start_frame_index, dv_y_cfg, wav_sr)
        tau_2, vuv_2 = cal_seq_win_tau_vuv(pitch_loc_data, utter_start_frame_index, dv_y_cfg, wav_sr)

        tau_diff = tau - tau_2
        if ((tau_diff!=0).any()):
            print('tau_diff')
            print(utter_start_frame_index)
            print(tau_diff[tau_diff!=0])
            print(numpy.argwhere(tau_diff!=0))

        vuv_diff = vuv - vuv_2
        if ((vuv_diff!=0).any()):
            print('vuv_diff')
            print(utter_start_frame_index)
            print(vuv_diff[vuv_diff!=0])
            print(numpy.argwhere(vuv_diff!=0))

    ''' Code below used to be in make_feed_dict_y_wav_sinenet_train '''

        tau_2, vuv_2 = cal_seq_win_tau_vuv(pitch_loc_data, utter_start_frame_index, dv_y_cfg, wav_sr)
        tau_diff = tau_spk - tau_2
        if ((tau_diff!=0).any()):

            print(pitch_file_name)

            wrong_index_list = numpy.argwhere(tau_diff!=0)

            l = pitch_loc_data.shape[0]
            tau = numpy.zeros((dv_y_cfg.spk_num_seq, dv_y_cfg.seq_num_win))
            vuv = numpy.zeros((dv_y_cfg.spk_num_seq, dv_y_cfg.seq_num_win))
            
            pitch_max = dv_y_cfg.seq_win_len / float(wav_sr)

            win_start_0 = dv_y_cfg.return_win_start_0_matrix()
            win_start = win_start_0 + utter_start_frame_index
            t_start = win_start / float(wav_sr)

            t_start = numpy.repeat(t_start, l, axis=2)

            pitch_start = pitch_loc_data - t_start
            pitch_start[pitch_start <= 0.] = pitch_max

            print(utter_start_frame_index)
            for wrong_index in wrong_index_list:
                print(wrong_index)
                print(pitch_start[wrong_index[0], wrong_index[1]])

            pitch_start_min = numpy.amin(pitch_start, axis=2)

            vuv[pitch_start_min < pitch_max] = 1
            pitch_start_min[pitch_start_min >= pitch_max] = 0.
            print(pitch_start_min[wrong_index[0], wrong_index[1]])

        tau_diff = tau_spk - tau_2
        if ((tau_diff!=0).any()):
            print(utter_start_frame_index)
            print('tau_diff')
            print(tau_spk[tau_diff!=0])
            print(tau_2[tau_diff!=0])
            print(numpy.argwhere(tau_diff!=0))
        vuv_diff = vuv_spk - vuv_2
        if ((vuv_diff!=0).any()):
            print(utter_start_frame_index)
            print('vuv_diff')
            print(utter_start_frame_index)
            print(vuv_spk[vuv_diff!=0])
            print(vuv_2[vuv_diff!=0])
            print(numpy.argwhere(vuv_diff!=0))

def make_feed_dict_y_wav_sinenet_train_voiced_only(dv_y_cfg, file_list_dict, file_dir_dict, batch_speaker_list, utter_tvt, all_utt_start_frame_index=None, return_one_hot=False, return_y=False, return_frame_index=False, return_file_name=False):
    ''' This make_dict method returns 200ms windows of voiced only '''
    logger = make_logger("make_dict"),

    '''
    Draw Utterances; Load Data
    Draw starting frame; Slice; Fit into numpy holders
    '''
    feat_name_list = ['wav'] # Load wav
    feat_dim_list  = [1]
    # Make i/o shape arrays
    # This is numpy shape, not Tensor shape!
    wav = numpy.zeros((dv_y_cfg.batch_num_spk, dv_y_cfg.spk_num_seq, dv_y_cfg.seq_num_win, dv_y_cfg.seq_win_len))
    nlf = numpy.zeros((dv_y_cfg.batch_num_spk, dv_y_cfg.spk_num_seq, dv_y_cfg.seq_num_win))
    tau = numpy.zeros((dv_y_cfg.batch_num_spk, dv_y_cfg.spk_num_seq, dv_y_cfg.seq_num_win))
    one_hot = numpy.zeros((dv_y_cfg.batch_num_spk))

    wav_sr  = dv_y_cfg.cfg.wav_sr
    cmp_sr  = dv_y_cfg.cfg.frame_sr
    wav_cmp_ratio = int(wav_sr / cmp_sr)
    # Do not use silence frames at the beginning or the end
    total_sil_one_side_cmp = dv_y_cfg.frames_silence_to_keep + dv_y_cfg.sil_pad  # This is at 200Hz
    total_sil_one_side_wav = total_sil_one_side_cmp * wav_cmp_ratio              # This is at 16kHz
    min_file_len = dv_y_cfg.batch_seq_total_len + 2 * total_sil_one_side_wav # This is at 16kHz
    voiced_seq_win_threshold = int(dv_y_cfg.use_voiced_threshold * dv_y_cfg.seq_num_win)

    file_name_list = [[] for i in range(dv_y_cfg.batch_num_spk)]
    start_frame_index_list = [[] for i in range(dv_y_cfg.batch_num_spk)]
    
    for speaker_idx in range(dv_y_cfg.batch_num_spk):

        speaker_id = batch_speaker_list[speaker_idx]
        # Make classification targets, index sequence
        true_speaker_index = dv_y_cfg.speaker_id_list_dict['train'].index(speaker_id)
        one_hot[speaker_idx] = true_speaker_index

        spk_num_seq_need = dv_y_cfg.spk_num_seq
        spk_seq_index = 0

        while spk_num_seq_need > 0:
            # Draw 1 utterance
            # Draw multiple windows per utterance: dv_y_cfg.spk_num_seq
            # Check vuv of all sub-windows, find "good" windows
            # Use all, or draw randomly if more than needed
            # Stach them along B

            speaker_file_name_list, speaker_utter_len_list, speaker_utter_list = get_utters_from_binary_dict(1, file_list_dict[(speaker_id, utter_tvt)], file_dir_dict, feat_name_list=feat_name_list, feat_dim_list=feat_dim_list, min_file_len=min_file_len, random_seed=None)
            file_name_list[speaker_idx].extend(speaker_file_name_list)

            file_name = speaker_file_name_list[0]
            wav_file  = speaker_utter_list['wav'][0] # T * 1; 16kHz
            wav_file  = numpy.squeeze(wav_file, axis=1)      # T*1 -> T
            wav_file_len = speaker_utter_len_list[0]

            # Find start frame index, random if None
            if all_utt_start_frame_index is None:
                extra_file_len = wav_file_len - min_file_len
                utter_start_frame_index = numpy.random.randint(low=total_sil_one_side_wav, high=total_sil_one_side_wav+extra_file_len+1)
            else:
                utter_start_frame_index = total_sil_one_side_wav + all_utt_start_frame_index
            start_frame_index_list[speaker_idx].append(utter_start_frame_index)

            # Load cmp and pitch data
            cmp_file_name = os.path.join(file_dir_dict['cmp'], file_name+'.cmp')
            lf0_index     = dv_y_cfg.cfg.acoustic_start_index['lf0']
            cmp_dim       = dv_y_cfg.cfg.nn_feature_dims['cmp']
            lf0_norm_data = load_cmp_file(cmp_file_name, cmp_dim=cmp_dim, feat_dim_index=lf0_index)
            pitch_file_name = os.path.join(file_dir_dict['pitch'], file_name+'.pm')
            pitch_loc_data = read_pitch_file(pitch_file_name)

            # Load pitch data and find good windows first
            voiced_win_idx_list = []
            for seq_idx in range(dv_y_cfg.spk_num_seq):
                seq_start = utter_start_frame_index + seq_idx * dv_y_cfg.batch_seq_shift
                voiced_count = 0
                for win_idx in range(dv_y_cfg.seq_num_win):
                    win_start = seq_start + win_idx * dv_y_cfg.seq_win_shift
                    win_end   = win_start + dv_y_cfg.seq_win_len - 1 # Inclusive index
                    t_start = (win_start) / wav_sr
                    t_end   = (win_end+1) / wav_sr

                    win_pitch_loc, vuv_temp = cal_win_pitch_loc(pitch_loc_data, t_start, t_end)
                    if vuv_temp:
                        voiced_count += 1
                if voiced_count >= voiced_seq_win_threshold:
                    voiced_win_idx_list.append(seq_idx)

            # Use all, or draw randomly if more than needed
            num_voiced_win = len(voiced_win_idx_list)
            if num_voiced_win < spk_num_seq_need:
                # Use all
                spk_num_seq_need -= num_voiced_win
            else:
                # Draw randomly
                voiced_win_idx_list = numpy.random.choice(voiced_win_idx_list, spk_num_seq_need, replace=False)
                spk_num_seq_need = 0

            # Slice data into seq and win
            for seq_idx in voiced_win_idx_list:
                seq_start = utter_start_frame_index + seq_idx * dv_y_cfg.batch_seq_shift
                for win_idx in range(dv_y_cfg.seq_num_win):
                    win_start = seq_start + win_idx * dv_y_cfg.seq_win_shift
                    win_end   = win_start + dv_y_cfg.seq_win_len - 1 # Inclusive index
                    t_start = (win_start) / wav_sr
                    t_end   = (win_end+1) / wav_sr

                    lf0_mid = cal_win_lf0_mid(lf0_norm_data, cmp_sr, t_start, t_end) # lf0_norm_data should have same length as wav_file
                    win_pitch_loc, vuv_temp = cal_win_pitch_loc(pitch_loc_data, t_start, t_end)
                
                    wav[speaker_idx, spk_seq_index, win_idx, :] = wav_file[win_start:win_end+1]
                    nlf[speaker_idx, spk_seq_index, win_idx] = lf0_mid
                    tau[speaker_idx, spk_seq_index, win_idx] = win_pitch_loc
                spk_seq_index += 1
    
    # S,B,M,D
    x_val = wav
    if dv_y_cfg.train_by_window:
        # S --> S*B
        y_val = numpy.repeat(one_hot, dv_y_cfg.spk_num_seq)
        batch_size = dv_y_cfg.batch_num_spk * dv_y_cfg.spk_num_seq
    else:
        y_val = one_hot
        batch_size = dv_y_cfg.batch_num_spk

    feed_dict = {'x':x_val, 'y':y_val}
    feed_dict['nlf'] = nlf
    feed_dict['tau'] = tau
    return_list = [feed_dict, batch_size]
    
    if return_one_hot:
        return_list.append(one_hot)
    if return_y:
        return_list.append(y)
    if return_frame_index:
        return_list.append(start_frame_index_list)
    if return_file_name:
        return_list.append(file_name_list)
    return return_list

def make_feed_dict_y_wav_sinenet_test(dv_y_cfg, file_dir_dict, speaker_id, file_name, start_frame_index, BTD_feat_remain):
    logger = make_logger("make_dict")

    '''Load Data; load starting frame; Slice; Fit into numpy holders
    '''
    # BTD_feat_remain is a tuple now,
    # BTD_feat_remain = (y_feat_remain, nlf_feat_remain, tau_feat_remain)
    feat_name_list = ['wav'] # Load wav
    feat_dim_list  = [1]
    assert dv_y_cfg.batch_num_spk == 1
    # Make i/o shape arrays
    # This is numpy shape, not Tensor shape!
    wav = numpy.zeros((dv_y_cfg.batch_num_spk, dv_y_cfg.spk_num_seq, dv_y_cfg.seq_num_win, dv_y_cfg.seq_win_len))
    nlf = numpy.zeros((dv_y_cfg.batch_num_spk, dv_y_cfg.spk_num_seq, dv_y_cfg.seq_num_win))
    tau = numpy.zeros((dv_y_cfg.batch_num_spk, dv_y_cfg.spk_num_seq, dv_y_cfg.seq_num_win))
    one_hot = numpy.zeros((dv_y_cfg.batch_num_spk))

    # Make classification targets, index sequence
    try: true_speaker_index = dv_y_cfg.speaker_id_list_dict['train'].index(speaker_id)
    except ValueError: true_speaker_index = 0 # At generation time, since one_hot is not used, a non-train speaker is given an arbituary speaker index
    one_hot[0] = true_speaker_index

    if BTD_feat_remain is not None:
        wav_feat_current, nlf_feat_current, tau_feat_current = BTD_feat_remain
        B_total = wav_feat_current.shape[0]
    else:
        # Get new file, make BTD
        file_min_len, features = get_one_utter_by_name(file_name, file_dir_dict, feat_name_list=feat_name_list, feat_dim_list=feat_dim_list)

        wav_file = features['wav'] # T * 1; 16kHz
        wav_file = numpy.squeeze(wav_file, axis=1)      # T*1 -> T
        wav_file_len = file_min_len
        if start_frame_index > 0:
            # Discard some features at beginning
            wav_file = wav_file[start_frame_index:]
            wav_file_len -= start_frame_index

        wav_sr = dv_y_cfg.cfg.wav_sr
        cmp_sr = dv_y_cfg.cfg.frame_sr
        wav_cmp_ratio = int(wav_sr / cmp_sr)

        # Do not use silence frames at the beginning or the end
        total_sil_one_side_cmp = dv_y_cfg.frames_silence_to_keep + dv_y_cfg.sil_pad
        total_sil_one_side_wav = total_sil_one_side_cmp * wav_cmp_ratio
        len_no_sil_wav = wav_file_len - 2 * total_sil_one_side_wav

        # Make numpy holders for no_sil data
        wav_features_no_sil = wav_file[total_sil_one_side_wav:total_sil_one_side_wav+len_no_sil_wav]
        B_total = int((len_no_sil_wav - dv_y_cfg.batch_seq_len) / dv_y_cfg.batch_seq_shift) + 1
        wav_feat_current = numpy.zeros((B_total, dv_y_cfg.seq_num_win, dv_y_cfg.seq_win_len))
        nlf_feat_current = numpy.zeros((B_total, dv_y_cfg.seq_num_win))
        tau_feat_current = numpy.zeros((B_total, dv_y_cfg.seq_num_win))

        # Load cmp and pitch data
        cmp_file_name = os.path.join(file_dir_dict['cmp'], file_name+'.cmp')
        lf0_index     = dv_y_cfg.cfg.acoustic_start_index['lf0']
        cmp_dim       = dv_y_cfg.cfg.nn_feature_dims['cmp']
        lf0_norm_data = load_cmp_file(cmp_file_name, cmp_dim=cmp_dim, feat_dim_index=lf0_index)
        pitch_file_name = os.path.join(file_dir_dict['pitch'], file_name+'.pm')
        pitch_loc_data = read_pitch_file(pitch_file_name)

        # Slice data into seq and win
        for seq_idx in range(B_total):
            spk_seq_index = seq_idx
            seq_start = 0 + seq_idx * dv_y_cfg.batch_seq_shift
            for win_idx in range(dv_y_cfg.seq_num_win):
                win_start = seq_start + win_idx * dv_y_cfg.seq_win_shift
                win_end   = win_start + dv_y_cfg.seq_win_len - 1 # Inclusive index
                t_start = (win_start) / wav_sr
                t_end   = (win_end+1) / wav_sr

                lf0_mid = cal_win_lf0_mid(lf0_norm_data, cmp_sr, t_start, t_end) # lf0_norm_data should have same length as wav_file
                win_pitch_loc, vuv = cal_win_pitch_loc(pitch_loc_data, t_start, t_end)
            
                wav_feat_current[spk_seq_index, win_idx, :] = wav_file[win_start:win_end+1]
                nlf_feat_current[spk_seq_index, win_idx] = lf0_mid
                tau_feat_current[spk_seq_index, win_idx] = win_pitch_loc

    if B_total > dv_y_cfg.spk_num_seq:
        B_actual = dv_y_cfg.spk_num_seq
        B_remain = B_total - B_actual
        gen_finish = False
        wav_feat_remain = wav_feat_current[B_actual:]
        nlf_feat_remain = nlf_feat_current[B_actual:]
        tau_feat_remain = tau_feat_current[B_actual:]
        BTD_feat_remain = (wav_feat_remain, nlf_feat_remain, tau_feat_remain)
    else:
        B_actual = B_total
        B_remain = 0
        gen_finish = True
        BTD_feat_remain = None

    wav[0,:B_actual] = wav_feat_current[:B_actual]
    nlf[0,:B_actual] = nlf_feat_current[:B_actual]
    tau[0,:B_actual] = tau_feat_current[:B_actual]
    batch_size = B_actual

    # B,T,D --> S(1),B,T*D
    x_val = wav
    # B,1,1 --> S(1),B,1,1
    nlf_val = nlf
    tau_val = tau
    if dv_y_cfg.train_by_window:
        # S --> S*B
        y_val = numpy.repeat(one_hot, dv_y_cfg.spk_num_seq)
    else:
        y_val = one_hot

    feed_dict = {'x':x_val, 'y':y_val, 'nlf':nlf_val, 'tau':tau_val}
    return_list = [feed_dict, gen_finish, batch_size, BTD_feat_remain]
    return return_list

def make_feed_dict_y_wav_sinenet_test_voiced_only(dv_y_cfg, file_dir_dict, speaker_id, file_name, start_frame_index, BTD_feat_remain):
    logger = make_logger("make_dict")

    '''Load Data; load starting frame; Slice; Fit into numpy holders
    '''
    # BTD_feat_remain is a tuple now,
    # BTD_feat_remain = (y_feat_remain, nlf_feat_remain, tau_feat_remain)
    feat_name_list = ['wav'] # Load wav
    feat_dim_list  = [1]
    assert dv_y_cfg.batch_num_spk == 1
    # Make i/o shape arrays
    # This is numpy shape, not Tensor shape!
    wav = numpy.zeros((dv_y_cfg.batch_num_spk, dv_y_cfg.spk_num_seq, dv_y_cfg.seq_num_win, dv_y_cfg.seq_win_len))
    nlf = numpy.zeros((dv_y_cfg.batch_num_spk, dv_y_cfg.spk_num_seq, dv_y_cfg.seq_num_win))
    tau = numpy.zeros((dv_y_cfg.batch_num_spk, dv_y_cfg.spk_num_seq, dv_y_cfg.seq_num_win))
    one_hot = numpy.zeros((dv_y_cfg.batch_num_spk))

    # Make classification targets, index sequence
    try: true_speaker_index = dv_y_cfg.speaker_id_list_dict['train'].index(speaker_id)
    except ValueError: true_speaker_index = 0 # At generation time, since one_hot is not used, a non-train speaker is given an arbituary speaker index
    one_hot[0] = true_speaker_index

    if BTD_feat_remain is not None:
        wav_feat_current, nlf_feat_current, tau_feat_current = BTD_feat_remain
        B_total = wav_feat_current.shape[0]
    else:
        # Get new file, make BTD
        file_min_len, features = get_one_utter_by_name(file_name, file_dir_dict, feat_name_list=feat_name_list, feat_dim_list=feat_dim_list)

        wav_file = features['wav'] # T * 1; 16kHz
        wav_file = numpy.squeeze(wav_file, axis=1)      # T*1 -> T
        wav_file_len = file_min_len
        if start_frame_index > 0:
            # Discard some features at beginning
            wav_file = wav_file[start_frame_index:]
            wav_file_len -= start_frame_index

        wav_sr = dv_y_cfg.cfg.wav_sr
        cmp_sr = dv_y_cfg.cfg.frame_sr
        wav_cmp_ratio = int(wav_sr / cmp_sr)

        # Do not use silence frames at the beginning or the end
        total_sil_one_side_cmp = dv_y_cfg.frames_silence_to_keep + dv_y_cfg.sil_pad
        total_sil_one_side_wav = total_sil_one_side_cmp * wav_cmp_ratio
        len_no_sil_wav = wav_file_len - 2 * total_sil_one_side_wav
        voiced_seq_win_threshold = int(dv_y_cfg.use_voiced_threshold * dv_y_cfg.seq_num_win)

        # Find number of sequences, then select the ones with good voicing        
        wav_features_no_sil = wav_file[total_sil_one_side_wav:total_sil_one_side_wav+len_no_sil_wav]
        B_total = int((len_no_sil_wav - dv_y_cfg.batch_seq_len) / dv_y_cfg.batch_seq_shift) + 1

        # Load cmp and pitch data
        cmp_file_name = os.path.join(file_dir_dict['cmp'], file_name+'.cmp')
        lf0_index     = dv_y_cfg.cfg.acoustic_start_index['lf0']
        cmp_dim       = dv_y_cfg.cfg.nn_feature_dims['cmp']
        lf0_norm_data = load_cmp_file(cmp_file_name, cmp_dim=cmp_dim, feat_dim_index=lf0_index)
        pitch_file_name = os.path.join(file_dir_dict['pitch'], file_name+'.pm')
        pitch_loc_data = read_pitch_file(pitch_file_name)

        # Load pitch data and find good windows first
        voiced_win_idx_list = []
        for seq_idx in range(B_total):
            seq_start = 0 + seq_idx * dv_y_cfg.batch_seq_shift
            voiced_count = 0
            for win_idx in range(dv_y_cfg.seq_num_win):
                win_start = seq_start + win_idx * dv_y_cfg.seq_win_shift
                win_end   = win_start + dv_y_cfg.seq_win_len - 1 # Inclusive index
                t_start = (win_start) / wav_sr
                t_end   = (win_end+1) / wav_sr

                win_pitch_loc, vuv_temp = cal_win_pitch_loc(pitch_loc_data, t_start, t_end)
                if vuv_temp:
                    voiced_count += 1
            if voiced_count >= voiced_seq_win_threshold:
                voiced_win_idx_list.append(seq_idx)

        # Make numpy holders for no_sil data
        B_total = len(voiced_win_idx_list)
        wav_feat_current = numpy.zeros((B_total, dv_y_cfg.seq_num_win, dv_y_cfg.seq_win_len))
        nlf_feat_current = numpy.zeros((B_total, dv_y_cfg.seq_num_win))
        tau_feat_current = numpy.zeros((B_total, dv_y_cfg.seq_num_win))

        # Slice data into seq and win
        for seq_idx in range(B_total):
            spk_seq_index = seq_idx
            seq_start = voiced_win_idx_list[seq_idx]
            for win_idx in range(dv_y_cfg.seq_num_win):
                win_start = seq_start + win_idx * dv_y_cfg.seq_win_shift
                win_end   = win_start + dv_y_cfg.seq_win_len - 1 # Inclusive index
                t_start = (win_start) / wav_sr
                t_end   = (win_end+1) / wav_sr

                lf0_mid = cal_win_lf0_mid(lf0_norm_data, cmp_sr, t_start, t_end) # lf0_norm_data should have same length as wav_file
                win_pitch_loc, vuv = cal_win_pitch_loc(pitch_loc_data, t_start, t_end)
            
                wav_feat_current[spk_seq_index, win_idx, :] = wav_file[win_start:win_end+1]
                nlf_feat_current[spk_seq_index, win_idx] = lf0_mid
                tau_feat_current[spk_seq_index, win_idx] = win_pitch_loc

    if B_total > dv_y_cfg.spk_num_seq:
        B_actual = dv_y_cfg.spk_num_seq
        B_remain = B_total - B_actual
        gen_finish = False
        wav_feat_remain = wav_feat_current[B_actual:]
        nlf_feat_remain = nlf_feat_current[B_actual:]
        tau_feat_remain = tau_feat_current[B_actual:]
        BTD_feat_remain = (wav_feat_remain, nlf_feat_remain, tau_feat_remain)
    else:
        B_actual = B_total
        B_remain = 0
        gen_finish = True
        BTD_feat_remain = None

    wav[0,:B_actual] = wav_feat_current[:B_actual]
    nlf[0,:B_actual] = nlf_feat_current[:B_actual]
    tau[0,:B_actual] = tau_feat_current[:B_actual]
    batch_size = B_actual

    # B,T,D --> S(1),B,T*D
    x_val = wav
    # B,1,1 --> S(1),B,1,1
    nlf_val = nlf
    tau_val = tau
    if dv_y_cfg.train_by_window:
        # S --> S*B
        y_val = numpy.repeat(one_hot, dv_y_cfg.spk_num_seq)
    else:
        y_val = one_hot

    feed_dict = {'x':x_val, 'y':y_val, 'nlf':nlf_val, 'tau':tau_val}
    return_list = [feed_dict, gen_finish, batch_size, BTD_feat_remain]
    return return_list

def make_feed_dict_y_wav_sinenet_distance(dv_y_cfg, file_list_dict, file_dir_dict, batch_speaker_list, utter_tvt, all_utt_start_frame_index=None,  return_y=False, return_frame_index=False, return_file_name=False):
    logger = make_logger("make_dict")

    '''
    Draw Utterances; Load Data
    Draw starting frame; Slice; Fit into numpy holders
    '''
    feat_name_list = ['wav'] # Load wav
    feat_dim_list  = [1]
    # Make i/o shape arrays
    # This is numpy shape, not Tensor shape!
    wav_list = []
    nlf_list = []
    tau_list = []
    for plot_idx in range(dv_y_cfg.num_to_plot + 1):
        wav = numpy.zeros((dv_y_cfg.batch_num_spk, dv_y_cfg.spk_num_seq, dv_y_cfg.seq_num_win, dv_y_cfg.seq_win_len))
        nlf = numpy.zeros((dv_y_cfg.batch_num_spk, dv_y_cfg.spk_num_seq, dv_y_cfg.seq_num_win))
        tau = numpy.zeros((dv_y_cfg.batch_num_spk, dv_y_cfg.spk_num_seq, dv_y_cfg.seq_num_win))
        wav_list.append(wav)
        nlf_list.append(nlf)
        tau_list.append(tau)

    wav_sr  = dv_y_cfg.cfg.wav_sr
    cmp_sr  = dv_y_cfg.cfg.frame_sr
    wav_cmp_ratio = int(wav_sr / cmp_sr)
    # Do not use silence frames at the beginning or the end
    total_sil_one_side_cmp = dv_y_cfg.frames_silence_to_keep + dv_y_cfg.sil_pad  # This is at 200Hz
    total_sil_one_side_wav = total_sil_one_side_cmp * wav_cmp_ratio              # This is at 16kHz
    min_file_len = dv_y_cfg.batch_seq_total_len + 2 * total_sil_one_side_wav # This is at 16kHz
    # Add extra for shift distance test
    min_file_len = min_file_len + dv_y_cfg.max_len_to_plot

    file_name_list = []
    start_frame_index_list = [[]]*dv_y_cfg.batch_num_spk
    
    for speaker_idx in range(dv_y_cfg.batch_num_spk):
        speaker_id = batch_speaker_list[speaker_idx]

        # Draw 1 utterances per speaker
        # Draw multiple windows per utterance:  dv_y_cfg.spk_num_seq
        # Stack them along B
        speaker_file_name_list, speaker_utter_len_list, speaker_utter_list = get_utters_from_binary_dict(1, file_list_dict[(speaker_id, utter_tvt)], file_dir_dict, feat_name_list=feat_name_list, feat_dim_list=feat_dim_list, min_file_len=min_file_len, random_seed=None)
        file_name_list.append(speaker_file_name_list)

        for utter_idx in range(dv_y_cfg.spk_num_utter):
            file_name = speaker_file_name_list[utter_idx]
            wav_file  = speaker_utter_list['wav'][utter_idx] # T * 1; 16kHz
            wav_file  = numpy.squeeze(wav_file, axis=1)      # T*1 -> T
            wav_file_len = speaker_utter_len_list[utter_idx]

            # Find start frame index, random if None
            if all_utt_start_frame_index is None:
                extra_file_len = wav_file_len - min_file_len
                utter_start_frame_index = numpy.random.randint(low=total_sil_one_side_wav, high=total_sil_one_side_wav+extra_file_len+1)
            else:
                utter_start_frame_index = total_sil_one_side_wav + all_utt_start_frame_index
            start_frame_index_list[speaker_idx].append(utter_start_frame_index)

            # Load cmp and pitch data
            cmp_file_name = os.path.join(file_dir_dict['cmp'], file_name+'.cmp')
            lf0_index     = dv_y_cfg.cfg.acoustic_start_index['lf0']
            cmp_dim       = dv_y_cfg.cfg.nn_feature_dims['cmp']
            lf0_norm_data = load_cmp_file(cmp_file_name, cmp_dim=cmp_dim, feat_dim_index=lf0_index)
            pitch_file_name = os.path.join(file_dir_dict['pitch'], file_name+'.pm')
            pitch_loc_data = read_pitch_file(pitch_file_name)

            for plot_idx in range(dv_y_cfg.num_to_plot+1):
                plot_start_frame_index = utter_start_frame_index + plot_idx * dv_y_cfg.gap_len_to_plot
                # Slice data into seq and win
                for seq_idx in range(dv_y_cfg.spk_num_seq):
                    spk_seq_index = utter_idx * dv_y_cfg.spk_num_seq + seq_idx
                    seq_start = plot_start_frame_index + seq_idx * dv_y_cfg.batch_seq_shift
                    for win_idx in range(dv_y_cfg.seq_num_win):
                        win_start = seq_start + win_idx * dv_y_cfg.seq_win_shift
                        win_end   = win_start + dv_y_cfg.seq_win_len - 1 # Inclusive index
                        t_start = (win_start) / wav_sr
                        t_end   = (win_end+1) / wav_sr

                        lf0_mid = cal_win_lf0_mid(lf0_norm_data, cmp_sr, t_start, t_end) # lf0_norm_data should have same length as wav_file
                        win_pitch_loc, vuv = cal_win_pitch_loc(pitch_loc_data, t_start, t_end)
                    
                        wav_list[plot_idx][speaker_idx, spk_seq_index, win_idx, :] = wav_file[win_start:win_end+1]
                        nlf_list[plot_idx][speaker_idx, spk_seq_index, win_idx] = lf0_mid
                        tau_list[plot_idx][speaker_idx, spk_seq_index, win_idx] = win_pitch_loc

    # S,B,M,D
    feed_dict_list = [{}] * (dv_y_cfg.num_to_plot+1)
    for plot_idx in range(dv_y_cfg.num_to_plot+1):
        x_val = wav_list[plot_idx]
        feed_dict_list[plot_idx] = {'x':x_val}
        feed_dict_list[plot_idx]['nlf'] = nlf_list[plot_idx]
        feed_dict_list[plot_idx]['tau'] = tau_list[plot_idx]
    batch_size = dv_y_cfg.batch_num_spk * dv_y_cfg.spk_num_seq

    return_list = [feed_dict_list, batch_size]
    
    if return_y:
        return_list.append(y)
    if return_frame_index:
        return_list.append(start_frame_index_list)
    if return_file_name:
        return_list.append(file_name_list)
    return return_list

def cal_seq_win_lf0_mid_old(lf0_norm_data, utter_start_frame_index, dv_y_cfg, wav_cmp_ratio):
    lf0_norm_data = numpy.array(range(30))
    wav_sr  = dv_y_cfg.cfg.wav_sr
    cmp_sr  = dv_y_cfg.cfg.frame_sr
    
    nlf = numpy.zeros((dv_y_cfg.spk_num_seq, dv_y_cfg.seq_num_win))
    # Slice data into seq and win
    for seq_idx in range(dv_y_cfg.spk_num_seq):
        spk_seq_index = seq_idx
        seq_start = utter_start_frame_index + seq_idx * dv_y_cfg.batch_seq_shift
        for win_idx in range(dv_y_cfg.seq_num_win):
            win_start = seq_start + win_idx * dv_y_cfg.seq_win_shift
            t_start = (win_start) / float(wav_sr)
            t_end   = (win_start+dv_y_cfg.seq_win_len) / float(wav_sr)

            t_mid = (t_start + t_end) / 2.
            n_mid = t_mid * float(cmp_sr)
            # e.g. 1.3 is between 0.5, 1.5; n_l=0, n_r=1
            n_l = int(n_mid-0.5)
            n_r = n_l + 1
            l = lf0_norm_data.shape[0]
            if n_r >= l:
                lf0_mid = lf0_norm_data[-1]
            else:
                lf0_l = lf0_norm_data[n_l]
                lf0_r = lf0_norm_data[n_r]
                r = n_mid - ( n_l + 0.5 )
                lf0_mid = (r * lf0_r) + ((1-r) * lf0_l)

            nlf[spk_seq_index, win_idx] = lf0_mid

    return nlf

def cal_seq_win_tau_vuv_old(pitch_loc_data, utter_start_frame_index, dv_y_cfg, wav_sr):
    tau = numpy.zeros((dv_y_cfg.spk_num_seq, dv_y_cfg.seq_num_win))
    vuv = numpy.zeros((dv_y_cfg.spk_num_seq, dv_y_cfg.seq_num_win))
    # Slice data into seq and win
    for seq_idx in range(dv_y_cfg.spk_num_seq):
        seq_start = utter_start_frame_index + seq_idx * dv_y_cfg.batch_seq_shift
        for win_idx in range(dv_y_cfg.seq_num_win):
            win_start = seq_start + win_idx * dv_y_cfg.seq_win_shift
            win_end   = win_start + dv_y_cfg.seq_win_len - 1 # Inclusive index
            t_start = (win_start) / float(wav_sr)
            t_end   = (win_start + dv_y_cfg.seq_win_len) / float(wav_sr)

            # No pitch, return 0
            win_pitch_loc = 0.
            vuv_temp = 0
            for t in pitch_loc_data:
                if t > (t_start):
                    if t < (t_end):
                        t_r = t - t_start
                        win_pitch_loc = t_r
                        vuv_temp = 1
                        break
                elif t > t_end:
                    # No pitch found in interval
                    win_pitch_loc = 0.
                    vuv_temp = 0
                    break

            tau[seq_idx, win_idx] = win_pitch_loc
            vuv[seq_idx, win_idx] = vuv_temp
    return tau, vuv


def cal_win_pitch_loc(pitch_loc_data, t_start, t_end, t_removed=0):
    t_start_total = t_start + t_removed
    t_end_total   = t_end + t_removed
    for t in pitch_loc_data:
        if t > (t_start_total):
            if t < (t_end_total):
                t_r = t - t_start_total
                return t_r, 1
        elif t > t_end_total:
            # No pitch found in interval
            return 0, 0
    # No pitch, return 0
    return 0, 0

def cal_win_lf0_mid(lf0_norm_data, cmp_sr, t_start, t_end):
    # 1. Find central time t_mid
    # 2. Find 2 frames left and right of t_mid
    # 3. Find interpolated lf0 value at t_mid
    t_mid = (t_start + t_end) / 2
    n_mid = t_mid * cmp_sr
    # e.g. 1.3 is between 0.5, 1.5; n_l=0, n_r=1
    n_l = int(n_mid-0.5)
    n_r = n_l + 1
    l = lf0_norm_data.shape[0]
    if n_r >= l:
        return lf0_norm_data[-1]
    else:
        lf0_l = lf0_norm_data[n_l]
        lf0_r = lf0_norm_data[n_r]
        r = n_mid - n_l
        lf0_mid = r * lf0_r + (1-r) * lf0_l
        return lf0_mid



###############
# Data Loader #
###############

class dv_y_data_loader(object):
    """ Data loader for dv_y training and tests """
    def __init__(self, dv_y_cfg, file_list_dict, nn_feat_scratch_dirs, speaker_id_list):
        self.dv_y_cfg = dv_y_cfg
        self.file_list_dict = file_list_dict
        self.nn_feat_scratch_dirs = nn_feat_scratch_dirs
        self.speaker_id_list = speaker_id_list
        self.speaker_loader  = list_random_loader(speaker_id_list)

class dv_y_wav_data_loader(dv_y_data_loader):
    def __init__(self, dv_y_cfg, file_list_dict, nn_feat_scratch_dirs, speaker_id_list):
        super().__init__(dv_y_cfg, file_list_dict, nn_feat_scratch_dirs, speaker_id_list)

        self.feat_name_list = ['wav'] # Load wav
        self.feat_dim_list  = [1]
        
        self.wav_sr  = dv_y_cfg.cfg.wav_sr
        self.cmp_sr  = dv_y_cfg.cfg.frame_sr
        self.wav_cmp_ratio = int(self.wav_sr / self.cmp_sr)
        # Do not use silence frames at the beginning or the end
        total_sil_one_side_cmp = dv_y_cfg.frames_silence_to_keep + dv_y_cfg.sil_pad  # This is at 200Hz
        self.total_sil_one_side_wav = total_sil_one_side_cmp * self.wav_cmp_ratio              # This is at 16kHz
        self.min_file_len = dv_y_cfg.batch_seq_total_len + 2 * self.total_sil_one_side_wav # This is at 16kHz

    def make_feed_dict_y_wav_train(self, batch_speaker_list=None, utter_tvt='train', feat_feed_dict=['wav'], input_utt_start_frame_index_list=None, return_frame_index=False, return_file_name=False):
        '''
          Load wav file, 1 per speaker. Check if long enough
          Use or draw random starting frame index
        '''
        dv_y_cfg = self.dv_y_cfg
        logger = make_logger("make_dict")
        # Make i/o shape arrays
        # This is numpy shape, not Tensor shape!
        one_hot = numpy.zeros((dv_y_cfg.batch_num_spk))
        wav = numpy.zeros((dv_y_cfg.batch_num_spk, dv_y_cfg.batch_seq_total_len))

        if batch_speaker_list is None:
            batch_num_spk = self.dv_y_cfg.batch_num_spk
            batch_speaker_list = self.speaker_loader.draw_n_samples(dv_y_cfg.batch_num_spk)
        else:
            batch_num_spk = len(batch_speaker_list)

        # One Hot
        for speaker_idx in range(batch_num_spk):
            speaker_id = batch_speaker_list[speaker_idx]
            # Make classification targets, index sequence
            true_speaker_index = dv_y_cfg.speaker_id_list_dict['train'].index(speaker_id)
            one_hot[speaker_idx] = true_speaker_index

        if dv_y_cfg.train_by_window:
            # S --> S*B
            y_val = numpy.repeat(one_hot, dv_y_cfg.spk_num_seq)
            batch_size = dv_y_cfg.batch_num_spk * dv_y_cfg.spk_num_seq
        else:
            y_val = one_hot
            batch_size = dv_y_cfg.batch_num_spk
        feed_dict = {'y':y_val}

        # Wav
        file_name_list = [[] for i in range(dv_y_cfg.batch_num_spk)]
        start_frame_index_list = [[] for i in range(dv_y_cfg.batch_num_spk)]
        min_file_len = self.min_file_len
        total_sil_one_side_wav = self.total_sil_one_side_wav

        for speaker_idx in range(batch_num_spk):
            speaker_id = batch_speaker_list[speaker_idx]
            # Draw 1 utterance per speaker
            # Draw multiple windows per utterance:  dv_y_cfg.spk_num_seq
            # Stack them along B
            speaker_file_name_list, speaker_utter_len_list, speaker_utter_list = get_utters_from_binary_dict(1, self.file_list_dict[(speaker_id, utter_tvt)], self.nn_feat_scratch_dirs, feat_name_list=self.feat_name_list, feat_dim_list=self.feat_dim_list, min_file_len=min_file_len, random_seed=None)
            file_name_list[speaker_idx].extend(speaker_file_name_list)

            file_name = speaker_file_name_list[0]
            wav_file  = speaker_utter_list['wav'][0] # T * 1; 16kHz
            wav_file  = numpy.squeeze(wav_file, axis=1)      # T*1 -> T
            wav_file_len = speaker_utter_len_list[0]

            # Find start frame index, random if None
            if input_utt_start_frame_index_list is None:
                extra_file_len = wav_file_len - min_file_len
                utter_start_frame_index = numpy.random.randint(low=total_sil_one_side_wav, high=total_sil_one_side_wav+extra_file_len+1)
            else:
                utter_start_frame_index = total_sil_one_side_wav + input_utt_start_frame_index_list[speaker_idx]
            start_frame_index_list[speaker_idx].append(utter_start_frame_index)
            wav[speaker_idx, :] = wav_file[utter_start_frame_index:utter_start_frame_index+dv_y_cfg.batch_seq_total_len]

        x_val = wav
        feed_dict['x'] = x_val

        # nlf
        if 'nlf' in feat_feed_dict:
            nlf = numpy.zeros((dv_y_cfg.batch_num_spk, dv_y_cfg.spk_num_seq, dv_y_cfg.seq_num_win))
            for speaker_idx in range(batch_num_spk):
                file_name = file_name_list[speaker_idx][0]
                utter_start_frame_index = start_frame_index_list[speaker_idx][0]
                # Load cmp data
                cmp_file_name = os.path.join(self.nn_feat_scratch_dirs['cmp'], file_name+'.cmp')
                lf0_index     = dv_y_cfg.cfg.acoustic_start_index['lf0']
                cmp_dim       = dv_y_cfg.cfg.nn_feature_dims['cmp']
                lf0_norm_data = load_cmp_file(cmp_file_name, cmp_dim=cmp_dim, feat_dim_index=lf0_index)

                # Get lf0_mid data in forms of numpy array operations, faster than for loops
                n_mid_0 = self.return_n_mid_0_matrix()
                nlf[speaker_idx] = cal_seq_win_lf0_mid(lf0_norm_data, utter_start_frame_index, n_mid_0, wav_cmp_ratio)
            feed_dict['nlf'] = nlf

        # tau and vuv
        if ('tau' in feat_feed_dict) or ('vuv' in feat_feed_dict):
            tau = numpy.zeros((dv_y_cfg.batch_num_spk, dv_y_cfg.spk_num_seq, dv_y_cfg.seq_num_win))
            vuv = numpy.zeros((dv_y_cfg.batch_num_spk, dv_y_cfg.spk_num_seq, dv_y_cfg.seq_num_win))
            for speaker_idx in range(batch_num_spk):
                file_name = file_name_list[speaker_idx][0]
                utter_start_frame_index = start_frame_index_list[speaker_idx][0]
                # Load pitch data
                pitch_file_name = os.path.join(self.nn_feat_scratch_dirs['pitch'], file_name+'.pm')
                pitch_loc_data = read_pitch_file(pitch_file_name)

                # Get lf0_mid data in forms of numpy array operations, faster than for loops
                win_start_0 = self.return_win_start_0_matrix()
                tau_spk, vuv_spk = cal_seq_win_tau_vuv(pitch_loc_data, utter_start_frame_index, dv_y_cfg.spk_num_seq, dv_y_cfg.seq_num_win, dv_y_cfg.seq_win_len, win_start_0, dv_y_cfg.cfg.wav_sr)
                tau[speaker_idx] = tau_spk
                vuv[speaker_idx] = vuv_spk
            feed_dict['tau'] = tau
            if dv_y_cfg.use_voiced_only:
                # Current method: Some b in B are voiced, use vuv as error mask
                assert dv_y_cfg.train_by_window
                # Make binary S * B matrix
                vuv_S_B = (vuv>0).all(axis=2)
                # Reshape to SB for pytorch cross-entropy function
                vuv_SB = numpy.reshape(vuv_S_B, (dv_y_cfg.batch_num_spk * dv_y_cfg.spk_num_seq))
                feed_dict['vuv'] = vuv_SB

        return_list = [feed_dict, batch_size]
        if return_frame_index:
            return_list.append(start_frame_index_list)
        if return_file_name:
            return_list.append(file_name_list)
        if return_vuv:
            return_list.append(vuv)
        return return_list

    def return_n_mid_0_matrix(self):
        try:
            return self.n_mid_0
        except AttributeError:
            self.make_n_mid_0_matrix()
            return self.n_mid_0

    def make_n_mid_0_matrix(self):
        dv_y_cfg = self.dv_y_cfg
        wav_sr  = dv_y_cfg.cfg.wav_sr
        cmp_sr  = dv_y_cfg.cfg.frame_sr
        wav_cmp_ratio = int(wav_sr / cmp_sr)
        self.n_mid_0 = numpy.zeros([dv_y_cfg.spk_num_seq, dv_y_cfg.seq_num_win])
        for seq_idx in range(dv_y_cfg.spk_num_seq):
            for win_idx in range(dv_y_cfg.seq_num_win):
                seq_start = seq_idx * dv_y_cfg.batch_seq_shift
                win_start = seq_start + win_idx * dv_y_cfg.seq_win_shift
                t_start = (win_start) / float(wav_sr)
                t_end   = (win_start+dv_y_cfg.seq_win_len) / float(wav_sr)

                t_mid = (t_start + t_end) / 2.
                n_mid = t_mid * float(cmp_sr)

                self.n_mid_0[seq_idx, win_idx] = n_mid

    def return_win_start_0_matrix(self):
        try:
            return self.win_start_matrix
        except AttributeError:
            self.make_win_start_0_matrix()
            return self.win_start_matrix

    def make_win_start_0_matrix(self):
        dv_y_cfg = self.dv_y_cfg
        wav_sr  = dv_y_cfg.cfg.wav_sr
        cmp_sr  = dv_y_cfg.cfg.frame_sr
        wav_cmp_ratio = int(wav_sr / cmp_sr)
        self.win_start_matrix = numpy.zeros([dv_y_cfg.spk_num_seq, dv_y_cfg.seq_num_win, 1])
        for seq_idx in range(dv_y_cfg.spk_num_seq):
            for win_idx in range(dv_y_cfg.seq_num_win):
                seq_start = seq_idx * dv_y_cfg.batch_seq_shift
                self.win_start_matrix[seq_idx, dv_y_cfg, 0] = seq_start + dv_y_cfg * dv_y_cfg.seq_win_shift

def cal_seq_win_lf0_mid(lf0_norm_data, utter_start_frame_index, n_mid_0, wav_cmp_ratio):
    ''' Derive position using utter_start_frame_index and n_mid_0 matrix, then extract from lf0_norm_data '''
    l = lf0_norm_data.shape[0]
    n_mid = n_mid_0 + (utter_start_frame_index / float(wav_cmp_ratio))
    n_l = (n_mid-0.5).astype(int)
    n_r = n_l + 1
    r = n_mid - (n_l + 0.5)
    n_l[n_r>= l] = -1
    n_r[n_r>= l] = -1
    lf0_l = lf0_norm_data[n_l]
    lf0_r = lf0_norm_data[n_r]
    lf0_mid = r * lf0_r + (1-r) * lf0_l
    return lf0_mid

def cal_seq_win_tau_vuv(pitch_loc_data, utter_start_frame_index, spk_num_seq, seq_num_win, seq_win_len, win_start_0, wav_sr):
    ''' Calculate pitch location per window; if not found then vuv=0 '''
    l = pitch_loc_data.shape[0]
    tau = numpy.zeros((spk_num_seq, seq_num_win))
    vuv = numpy.zeros((spk_num_seq, seq_num_win))
    
    pitch_max = seq_win_len / float(wav_sr)

    win_start = win_start_0 + utter_start_frame_index
    t_start = win_start / float(wav_sr)

    t_start = numpy.repeat(t_start, l, axis=2)

    pitch_start = pitch_loc_data - t_start
    pitch_start[pitch_start <= 0.] = pitch_max
    pitch_start_min = numpy.amin(pitch_start, axis=2)

    vuv[pitch_start_min < pitch_max] = 1
    pitch_start_min[pitch_start_min >= pitch_max] = 0.
    # pitch_start_min = pitch_start_min * vuv
    tau = pitch_start_min

    return tau, vuv

def cal_seq_win_tau_vuv_old(pitch_loc_data, utter_start_frame_index, dv_y_cfg, wav_sr):
    # tau_spk, vuv_spk = cal_seq_win_tau_vuv_old(pitch_loc_data, utter_start_frame_index, dv_y_cfg, wav_sr)
    tau = numpy.zeros((dv_y_cfg.spk_num_seq, dv_y_cfg.seq_num_win))
    vuv = numpy.zeros((dv_y_cfg.spk_num_seq, dv_y_cfg.seq_num_win))
    # Slice data into seq and win
    for seq_idx in range(dv_y_cfg.spk_num_seq):
        seq_start = utter_start_frame_index + seq_idx * dv_y_cfg.batch_seq_shift
        for win_idx in range(dv_y_cfg.seq_num_win):
            win_start = seq_start + win_idx * dv_y_cfg.seq_win_shift
            win_end   = win_start + dv_y_cfg.seq_win_len - 1 # Inclusive index
            t_start = (win_start) / float(wav_sr)
            t_end   = (win_start + dv_y_cfg.seq_win_len) / float(wav_sr)

            # No pitch, return 0
            win_pitch_loc = 0.
            vuv_temp = 0
            for t in pitch_loc_data:
                if t > (t_start):
                    if t < (t_end):
                        t_r = t - t_start
                        win_pitch_loc = t_r
                        vuv_temp = 1
                        break
                elif t > t_end:
                    # No pitch found in interval
                    win_pitch_loc = 0.
                    vuv_temp = 0
                    break

            tau[seq_idx, win_idx] = win_pitch_loc
            vuv[seq_idx, win_idx] = vuv_temp
    return tau, vuv

def cal_seq_win_lf0_mid_old(lf0_norm_data, utter_start_frame_index, dv_y_cfg, wav_cmp_ratio):
    # nlf[speaker_idx] = cal_seq_win_lf0_mid_old(lf0_norm_data, utter_start_frame_index, dv_y_cfg, wav_cmp_ratio)
    wav_sr  = dv_y_cfg.cfg.wav_sr
    cmp_sr  = dv_y_cfg.cfg.frame_sr
    
    nlf = numpy.zeros((dv_y_cfg.spk_num_seq, dv_y_cfg.seq_num_win))
    # Slice data into seq and win
    for seq_idx in range(dv_y_cfg.spk_num_seq):
        spk_seq_index = seq_idx
        seq_start = utter_start_frame_index + seq_idx * dv_y_cfg.batch_seq_shift
        for win_idx in range(dv_y_cfg.seq_num_win):
            win_start = seq_start + win_idx * dv_y_cfg.seq_win_shift
            t_start = (win_start) / float(wav_sr)
            t_end   = (win_start+dv_y_cfg.seq_win_len) / float(wav_sr)

            t_mid = (t_start + t_end) / 2.
            n_mid = t_mid * float(cmp_sr)
            # e.g. 1.3 is between 0.5, 1.5; n_l=0, n_r=1
            n_l = int(n_mid-0.5)
            n_r = n_l + 1
            l = lf0_norm_data.shape[0]
            if n_r >= l:
                lf0_mid = lf0_norm_data[-1]
            else:
                lf0_l = lf0_norm_data[n_l]
                lf0_r = lf0_norm_data[n_r]
                r = n_mid - ( n_l + 0.5 )
                lf0_mid = (r * lf0_r) + ((1-r) * lf0_l)

            nlf[spk_seq_index, win_idx] = lf0_mid

    return nlf



import os, pickle
import numpy
from scipy.stats.stats import pearsonr

dnn_dir = '/home/dawna/tts/mw545/TorchDV/dv_wav_subwin/dv_y_wav_lr_0.000100_ReL80_LRe256BN_LRe256BN_LRe8DR_DV8S100B23T3200D1'
sin_dir = '/home/dawna/tts/mw545/TorchDV/dv_wav_sinenet_v3/dv_y_wav_lr_0.000100_Sin81f16_LRe256BN_LRe256BN_LRe8DR_DV8S100B12T3200D1'

nlf_var_list_dnn = pickle.load(open(os.path.join(dnn_dir, 'nlf_var_list.data'), 'rb'))
nlf_var_list_sin = pickle.load(open(os.path.join(sin_dir, 'nlf_var_list.data'), 'rb'))
ce_list_dnn = pickle.load(open(os.path.join(dnn_dir, 'ce_list.data'), 'rb'))
ce_list_sin = pickle.load(open(os.path.join(sin_dir, 'ce_list.data'), 'rb'))

assert nlf_var_list_dnn['train'][108] == nlf_var_list_sin['train'][108]

for utter_tvt_name in ['train', 'valid', 'test']:
    ce_diff_list = []
    for i in range(len(ce_list_dnn[utter_tvt_name])):
        ce_dnn = ce_list_dnn[utter_tvt_name][i]
        ce_sin = ce_list_sin[utter_tvt_name][i]
        ce_diff = ce_dnn - ce_sin
        ce_diff_list.append(ce_diff)
    corr_coef = pearsonr(nlf_var_list_dnn[utter_tvt_name], ce_list[utter_tvt_name])
    print(corr_coef)
