
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
    dv_y_cfg.change_to_gen_h_mode()
    log_class_attri(dv_y_cfg, logger, except_list=dv_y_cfg.log_except_list)

    dv_y_model = torch_initialisation(dv_y_cfg)
    dv_y_model.load_nn_model(dv_y_cfg.nnets_file_name)
    dv_y_model.eval()

    batch_speaker_list = dv_y_cfg.batch_speaker_list
    file_list_dict     = dv_y_cfg.file_list_dict
    make_feed_dict_method_train = dv_y_cfg.make_feed_dict_method_train

    feed_dict, batch_size = make_feed_dict_method_train(dv_y_cfg, file_list_dict, cfg.nn_feat_scratch_dirs, batch_speaker_list, utter_tvt='gen')

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