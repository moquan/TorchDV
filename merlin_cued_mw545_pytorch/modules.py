# modules.py

import os, sys, pickle, time, shutil, logging
import math, numpy, scipy, scipy.io.wavfile #, sigproc, sigproc.pystraight
numpy.random.seed(545)

'''
This file contains handy modules of using Merlin
All file lists and directories should be provided elsewhere
'''

def make_logger(logger_name):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    # create console handler and set level to debug
    if not logger.handlers:
        ch = logging.StreamHandler(sys.stdout)
        # ch.setLevel(logging.DEBUG)
        # create formatter
        formatter = logging.Formatter('%(asctime)s %(levelname)8s%(name)15s: %(message)s')
        # add formatter to ch
        ch.setFormatter(formatter)
        # add ch to logger
        logger.addHandler(ch)
    return logger

def find_index_list_for_parallel(num_threads, in_file_list):
    num_files = len(in_file_list)
    num_files_per_thread = int(num_files/num_threads)
    index_list = []
    start_index = 0
    for i in range(num_threads-1):
        end_index = start_index + num_files_per_thread
        index_list.append([start_index, end_index])
        start_index = start_index + num_files_per_thread
    end_index = num_files
    index_list.append([start_index, end_index])
    assert len(index_list) == num_threads
    return index_list

def make_held_out_file_number(last_index, start_index=1):
    held_out_file_number = []
    for i in range(start_index, last_index+1):
        held_out_file_number.append('0'*(3-len(str(i)))+str(i))
    return held_out_file_number

def read_file_list(file_name):
    logger = make_logger("read_file_list")
    file_lists = []
    fid = open(file_name)
    for line in fid.readlines():
        line = line.strip()
        if len(line) < 1:
            continue
        file_lists.append(line)
    fid.close()
    logger.info('Read file list from %s' % file_name)
    return  file_lists

def get_iv_values_from_file(iv_file_name, file_type='text'):
    if file_type == 'text':
        iv_values = {}
        with open(iv_file_name, 'r') as f:
            f_lines = f.readlines()
        for x in f_lines:
            x_id = x.split(':')[0][1:-1]
            y = x.split(':')[1].strip()[1:-2].split(',')
            x_value = [float(i) for i in y]
            iv_values[x_id] = numpy.asarray(x_value,dtype=numpy.float32).reshape([1,-1])
    elif file_type == 'pickle':
        iv_values = pickle.load(open(iv_file_name, 'rb'))
    return iv_values

def save_iv_values_to_file(iv_values, iv_file_name, file_type='text'):
    if file_type == 'text':
        speaker_id_list = iv_values.keys()
        with open(iv_file_name, 'w') as f:
            for speaker_id in speaker_id_list:
                f.write("'"+speaker_id+"': [")
                iv_size = len(iv_values[speaker_id])
                for i in range(iv_size):
                    if i > 0:
                        f.write(',')
                    f.write(str(iv_values[speaker_id][i]))
                f.write("],\n")
    elif file_type == 'pickle':
        pickle.dump(iv_values, open(iv_file_name, 'wb'))

def check_and_change_to_list(sub_list):
    if not isinstance(sub_list, list):
        sub_list = [sub_list]
    return sub_list

def keep_by_speaker(source_list, sub_list):
    target_list = []
    sub_list = check_and_change_to_list(sub_list)
    for y in source_list:
        speaker_id = y.split('/')[-1].split('.')[0].split('_')[0]
        if speaker_id in sub_list:
            target_list.append(y)
    return target_list

def remove_by_speaker(source_list, sub_list):
    target_list = []
    sub_list = check_and_change_to_list(sub_list)
    for y in source_list:
        speaker_id = y.split('/')[-1].split('.')[0].split('_')[0]
        if speaker_id not in sub_list:
            target_list.append(y)
    return target_list

def keep_by_file_number(source_list, sub_list):
    target_list = []
    sub_list = check_and_change_to_list(sub_list)
    for y in source_list:
        file_number = y.split('/')[-1].split('.')[0].split('_')[1]
        if file_number in sub_list:
            target_list.append(y)
    return target_list

def remove_by_file_number(source_list, sub_list):
    target_list = []
    sub_list = check_and_change_to_list(sub_list)
    for y in source_list:
        file_number = y.split('/')[-1].split('.')[0].split('_')[1]
        if file_number not in sub_list:
            target_list.append(y)
    return target_list

def keep_by_min_max_file_number(source_list, min_file_number, max_file_number):
    min_minus_1 = int(min_file_number) - 1
    max_plus_1  = int(max_file_number) + 1
    target_list = []
    for y in source_list:
        file_number = int(y.split('/')[-1].split('.')[0].split('_')[1])
        if (file_number > min_minus_1) and (file_number < max_plus_1):
            target_list.append(y)
    return target_list

def prepare_file_path(file_dir, new_dir_switch=True, script_name=''):
    if not os.path.exists(file_dir) and new_dir_switch:
        os.makedirs(file_dir)
    if len(script_name) > 0:
        target_script_name = os.path.join(file_dir, os.path.basename(script_name))
        if os.path.exists(target_script_name):
            for i in range(100000):
                temp_script_name = target_script_name+'_'+str(i)
                if not os.path.exists(temp_script_name):
                    os.rename(target_script_name, temp_script_name)
                    break
        shutil.copyfile(script_name, target_script_name)

def prepare_file_path_list(file_id_list, file_dir, file_extension, new_dir_switch=True):
    prepare_file_path(file_dir, new_dir_switch)
    file_name_list = []
    for file_id in file_id_list:
        file_name = file_dir + '/' + file_id + file_extension
        file_name_list.append(file_name)
    return  file_name_list

def clean_directory(target_dir):
    # remove all files under this directory
    # but keep this directory
    file_list = os.listdir(target_dir)
    for file_name in file_list:
        full_file_name = os.path.join(target_dir, file_name)
        os.remove(full_file_name)

def copy_to_scratch(cfg, file_id_list):
    for feat_name in cfg.nn_features:
        if feat_name == 'wav':
            # nn_resil_norm_file_list         = prepare_file_path_list(file_id_list, cfg.nn_feat_resil_norm_dirs[feat_name], '.mu.'+feat_name)
            # nn_resil_norm_file_list_scratch = prepare_file_path_list(file_id_list, cfg.nn_feat_scratch_dirs[feat_name], '.mu.'+feat_name)
            nn_resil_norm_file_list         = prepare_file_path_list(file_id_list, cfg.nn_feat_resil_norm_dirs[feat_name], '.'+feat_name)
            nn_resil_norm_file_list_scratch = prepare_file_path_list(file_id_list, cfg.nn_feat_scratch_dirs[feat_name], '.'+feat_name)
        else:
            nn_resil_norm_file_list         = prepare_file_path_list(file_id_list, cfg.nn_feat_resil_norm_dirs[feat_name], '.'+feat_name)
            nn_resil_norm_file_list_scratch = prepare_file_path_list(file_id_list, cfg.nn_feat_scratch_dirs[feat_name], '.'+feat_name)

        for x, y in zip(nn_resil_norm_file_list, nn_resil_norm_file_list_scratch):
            shutil.copyfile(x, y)

def check_within_range(in_data, value_max, value_min):
    temp_max = max(in_data)
    temp_min = min(in_data)
    assert temp_max <= value_max
    assert temp_min >= value_min

def reduce_silence_reaper_output(cfg, reaper_output_file='/home/dawna/tts/mw545/Data/Data_Voicebank_48kHz_Pitch/p7_345.used.pm', label_align_file='/data/vectra2/tts/mw545/Data/data_voicebank/label_state_align/p7_345.lab', out_file='/home/dawna/tts/mw545/Data/Data_Voicebank_48kHz_Pitch_Resil/p7_345.pm', silence_pattern=['*-#+*']):
    logger = make_logger("reduce_silence_reaper")
    from frontend.silence_reducer_keep_sil import SilenceReducer
    remover = SilenceReducer(n_cmp = 1, silence_pattern = silence_pattern)
    nonsilence_indices = remover.load_alignment(label_align_file)
    start_time = float(nonsilence_indices[0]) / float(cfg.frame_sr)
    end_time   = float(nonsilence_indices[-1]+1.) / float(cfg.frame_sr)
    with open(reaper_output_file, 'r') as f:
        file_lines = f.readlines()
    with open(out_file, 'w') as f:
        for l in file_lines:
            x = l.strip().split(' ')
            # Content lines should have 3 values
            # Time stamp, vuv, F0 value
            if len(x) == 3:
                t = float(x[0])
                if (t >= start_time) and (t <= end_time):
                    t_new = t - start_time
                    f.write(str(t_new)+' '+x[1]+' '+x[2]+'\n')

def reduce_silence_reaper_output_list(cfg, file_id_list, reaper_output_dir, label_align_dir, out_dir, reaper_output_ext='.used.pm', label_align_ext='.lab', out_ext='.pm', silence_pattern=['*-#+*']):
    for file_id in file_id_list:
        reaper_output_file = os.path.join(reaper_output_dir, file_id + reaper_output_ext)
        label_align_file   = os.path.join(label_align_dir, file_id + label_align_ext)
        out_file           = os.path.join(out_dir, file_id + out_ext)
        reduce_silence_reaper_output(cfg, reaper_output_file, label_align_file, out_file, silence_pattern)

def reduce_silence_list(cfg, feature_dim, in_file_list, label_align_file_list, out_file_list, silence_pattern=['*-#+*']):
    logger = make_logger("reduce_silence_list")
    from frontend.silence_reducer_keep_sil import SilenceReducer
    remover = SilenceReducer(n_cmp = feature_dim, silence_pattern = silence_pattern)
    remover.reduce_silence(in_file_list, label_align_file_list, out_file_list, frames_silence_to_keep=cfg.frames_silence_to_keep,sil_pad=cfg.sil_pad)

def compute_min_max_normaliser(feature_dim, in_file_list, norm_file, min_value=0.01, max_value=0.99):
    logger = make_logger("compute_min_max_normaliser")
    from frontend.min_max_norm import MinMaxNormalisation
    min_max_normaliser = MinMaxNormalisation(feature_dimension=feature_dim, min_value=min_value, max_value=max_value)
    min_max_normaliser.find_min_max_values(in_file_list)
    min_vector = min_max_normaliser.min_vector
    max_vector = min_max_normaliser.max_vector

    norm_info = numpy.concatenate((min_vector, max_vector), axis=0)
    norm_info = numpy.array(norm_info, 'float32')
    fid = open(norm_file, 'wb')
    norm_info.tofile(fid)
    fid.close()
    logger.info('saved %s vectors to %s' %(min_vector.size, norm_file))

def perform_min_max_normlisation_list(feature_dim, norm_file, in_file_list, out_file_list, min_value=0.01, max_value=0.99):
    logger = make_logger("perform_min_max_normlisation_list")
    from frontend.min_max_norm import MinMaxNormalisation
    min_max_normaliser = MinMaxNormalisation(feature_dimension=feature_dim, min_value=min_value, max_value=max_value)
    if norm_file is None:
        compute_min_max_normaliser(feature_dim, in_file_list, norm_file, min_value, max_value)
    min_max_normaliser.load_min_max_values(norm_file)
    min_max_normaliser.normalise_data(in_file_list, out_file_list)

def perform_min_max_denormlisation_list(feature_dim, norm_file, in_file_list, out_file_list, min_value=0.01, max_value=0.99):
    from frontend.min_max_norm import MinMaxNormalisation
    min_max_normaliser = MinMaxNormalisation(feature_dimension=feature_dim, min_value=min_value, max_value=max_value)
    min_max_normaliser.load_min_max_values(norm_file)
    min_max_normaliser.denormalise_data(in_file_list, out_file_list)

def compute_mean_var_normaliser(feature_dim, in_file_list, norm_file, var_file_dict=None, acoustic_out_dimension_dict=None):
    logger = make_logger("compute_mean_var_normaliser")
    from frontend.mean_variance_norm import MeanVarianceNorm
    mean_var_normaliser = MeanVarianceNorm(feature_dimension=feature_dim)
    mean_vector = mean_var_normaliser.compute_mean(in_file_list, 0, feature_dim)
    std_vector  = mean_var_normaliser.compute_std(in_file_list, mean_vector, 0, feature_dim)

    norm_info = numpy.concatenate((mean_vector, std_vector), axis=0)
    norm_info = numpy.array(norm_info, 'float32')
    fid = open(norm_file, 'wb')
    norm_info.tofile(fid)
    fid.close()
    logger.info('saved %s vectors to %s' %('MVN', norm_file))

    # Store variance for each feature separately
    # Store Variance instead of STD
    if var_file_dict:
        feature_index = 0
        for feature_name in var_file_dict.keys():
            feature_std_vector = numpy.array(std_vector[:,feature_index:feature_index+acoustic_out_dimension_dict[feature_name]], 'float32')
            fid = open(var_file_dict[feature_name], 'w')
            feature_var_vector = feature_std_vector ** 2
            feature_var_vector.tofile(fid)
            fid.close()
            logger.info('saved %s variance vector to %s' %(feature_name, var_file_dict[feature_name]))
            feature_index += acoustic_out_dimension_dict[feature_name]

def perform_mean_var_normlisation_list(feature_dim, norm_file, in_file_list, out_file_list):
    from frontend.mean_variance_norm import MeanVarianceNorm
    mean_var_normaliser = MeanVarianceNorm(feature_dimension=feature_dim)
    if norm_file is None:
        compute_mean_var_normaliser(feature_dim, in_file_list, norm_file, var_file_dict=None, acoustic_out_dimension_dict=None)
    mean_var_normaliser.load_mean_var_values(norm_file)
    mean_var_normaliser.feature_normalisation(in_file_list, out_file_list)

def perform_mean_var_denormlisation_list(feature_dim, norm_file, in_file_list, out_file_list):
    from frontend.mean_variance_norm import MeanVarianceNorm
    mean_var_normaliser = MeanVarianceNorm(feature_dimension=feature_dim)
    mean_var_normaliser.load_mean_var_values(norm_file)
    mean_var_normaliser.feature_denormalisation(in_file_list, out_file_list, mean_var_normaliser.mean_vector, mean_var_normaliser.std_vector)

def label_align_2_binary_label_list(cfg, in_label_align_file_list, out_binary_label_file_list):
    logger = make_logger("label_align_2_binary_label_list")
    from frontend.label_normalisation import HTSLabelNormalisation
    # Make label_normaliser first
    label_normaliser = HTSLabelNormalisation(question_file_name=cfg.question_file_name)
    lab_dim = label_normaliser.dimension
    logger.info('Input label dimension is %d' % lab_dim)
    cfg.lab_dim = lab_dim
    label_normaliser.perform_normalisation(in_label_align_file_list, out_binary_label_file_list)
    # Single file function: label_normaliser.extract_linguistic_features(in_file, out_file)

def acoustic_2_cmp_list(cfg, in_file_list_dict, out_cmp_file_list):
    ''' Computes delta and ddelta, and stack to form cmp '''
    logger = make_logger("acoustic_2_cmp_list")
    logger.info('creating acoustic (output) features')
    from frontend.acoustic_composition import AcousticComposition
    delta_win = cfg.delta_win #[-0.5, 0.0, 0.5]
    acc_win = cfg.acc_win         #[1.0, -2.0, 1.0]
    acoustic_worker = AcousticComposition(delta_win = delta_win, acc_win = acc_win)
    acoustic_worker.prepare_nn_data(in_file_list_dict, out_cmp_file_list, cfg.acoustic_in_dimension_dict, cfg.acoustic_out_dimension_dict)

def cmp_2_acoustic_list(cfg, in_file_list, out_dir, do_MLPG=False):
    from frontend.parameter_generation_new import ParameterGeneration
    generator = ParameterGeneration(gen_wav_features = cfg.acoustic_features)
    generator.acoustic_decomposition(in_file_list, cfg.nn_feature_dims['cmp'], cfg.acoustic_out_dimension_dict, cfg.acoustic_file_ext_dict, cfg.var_file_dict, do_MLPG, out_dir)

def wav_2_wav_cmp(in_file_name, out_file_name, label_rate=200):
    from io_funcs.binary_io import BinaryIOCollection
    ''' Strip waveform header first '''
    ''' Make "cmp" style file, by reshaping waveform '''
    # find frame number, remove residual to make whole frames, quantise
    sr, data = scipy.io.wavfile.read(in_file_name)
    dim = sr / label_rate
    assert len(data.shape) == 1
    num_frames = int(data.shape[0] / dim)
    # remove residual samples i.e. less than a frame
    num_samples = dim * num_frames
    new_data = numpy.array(data[:num_samples], dtype='float32')
    BIC = BinaryIOCollection()
    BIC.array_to_binary_file(new_data, out_file_name)
    return sr

def wav_2_wav_cmp_list(in_file_list, out_file_list, label_rate=200):
    for (in_file_name, out_file_name) in zip(in_file_list, out_file_list):
        sr = wav_2_wav_cmp(in_file_name, out_file_name, label_rate)
    return sr

def wav_cmp_2_wav(in_file_name, out_file_name, sr=16000):
    from io_funcs.binary_io import BinaryIOCollection
    BIC = BinaryIOCollection()
    cmp_data = BIC.load_binary_file(in_file_name, 1)
    cmp_data = numpy.array(cmp_data, dtype='int16')
    scipy.io.wavfile.write(out_file_name, sr, cmp_data)

def make_wav_min_max_normaliser(norm_file, feature_dim, wav_max=32768, wav_min=-32768):
    logger = make_logger("make_wav_min_max_normaliser")
    min_max_vector = numpy.zeros(feature_dim * 2)
    min_max_vector[0:feature_dim] = wav_min
    min_max_vector[feature_dim:]  = wav_max
    min_max_vector = numpy.array(min_max_vector, dtype='float32')
    fid = open(norm_file, 'wb')
    min_max_vector.tofile(fid)
    fid.close()
    logger.info('saved %s vectors to %s' %(feature_dim, norm_file))

def perform_mu_law(in_file_name, out_file_name, mu_value=255.):
    from io_funcs.binary_io import BinaryIOCollection
    BIC = BinaryIOCollection()
    ori_data = BIC.load_binary_file(in_file_name, 1)
    # apply mu-law (ITU-T, 1988)
    mu_data = numpy.sign(ori_data) * numpy.log(1.+mu_value*numpy.abs(ori_data)) / numpy.log(1.+mu_value)
    check_within_range(mu_data, 1, -1)
    BIC.array_to_binary_file(mu_data, out_file_name)

def perform_mu_law_list(in_file_list, out_file_list, mu_value=255.):
    for (in_file_name, out_file_name) in zip(in_file_list, out_file_list):
        perform_mu_law(in_file_name, out_file_name, mu_value)

def invert_mu_law(in_file_name, out_file_name, mu_value=255.):
    from io_funcs.binary_io import BinaryIOCollection
    BIC = BinaryIOCollection()
    mu_data = BIC.load_binary_file(in_file_name, 1)
    # apply mu-law (ITU-T, 1988)
    ori_data = numpy.sign(mu_data) * (1./mu_value) * ( numpy.power((1.+mu_value), numpy.abs(mu_data)) - 1.)
    check_within_range(ori_data, 1, -1)
    BIC.array_to_binary_file(ori_data, out_file_name)

def wav_2_acoustic(in_file_name, out_file_dict, acoustic_in_dimension_dict, verbose_level=0):
    from pulsemodel.analysis import analysisf
    analysisf(in_file_name,
        shift=0.005, dftlen=4096,
        finf0txt=None, f0_min=60, f0_max=600, ff0=out_file_dict['lf0'], f0_log=True, finf0bin=None,
        fspec=out_file_dict['mgc'], spec_mceporder=acoustic_in_dimension_dict['mgc']-1, spec_fwceporder=None, spec_nbfwbnds=None,
        fpdd=None, pdd_mceporder=None, fnm=out_file_dict['bap'], nm_nbfwbnds=acoustic_in_dimension_dict['bap'],
        verbose=verbose_level)

def acoustic_2_wav(in_file_dict, synthesis_wav_sr, out_file_name, verbose_level=0):
    from pulsemodel.synthesis import synthesizef
    synthesizef(synthesis_wav_sr, shift=0.005, dftlen=4096, 
        ff0=None, flf0=in_file_dict['lf0'], 
        fspec=None, ffwlspec=None, ffwcep=None, fmcep=in_file_dict['mgc'], 
        fnm=None, ffwnm=in_file_dict['bap'], nm_cont=False, fpdd=None, fmpdd=None, 
        fsyn=out_file_name, verbose=verbose_level)

def acoustic_2_wav_cfg(cfg, in_file_dict, out_file_name, verbose_level=0):
    acoustic_2_wav(in_file_dict, cfg.synthesis_wav_sr, out_file_name, verbose_level=0)

def wav_2_acoustic_cfg(cfg, in_file_name, out_file_dict, verbose_level=0):
    wav_2_acoustic(in_file_name, out_file_dict, cfg.acoustic_in_dimension_dict, verbose_level=0)

def wav_2_norm_cmp(cfg, wav_file, target_dir, lab_file, cmp_norm_file):
    prepare_file_path(target_dir)
    file_name = os.path.basename(wav_file).split('.')[0]
    ''' 1. wav to acoustic '''
    acoustic_file_dict = {}
    for feat_name in cfg.acoustic_features:
        acoustic_file_dict[feat_name] = os.path.join(target_dir, file_name+cfg.acoustic_file_ext_dict[feat_name])
    wav_2_acoustic(wav_file, acoustic_file_dict, cfg.acoustic_in_dimension_dict)

    ''' 2. acoustic to cmp '''
    acoustic_file_list_dict = {}
    for feat_name in cfg.acoustic_features:
        acoustic_file_list_dict[feat_name] = prepare_file_path_list([file_name], target_dir, cfg.acoustic_file_ext_dict[feat_name])
    cmp_file_list = prepare_file_path_list([file_name], target_dir, '.cmp')
    acoustic_2_cmp_list(cfg, acoustic_file_list_dict, cmp_file_list)

    ''' 3. cmp to resil_cmp (requires label file) '''
    from modules_2 import resil_nn_file_list
    feat_name = 'cmp'
    label_align_file_list = [lab_file]
    cmp_resil_file_list = prepare_file_path_list([file_name], target_dir, '.cmp.resil')
    reduce_silence_list(cfg, cfg.nn_feature_dims[feat_name], cmp_file_list, label_align_file_list, cmp_resil_file_list)

    ''' 4. resil_cmp to norm_cmp (requires cmp_norm_info file) '''
    cmp_resil_norm_file_list = prepare_file_path_list([file_name], target_dir, '.cmp.resil.norm')
    perform_mean_var_normlisation_list(cfg.nn_feature_dims[feat_name], cfg.nn_feat_resil_norm_files[feat_name], cmp_resil_file_list, cmp_resil_norm_file_list)

def norm_cmp_2_wav(cfg, cmp_resil_norm_file, target_dir, cmp_norm_file):
    prepare_file_path(target_dir)
    file_name = os.path.basename(cmp_resil_norm_file).split('.')[0]
    feat_name = 'cmp'
    ''' 1. norm_cmp to resil_cmp (requires cmp_norm_info file) '''
    cmp_resil_norm_file_list = [cmp_resil_norm_file]
    cmp_resil_file_list      = prepare_file_path_list([file_name], target_dir, '.cmp')
    perform_mean_var_denormlisation_list(cfg.nn_feature_dims[feat_name], cmp_norm_file, cmp_resil_norm_file_list, cmp_resil_file_list)

    ''' 2. cmp to acoustic '''
    cmp_2_acoustic_list(cfg, cmp_resil_file_list, target_dir, do_MLPG=False)

    ''' 3. acoustic to wav '''
    acoustic_file_dict = {}
    for feat_name in cfg.acoustic_features:
        acoustic_file_dict[feat_name] = os.path.join(target_dir, file_name+cfg.acoustic_file_ext_dict[feat_name])
    wav_file = os.path.join(target_dir, file_name+'.wav')
    acoustic_2_wav(acoustic_file_dict, cfg.synthesis_wav_sr, wav_file)


def cal_mcd_dir(cfg, ref_data_dir, gen_denorm_no_sil_dir, file_id_list):
    logger = make_logger("cal_mcd_dir")
    logger.info('calculating MCD')
    from utils.compute_distortion import IndividualDistortionComp
    calculator = IndividualDistortionComp()

    error_dict = {}
    for tvt in file_id_list.keys():
        for feat_name in cfg.acoustic_features:
            if feat_name == 'lf0':
                f0_mse, f0_corr, vuv_error  = calculator.compute_distortion(file_id_list[tvt], ref_data_dir, gen_denorm_no_sil_dir, cfg.acoustic_file_ext_dict[feat_name], cfg.acoustic_in_dimension_dict[feat_name])
            else:
                error_dict[(tvt, feat_name)] = calculator.compute_distortion(file_id_list[tvt], ref_data_dir, gen_denorm_no_sil_dir, cfg.acoustic_file_ext_dict[feat_name], cfg.acoustic_in_dimension_dict[feat_name])
        if cfg.acoustic_feature_type == 'STRAIGHT':
            error_dict[(tvt, 'mgc')] = error_dict[(tvt, 'mgc')] * (10 /numpy.log(10)) * numpy.sqrt(2.0)    ##MCD
            error_dict[(tvt, 'bap')] = error_dict[(tvt, 'bap')] / 10.0    ##Cassia's bap is computed from 10*log|S(w)|. if use HTS/SPTK style, do the same as MGC
        logger.info('%s: DNN -- MCD: %.3f dB; BAP: %.3f dB; F0:- RMSE: %.3f Hz; CORR: %.3f; VUV: %.3f%%' \
                %(tvt, error_dict[(tvt, 'mgc')], error_dict[(tvt, 'bap')], f0_mse, f0_corr, vuv_error*100.))


    if cfg.acoustic_in_dimension_dict.has_key('mgc'):
        valid_spectral_distortion = calculator.compute_distortion(valid_file_id_list, ref_data_dir, gen_denorm_no_sil_dir, cfg.acoustic_file_ext_dict['mgc'], cfg.acoustic_in_dimension_dict['mgc'])
        test_spectral_distortion  = calculator.compute_distortion(test_file_id_list , ref_data_dir, gen_denorm_no_sil_dir, cfg.acoustic_file_ext_dict['mgc'], cfg.acoustic_in_dimension_dict['mgc'])
        
    if cfg.acoustic_in_dimension_dict.has_key('bap'):
        valid_bap_mse        = calculator.compute_distortion(valid_file_id_list, ref_data_dir, gen_denorm_no_sil_dir, cfg.bap_ext, cfg.bap_dim)
        test_bap_mse         = calculator.compute_distortion(test_file_id_list , ref_data_dir, gen_denorm_no_sil_dir, cfg.bap_ext, cfg.bap_dim)

    if cfg.acoustic_feature_type == 'STRAIGHT':
        valid_spectral_distortion *= (10 /numpy.log(10)) * numpy.sqrt(2.0)    ##MCD
        test_spectral_distortion  *= (10 /numpy.log(10)) * numpy.sqrt(2.0)    ##MCD
        valid_bap_mse = valid_bap_mse / 10.0    ##Cassia's bap is computed from 10*log|S(w)|. if use HTS/SPTK style, do the same as MGC
        test_bap_mse  = test_bap_mse / 10.0    ##Cassia's bap is computed from 10*log|S(w)|. if use HTS/SPTK style, do the same as MGC
            
    if cfg.acoustic_in_dimension_dict.has_key('lf0'):
        valid_f0_mse, valid_f0_corr, valid_vuv_error  = calculator.compute_distortion(valid_file_id_list, ref_data_dir, gen_denorm_no_sil_dir, cfg.lf0_ext, cfg.lf0_dim)
        test_f0_mse , test_f0_corr, test_vuv_error    = calculator.compute_distortion(test_file_id_list , ref_data_dir, gen_denorm_no_sil_dir, cfg.lf0_ext, cfg.lf0_dim)

    logger.info('Valid: DNN -- MCD: %.3f dB; BAP: %.3f dB; F0:- RMSE: %.3f Hz; CORR: %.3f; VUV: %.3f%%' \
                %(valid_spectral_distortion, valid_bap_mse, valid_f0_mse, valid_f0_corr, valid_vuv_error*100.))
    logger.info('Test : DNN -- MCD: %.3f dB; BAP: %.3f dB; F0:- RMSE: %.3f Hz; CORR: %.3f; VUV: %.3f%%' \
                %(test_spectral_distortion , test_bap_mse , test_f0_mse , test_f0_corr, test_vuv_error*100.))










def reduce_silence(cfg, feature_dim, in_file, label_align_file, out_file, silence_pattern=['*-#+*']):
    from frontend.silence_reducer_keep_sil import SilenceReducer
    remover = SilenceReducer(n_cmp = feature_dim, silence_pattern = silence_pattern)
    remover.reduce_silence([in_file], [label_align_file], [out_file], frames_silence_to_keep=cfg.frames_silence_to_keep,sil_pad=cfg.sil_pad)

def reduce_silence_list_parallel(cfg, feature_dim, in_file_list, label_align_file_list, out_file_list, silence_pattern=['*-#+*'], num_threads=20):
    logger = make_logger("reduce_silence_list_parallel")
    # from multiprocessing import Pool
    from pathos.multiprocessing import ProcessingPool as Pool
    def reduce_silence_list_wrapper(args):
        cfg, feature_dim, in_file_list, label_align_file_list, out_file_list, silence_pattern = args
        reduce_silence_list(cfg, feature_dim, in_file_list, label_align_file_list, out_file_list)

    args_list = []
    index_list = find_index_list_for_parallel(num_threads, in_file_list)
    for i in range(num_threads):
        # Make sub-lists
        x = index_list[i][0]
        y = index_list[i][1]
        args = (cfg, feature_dim, in_file_list[x: y], label_align_file_list[x: y], out_file_list[x: y])
        args_list.append(args)
    with Pool(num_threads) as p:
        p.map(reduce_silence_list_wrapper, args_list)
