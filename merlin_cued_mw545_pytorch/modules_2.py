# modules_2.py

import os, sys, pickle, time, shutil, logging
import math, numpy, scipy
from io_funcs.binary_io import BinaryIOCollection
io_fun = BinaryIOCollection()

from modules import make_logger, read_file_list, prepare_file_path, prepare_file_path_list, make_held_out_file_number, copy_to_scratch
from modules import keep_by_speaker, remove_by_speaker, keep_by_file_number, remove_by_file_number


def compute_feat_dim(model_cfg, cfg, feat_list):
    feat_dim = 0
    feat_index = []
    if 'wav' in feat_list:
        feat_dim = 1
        feat_index = [0]
    elif 'lab' in feat_list:
        feat_dim = cfg.nn_feature_dims['lab']
        feat_index = range(cfg.nn_feature_dims['lab'])
    else:
        if model_cfg.cmp_use_delta:
            for feat in feat_list:
                feat_dim += cfg.acoustic_in_dimension_dict[feat] * 3
                feat_index.extend(range(cfg.acoustic_start_index[feat], cfg.acoustic_start_index[feat] + cfg.acoustic_in_dimension_dict[feat] * 3))
        else:
            for feat in feat_list:
                feat_dim += cfg.acoustic_in_dimension_dict[feat]
                feat_index.extend(range(cfg.acoustic_start_index[feat], cfg.acoustic_start_index[feat] + cfg.acoustic_in_dimension_dict[feat]))
    
    feat_index = numpy.array(feat_index)
    return feat_dim, feat_index
       
def log_class_attri(cfg, logger, except_list=['feat_index']):
    attri_list = vars(cfg)
    for i in attri_list.keys():
        if i not in except_list:
            logger.info(i+ ' is '+str(attri_list[i]))

def resil_nn_file_list(feat_name, cfg, file_id_list, nn_file_list={}, nn_resil_file_list={}):
    from modules import reduce_silence_list
    try:    nn_file_list[feat_name]
    except: nn_file_list[feat_name] = prepare_file_path_list(file_id_list, cfg.nn_feat_dirs[feat_name], '.'+feat_name)
    nn_resil_file_list[feat_name]= prepare_file_path_list(file_id_list, cfg.nn_feat_resil_dirs[feat_name], '.'+feat_name)
    label_align_file_list  = prepare_file_path_list(file_id_list, cfg.lab_dir, '.lab')
    reduce_silence_list(cfg, cfg.nn_feature_dims[feat_name], nn_file_list[feat_name], label_align_file_list, nn_resil_file_list[feat_name])

def norm_nn_file_list(feat_name, cfg, file_id_list, nn_resil_file_list={}, nn_resil_norm_file_list={}, compute_normaliser=True, norm_type='MinMax'):
    held_out_file_number = cfg.held_out_file_number
    try:    nn_resil_file_list[feat_name]
    except: nn_resil_file_list[feat_name] = prepare_file_path_list(file_id_list, cfg.nn_feat_resil_dirs[feat_name], '.'+feat_name)
    try:    nn_resil_norm_file_list[feat_name]
    except: nn_resil_norm_file_list[feat_name] = prepare_file_path_list(file_id_list, cfg.nn_feat_resil_norm_dirs[feat_name], '.'+feat_name)
    
    if compute_normaliser:
        nn_resil_file_list[feat_name+'_train'] = keep_by_speaker(nn_resil_file_list[feat_name], cfg.train_speaker_list)
        nn_resil_file_list[feat_name+'_train'] = remove_by_file_number(nn_resil_file_list[feat_name+'_train'], held_out_file_number)
        if norm_type == 'MinMax':
            if feat_name == 'wav':
                from modules import make_wav_min_max_normaliser
                make_wav_min_max_normaliser(cfg.nn_feat_resil_norm_files[feat_name], cfg.nn_feature_dims[feat_name])
            else:    
                from modules import compute_min_max_normaliser
                compute_min_max_normaliser(cfg.nn_feature_dims[feat_name], nn_resil_file_list[feat_name+'_train'], cfg.nn_feat_resil_norm_files[feat_name], min_value=0.01, max_value=0.99)
        elif norm_type == 'MeanVar':
            from modules import compute_mean_var_normaliser
            if feat_name == 'cmp':
                var_file_dict = cfg.var_file_dict
                acoustic_out_dimension_dict = cfg.acoustic_out_dimension_dict
            else:
                var_file_dict = None
                acoustic_out_dimension_dict = None
            compute_mean_var_normaliser(cfg.nn_feature_dims[feat_name], nn_resil_file_list[feat_name+'_train'], cfg.nn_feat_resil_norm_files[feat_name], var_file_dict, acoustic_out_dimension_dict)
    else:
        print("Using norm file " + cfg.nn_feat_resil_norm_files[feat_name])

    if norm_type == 'MinMax':
        from modules import perform_min_max_normlisation_list
        if feat_name == 'wav':
            perform_min_max_normlisation_list(cfg.nn_feature_dims[feat_name], cfg.nn_feat_resil_norm_files[feat_name], nn_resil_file_list[feat_name], nn_resil_norm_file_list[feat_name], min_value=-3.99, max_value=3.99)
        else:
            perform_min_max_normlisation_list(cfg.nn_feature_dims[feat_name], cfg.nn_feat_resil_norm_files[feat_name], nn_resil_file_list[feat_name], nn_resil_norm_file_list[feat_name], min_value=0.01, max_value=0.99)
    elif norm_type == 'MeanVar':
        from modules import perform_mean_var_normlisation_list
        perform_mean_var_normlisation_list(cfg.nn_feature_dims[feat_name], cfg.nn_feat_resil_norm_files[feat_name], nn_resil_file_list[feat_name], nn_resil_norm_file_list[feat_name])

def get_utters_from_binary(file_list, num_files, min_file_len, feat_dim):
    # Draw n files from a list of full file paths
    # TODO: merge with the method below later; this method only takes one feature, 
    # not a list of features corresponding to same file
    num_to_draw = num_files
    final_file_list = []
    final_len_list  = []
    while num_to_draw > 0:
        temp_file_list = numpy.random.choice(file_list, num_to_draw)
        # Check if they are long enough
        new_num_to_draw = 0
        for i in range(num_to_draw):
            features, frame_number = io_fun.load_binary_file_frame(temp_file_list[i], feat_dim)
            if frame_number < min_file_len:
                new_num_to_draw += 1
            else:
                final_file_list.append(features)
                final_len_list.append(frame_number)
        num_to_draw = new_num_to_draw
    # assert len(final_file_list) == num_files
    return (final_file_list, final_len_list)

def get_utters_from_binary_dict(spk_num_utter, file_list, file_dir_dict, feat_name_list, feat_dim_list, min_file_len=0, random_seed=None):
    if random_seed is not None:
        numpy.random.seed(random_seed)
    file_name_list = []
    speaker_utter_len_list = []
    speaker_utter_list = {}
    # speaker_utter_list is first sorted by feature name, then a list of utterances
    for feat_name in feat_name_list:
        speaker_utter_list[feat_name] = []
    utter_counter = 0
    while utter_counter < spk_num_utter:
        file_name, new_utter_len, feat_file_list = get_one_utter_from_binary_dict(file_list, file_dir_dict, feat_name_list, feat_dim_list)
        if new_utter_len >= min_file_len:
            utter_counter += 1
            file_name_list.append(file_name)
            speaker_utter_len_list.append(new_utter_len)
            for feat_name in feat_name_list:
                speaker_utter_list[feat_name].append(feat_file_list[feat_name])
    return file_name_list, speaker_utter_len_list, speaker_utter_list        

def get_one_utter_from_binary_dict(file_list, file_dir_dict, feat_name_list, feat_dim_list):
    # Draw a random file from file_list
    file_name = numpy.random.choice(file_list)
    frame_number, feature_files = get_one_utter_by_name(file_name, file_dir_dict, feat_name_list, feat_dim_list)
    return file_name, frame_number, feature_files

def get_one_utter_by_name(file_name, file_dir_dict, feat_name_list, feat_dim_list):
    # Given file_name and a list of directories, extension names and extension dimensions
    # Return the file length and the binary files
    feature_files = {}
    len_list       = []
    for feat_name, feat_dim in zip(feat_name_list, feat_dim_list):
        full_file_name = os.path.join(file_dir_dict[feat_name], file_name+'.'+feat_name)
        features, frame_number = io_fun.load_binary_file_frame(full_file_name, feat_dim)
        len_list.append(frame_number)
        feature_files[feat_name] = features
    # Check for length consistency; and use shortest
    min_len = 30000
    max_len = -1
    for l in len_list:
        if l > max_len:  max_len = l
        if l < min_len:  min_len = l
    # assert max_len - min_len < 10, full_file_name
    # Use shortest
    for feat_name in feat_name_list:
        feature_files[feat_name] = feature_files[feat_name][:min_len, :]
    return min_len, feature_files

def shift_distance(y, d, l):
    if len(y.shape) == 4:
        S = y.shape[0]
        B = y.shape[1]
        T = y.shape[2]
        D = y.shape[3]
        y_td = y.reshape(S,B,T*D)
        y_td_temp = y_td[:,:,d:d+l*D]
        y_temp = y_td_temp.reshape(S,B,l,D)
    elif len(y.shape) == 3:
        B = y.shape[0]
        T = y.shape[1]
        D = y.shape[2]
        y_td = y.reshape(B,T*D)
        y_td_temp = y_td[:,d:d+l*D]
        y_temp = y_td_temp.reshape(B,l,D)
    return y_temp

def compute_cosine_distance(lambda_1, lambda_2):
    d = 0.
    S = lambda_1.shape[0]
    B = lambda_1.shape[1]
    D = lambda_1.shape[2]
    for i in range(S):
        for j in range(B):
            d += scipy.spatial.distance.cosine(lambda_1[i,j], lambda_2[i,j])
    return d

def get_file_id_from_file_name(file_name):
    file_id = file_name.split('/')[-1].split('.')[0]
    return file_id

def linear_interpolate(lf0_data, t_space, t_start, t_end):
    # t_mid = float(t_start + t_end) / 2.
    t_mid = float(t_start) / 2.
    n = int(t_mid / t_space)
    r = float(t_mid / t_space) - n
    lf0_mid = lf0_data[n] * (1-r) + lf0_data[n+1] * r
    return lf0_mid

def find_pitch_time(pitch_loc_list, t_start, t_end):
    for t in pitch_loc_list:
        if t > t_start:
            if t <= t_end:
                return t
        elif t > t_end:
            return 0
    return 0


def count_male_female_class_errors(total_wrong_class, male_speaker_list):
    wrong_list = {'mm':0, 'ff':0, 'mf':0, 'fm':0}
    for (x,y) in total_wrong_class.keys():
        count_temp = total_wrong_class[(x,y)]
        list_temp = ['f','f']
        if x in male_speaker_list:
            list_temp[0] = 'm'
        if y in male_speaker_list:
            list_temp[1] = 'm'
        str_temp = list_temp[0] + list_temp[1]
        wrong_list[str_temp] += count_temp
    return wrong_list