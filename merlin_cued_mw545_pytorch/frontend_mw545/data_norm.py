# data_norm.py

import os, sys, pickle, time, shutil, logging, copy
import math, numpy
numpy.random.seed(545)

from frontend_mw545.modules import make_logger, read_file_list, prepare_file_path_list
from frontend_mw545.data_io import Data_File_IO

##########################
# TODO: not finished yet #
# Need 2 more: lab, cmp  #
##########################

###########
# Vocoder #
###########

class Data_Mean_Var_Normaliser(object):
    """docstring for Data_Mean_Var_Normaliser"""
    def __init__(self, cfg=None):
        super(Data_Mean_Var_Normaliser, self).__init__()
        self.logger = make_logger("DataNorm")

        self.cfg = cfg
        self.DF_IO = Data_File_IO(cfg)

    def load_mean_std_values(self, norm_info_file=None, feat_dim=None):
        if norm_info_file is None:
            norm_info_file = self.cfg.nn_feat_resil_norm_files['cmp']
        if feat_dim is None:
            feat_dim = self.cfg.nn_feature_dims['cmp']

        mean_std_vector, frame_number = self.DF_IO.load_data_file_frame(norm_info_file, 1)
        assert frame_number == feat_dim * 2

        mean_std_vector = numpy.reshape(mean_std_vector, (-1))
        self.mean_vector = mean_std_vector[:feat_dim]
        self.std_vector  = mean_std_vector[feat_dim:]

        self.logger.info('Loaded mean std values from %s' % norm_info_file)

    def save_mean_std_values(self, norm_info_file=None, feat_dim=None):
        if norm_info_file is None:
            norm_info_file = self.cfg.nn_feat_resil_norm_files['cmp']
        if feat_dim is None:
            feat_dim = self.cfg.nn_feature_dims['cmp']

        mean_std_vector = numpy.zeros(feat_dim*2)
        mean_std_vector[:feat_dim] = self.mean_vector
        mean_std_vector[feat_dim:] = self.std_vector

        self.DF_IO.save_data_file(mean_std_vector, norm_info_file)

        self.logger.info('Saved mean_std_values to %s' % norm_info_file)

    def norm_file(self, in_file_name, out_file_name, norm_info_file=None, feat_name='cmp', feat_dim=None):
        '''
        if norm_info_file is None: self.mean_vector and self.std_vector already loaded
        if feat_dim is None: extract from cfg
        '''
        if norm_info_file is not None:
            self.load_mean_std_values(norm_info_file, feat_dim)
        if feat_dim is None:
            feat_dim = self.cfg.nn_feature_dims[feat_name]

        # Floor std
        self.std_vector[self.std_vector<=0] = 1.

        in_data, frame_number = self.DF_IO.load_data_file_frame(in_file_name, feat_dim)

        mean_matrix = numpy.tile(self.mean_vector, (frame_number, 1))
        std_matrix = numpy.tile(self.std_vector, (frame_number, 1))
        
        norm_features = (in_data - mean_matrix) / std_matrix
        self.DF_IO.save_data_file(norm_features, out_file_name)

    def denorm_file(self, in_file_name, out_file_name, norm_info_file=None, feat_name='cmp', feat_dim=None):
        '''
        if norm_info_file is None: self.mean_vector and self.std_vector already loaded
        if feat_dim is None: extract from cfg
        '''
        if norm_info_file is not None:
            self.load_mean_std_values(norm_info_file, feat_dim)
        if feat_dim is None:
            feat_dim = self.cfg.nn_feature_dims[feat_name]

        # Floor std
        self.std_vector[self.std_vector<=0] = 1.

        in_data, frame_number = self.DF_IO.load_data_file_frame(in_file_name, feat_dim)

        mean_matrix = numpy.tile(self.mean_vector, (frame_number, 1))
        std_matrix = numpy.tile(self.std_vector, (frame_number, 1))
        
        denorm_features = in_data * std_matrix + mean_matrix
        self.DF_IO.save_data_file(denorm_features, out_file_name)


    def compute_mean(self, file_id_list=None, file_dir=None, feat_name='cmp', feat_dim=None):
        if file_id_list is None:
            file_id_list = read_file_list(self.cfg.file_id_list_file['compute_norm_info'])
        if file_dir is None:
            file_dir = self.cfg.nn_feat_resil_dirs[feat_name]
        if feat_dim is None:
            feat_dim = self.cfg.nn_feature_dims[feat_name]
        

        mean_vector = numpy.zeros(feat_dim)
        all_frame_number = 0.

        for file_id in file_id_list:
            speaker_id = file_id.split('_')[0]
            in_file_name = os.path.join(file_dir, speaker_id, file_id + '.'+feat_name)
            in_data, frame_number = self.DF_IO.load_data_file_frame(in_file_name, feat_dim)

            mean_vector += numpy.sum(in_data, axis=0)
            all_frame_number += frame_number
            
        mean_vector /= float(all_frame_number)

        self.logger.info('computed mean vector of length %d :' % mean_vector.shape[0] )
        self.logger.info('mean: %s' % mean_vector)
        
        self.mean_vector = mean_vector
        
        return  mean_vector

    def compute_std(self, file_id_list=None, file_dir=None, feat_name='cmp', feat_dim=None):
        if file_id_list is None:
            file_id_list = read_file_list(self.cfg.file_id_list_file['compute_norm_info'])
        if file_dir is None:
            file_dir = self.cfg.nn_feat_resil_dirs[feat_name]
        if feat_dim is None:
            feat_dim = self.cfg.nn_feature_dims[feat_name]

        var_vector = numpy.zeros(feat_dim)
        all_frame_number = 0.

        for file_id in file_id_list:
            speaker_id = file_id.split('_')[0]
            in_file_name = os.path.join(file_dir, speaker_id, file_id + '.'+feat_name)
            in_data, frame_number = self.DF_IO.load_data_file_frame(in_file_name, feat_dim)

            mean_matrix = numpy.tile(self.mean_vector, (frame_number, 1))
            var_vector += numpy.sum((in_data - mean_matrix) ** 2, axis=0)
            all_frame_number += frame_number

        var_vector /= float(all_frame_number)
        std_vector = var_vector ** 0.5

        self.logger.info('computed std vector of length %d :' % std_vector.shape[0] )
        self.logger.info('std: %s' % std_vector)
        
        self.std_vector = std_vector
        
        return  std_vector
    
#######
# Lab #
#######

class MinMaxNormalisation(object):
    def __init__(self, feature_dimension, min_value = 0.01, max_value = 0.99, min_vector = 0.0, max_vector = 0.0, exclude_columns=[]):

        # this is the wrong name for this logger because we can also normalise labels here too
        logger = logging.getLogger("acoustic_norm")

        self.target_min_value = min_value
        self.target_max_value = max_value

        self.feature_dimension = feature_dimension

        self.min_vector = min_vector
        self.max_vector = max_vector

        self.exclude_columns = exclude_columns

        if type(min_vector) != float:
            try:
                assert( len(self.min_vector) == self.feature_dimension)
            except AssertionError:
                logger.critical('inconsistent feature_dimension (%d) and length of min_vector (%d)' % (self.feature_dimension,len(self.min_vector)))
                raise
            
        if type(max_vector) != float:
            try:
                assert( len(self.max_vector) == self.feature_dimension)
            except AssertionError:
                logger.critical('inconsistent feature_dimension (%d) and length of max_vector (%d)' % (self.feature_dimension,len(self.max_vector)))
                raise

        logger.debug('MinMaxNormalisation created for feature dimension of %d' % self.feature_dimension)

    def load_min_max_values(self, label_norm_file):

        logger = logging.getLogger("acoustic_norm")

        io_funcs = BinaryIOCollection()
        min_max_vector, frame_number = io_funcs.load_binary_file_frame(label_norm_file, 1)
        min_max_vector = numpy.reshape(min_max_vector, (-1, ))
        feature_dimension = int(frame_number/2)
        self.min_vector = min_max_vector[0:feature_dimension]
        self.max_vector = min_max_vector[feature_dimension:]

        logger.info('Loaded min max values from the trained data for feature dimension of %d' % self.feature_dimension)

    def find_min_max_values(self, in_file_list):

        logger = logging.getLogger("acoustic_norm")

        file_number = len(in_file_list)
        min_value_matrix = numpy.zeros((file_number, self.feature_dimension))
        max_value_matrix = numpy.zeros((file_number, self.feature_dimension))
        io_funcs = BinaryIOCollection()
        for i in range(file_number):
            features = io_funcs.load_binary_file(in_file_list[i], self.feature_dimension)
            
            temp_min = numpy.amin(features, axis = 0)
            temp_max = numpy.amax(features, axis = 0)
            
            min_value_matrix[i, ] = temp_min;
            max_value_matrix[i, ] = temp_max;

        self.min_vector = numpy.amin(min_value_matrix, axis = 0)
        self.max_vector = numpy.amax(max_value_matrix, axis = 0)
        self.min_vector = numpy.reshape(self.min_vector, (1, self.feature_dimension))
        self.max_vector = numpy.reshape(self.max_vector, (1, self.feature_dimension))

        # po=numpy.get_printoptions()
        # numpy.set_printoptions(precision=2, threshold=20, linewidth=1000, edgeitems=4)
        logger.info('across %d files found min/max values of length %d:' % (file_number,self.feature_dimension) )
        logger.info('  min: %s' % self.min_vector)
        logger.info('  max: %s' % self.max_vector)
        # restore the print options
        # numpy.set_printoptions(po)

    def normalise_data(self, in_file_list, out_file_list):
        file_number = len(in_file_list)

        fea_max_min_diff = self.max_vector - self.min_vector
        diff_value = self.target_max_value - self.target_min_value
        fea_max_min_diff = numpy.reshape(fea_max_min_diff, (1, self.feature_dimension))

        target_max_min_diff = numpy.zeros((1, self.feature_dimension))
        target_max_min_diff.fill(diff_value)
        
        target_max_min_diff[fea_max_min_diff <= 0.0] = 1.0
        fea_max_min_diff[fea_max_min_diff <= 0.0] = 1.0
        
        io_funcs = BinaryIOCollection()
        for i in range(file_number):
            features = io_funcs.load_binary_file(in_file_list[i], self.feature_dimension)

            frame_number = int(features.size / self.feature_dimension)
            fea_min_matrix = numpy.tile(self.min_vector, (frame_number, 1))
            target_min_matrix = numpy.tile(self.target_min_value, (frame_number, self.feature_dimension))
            
            fea_diff_matrix = numpy.tile(fea_max_min_diff, (frame_number, 1))
            diff_norm_matrix = numpy.tile(target_max_min_diff, (frame_number, 1)) / fea_diff_matrix

            norm_features = diff_norm_matrix * (features - fea_min_matrix) + target_min_matrix

            ## If we are to keep some columns unnormalised, use advanced indexing to 
            ## reinstate original values:
            m,n = numpy.shape(features)
            for col in self.exclude_columns:
                norm_features[range(m),[col]*m] = features[range(m),[col]*m]
                
            io_funcs.array_to_binary_file(norm_features, out_file_list[i])

class Data_Min_Max_Normaliser(object):
    """docstring for Data_Normaliser"""
    def __init__(self, cfg=None):
        super(Data_Normaliser, self).__init__()
        self.logger = make_logger("DataNorm")

        self.cfg = cfg
        self.DF_IO = Data_File_IO(cfg)

    def compute_min_max_normaliser(self, feature_dim, in_file_list, norm_file, min_value=0.01, max_value=0.99):
        self.logger.info("compute_min_max_normaliser")
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
        self.logger.info('saved %s vectors to %s' %(min_vector.size, norm_file))

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


    def perform_min_max_normlisation_list(self, feature_dim, norm_file, in_file_list, out_file_list, min_value=0.01, max_value=0.99):
        logger = make_logger("perform_min_max_normlisation_list")
        from frontend.min_max_norm import MinMaxNormalisation
        min_max_normaliser = MinMaxNormalisation(feature_dimension=feature_dim, min_value=min_value, max_value=max_value)
        if norm_file is None:
            self.compute_min_max_normaliser(feature_dim, in_file_list, norm_file, min_value, max_value)
        min_max_normaliser.load_min_max_values(norm_file)
        min_max_normaliser.normalise_data(in_file_list, out_file_list)

    def perform_min_max_denormlisation_list(self, feature_dim, norm_file, in_file_list, out_file_list, min_value=0.01, max_value=0.99):
        from frontend.min_max_norm import MinMaxNormalisation
        min_max_normaliser = MinMaxNormalisation(feature_dimension=feature_dim, min_value=min_value, max_value=max_value)
        min_max_normaliser.load_min_max_values(norm_file)
        min_max_normaliser.denormalise_data(in_file_list, out_file_list)

    def norm_nn_file_list(feat_name, cfg, file_id_list, nn_resil_file_list={}, nn_resil_norm_file_list={}, compute_normaliser=True, norm_type='MinMax'):
        held_out_file_number = cfg.held_out_file_number
        try:    nn_resil_file_list[feat_name]
        except: nn_resil_file_list[feat_name] = prepare_file_path_list(file_id_list, cfg.nn_feat_resil_dirs[feat_name], '.'+feat_name)
        try:    nn_resil_norm_file_list[feat_name]
        except: nn_resil_norm_file_list[feat_name] = prepare_file_path_list(file_id_list, cfg.nn_feat_resil_norm_dirs[feat_name], '.'+feat_name)
        
        if compute_normaliser:
            nn_resil_file_list[feat_name+'_train'] = keep_by_speaker(nn_resil_file_list[feat_name], cfg.speaker_id_list_dict['train'])
            nn_resil_file_list[feat_name+'_train'] = remove_by_file_number(nn_resil_file_list[feat_name+'_train'], held_out_file_number)
            if norm_type == 'MinMax':
                if feat_name == 'wav':
                    print('Computing %s norm file for wav' % norm_type)
                    from modules import make_wav_min_max_normaliser
                    make_wav_min_max_normaliser(cfg.nn_feat_resil_norm_files[feat_name], cfg.nn_feature_dims[feat_name])
                else:
                    print('Computing %s norm file' % norm_type)
                    from modules import compute_min_max_normaliser
                    compute_min_max_normaliser(cfg.nn_feature_dims[feat_name], nn_resil_file_list[feat_name+'_train'], cfg.nn_feat_resil_norm_files[feat_name], min_value=0.01, max_value=0.99)
            elif norm_type == 'MeanVar':
                from modules import compute_mean_var_normaliser
                if feat_name == 'cmp':
                    print('Computing %s norm file for cmp' % norm_type)
                    var_file_dict = cfg.var_file_dict
                    acoustic_out_dimension_dict = cfg.acoustic_out_dimension_dict
                else:
                    print('Computing %s norm file' % norm_type)
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

############
# Waveform #
############

class Data_Wav_Min_Max_Normaliser(object):
    """
    This normaliser is for waveform data only
    1. Treat data as 1D
    2. Input Min-Max: +-2e15=32768, 16-bit signed integer PCM
    3. Output Min-Max: +-4., according to estimated std
    """
    def __init__(self, cfg=None):
        super(Data_Wav_Min_Max_Normaliser, self).__init__()
        self.logger = make_logger("DataNorm")

        self.cfg = cfg
        self.DF_IO = Data_File_IO(cfg)

        self.load_min_max_values()

    def load_min_max_values(self):
        '''
        Only values here, for waveform 1D data
        '''
        self.wav_max  = 32768.
        self.wav_min  = -32768.
        self.wav_diff = self.wav_max - self.wav_min

    def norm_file(self, in_file_name, out_file_name, min_value=-4., max_value=4.):
        value_diff = max_value - min_value
        in_data, in_num_samples = self.DF_IO.load_data_file_frame(in_file_name, 1)
        out_data = (in_data - self.wav_min) * (value_diff / self.wav_diff) + min_value
        self.DF_IO.save_data_file(out_data, out_file_name)

    def denorm_file(self, in_file_name, out_file_name, min_value=-4., max_value=4.):
        value_diff = max_value - min_value
        in_data, in_num_samples = self.DF_IO.load_data_file_frame(in_file_name, 1)
        # out_data = (in_data - self.wav_min) * (value_diff / self.wav_diff) + min_value
        out_data = (in_data - min_value) * (self.wav_diff / value_diff) + self.wav_min
        self.DF_IO.save_data_file(out_data, out_file_name)

class Data_Wav_List_Min_Max_Normaliser(object):
    """docstring for Data_Silence_List_Reducer"""
    def __init__(self, cfg=None):
        super(Data_Wav_List_Min_Max_Normaliser, self).__init__()
        self.logger = make_logger("DataNorm_list")

        self.cfg = cfg
        self.DWMMN = Data_Wav_Min_Max_Normaliser(cfg)

        file_id_list_file = cfg.file_id_list_file['used']
        # file_id_list_file = cfg.file_id_dv_test_list_file
        
        self.logger.info('Reading file list from %s' % file_id_list_file)
        self.file_id_list = read_file_list(file_id_list_file)

    def norm_file_list(self):

        cfg = self.cfg
        feat_name = 'wav'

        in_file_dir = cfg.nn_feat_resil_dirs[feat_name]
        out_file_dir = cfg.nn_feat_resil_norm_dirs[feat_name]
        file_ext = '.' + feat_name


        for file_id in self.file_id_list:
            in_file_name  = os.path.join(in_file_dir,  file_id + file_ext)
            out_file_name = os.path.join(out_file_dir, file_id + file_ext)

            self.logger.info('Saving to file %s' % out_file_name)
            self.DWMMN.norm_file(in_file_name, out_file_name)

        
#########################
# Main function to call #
#########################

def run_Data_List_Normaliser(cfg):
    ''' 
    Normalise waveform
    TODO: other features; MinMax for label, MeanVar for vocoder
    '''
    DWMMN_List = Data_Wav_List_Min_Max_Normaliser(cfg)
    DWMMN_List.norm_file_list()
