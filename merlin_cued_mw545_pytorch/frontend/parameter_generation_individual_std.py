################################################################################
#           The Neural Network (NN) based Speech Synthesis System
#                https://svn.ecdf.ed.ac.uk/repo/inf/dnn_tts/
#                
#                Centre for Speech Technology Research                 
#                     University of Edinburgh, UK                       
#                      Copyright (c) 2014-2015
#                        All Rights Reserved.                           
#                                                                       
# The system as a whole and most of the files in it are distributed
# under the following copyright and conditions
#
#  Permission is hereby granted, free of charge, to use and distribute  
#  this software and its documentation without restriction, including   
#  without limitation the rights to use, copy, modify, merge, publish,  
#  distribute, sublicense, and/or sell copies of this work, and to      
#  permit persons to whom this work is furnished to do so, subject to   
#  the following conditions:
#  
#   - Redistributions of source code must retain the above copyright  
#     notice, this list of conditions and the following disclaimer.   
#   - Redistributions in binary form must reproduce the above         
#     copyright notice, this list of conditions and the following     
#     disclaimer in the documentation and/or other materials provided 
#     with the distribution.                                          
#   - The authors' names may not be used to endorse or promote products derived 
#     from this software without specific prior written permission.   
#                                  
#  THE UNIVERSITY OF EDINBURGH AND THE CONTRIBUTORS TO THIS WORK        
#  DISCLAIM ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING      
#  ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO EVENT   
#  SHALL THE UNIVERSITY OF EDINBURGH NOR THE CONTRIBUTORS BE LIABLE     
#  FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES    
#  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN   
#  AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,          
#  ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF       
#  THIS SOFTWARE.
################################################################################

## Added FAST_MLPG as a variable here, in case someone wants to use the slow version, but perhaps we 
## should always use the bandmat version?
FAST_MLPG = True
#io_funcs.

from io_funcs.binary_io import  BinaryIOCollection
import os, numpy
import logging

if FAST_MLPG:
    from mlpg_fast_new import MLParameterGenerationFast as MLParameterGeneration
else:
    from mlpg import MLParameterGeneration

class   ParameterGeneration(object):

    def __init__(self, gen_wav_features = ['mgc', 'lf0', 'bap']):
        self.gen_wav_features = gen_wav_features
        self.inf_float = -1.0e+10

        # not really necessary to have the logger rembered in the class - can easily obtain it by name instead
        # self.logger = logging.getLogger('param_generation')

        self.var = {}
        
    def acoustic_decomposition(self, in_file_list, in_std_file_list, dimension, out_dimension_dict, file_extension_dict, var_file_dict):

        logger = logging.getLogger('param_generation')

        logger.debug('acoustic_decomposition for %d files' % len(in_file_list) )

        self.load_covariance(var_file_dict, out_dimension_dict)

        stream_start_index = {}
        dimension_index = 0
        recorded_vuv = False
        vuv_dimension = None

        for feature_name in out_dimension_dict.keys():
#            if feature_name != 'vuv':
            stream_start_index[feature_name] = dimension_index
#            else:
#                vuv_dimension = dimension_index
#                recorded_vuv = True
            
            dimension_index += out_dimension_dict[feature_name]

        io_funcs = BinaryIOCollection()

        mlpg_algo = MLParameterGeneration()

        findex=0
        flen=len(in_file_list)
        for file_name in in_file_list:
            
            dir_name = os.path.dirname(file_name)
            file_id = os.path.splitext(os.path.basename(file_name))[0]

            features, frame_number = io_funcs.load_binary_file_frame(file_name, dimension)
            # DONE: read file for std
            std_features = io_funcs.load_binary_file(in_std_file_list[findex], dimension)
            
            logger.info('processing %4d of %4d: %s' % (findex,flen,file_name) )

            for feature_name in self.gen_wav_features:
                
                logger.debug(' feature: %s' % feature_name)
                
                current_features = features[:, stream_start_index[feature_name]:stream_start_index[feature_name]+out_dimension_dict[feature_name]]

                if FAST_MLPG:
                    ### fast version wants variance per frame, not single global one:
                    # DONE: choose which features' variance are from DNN output
                    if feature_name in ['mcep','mgc','bap','bndap', 'lf0', 'f0']:
                        var = (std_features[:, stream_start_index[feature_name]:stream_start_index[feature_name]+out_dimension_dict[feature_name]]) ** 2
                    else:
                        var = self.var[feature_name]
                        var = numpy.transpose(numpy.tile(var,frame_number))

                else:
                    # TODO: make a good version? Not necessary yet
                    var = self.var[feature_name]
                    
                gen_features, gen_features_std = mlpg_algo.generation(current_features, var, out_dimension_dict[feature_name]/3)

                logger.debug(' feature dimensions: %d by %d' %(gen_features.shape[0], gen_features.shape[1]))

                if feature_name in ['lf0', 'F0']:
                    if stream_start_index.has_key('vuv'):
                        vuv_feature = features[:, stream_start_index['vuv']:stream_start_index['vuv']+1]

                        for i in xrange(frame_number):
                            if vuv_feature[i, 0] < 0.5:
                                gen_features[i, 0] = self.inf_float

                new_file_name = os.path.join(dir_name, file_id + file_extension_dict[feature_name])
                new_std_file_name = new_file_name + '_std'

                io_funcs.array_to_binary_file(gen_features, new_file_name)
                logger.debug(' wrote to file %s' % new_file_name)
                io_funcs.array_to_binary_file(gen_features_std, new_std_file_name)
                logger.debug(' wrote to file %s' % new_std_file_name)

            findex=findex+1


    def load_covariance(self, var_file_dict, out_dimension_dict):

        io_funcs = BinaryIOCollection()
        for feature_name in var_file_dict.keys():
            var_values, dimension = io_funcs.load_binary_file_frame(var_file_dict[feature_name], 1)

            var_values = numpy.reshape(var_values, (out_dimension_dict[feature_name], 1))

            self.var[feature_name] = var_values
